"""Complex road-constrained calibration trials, retaining independent-band controls."""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from coherent_fit import CoherentProblem,information
from fit_geometry import observations
from geographic_io import read_profiles
from phase_fit import load_histories
from range_fit import load_fields
from road_geometry import RoadCurve
from road_model import RoadTrajectory,RoadProblem
from transmitter_config import candidate_indices
from site_hypotheses import licensed_geometry
from terrain import RasterTerrain
from complex_likelihood import projected_profile
from calibration_solver import solve


def load_trial(saved,routes,histories,patches,templates,allow_time_dependent=False):
    """Rebuild the deterministic likelihood associated with a saved trial."""
    if saved.get('amplitude_block_seconds') and not allow_time_dependent:
        raise ValueError('This fit has time-dependent scattering coefficients; use its block likelihood, not a constant-amplitude model')
    sites=np.array(saved['site_positions_enu_m']);offsets=np.array(saved['channel_offsets_enu_m'])
    roads=json.loads(Path(routes).read_text())['routes'];_,records=read_profiles(templates)
    problems=[]
    for label,coarse,selected in zip(saved['labels'],observations(histories,saved['labels']),saved['selected_roads']):
        channels,description=load_histories(histories,label,1.)
        model=RoadTrajectory(channels[1]['seed'],RoadCurve(roads[selected['route']]),selected['coarse_state'],saved['assignment'],offsets)
        model.receiver_enu_m=np.array(saved.get('receiver_correction_enu_m',[0.,0.,0.]))
        model.receiver_channel_offsets_enu_m=np.array(saved.get('receiver_channel_offsets_enu_m',np.zeros((5,3))))
        problems.append(RoadProblem(channels,coarse,sites,load_fields(patches,label,description,records),model))
    return CoherentProblem(problems,range_bias=saved.get('range_bias',False),
                           site_prior_sigma=np.array(saved['site_prior_sigma_m']))


def seed_state(problem,channel=None):
    model=problem.model
    # Smooth the reference phase only inside the explicitly retained Doppler
    # subspace. This seeds the fit; all bands/range cells decide the final fit.
    strengths=[]
    for d in problem.channels:
        strengths.append(np.linalg.norm(d['operator']@d['value'])**2)
    ch=int(np.argmax(strengths)) if channel is None else channel
    d=problem.channels[ch]
    q,b=d['operator'],d['basis']
    smoothed=b@np.linalg.lstsq(q@b,q@d['value'],rcond=None)[0]
    phase=np.unwrap(np.angle(smoothed))
    phase-=np.interp(model.seed['epoch'],d['time'],phase)
    p=np.zeros(model.count)
    middle=model.paths(p,np.array([model.seed['epoch']]),np.array([ch]),problem.sites)[0][0]
    wanted=middle+(d['seed_delta']-np.interp(model.seed['epoch'],d['time'],d['seed_delta']))-phase/d['k']
    lo,hi=model.bounds()
    for _ in range(3):
        path,j=model.paths(p,d['time'],np.full(len(d['time']),ch),problem.sites)
        w=np.minimum(abs(smoothed)/np.sqrt(d['variance']),30.)
        delta=np.linalg.lstsq(j[:,:model.radial_count]*w[:,None],(wanted-path)*w,rcond=None)[0]
        p[:model.radial_count]=np.clip(p[:model.radial_count]+delta,lo[:model.radial_count]+1e-6,hi[:model.radial_count]-1e-6)
    return p


def describe(problem,p):
    pieces=[]
    for ch in range(5):
        u,_=problem.channel_model(p,ch)
        r,a=projected_profile(problem.channel_data(ch),u)
        pieces.append({'captured_energy':float(abs(a)**2*np.vdot(u,u).real),
            'data_energy':float(np.vdot(problem.channel_data(ch),problem.channel_data(ch)).real),
            'residual_energy':float(np.vdot(r,r).real/2),'amplitude':[float(a.real),float(a.imag)]})
    values,_=problem.model.kinematics(p,problem.sites)
    mid=problem.model.positions(p,np.array([problem.model.seed['epoch']]),problem.sites)[0][0]
    return {'channels':pieces,'max_speed_mps':float(np.linalg.norm(values[0],axis=1).max()),
        'acceleration_rms_mps2':float(np.sqrt(np.mean(np.sum(values[1]**2,axis=1)))),
        'jerk_rms_mps3':float(np.sqrt(np.mean(np.sum(values[2]**2,axis=1)))),
        'midpoint_enu_m':mid.tolist(),'lateral_offset_m':float(p[problem.model.radial_count]),
        'lateral_slope_m_per_half_interval':float(p[problem.model.radial_count+1]),
        'height_above_dem_m':float(2+p[problem.model.radial_count+2])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('routes','histories','patches','templates','search','output','prior-directory','terrain','dem'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--labels',nargs='+',required=True)
    parser.add_argument('--assignments',nargs='+',default=['00002'],help='Five site indices, or all')
    parser.add_argument('--roads-per-target',type=int,default=4)
    parser.add_argument('--road-names',nargs='+',help='Explicit road-association sensitivity hypothesis; all selected trajectories must use one of these names')
    parser.add_argument('--iterations',type=int,default=120)
    parser.add_argument('--joint',action='store_true')
    parser.add_argument('--joint-starts',type=int,default=12)
    parser.add_argument('--solver',choices=['normal','sparse'],default='normal')
    parser.add_argument('--joint-refine',type=int,default=3)
    parser.add_argument('--joint-tag',default='joint')
    parser.add_argument('--require-independent',action='store_true',help='Only run joint trials whose independent road fits are already saved')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    geometry=json.loads((args.prior_directory/'geometry-prior.json').read_text())
    terrain=RasterTerrain(args.terrain,geometry,args.dem)
    roads=[RoadCurve(r) for r in json.loads(args.routes.read_text())['routes']]
    _,records=read_profiles(args.templates)
    data=[];clock=time.monotonic()
    for label,coarse in zip(args.labels,observations(args.histories,args.labels)):
        channels,description=load_histories(args.histories,label,1.)
        fields=load_fields(args.patches,label,description,records)
        data.append((label,coarse,channels,description,fields))
    assignments=list(itertools.product(*candidate_indices(geometry))) if args.assignments==['all'] else [tuple(map(int,s)) for s in args.assignments]
    warm={}
    for saved in sorted(args.output.glob('?????-?.json')):
        result=json.loads(saved.read_text())
        if result.get('initializer_version')!=2:continue
        for trial in result['trials']:
            warm[(result['label'],trial['route'])]=(trial['coarse_state'],trial['state'])
    print('Loaded fields',len(data),'elapsed',round(time.monotonic()-clock,2),flush=True)
    for assignment in assignments:
        code=''.join(map(str,assignment))
        if args.require_independent and any(not (args.output/f'{code}-{label}.json').exists() for label in args.labels):
            print(code,'waiting for independent road fits; skipped in this pass',flush=True)
            continue
        sites,offsets,channel_descriptions=licensed_geometry(args.prior_directory,args.terrain,assignment,terrain)
        screens=json.loads((args.search/f'sites-{code}.json').read_text())
        screens={t['label']:t['trials'] for t in screens['targets']}
        problems,states,selected=[],[],[]
        for label,coarse,channels,description,fields in data:
            path=args.output/f'{code}-{label}.json'
            if path.exists() and json.loads(path.read_text()).get('initializer_version')==2:
                trials=json.loads(path.read_text())['trials']
            else:
                trials=[]
            retained={trial['route'] for trial in trials}
            allowed=[screen for screen in screens[label] if not args.road_names or screen['name'] in args.road_names]
            missing=[screen for screen in allowed[:args.roads_per_target] if screen['route'] not in retained]
            if missing:
                for screen in missing:
                    try:
                        model=RoadTrajectory(channels[1]['seed'],roads[screen['route']],screen['state'],assignment,offsets)
                    except ValueError:
                        continue
                    problem=RoadProblem(channels,coarse,sites,fields,model)
                    seeds=[seed_state(problem,ch) for ch in (0,1,2)]
                    key=(label,screen['route'])
                    if key in warm:
                        old_coarse,old_state=warm[key]
                        initial=np.array(old_state)
                        t=np.linspace(-1,1,61)
                        change=np.polynomial.polynomial.polyval(t,np.array(old_coarse)-np.array(screen['state']))
                        initial[:model.radial_count]+=np.linalg.lstsq(model.spline(t),change,rcond=None)[0]
                        low,high=model.bounds()
                        seeds.append(np.clip(initial,low+1e-6,high-1e-6))
                    seeds.sort(key=lambda p:np.linalg.norm(problem.residual(p)))
                    fits=[problem.fit(start,max_nfev=args.iterations) for start in seeds[:2]]
                    fit=min(fits,key=lambda f:f.cost)
                    warm[key]=(screen['state'],fit.x.tolist())
                    result={'route':screen['route'],'name':screen['name'],'coarse_state':screen['state'],
                        'state':fit.x.tolist(),'cost':float(fit.cost),'converged':bool(fit.success),
                        'evaluations':fit.nfev,'optimality':float(fit.optimality),'description':describe(problem,fit.x)}
                    trials.append(result)
                    print(code,label,screen['name'],'cost',round(fit.cost,2),'n',fit.nfev,'speed',round(result['description']['max_speed_mps'],2),'elapsed',round(time.monotonic()-clock,2),flush=True)
                trials.sort(key=lambda t:t['cost'])
                path.write_text(json.dumps({'assignment':assignment,'label':label,'trials':trials,'initializer_version':2,
                    'status':'independent-band complex fit; candidate road association'},indent=2))
            eligible=[trial for trial in trials if not args.road_names or trial['name'] in args.road_names]
            if not eligible:
                raise ValueError(f'No fully supported road trajectory for {label} under {code}')
            best=min(eligible,key=lambda trial:trial['cost'])
            model=RoadTrajectory(channels[1]['seed'],roads[best['route']],best['coarse_state'],assignment,offsets)
            problems.append(RoadProblem(channels,coarse,sites,fields,model));states.append(np.array(best['state']));selected.append(best)
        if args.joint:
            path=args.output/f'{code}-{args.joint_tag}.json'
            if path.exists():
                cached=json.loads(path.read_text())
                if cached['labels']!=args.labels:
                    raise ValueError('Saved joint trial uses a different target set; select a new tag')
                if cached.get('road_association_names')!=args.road_names:
                    raise ValueError('Saved joint trial uses a different road hypothesis; select a new tag')
                continue
            problem=CoherentProblem(problems,site_prior_sigma=np.array([10.,10.,10.]))
            initial=problem.initial(states)
            candidates=problem.constant_phase_starts(initial,args.joint_starts)
            print(code,'phase starts retained',len(candidates),'best full cost',round(candidates[0][0],2),flush=True)
            low,high=problem.bounds();results=[]
            for cost,state in candidates[:args.joint_refine]:
                state=np.clip(state,low+1e-7,high-1e-7)
                def progress(intermediate_result):
                    if intermediate_result.nfev%10==0:
                        print(code,'joint progress',intermediate_result.nfev,round(intermediate_result.cost,2),flush=True)
                        checkpoint={'assignment':assignment,'labels':args.labels,'state':intermediate_result.x.tolist(),
                            'cost':float(intermediate_result.cost),'evaluations':intermediate_result.nfev,
                            'status':'incomplete optimizer checkpoint, not a calibration'}
                        (args.output/f'{code}-checkpoint.json').write_text(json.dumps(checkpoint))
                fit=(solve(problem.residual,state,problem.bounds(),args.iterations*2,progress) if args.solver=='normal'
                     else problem.fit(state,max_nfev=args.iterations*2,sparse=True,callback=progress))
                results.append(fit)
                print(code,'joint cost',round(fit.cost,2),'n',fit.nfev,'elapsed',round(time.monotonic()-clock,2),flush=True)
            best=min(results,key=lambda f:f.cost)
            result={'assignment':assignment,'channels':channel_descriptions,'labels':args.labels,
                'road_association_names':args.road_names,
                'state':best.x.tolist(),'cost':float(best.cost),'converged':bool(best.success),'evaluations':best.nfev,
                'solver':args.solver,'optimality':float(best.optimality),'termination':str(best.message),
                'site_prior_sigma_m':[10.,10.,10.],'site_positions_enu_m':sites.tolist(),
                'site_corrections_enu_m':(best.x[:9].reshape(3,3)*[30,30,75]).tolist(),
                'channel_offsets_enu_m':offsets.tolist(),'selected_roads':selected,
                'information':information(problem,best.x),
                'status':'road-constrained coherent trial; validation required before adoption'}
            path.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
