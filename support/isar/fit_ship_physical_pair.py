"""Test one common physical F01 echo with per-band overlapping-echo nuisance.

Antennas start fixed at user priors; an explicit refinement control can then
free them under the existing physical prior. All 32 component assignments
are considered for initialization using conditional
training-only local summaries. The chosen starts then fit original complex
observations, not those summaries. No geometric calibration is accepted.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from fit_joint_sea_arcs import prepare_problem
from motion import C
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_pair import SharedPhysicalPair


def build_model(case,origin_bounds=(-.8,2.8),label='F01'):
    i=next(i for i,e in enumerate(case['descriptions']) if e['label']==label)
    entry=case['descriptions'][i];midpoint=(entry['first_s']+entry['last_s'])/2
    g=case['geometries'][i]
    bands=[PhysicalAmplitudeArc(f,tr,knots,g,ch,midpoint) for ch,(f,tr,knots) in
        enumerate(zip(case['fields'][i],case['training'][i],case['knots'][i]))]
    return SharedPhysicalPair(bands,origin_bounds=origin_bounds),i


def initial_states(model,source):
    g=model.geometry;best=[ch['trials'][ch['best_index']] for ch in source['channels']]
    moments=[];whitening=[]
    for pair,row in zip(model.pairs,best):
        base=pair.secondary;u=(pair.base.midpoint-base.epoch)/base.scale
        transform=np.array([[C/8e6,0,u,u*u],[0,0,1/base.scale,2*u/base.scale],[0,0,0,2/base.scale**2]])
        covariance=np.asarray(row['conditional_covariance']);per_component=[]
        for k in range(2):
            cov=transform@covariance[k*4:(k+1)*4,k*4:(k+1)*4]@transform.T
            per_component.append(solve_triangular(np.linalg.cholesky(cov),np.eye(3),lower=True))
        moments.append(np.asarray(row['observables']));whitening.append(per_component)
    starts=[];lo,hi=g.bounds()
    for assignment in itertools.product(range(2),repeat=len(model.bands)):
        observed=np.array([m[k] for m,k in zip(moments,assignment)])
        def residual(q):
            prediction=g.prediction(np.r_[np.zeros(g.nantenna),q])
            return np.r_[np.concatenate([w[k]@(p-y) for w,k,p,y in zip(whitening,assignment,prediction,observed)]),q]
        fit=least_squares(residual,np.clip(np.zeros(10),lo[18:],hi[18:]),jac='cs',bounds=(lo[18:],hi[18:]),
            max_nfev=200,ftol=1e-11,xtol=1e-11,gtol=1e-9)
        starts.append(dict(assignment=list(assignment),cost=float(fit.fun@fit.fun),
            parameters=np.r_[np.zeros(18),fit.x].tolist(),optimality=float(fit.optimality)))
    return sorted(starts,key=lambda r:r['cost'])


def raw_seed(model,source,initial):
    best=[ch['trials'][ch['best_index']] for ch in source['channels']]
    seed=np.r_[initial['parameters'],np.zeros(5),np.zeros(4*len(model.bands))]
    for pair,index,row,component in zip(model.pairs,model.indices,best,initial['assignment']):
        secondary=np.asarray(row['parameters']).reshape(2,4)[1-component]
        seed[index[-4:]]=secondary
        raw,_=pair.base.response(seed[:model.primary_count])
        secondary=secondary.copy();secondary[1]=0
        other,_=pair.secondary.response(secondary)
        columns=[pair.base.nuisance.remove(pair.base.reduced(v[:,:,None])[pair.base.selected,0]) for v in (raw,other)]
        amplitude=np.linalg.lstsq(np.column_stack(columns),pair.base.clean,rcond=1e-10)[0]
        seed[model.geometry.count+pair.base.channel]=np.angle(amplitude[0])
        seed[index[-3]]=np.angle(amplitude[1])
    return seed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/ship-amplitude-pair-F01/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-physical-pair-F01'))
    parser.add_argument('--starts',type=int,default=3)
    parser.add_argument('--max-evaluations',type=int,default=300)
    parser.add_argument('--seed-fit',type=Path,help='Continue each saved primary/secondary solution')
    parser.add_argument('--free-antennas',action='store_true')
    parser.add_argument('--origin-bounds',nargs=2,type=float,default=[-.8,2.8],metavar=('LOW','HIGH'))
    parser.add_argument('--interval',choices=['F01','F03','Flash'],default='F01')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);output=args.output/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    source=json.loads(args.source.read_text());case=prepare_problem();model,i=build_model(case,args.origin_bounds,args.interval)
    if source['interval']['label']!=args.interval:raise ValueError('Source components belong to another interval')
    continuation=json.loads(args.seed_fit.read_text()) if args.seed_fit else None
    if continuation is not None and not continuation['complete']:raise ValueError('Seed fits must be complete')
    starts=continuation['all_initial_states'] if continuation is not None else initial_states(model,source)
    files=[args.source,Path(__file__),case['track_path'],case['geometry_directory']/'geometry-prior.json',
        case['geometry_directory']/'sites-main.json',case['geometry_directory']/'acma-transmitters.json']
    files += [Path(__file__).with_name(n+'.py') for n in ('amplitude_pair','physical_pair','physical_amplitude_arc',
        'amplitude_arc','field','fit_joint_sea_arcs','flash_geometry')]
    if args.seed_fit:files.append(args.seed_fit)
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,assignment='00002',
        source=str(args.source),interval=case['descriptions'][i],all_initial_states=starts,trials=[],
        antenna_policy=('Refine antennas within the existing physical prior and bounds.' if args.free_antennas else
                        'Fixed user-supplied coordinates, zero antenna corrections.'),
        free_antennas=args.free_antennas,secondary_origin_bounds_samples=args.origin_bounds,
        seed_fit=str(args.seed_fit) if args.seed_fit else None,
        validation_scope='Unused blocks in already inspected data; paired conditional prediction, not untouched external validation.',
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files})
    output.write_text(json.dumps(result,indent=2,allow_nan=False));started=time.perf_counter()
    for rank,initial in enumerate(starts[:args.starts]):
        seed=raw_seed(model,source,initial) if continuation is None else np.asarray(continuation['trials'][rank]['parameters'])
        value,gradient=model.objective(seed);direction=np.cos(np.arange(model.count))*.01
        if not args.free_antennas:direction[:18]=0
        step=5e-4
        finite=(model.objective(seed+step*direction)[0]-model.objective(seed-step*direction)[0])/(2*step)
        error=float(abs(finite-gradient@direction)/max(abs(finite),1e-8))
        if error>2e-4:raise ValueError(f'Raw objective derivative check failed: {error}')
        print('starting',rank,initial['assignment'],'cost',value*model.energy,'gradient_error',error,flush=True)
        parameters,audit=model.fit(seed,free_antennas=args.free_antennas,max_evaluations=args.max_evaluations)
        g=model.geometry;state=g.state(parameters[:g.count]);sigma=g.nuisance_sigma
        prediction=model.prediction(parameters,case['testing'][i])
        record=dict(start_rank=rank,initial_state=initial,initial_objective=value,
            initial_directional_gradient_relative_error=error,parameters=parameters.tolist(),audit=audit,
            data_energy=model.energy,cost=audit['objective']*model.energy,prediction=prediction,
            primary_observables=g.prediction(parameters[:g.count]).tolist(),information=model.information(parameters),
            antenna_fit=g.antennas.describe(parameters[:18]),
            target_state=dict(position_correction_enu_m=(parameters[18:21]*sigma[:3]).tolist(),
                velocity_correction_enu_mps=(parameters[21:24]*sigma[3:6]).tolist(),
                acceleration_correction_enu_mps2=(parameters[24:27]*sigma[6:9]).tolist(),
                ais_shift_s=float(parameters[27]*sigma[-1]),position_local_enu_m=state[0].tolist(),
                velocity_local_enu_mps=state[1].tolist(),acceleration_local_enu_mps2=state[2].tolist()))
        result['trials'].append(record);result['seconds']=time.perf_counter()-started
        output.write_text(json.dumps(result,indent=2,allow_nan=False))
        print('finished',rank,'cost',record['cost'],'gradient',audit['projected_gradient'],
            'bounds',audit['bound_indices'],'heldout',sum(p['heldout_error'] for p in prediction),flush=True)
    result['complete']=True;output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
