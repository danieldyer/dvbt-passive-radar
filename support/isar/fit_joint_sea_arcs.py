"""Compare direct and declared sea-assisted models with shared ship calibration.

Three disjoint early arcs share transmitter/receiver coordinates. Each has an
independent physical point state, a common physical scattering phase across
bands and real positive brightness knots. This is a conditional propagation
test, not an exhaustive site/cycle search or an adopted phase-centre survey.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar

from ais import read_csv
from cluster_phase import crop
from common_physical_arc import CommonPhysicalFit
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH
from imaging import sites
from joint_physical_arcs import JointPhysicalArcs
from physical_amplitude_arc import PhysicalAmplitudeArc,PhysicalAmplitudeFit
from sea_physical_arc import SeaPhysicalArc
from referenced_common_arc import ReferencedCommonArc
from ship_site_signature import licensed_options


def phase_start(model,parameters):
    """One-dimensional training-only start; no new phase freedom in the model."""
    def cost(phi):
        trial=parameters.copy();trial[-1]=phi
        return model.objective(trial)[0]
    grid=np.linspace(-np.pi,np.pi,33)[:-1]
    scores=[cost(phi) for phi in grid];index=int(np.argmin(scores));step=2*np.pi/32
    fit=minimize_scalar(cost,bounds=(grid[index]-step,grid[index]+step),method='bounded',options={'xatol':1e-9})
    result=parameters.copy();result[-1]=fit.x
    return result


def prepare_problem():
    root=Path('docs/data/isar_20261017');geometry_directory=Path('docs/data/calibration_20261016_sites')
    source_paths=[root/'ship-common-phase-polished/fits.json',root/'ship-phase-continuity/fits.json',
        root/'ship-common-phase-transfer/fits.json',root/'full-capture-ship-arc-validation/fits.json',
        Path('docs/data/isar_20261016/flash-reflection/analysis.json')]
    source,continuity,transfer,discovery,reflection=[json.loads(p.read_text()) for p in source_paths]
    flash=next(r for r in continuity['intervals'] if [r['first_s'],r['last_s']]==source['interval_s'])
    descriptions=[dict(label='Flash',first_s=flash['first_s'],last_s=flash['last_s'],centre_sample=0),
        *[next(c for c in discovery['candidates'] if c['label']==label) for label in ('F01','F03')]]
    _,receiver,_=sites(geometry_directory)
    rotation=np.asarray(json.loads((geometry_directory/'geometry-prior.json').read_text())['ecef_to_enu'])
    track_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(track_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    code=lambda a:''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a)
    assignment=next(a for a in assignments(licensed_options(geometry_directory)) if code(a)=='00002')
    water=reflection['water'];centre=np.asarray(water['centre_ecef_m']);radius=water['radius_m']
    fields=[];training=[];testing=[];knots=[];geometries=[];field_paths=[]
    for entry in descriptions:
        first,last=entry['first_s'],entry['last_s'];midpoint=(first+last)/2
        g=Flash(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,EPOCH+midpoint,rotation,receiver,assignment,
            receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.))
        p=g.receiver+g.state(np.zeros(g.count))[0]@g.rotation
        height=np.linalg.norm(p-centre)-radius
        # A conservative midpoint interval leaves room for allowed horizontal
        # displacement and vertical velocity/acceleration over the three seconds.
        sensitivity=20*np.dot((p-centre)/np.linalg.norm(p-centre),g.rotation[2])
        g.lower[20]=max(g.lower[20],(8.-height)/sensitivity)
        g.upper[20]=min(g.upper[20],(57.-height)/sensitivity)
        geometries.append(g);fs=[];tr=[];te=[];ks=[]
        for ch in range(5):
            path=Path(f'build/isar-20261015/full-fields/channel-{ch}.npz');field_paths.append(path)
            with np.load(path) as data:
                f=crop(data,first,last,np.arange(entry['centre_sample']-2,entry['centre_sample']+3))
            fs.append(f)
            if entry['label']=='Flash':
                row=flash['channels'][ch];fit=row['alternatives'][row['chosen_index']]
                tr.append(np.asarray(fit['test']['training']));te.append(np.asarray(fit['test']['testing']))
                ks.append(fit['test']['amplitude_knots_s'])
            else:
                tr.append(((f.time-first)//.125).astype(int)%2==0);te.append(~tr[-1])
                ks.append(np.r_[f.data['row_time'][0],np.arange(first+.5,last,.5),f.data['row_time'][-1]])
        fields.append(fs);training.append(tr);testing.append(te);knots.append(ks)
    return {
        'source':source,
        'descriptions':descriptions,
        'geometries':geometries,
        'fields':fields,
        'training':training,
        'testing':testing,
        'knots':knots,
        'water':water,
        'source_paths':source_paths,
        'field_paths':field_paths,
        'track_path':track_path,
        'geometry_directory':geometry_directory,
    }


def build_arcs(case,family,reference_band=None):
    arcs=[]
    for entry,g,fs,tr,ks in zip(case['descriptions'],case['geometries'],case['fields'],case['training'],case['knots']):
        midpoint=(entry['first_s']+entry['last_s'])/2
        bands=[(PhysicalAmplitudeArc if family=='direct' else SeaPhysicalArc)(f,t,k,g,ch,midpoint,
            **({} if family=='direct' else dict(water=case['water'],family=family))) for ch,(f,t,k) in enumerate(zip(fs,tr,ks))]
        independent=PhysicalAmplitudeFit(bands)
        arcs.append(CommonPhysicalFit(independent) if reference_band is None else ReferencedCommonArc(independent,reference_band))
    return arcs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families',nargs='+',choices=['direct','transmit','both'],default=['direct','transmit'])
    parser.add_argument('--cycles',nargs='+',type=int,default=[1,8])
    parser.add_argument('--max-evaluations',type=int,default=400)
    parser.add_argument('--reference-band',type=int,choices=range(5),help='Exact midpoint conditioning of the common phase')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'fits.json').exists():raise ValueError('Use a fresh output')
    case=prepare_problem()
    source=case['source']
    transfer=json.loads((Path('docs/data/isar_20261017')/'ship-common-phase-transfer/fits.json').read_text())
    descriptions=case['descriptions']
    geometries=case['geometries']
    fields=case['fields']
    training=case['training']
    testing=case['testing']
    knots=case['knots']
    water=case['water']
    centre=np.asarray(water['centre_ecef_m']);radius=water['radius_m']
    source_paths=case['source_paths']
    field_paths=case['field_paths']
    track_path=case['track_path']
    geometry_directory=case['geometry_directory']
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,assignment='00002',
        intervals=[{k:e[k] for k in ('label','first_s','last_s','centre_sample')} for e in descriptions],
        water=water,relative_permittivity=80.,conductivity_s_per_m=4.,trials=[],
        height_policy='Midpoint 8-57 m above declared water, with full-row extrema reported; same bounds in direct and sea models.',
        phase_policy='One common physical target phase per interval; no independent band phases or rate corrections.',
        validation_scope='Unused blocks in already inspected data; paired conditional prediction, not untouched external validation.')
    result['common_phase_reference_band']=args.reference_band
    started=time.perf_counter()
    for family in args.families:
        arcs=build_arcs(case,family,args.reference_band)
        joint=JointPhysicalArcs(arcs)
        for cycle in args.cycles:
            anchor=next(r for r in source['trials'] if r['assignment']=='00002' and r['branch_index']==cycle and r['converged'])
            seed_parts=[np.array(anchor['parameters'])[:18]]
            for entry,g,model in zip(descriptions,geometries,arcs):
                if entry['label']=='Flash':state=np.array(anchor['parameters'])[18:]
                else:
                    row=next(r for r in transfer['trials'] if r['label']==entry['label'] and r['calibration']==f'00002-cycle-{cycle}')
                    state=np.asarray(min(row['common_trials'],key=lambda r:r['cost'])['parameters'])
                local=np.r_[seed_parts[0],state]
                lo,hi=g.bounds();local[:-1]=np.clip(local[:-1],lo,hi)
                if args.reference_band is not None:local=model.from_physical(local)
                seed_parts.append(phase_start(model,local)[18:])
            seed=np.concatenate(seed_parts)
            initial=joint.objective(seed)[0]
            value,gradient=joint.objective(seed)
            direction=np.sin(np.arange(len(seed))+.3)*.01;step=1e-5
            finite=(joint.objective(seed+step*direction)[0]-joint.objective(seed-step*direction)[0])/(2*step)
            gradient_error=float(abs(finite-gradient@direction)/max(abs(finite),1e-8))
            if gradient_error>2e-4:raise ValueError(f'Joint raw gradient check failed: {gradient_error}')
            print('starting',family,cycle,'cost',round(initial*joint.energy,3),'gradient_error',gradient_error,flush=True)
            fitted,audit=joint.fit(seed,max_evaluations=args.max_evaluations)
            predictions=[];heights=[]
            for entry,g,model,index,tests in zip(descriptions,geometries,arcs,joint.indices,testing):
                local=fitted[index]
                predictions.append(dict(label=entry['label'],channels=model.prediction(local,tests),
                    parameters=local.tolist(),physical_parameters=(local if args.reference_band is None else model.physical_parameters(local)).tolist(),
                    observables=g.prediction(local[:-1]).tolist()))
                p,v,a,_,_=g.state(local[:-1]);elapsed=model.independent.bands[0].field.data['row_time']-(entry['first_s']+entry['last_s'])/2
                positions=g.receiver+(p+elapsed[:,None]*v+.5*elapsed[:,None]**2*a)@g.rotation
                h=np.linalg.norm(positions-centre,axis=1)-radius
                heights.append(dict(label=entry['label'],minimum_m=float(h.min()),maximum_m=float(h.max())))
            record=dict(family=family,seed_cycle=cycle,parameters=fitted.tolist(),initial_objective=initial,
                initial_directional_gradient_relative_error=gradient_error,
                cost=audit['objective']*joint.energy,data_energy=joint.energy,audit=audit,prediction=predictions,
                height_extrema=heights,antenna_fit=geometries[0].antennas.describe(fitted[:18]))
            result['trials'].append(record);result['seconds']=time.perf_counter()-started
            (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))
            print(family,cycle,'cost',round(record['cost'],3),'converged',audit['converged'],
                'KKT',audit['polishing'].get('final_kkt_residual'),'eval',audit['evaluations'],
                'test_error',round(sum(c['test_error'] for p in predictions for c in p['channels']),2),flush=True)
    result['complete']=True
    paths=[*source_paths,track_path,*set(field_paths),geometry_directory/'geometry-prior.json',
        geometry_directory/'sites-main.json',geometry_directory/'acma-transmitters.json',Path(__file__)]
    paths += [Path(__file__).with_name(n+'.py') for n in ('joint_physical_arcs','differentiable_sea','sea_physical_arc',
        'spherical_sea','profiled_arc_solver','common_physical_arc','referenced_common_arc','physical_amplitude_arc','amplitude_arc','field')]
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
