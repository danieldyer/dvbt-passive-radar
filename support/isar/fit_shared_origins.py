"""Matched control for a common per-band vehicle/ship envelope origin.

Keep the same observations, weights, roads, motion bounds, antenna prior and
parameter count. The five ship-origin coordinates now also affect every
vehicle's envelope. Their existing seed-centred bounds are retained explicitly;
these limits are not surveyed uncertainty on the physical delay reference.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from fit_physical_profile import solve
from fit_shared_target_geometry import load_experiment
from scene_dictionary import fingerprint
from shared_origin_geometry import SharedOriginGeometry
from stationarity import check
from validate_motion import summarize


def predictions(joint,original,parameters,labels):
    ship=original.validation(parameters[:joint.ship_count])
    vehicles={label:road.predict(parameters[section],parameters[joint.antenna_slice],parameters[joint.range_slice])
        for label,road,section in zip(labels,joint.roads,joint.road_slices)}
    return {'ship_channels':ship,'ship_summary':summarize(ship),'vehicles':vehicles,
        'scope':'Same withheld blocks as the legacy experiment; only the forward-model origin coupling changes.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('control','ship-fit','road-fit','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--fields',type=Path,default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--initial',type=Path)
    parser.add_argument('--origin-start',choices=['zero','inherited'],default='zero')
    parser.add_argument('--fixed-antennas',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--iterations',type=int,default=180)
    args=parser.parse_args();started=time.monotonic()
    if args.output.exists():raise ValueError('Preserve the existing experiment result')
    control=json.loads(args.control.read_text())
    if not control['complete'] or not control['fit']['converged']:raise ValueError('Use a completed stationary legacy control')
    for group in ('sources','implementation_sha256'):
        for file,digest in control['identity'][group].items():
            if fingerprint(Path(file))!=digest:raise ValueError('Legacy input or implementation changed: '+file)
    for file in (args.ship_fit,args.road_fit):
        if control['identity']['sources'].get(str(file))!=fingerprint(file):raise ValueError('Use the original source fits')
    legacy,original,bounds,labels,dropped=load_experiment(args.ship_fit,args.road_fit,args.fields,args.geometry,
        control['identity']['labels'],control['identity']['nodes'])
    joint=SharedOriginGeometry(legacy)
    matrix,lo,hi,lower,upper=bounds
    if args.fixed_antennas:lower[joint.antenna_slice]=upper[joint.antenna_slice]=0.
    files=[Path(__file__),Path(__file__).with_name('shared_origin_geometry.py'),
        Path('support/calibration/range_fit.py'),Path('support/calibration/road_model.py')]
    identity={'legacy_control':str(args.control),'legacy_control_sha256':fingerprint(args.control),
        'legacy_identity':control['identity'],'origin_model':'one_absolute_per_band_shared_by_ship_and_all_vehicles',
        'fixed_antennas':args.fixed_antennas,'labels':labels,'dropped':dropped,
        'implementation_sha256':{str(file):fingerprint(file) for file in files}}
    joint.initial=np.asarray(control['parameters']).copy()
    if args.origin_start=='zero':joint.initial[joint.range_slice]=0.
    if args.initial:
        seed=json.loads(args.initial.read_text())
        left,right=copy.deepcopy(seed['identity']),copy.deepcopy(identity)
        left.pop('fixed_antennas');right.pop('fixed_antennas')
        if left!=right:raise ValueError('Continuation changed the scientific objective or inputs')
        joint.initial=np.asarray(seed['parameters']).copy()
    if args.fixed_antennas:joint.initial[joint.antenna_slice]=0.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    base={'scope':__doc__,'identity':identity,'complete':False,'accepted_calibration':False,'yaw_fitted':False,
        'origin_start':args.origin_start,'initial_source':str(args.initial) if args.initial else str(args.control),
        'initial_source_sha256':fingerprint(args.initial or args.control),'parameter_count':len(joint.initial),
        'range_origin_bounds_m':[lower[joint.range_slice].tolist(),upper[joint.range_slice].tolist()],
        'legacy_vehicle_origins_m':[road.origins.tolist() for road in legacy.roads],
        'range_slice':[joint.range_slice.start,joint.range_slice.stop],
        'antenna_slice':[joint.antenna_slice.start,joint.antenna_slice.stop],
        'road_slices':[[s.start,s.stop] for s in joint.road_slices]}
    print(json.dumps({'stage':'loaded','parameters':len(joint.initial),'seconds':time.monotonic()-started}),flush=True)
    if args.audit_only:
        point=joint.initial.copy();point[joint.range_slice]=np.linspace(-1.3,1.7,len(joint.roads[0].training))
        result=joint.evaluate(point);checks=[]
        indices=[joint.antenna_slice.start,joint.antenna_slice.stop-1,*range(joint.range_slice.start,joint.range_slice.stop),joint.road_slices[0].start]
        for index in indices:
            delta=np.eye(len(point))[index]*1e-4
            numeric=(joint.objective(point+delta)[0]-joint.objective(point-delta)[0])/2e-4
            analytic=result['gradient'][index]
            if abs(numeric-analytic)>3e-7+abs(numeric)*5e-5:raise ValueError('Recorded-data gradient mismatch')
            checks.append({'parameter':index,'analytic':float(analytic),'numeric':float(numeric),'absolute_error':float(abs(numeric-analytic))})
        zero=point.copy();zero[joint.range_slice]=0.
        equivalence=joint.objective(zero)[0]-legacy.objective(zero)[0]
        if abs(equivalence)>1e-12:raise ValueError('Zero-origin control changed the legacy likelihood')
        base.update(complete=True,kind='recorded_gradient_audit',checks=checks,zero_origin_objective_delta=equivalence,
                    seconds=time.monotonic()-started)
        args.output.write_text(json.dumps(base,indent=2)+'\n');print(json.dumps({'stage':'audit_complete','checks':checks}),flush=True);return
    def checkpoint(parameters,trace):
        path=args.output.with_suffix('.checkpoint.json');temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({**base,'parameters':parameters.tolist(),'trace':trace},indent=2)+'\n');temporary.replace(path)
        print(json.dumps({'stage':'fit',**trace[-1]}),flush=True)
    parameters,fit=solve(joint,matrix,lo,hi,lower,upper,iterations=args.iterations,checkpoint=checkpoint)
    exact=original.evaluate(parameters[:joint.ship_count]);fast=joint.ship.evaluate(parameters[:joint.ship_count])
    gradient=joint.evaluate(parameters)['gradient'].copy()
    gradient[:joint.ship_count]+=(exact['gradient']-fast['gradient'])*original.energy/joint.energy
    kkt,violation=check(parameters,gradient,matrix,lo,hi,lower,upper)
    result={**base,'complete':True,'parameters':parameters.tolist(),'fit':fit,
        'range_origins_m':parameters[joint.range_slice].tolist(),
        'antenna_fit':joint.ship.antennas.describe(parameters[joint.antenna_slice]),
        'antenna_bound_hits':int(np.sum(abs(parameters[joint.antenna_slice])>3-1e-5)),
        'original_row_maximum_kkt_residual':kkt,'original_row_constraint_violation':violation,
        'validation':predictions(joint,original,parameters,labels),'seconds':time.monotonic()-started}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'stage':'complete','converged':fit['converged'],'kkt':kkt,
        'range_origins_m':result['range_origins_m'],'antenna_bound_hits':result['antenna_bound_hits'],
        'ship_validation':result['validation']['ship_summary'],'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
