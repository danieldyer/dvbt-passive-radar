"""Fit separated ship apertures with fixed N/E axes and common range origins.

Each aperture starts from an existing radar trajectory and reflector support.
No fitted antenna correction or changing heading is imported. This supplies
additional measured views for the next shared-geometry experiment; it does
not itself establish common reflectivity across frequencies or calibrated sites.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from ais import utc_seconds
from export_shared_motion import PHYSICAL_KEYS
from field import Field
from fit_physical_profile import solve
from fixed_frame_motion import geographic_frame,restrict_interval
from full_capture_motion import constrained_curve
from full_motion import Band
from imaging import motion_state,sites
from model_quadrature import quadrature
from physical_profile import PhysicalProfile
from radar_rate_initialization import rate_seed
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationarity import check
from validate_motion import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('seed','source','origins','fields','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--windows',type=int,nargs='+',required=True)
    p.add_argument('--iterations',type=int,default=180)
    p.add_argument('--initial-motion',choices=('inherited','zero','phase-rate'),default='inherited')
    args=p.parse_args();started=time.monotonic()
    if args.output.exists():raise ValueError('Use a new experiment directory')
    seed=json.loads(args.seed.read_text());origin=json.loads(args.origins.read_text())
    if not origin['complete'] or not origin['fit']['converged']:
        raise ValueError('Use a completed origin control')
    tx,rx,_=sites(Path(seed['geometry_directory']));fields=[]
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as f:fields.append(Field(dict(f),f['pulse_samples']))
    files=[args.seed,args.origins,Path(__file__),*[args.fields/f'channel-{c}.npz' for c in range(5)],
        *[Path(__file__).with_name(n) for n in ('physical_profile.py','full_motion.py','field.py','spline_motion.py',
          'fixed_frame_motion.py','model_quadrature.py','fit_physical_profile.py','stationarity.py',
          'refine_full_capture_motion.py','radar_rate_initialization.py')]]
    result={'complete':False,'scope':__doc__,'windows':args.windows,'fits':[],
            'source_sha256':{str(f):fingerprint(f) for f in files},'accepted_calibration':False,
            'initial_motion':args.initial_motion}
    args.output.mkdir(parents=True)
    for window in args.windows:
        source=args.source/f'window-{window:03d}.json';local=json.loads(source.read_text())
        if not local['complete'] or not local.get('linear_reference'):
            raise ValueError('Require an existing physical single-anchor radar initialization')
        centre=float(np.mean(local['interval_s']))
        model={**seed,**local};model.pop('path_reference_time_s',None)
        model=restrict_interval(model,[centre-15,centre+15]);model=geographic_frame(model)
        model['range_bias_m']=copy.deepcopy(origin['range_bias_m'])
        # Old whole-capture path gauges do not define this independent aperture.
        model.pop('path_reference_time_s',None)
        description={k:copy.deepcopy(model[k]) for k in PHYSICAL_KEYS if k in model}
        description.update(parameters=model['parameters'],range_bias_m=model['range_bias_m'],
            selected_points_body_m=model['selected_points_body_m'])
        curve=SplineMotion(description);exact=[];fast=[]
        for c,full in enumerate(fields):
            field=full.windowed(centre-15+.03,centre+15-.03)
            selected=np.floor((field.time-(centre-15))/.5).astype(int)%2==0
            family=[]
            for f in (field,quadrature(field,2)):
                prior=motion_state(None,utc_seconds(description['nominal_start_utc'])+f.data['row_time'],description)
                family.append(Band(f,prior,tx[description['reference_channel']],rx,
                    description['selected_points_body_m'][c],description,selected,range_bias_m=model['range_bias_m'][c]))
            family[1].nuisance,family[1].clean,family[1].energy=family[0].nuisance,family[0].clean,family[0].energy
            exact.append(family[0]);fast.append(family[1])
        original=PhysicalProfile(exact,curve.parameters);approximate=PhysicalProfile(fast,curve.parameters)
        _,matrix,lo,hi,lower,upper=constraints(description,common_path=True)
        frozen=(original.active>=curve.sections[1].start)&(original.active<curve.sections[1].stop)
        lower[frozen]=upper[frozen]=0.
        free=lower<upper;variable=np.any(matrix[:,free]!=0.,axis=1)
        initial=original.initial.copy()
        acquisition=None
        if args.initial_motion in ('zero','phase-rate'):initial.fill(0.)
        if args.initial_motion=='phase-rate':
            acquired,acquisition=rate_seed(exact,original.expand(initial))
            initial=acquired[original.active]
        initial[free]=constrained_curve(np.eye(free.sum()),initial[free],np.ones(free.sum()),
            np.r_[matrix[variable][:,free],np.eye(free.sum())],np.r_[hi[variable],upper[free]])
        initial[~free]=lower[~free];approximate.initial=initial.copy();original.initial=initial.copy()
        before=original.validation(initial)
        stage='quadrature'
        def checkpoint(parameters,trace):
            out={'window':window,'stage':stage,'parameters':parameters.tolist(),'trace':trace}
            (args.output/f'window-{window:03d}.checkpoint.json').write_text(json.dumps(out,indent=2)+'\n')
            print(json.dumps({'window':window,'stage':stage,**trace[-1]}),flush=True)
        print(json.dumps({'window':window,'stage':'loaded','interval':description['interval_s'],'seconds':time.monotonic()-started}),flush=True)
        parameters,coarse=solve(approximate,matrix,lo,hi,lower,upper,iterations=args.iterations,
                                checkpoint=checkpoint,polish_stationarity=True)
        original.initial=parameters.copy();stage='original_rows'
        parameters,fit=solve(original,matrix,lo,hi,lower,upper,iterations=40,checkpoint=checkpoint,
            polish_stationarity=True,curvature_gradient=lambda q:approximate.objective(q)[1])
        evaluation=original.evaluate(parameters);kkt,violation=check(parameters,evaluation['gradient'],matrix,lo,hi,lower,upper)
        after=original.validation(parameters)
        output={**description,'parameters':original.expand(parameters).tolist(),'complete':True,'fit':fit,
            'original_row_kkt':kkt,'constraint_violation':violation,'quadrature_fit':coarse,
            'before':before,'before_summary':summarize(before),'after':after,'after_summary':summarize(after),
            'source_sha256':{**result['source_sha256'],str(source):fingerprint(source)},
            'scope':__doc__,'phase_centres_validated':False,'channel_calibration_validated':False,
            'origins_frozen_from':str(args.origins),'stationary_nuisance_source':'Exact original-row fixed response basis'}
        output['initial_motion']=args.initial_motion
        output['acquisition']=acquisition
        path=args.output/f'window-{window:03d}.json';path.write_text(json.dumps(output,indent=2)+'\n')
        result['fits'].append({'window':window,'path':str(path),'sha256':fingerprint(path),'converged':fit['converged'],
                              'original_row_kkt':kkt,'summary':output['after_summary']})
        (args.output/'experiment.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'window':window,'stage':'complete','kkt':kkt,'converged':fit['converged'],'seconds':time.monotonic()-started}),flush=True)
    result.update(complete=True,seconds=time.monotonic()-started)
    (args.output/'experiment.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
