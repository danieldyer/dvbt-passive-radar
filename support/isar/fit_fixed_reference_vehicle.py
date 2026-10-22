"""Rebuild a vehicle calibration control on fixed, actually observed range cells.

Read C0/C1 directly from the original capture, including negative lags needed
by symmetric interpolation. Keep the road and antenna priors. No fitted range
offset, local noise normalization or extrapolated background coefficients.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from coherent_arc import CoherentArc
from extract_full_aperture import window
from fit_road_arc_raw import load_problem,solve,describe
from fixed_range_field import fixed_reference,stationary_profile_design
from report_measured_scatterer_phases import serial
from road_arc_raw import RawRoadGeometry


def rebuild(original,index,*,preserve_carrier=True):
    likelihoods=[];checks=[]
    capture=Path(index['identity'][0])
    for arc,bands in zip(original.geometry.arcs,original.likelihoods):
        epoch=(arc['first_s']+arc['last_s'])/2;new=[]
        for c,b in enumerate(bands):
            rows=b.field.data['row_time']
            aperture=window(capture,index,c,rows[0]-2e-8,rows[-1]+2e-8)
            offsets=np.arange(b.field.offsets[0]-6,b.field.offsets[-1]+7)
            omega=float(np.interp(epoch,rows,b.field.data['centre_omega']))
            f,change=fixed_reference(b.field,aperture,epoch,offsets,omega_reference=omega,preserve_carrier=preserve_carrier)
            np.testing.assert_array_equal(f.data['row_time'],rows)
            np.testing.assert_array_equal(f.data['row_information'],b.field.data['row_information'])
            np.testing.assert_array_equal(f.data['information'],b.field.data['information'])
            new.append(CoherentArc(f,b.training,stationary_design=stationary_profile_design(f)))
            checks.append(dict(channel=c,label=arc['label'],rows=len(rows),offsets=offsets,
                change=change,stationary_condition=float(new[-1].nuisance.singular_values[0]/new[-1].nuisance.singular_values[-1]),
                original_information_exact=True,phase_reference_offset_m=f.metadata['phase_reference_offset_m']))
        likelihoods.append(new)
    return RawRoadGeometry(original.geometry,likelihoods),checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',nargs='+',default=['A'])
    parser.add_argument('--assignment',default='02002')
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/fixed-reference-vehicle-A'))
    parser.add_argument('--fixed-carrier',action='store_true',help='Explicit loss-control experiment; normally retain the known carrier deramp')
    args=parser.parse_args();root=Path('docs/data/isar_20261017');out=args.output
    out.mkdir(exist_ok=True);target=out/'fits.json'
    if target.exists():raise ValueError('Use a fresh control directory')
    summary=root/'road-arc-geometry/fits.json';code=list(map(int,args.assignment))
    original,associations,sources=load_problem(summary,code,args.labels)
    index_path=Path('build/calibration-20261013/shared-seed-histories/archive-index.json')
    index=json.loads(index_path.read_text());model,checks=rebuild(original,index,preserve_carrier=not args.fixed_carrier)
    previous_path=root/'road-sites-32/fits.json';previous=json.loads(previous_path.read_text())
    previous_fit=next(t for t in previous['trials'] if t['assignment']==code)
    seed=model.initial.copy()
    for i,label in enumerate(args.labels):
        old=next(a for a in previous_fit['associations'] if a['label']==label)
        if old['route']!=associations[i]['route']:raise ValueError('Initial road association changed')
        seed[5*i:5*i+5]=np.asarray(previous_fit['road_states'][previous['labels'].index(label)])/model.geometry.scale
    seed=np.clip(seed,model.lower+1e-7,model.upper-1e-7)
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        labels=args.labels,assignment=code,associations=associations,observation_checks=checks,trials=[],
        original_carrier_deramp_retained=not args.fixed_carrier,
        policy='Fixed nominal antenna positions from approved user priors; no free range origins. One road point and independent complex band gains. Stationary coefficients live only in actually observed fixed cells, with condition one. Same original sample epochs/information; two physical motion starts ranked by training cost.',
        source_sha256={})
    files=[previous_path,index_path,Path(__file__),*sources,*[Path('support/isar')/(n+'.py') for n in
        ('fixed_range_field','extract_full_aperture','road_arc_raw','coherent_arc','road_arc_geometry','field')]]
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for p in files:
        result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshots/p.name)
    started=time.monotonic()
    def save():
        result['seconds']=time.monotonic()-started
        target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for start,initial in enumerate((seed,model.initial)):
        initial=np.clip(initial,model.lower+1e-7,model.upper-1e-7)
        r,j=model.evaluate(initial);direction=.002*np.cos(np.arange(len(initial)))
        errors=[]
        for h in (1e-4,1e-5,1e-6):
            fd=(model.evaluate(initial+h*direction)[0]-model.evaluate(initial-h*direction)[0])/(2*h)
            errors.append(float(np.linalg.norm(fd-j@direction)/max(np.linalg.norm(fd),1e-15)))
        if min(errors)>4e-4:raise ValueError('Fixed-reference road derivative failed')
        fitted=solve(model,initial,iterations=160);value=describe(model,fitted)
        value.update(start=start,directional_derivative_errors=errors)
        result['trials'].append(value);save()
        print(start,'cost',value['cost'],'optimality',value['optimality'],'bounds',value['bound_indices'],
            'unused',sum(c['error'] for a in value['prediction'] for c in a['channels']),flush=True)
    result['best_index']=int(np.argmin([f['cost'] for f in result['trials']]))
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
