"""Test common relative range at nominally co-sited flash transmitters.

This restricts the empirically shared B/E temporal paths by their necessary
co-sited relative envelope delay. Each carrier phase and brightness remains
free. The original data/operator are unchanged. These overlapping apertures
are controls, not independent calibration observations to multiply together.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from cluster_phase import SplinePath
from fit_piecewise_nearby_paths import guarded_model,acquire_phases,measured
from fit_shared_echo_cluster import subset_prediction
from joint_path_information import paths
from nearby_echo_control import load,ROOT
from phase_window import window_band
from relative_scatterer_delay import RelativeScattererDelay
from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    if out.exists():raise ValueError('Use a fresh output directory')
    source=ROOT/'piecewise-nearby-common-paths/fits.json';saved=json.loads(source.read_text())
    if not saved['complete']:raise ValueError('Wait for all predeclared piecewise controls')
    control=load();out.mkdir(parents=True);started=time.monotonic()
    files=[source,Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('fit_piecewise_nearby_paths','relative_scatterer_delay','test_relative_scatterer_delay',
            'phase_window','joint_path_information','test_joint_path_information')]]
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        source=str(source),assignment=saved['assignment'],windows=[],
        policy='Same complete original bins, known kernels, C0/C1 reference, conditional covariance, brightness knots, guards and train/testing splits as the short-path controls. A nominal co-sited pair has one relative delay in each site group; point phases remain independent across bands. Constant phase is acquired on training only. Minimum training cost among mapped-guard-valid starts selects each control; all failed gradients/bounds remain explicit. The selected restricted fit is then released into the larger unrestricted model as a nonlinear-initialization control.',
        limitation='Relative delays at different transmitter sites remain independent here. Actual per-channel antenna offsets and the exact physical point positions require the later full geometry. This test does not identify coordinates or establish a frequency-independent intrinsic scattering phase. Overlapping windows and previously inspected unused rows are not independent final validation.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    snap=out/'sources';snap.mkdir()
    for p in files:
        if p.suffix=='.py':shutil.copyfile(p,snap/p.name)
    def save():
        result['seconds']=time.monotonic()-started
        (out/'fits.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    groups=[tuple(c for c,site in enumerate(saved['assignment']) if site==label) for label in sorted(set(saved['assignment']))]
    for w in saved['windows']:
        first,last=w['interval_s'];bands=[];parts=[]
        for c,b in enumerate(control['bands']):
            short,part,_=window_band(b,first,last,complete_rows=True);keep=np.isin(b.field.time,short.field.time)
            np.testing.assert_array_equal(part,control['parts'][c][keep]);bands.append(short);parts.append(part)
        curve=SplinePath(first,last,w['knot_spacing_s']);base=guarded_model(bands,control,curve,True)
        original=w['common'][w['best_common_index']];p=np.asarray(original['parameters']);r,_=base.evaluate(p)
        if abs(r@r*base.energy-original['cost'])>1e-7:raise ValueError('Short source likelihood changed')
        independent=[guarded_model([b],control,curve,False) for b in bands];local=[]
        for c,m in enumerate(independent):
            entry=w['independent'][c];fit=entry['trials'][entry['best_index']];q=np.asarray(fit['parameters'])
            r,_=m.evaluate(q)
            if abs(r@r*m.energy-fit['cost'])>1e-7:raise ValueError('Short independent likelihood changed')
            local.append(q)
        model=RelativeScattererDelay(base,1,4,groups);frozen=base.local_path_indices[3][2]
        gaps=np.array([(p[10*c+8]-p[10*c+2]) for c in range(5)])
        proposed=[('current common',p.copy())]
        for name,gap in (('median same-site gap',float(np.median(gaps[list(groups[0])]))),
                         ('ABC gap initialization only',float(gaps[4]))):
            seed=p.copy();ref=groups[0][0];seed[10*ref+8]=seed[10*ref+2]+gap;proposed.append((name,seed))
        record=dict(interval_s=w['interval_s'],groups=groups,unrestricted_source_cost=original['cost'],
            unrestricted_source_testing_error=sum(v['heldout_error'] for v in original['prediction']['testing']),
            independent_testing_error=sum(e['trials'][e['best_index']]['prediction']['testing'][0]['heldout_error'] for e in w['independent']),
            trials=[])
        result['windows'].append(record);save()
        for name,seed in proposed:
            q=model.mapping(seed[model.kept]);q=acquire_phases(base,q,independent,local);seed=q[model.kept]
            r,j=model.evaluate(seed);direction=1e-4*np.cos(np.arange(model.count));direction[np.isin(model.kept,frozen)]=0.;step=1e-4
            finite=(model.evaluate(seed+step*direction)[0]-model.evaluate(seed-step*direction)[0])/(2*step)
            error=float(np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-15))
            if error>4e-4:raise ValueError('Original-row constrained derivative failed')
            print(first,last,'starting',name,'cost',r@r*model.energy,'derivative',error,flush=True)
            q,audit=model.fit(seed,frozen_base_indices=frozen,max_evaluations=240);full=model.mapping(q)
            fit=dict(seed=name,parameters=full,parameters_reduced=q,audit=audit,cost=model.objective(q)[0]*model.energy,
                prior_cost=base.prior_cost(full),prediction=subset_prediction(base,full,parts),derivative_error=error)
            record['trials'].append(fit);save()
            print(first,last,'done',name,'cost',fit['cost'],'gradient',audit['projected_gradient'],
                'guards',audit['mapped_envelope_guard'],flush=True)
        valid=[i for i,f in enumerate(record['trials']) if f['audit']['mapped_envelope_guard']['valid']]
        if valid:
            record['best_index']=min(valid,key=lambda i:record['trials'][i]['cost'])
            best=record['trials'][record['best_index']];q=np.asarray(best['parameters_reduced']);full=np.asarray(best['parameters'])
            record['measured'],record['local_B_E_component_to_sum_energy_ratio']=measured(base,full)
            record['information']=paths(base,full,transform=model.transform)
            error=sum(v['heldout_error'] for v in best['prediction']['testing'])
            record['unused_change_from_unrestricted_percent']=100*(error/record['unrestricted_source_testing_error']-1)
            record['unused_change_from_independent_percent']=100*(error/record['independent_testing_error']-1)
            print(first,last,'releasing best constrained seed',flush=True)
            free=np.setdiff1d(np.arange(base.count),frozen);released=base.fit(full,200,free_indices=free)
            rp=np.asarray(released['parameters']);released.update(prior_cost=base.prior_cost(rp),prediction=subset_prediction(base,rp,parts))
            record['released']=released
            record['released_measured'],record['released_local_B_E_component_to_sum_energy_ratio']=measured(base,rp)
            record['released_information']=paths(base,rp)
            record['complete']=True
            print('window complete',first,last,'restricted vs independent',record['unused_change_from_independent_percent'],
                'released cost',released['cost'],flush=True)
        else:record['complete']=True;record['no_valid_restricted_start']=True
        save()
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
