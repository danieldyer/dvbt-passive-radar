"""Fit a newly acquired near-main-return echo on the original symbol rows.

Retain all four existing independent returns and their observed-cell static
profile. B/D remain flexible; the additional path basis is declared explicitly.
Its envelope, carrier and Doppler all follow that same path before averaging.
No common phase or trajectory is imposed across bands in this control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from audit_joint_kernel_phase import known_band
from cluster_phase import SplinePath
from component_phase import component_measurements
from fit_fixed_grid_phases import prepare_fixed
from fit_shared_echo_cluster import subset_prediction
from release_fixed_grid_phase_guide import path_information
from report_measured_scatterer_phases import load_fields,serial
from smooth_echo_cluster import SmoothEchoCluster
from waveform_kernels import read_kernels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel',type=int,choices=range(5),required=True)
    parser.add_argument('--candidate',type=int,default=1,help='One-based index in the saved residual acquisition')
    parser.add_argument('--extra-curve',choices=('quadratic','flexible'),default='quadratic')
    parser.add_argument('--origin-bounds',type=float,nargs=2,help='Explicit extra-component envelope interval in original range samples')
    parser.add_argument('--resume-extra',type=Path,help='Re-express both completed same-data extra-component starts in this declared basis')
    parser.add_argument('--guide-extra',type=Path,help='Initialize B/E paths from another completed band, then fit every path independently')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    if out.exists() or (args.resume_extra and args.guide_extra):raise ValueError('Use a fresh output and at most one resume/guide source')
    root=Path('docs/data/isar_20261017');source=root/'longer-known-phases-6s/fits.json'
    acquisition_path=root/'main-return-residual-acquisition/analysis.json'
    acquired=json.loads(acquisition_path.read_text());hypothesis=acquired['candidate_hypotheses'][args.candidate-1]
    saved=json.loads(source.read_text());baseline_path=Path(saved['source']);baseline=json.loads(baseline_path.read_text())
    first,last=saved['interval_s'];std=saved['validation_selected_jerk_std'];group=next(g for g in saved['groups'] if g['jerk_std']==std)
    entry=group['independent'][args.channel];original=entry['trials'][entry['best_index']]
    geometry=Path('docs/data/calibration_20261016_sites')
    mean,parts,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    kernel=Path('build/isar-20261016/waveform-kernels-30s');_,records,pulses=read_kernels(kernel/'response.kernels')
    with np.load(kernel/'rows.npz') as grids:
        c=args.channel;band=known_band(mean[c],grids[f'channel_{c}'],records[c],pulses[c])
    curve=SplinePath(first,last,saved['knot_spacing_s']);origins=baseline['selected_origins_samples']
    old=SmoothEchoCluster([band],origins,(1,3),curve,shared=False,jerk_std=std);previous=np.array(original['parameters'])
    r,_=old.evaluate(previous);error=abs(r@r*old.energy-original['cost'])
    if error>1e-7:raise ValueError('Original likelihood changed')
    origin=hypothesis['origin_samples'];extra=len(origins)
    selected=(1,3,extra) if args.extra_curve=='flexible' else (1,3)
    model=SmoothEchoCluster([band],[*origins,origin],selected,curve,shared=False,jerk_std=std)
    seed=np.zeros(model.count);seed[:2*len(origins)]=previous[:2*len(origins)]
    seed[2*len(origins)]=origin
    for s in range(len(origins)):seed[model.local_path_indices[0][s]]=previous[old.local_path_indices[0][s]]
    origin_index=2*extra;phase_index=origin_index+1
    low,high=args.origin_bounds if args.origin_bounds else (origin-1.,origin+1.)
    if not low<high:raise ValueError('Ordered envelope bounds required')
    model.lower[origin_index]=low;model.upper[origin_index]=high
    frozen=model.local_path_indices[0][2] if c==3 else np.array([],int)
    free=np.setdiff1d(np.arange(model.count),frozen)
    v=old.clusters[0].evaluate(previous[old.indices[0]])
    residual=band.clean-band.nuisance.remove(v['design'][band.selected])@v['amplitude']
    resumed=None;guided=None;old_extra=None
    if args.resume_extra:
        resumed=json.loads(args.resume_extra.read_text())
        if (not resumed['complete'] or resumed['channel']!=c or resumed['source']!=str(source) or
                resumed['candidate_index']!=args.candidate):raise ValueError('Resume requires completed matching data/candidate starts')
        previous_selected=(1,3,extra) if resumed.get('extra_curve','quadratic')=='flexible' else (1,3)
        old_extra=SmoothEchoCluster([band],[*origins,origin],previous_selected,curve,shared=False,jerk_std=std)
    if args.guide_extra:
        guided=json.loads(args.guide_extra.read_text())
        if (not guided['complete'] or guided['source']!=str(source) or guided['candidate_index']!=args.candidate or
                guided['interval_s']!=saved['interval_s']):raise ValueError('Guide requires completed matching-candidate observations')
        previous_selected=(1,3,extra) if guided.get('extra_curve','quadratic')=='flexible' else (1,3)
        old_extra=SmoothEchoCluster([band],[*origins,origin],previous_selected,curve,shared=False,jerk_std=std)
        k=len(band.knots)
        main=v['design'][:,k:2*k]@v['amplitude'][k:2*k]
        guide_residual=residual+band.nuisance.remove(main[band.selected])
    out.mkdir(parents=True);target=out/'fits.json';started=time.monotonic()
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        channel=c,name=entry['name'],candidate_index=args.candidate,hypothesis=hypothesis,
        interval_s=saved['interval_s'],origins_samples=[*origins,origin],source=str(source),acquisition=str(acquisition_path),
        extra_curve=args.extra_curve,extra_origin_bounds_samples=[low,high],resume_extra=str(args.resume_extra) if args.resume_extra else None,
        guide_extra=str(args.guide_extra) if args.guide_extra else None,
        original_cost=original['cost'],original_prediction=original['prediction'],original_parameters=previous,
        maximum_original_objective_error=error,trials=[],
        policy='Training-only acquired candidate, added to the unchanged original four-return model. Same known kernels, weights, stationary design and B/D motion prior; declared extra path basis and envelope bounds are explicit fields. A guide supplies initialization only: all band phases, envelopes and trajectories are independent after initialization. A bound contact is not location information. Starts are selected by training cost only; compare untouched blocks afterward.',
        source_sha256={})
    files=[source,acquisition_path,baseline_path,Path(__file__),*([args.resume_extra] if args.resume_extra else []),*([args.guide_extra] if args.guide_extra else []),*[Path('support/isar')/(n+'.py') for n in
        ('smooth_echo_cluster','flexible_echo_cluster','shared_echo_cluster','amplitude_pair','component_phase',
         'release_fixed_grid_phase_guide','field','fixed_range_field','waveform_kernels')]]
    snapshot=out/'sources';snapshot.mkdir()
    for p in files:
        result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    def save():
        result['seconds']=time.monotonic()-started;target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for start in range(2):
        p=seed.copy();p[origin_index]+=.25*start
        coefficients=[hypothesis['reference_path_rate_mps']*band.scale,
            .5*hypothesis['acceleration_mps2']*band.scale**2]
        rows=band.field.data['row_time']
        if resumed:
            trial=resumed['trials'][start];previous_extra=np.array(trial['parameters'])
            p[:2*(extra+1)]=previous_extra[:2*(extra+1)]
            for s,index in enumerate(model.local_path_indices[0]):
                prior_curve=old_extra.clusters[0].arcs[s].base.curve;new_curve=model.clusters[0].arcs[s].base.curve
                values=prior_curve(rows)@previous_extra[old_extra.local_path_indices[0][s]]
                p[index]=np.linalg.lstsq(new_curve(rows),values,rcond=None)[0]
                if np.max(abs(new_curve(rows)@p[index]-values))>1e-10:raise ValueError('Changing basis changed an original-row phase path')
            p[frozen]=seed[frozen]
        elif guided:
            previous_extra=np.array(guided['trials'][start]['parameters'])
            guide_origin=np.array(guided['original_parameters'])[2]
            for s in (1,extra):
                p[2*s]=seed[2]+previous_extra[2*s]-guide_origin
                prior_curve=old_extra.clusters[0].arcs[s].base.curve;new_curve=model.clusters[0].arcs[s].base.curve
                values=prior_curve(rows)@previous_extra[old_extra.local_path_indices[0][s]]
                p[model.local_path_indices[0][s]]=np.linalg.lstsq(new_curve(rows),values,rcond=None)[0]
                if np.max(abs(new_curve(rows)@p[model.local_path_indices[0][s]]-values))>1e-10:
                    raise ValueError('Guide path is not represented in the declared basis')
                p[2*s+1]=0.
            responses=model.clusters[0].response_blocks(p[model.indices[0]])
            design=np.column_stack([band.nuisance.remove(band.reduced(responses[s][0][:,:,None])[:,0][band.selected]) for s in (1,extra)])
            coefficient=np.linalg.lstsq(design,guide_residual,rcond=None)[0]
            p[3],p[phase_index]=np.angle(coefficient)
            if np.any(p<model.lower) or np.any(p>model.upper):raise ValueError('Guide envelope is outside declared support')
        else:
            path=band.base.curve(rows)@coefficients;new_curve=model.clusters[0].arcs[extra].base.curve
            p[model.local_path_indices[0][extra]]=np.linalg.lstsq(new_curve(rows),path,rcond=None)[0]
            raw,_=band.response(np.r_[p[origin_index],0.,coefficients])
            h=band.nuisance.remove(band.reduced(raw[:,:,None])[:,0][band.selected])
            p[phase_index]=np.angle(np.vdot(h,residual))
        r,j=model.evaluate(p);initial_cost=float(r@r*model.energy)
        if not guided and initial_cost>original['cost']+1e-6:raise ValueError('Adding a nested echo increased its optimum linear cost')
        if resumed and abs(initial_cost-trial['cost'])>1e-7:raise ValueError('Resumed original-row likelihood changed')
        direction=.0001*np.cos(np.arange(model.count));direction[frozen]=0;step=1e-4
        numerical=(model.evaluate(p+step*direction)[0]-model.evaluate(p-step*direction)[0])/(2*step)
        derivative_error=float(np.linalg.norm(numerical-j@direction)/max(np.linalg.norm(numerical),1e-15))
        if derivative_error>4e-4:raise ValueError('Original-row added-echo derivative check failed')
        print('start',start,'initial',initial_cost,'derivative',derivative_error,flush=True)
        fit=model.fit(p,240,free_indices=free);q=np.array(fit['parameters'])
        np.testing.assert_array_equal(q[frozen],seed[frozen])
        rec=component_measurements(model.clusters[0],q[model.indices[0]])
        fit.update(start=start,initial_cost=initial_cost,directional_derivative_error=derivative_error,
            prediction=subset_prediction(model,q,[parts[c]]),prior_cost=model.prior_cost(q),
            information={str(s):path_information(model,q,component=s) for s in (1,extra)},
            measured={str(s):dict(time_s=rec[s]['time_s'],physical_path_change_m=rec[s]['physical_path_change_m'],
                complex_coefficient_real_imag=np.c_[rec[s]['complex_coefficient'].real,rec[s]['complex_coefficient'].imag],
                conditional_information=rec[s]['conditional_information'],
                prediction_power=np.sum(abs(rec[s]['frozen_component_prediction'])**2,axis=1)) for s in (1,extra)})
        result['trials'].append(fit);save()
        print('done',start,'cost',fit['cost'],'unused',fit['prediction']['testing'][0]['heldout_error'],
            'gradient',fit['maximum_projected_gradient'],'bounds',fit['bound_indices'],flush=True)
    result['best_index']=int(np.argmin([t['cost'] for t in result['trials']]))
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
