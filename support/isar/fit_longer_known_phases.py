"""Longer original-data phase comparison with flexible B and D trajectories.

Both bright returns get independent smooth paths, preventing the stronger
neighbour's quadratic approximation from driving the weaker point's phase.
Known original-weight waveform kernels follow original rows. A/C remain
independent quadratic nuisances; the previously unidentifiable Ten C path
stays fixed in this explicitly conditional control.
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
from shared_echo_cluster import SharedEchoCluster
from smooth_echo_cluster import SmoothEchoCluster
from waveform_kernels import read_kernels


def seed_from_quadratics(model,previous,old_bands):
    seed=np.zeros(model.count);n=model.components;checks=[]
    for c,(entry,b,old_b) in enumerate(zip(previous['independent'],model.bands,old_bands)):
        old=np.asarray(entry['trials'][entry['best_index']]['parameters'])
        seed[2*n*c:2*n*(c+1)]=old[:2*n]
        rows=b.field.data['row_time']
        for s,path in enumerate(model.local_path_indices[c]):
            curve=model.curve if s in model.selected_components else b.base.curve
            expected=old_b.base.curve(rows)@old[2*n+2*s:2*n+2*s+2]
            seed[path]=np.linalg.lstsq(curve(rows),expected,rcond=None)[0]
        # This exact check applies to the mean-pulse model before known kernels
        # change the response. Epoch, range, carrier and C0/C1 gauges all agree.
        old_model=SharedEchoCluster([old_b],previous['selected_origins_samples'])
        indices=np.searchsorted(rows,old_b.field.data['row_time'])
        np.testing.assert_array_equal(rows[indices],old_b.field.data['row_time'])
        for (expected,_),(actual,_) in zip(old_model.clusters[0].response_blocks(old[old_model.indices[0]]),
                model.clusters[c].response_blocks(seed[model.indices[c]])):
            error=float(np.max(abs(actual[indices]-expected)))
            if error>3e-12:raise ValueError('Aperture extension changed the original forward phase gauge')
            checks.append(error)
    return seed,checks


def phase_diagnostics(model,p,partitions):
    result=[]
    for c,(cluster,index,part) in enumerate(zip(model.clusters,model.indices,partitions)):
        measured=component_measurements(cluster,p[index]);b=cluster.base;entry=[]
        for s in (1,3):
            r=measured[s];path=r['physical_path_change_m']
            phase=np.angle(r['complex_coefficient']*np.exp(1j*b.field.k*path))
            weight=np.sum(abs(r['frozen_component_prediction'])**2,axis=1);use=(part==3)&(weight>0)
            arc=cluster.arcs[s];local=p[index][cluster.slices[s]];curve=arc.base.curve
            grid=np.linspace(b.field.data['row_time'][0],b.field.data['row_time'][-1],201)
            h=r['frozen_component_prediction'].ravel()[b.selected]
            retained=float(np.vdot(b.nuisance.remove(h),b.nuisance.remove(h)).real/np.vdot(h,h).real)
            entry.append(dict(component=s,label=('B' if s==1 else 'D'),time_s=r['time_s'],testing=part==3,
                physical_path_change_m=path,conditional_phase_residual_deg=np.degrees(phase),
                unused_conditional_phase_rms_deg=float(np.degrees(np.sqrt(np.average(phase[use]**2,weights=weight[use])))),
                stationary_projected_signal_energy_retained=retained,
                derivative_time_s=grid,reference_path_rate_mps=curve(grid,1)@local[2:],
                reference_path_acceleration_mps2=curve(grid,2)@local[2:]))
        result.append(entry)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-s',type=int,choices=(6,8),default=6)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/longer-known-phases-6s'))
    args=parser.parse_args();out=args.output;out.mkdir(exist_ok=True);target=out/'fits.json'
    if target.exists():raise ValueError('Use a fresh output directory')
    root=Path('docs/data/isar_20261017');source=root/'fixed-grid-multiple-phases/fits.json'
    previous=json.loads(source.read_text());first,last=165-args.duration_s/2,165+args.duration_s/2
    geometry=Path('docs/data/calibration_20261016_sites')
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    mean_bands,parts,_=prepare_fixed(fields,geometry,first,last,stationary=True,observation_directory=out/'observations')
    old_bands,old_parts,_=prepare_fixed(fields,geometry,*previous['interval_s'],stationary=True)
    origins=previous['selected_origins_samples'];curve=SplinePath(first,last,1.)
    mean_model=SmoothEchoCluster(mean_bands,origins,(1,3),curve,shared=False,jerk_std=.1)
    all_seed,forward_errors=seed_from_quadratics(mean_model,previous,old_bands)
    overlap_errors=[]
    for b,old,part,old_part in zip(mean_bands,old_bands,parts,old_parts):
        i=np.searchsorted(b.field.time,old.field.time)
        np.testing.assert_array_equal(b.field.time[i],old.field.time)
        np.testing.assert_array_equal(part[i],old_part)
        overlap_errors.append(float(np.max(abs(b.field.data['value'][i]-old.field.data['value']))))
    if max(overlap_errors)>1e-12:raise ValueError('Original observations changed on the extended aperture')
    kernel=Path('build/isar-20261016/waveform-kernels-30s');_,records,samples=read_kernels(kernel/'response.kernels')
    with np.load(kernel/'rows.npz') as grids:
        bands=[known_band(b,grids[f'channel_{c}'],records[c],samples[c]) for c,b in enumerate(mean_bands)]
    result=dict(scope=__doc__,complete=False,production_changed=False,accepted_calibration=False,
        interval_s=[first,last],selected_components=[1,3],knot_spacing_s=1.,groups=[],common=[],
        source=str(source),maximum_overlap_observation_error=max(overlap_errors),maximum_overlap_forward_error=max(forward_errors),
        policy='Same original rows, fixed observer at 165 s, known block kernels, unchanged mean covariance and observed-cell static design. B/D have flexible paths; A/C remain quadratic. Two white-jerk scales compare validation prediction; two starts compare training only. Final unused blocks select neither scales nor starts. All data have previously been inspected.',
        conditional_nuisance='Only the two Ten C path coefficients are fixed at their earlier independent training estimate, expressed in the new basis. No ABC path is externally constrained in the independent fits.',source_sha256={})
    files=[source,Path(__file__),kernel/'response.kernels',kernel/'rows.npz',*[Path('support/isar')/(n+'.py') for n in
        ('flexible_echo_cluster','smooth_echo_cluster','component_phase','waveform_kernels',
         'audit_joint_kernel_phase','fit_fixed_grid_phases','field','amplitude_pair','release_fixed_grid_phase_guide')]]
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for p in files:
        result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshots/p.name)
    started=time.monotonic()
    def save():
        result['seconds']=time.monotonic()-started
        target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for std in (.01,.1):
        group=dict(jerk_std=std,independent=[]);result['groups'].append(group);save()
        for c,b in enumerate(bands):
            model=SmoothEchoCluster([b],origins,(1,3),curve,shared=False,jerk_std=std)
            seed=np.zeros(model.count);seed[model.indices[0]]=all_seed[mean_model.indices[c]]
            frozen=model.local_path_indices[0][2] if c==3 else np.array([],int)
            free=np.setdiff1d(np.arange(model.count),frozen)
            entry=dict(channel=c,name=previous['independent'][c]['name'],trials=[]);group['independent'].append(entry);save()
            for start in range(2):
                initial=seed.copy()
                if start:
                    offset=np.linalg.lstsq(curve(b.field.data['row_time']),.15*(b.field.data['row_time']-165),rcond=None)[0]
                    initial[model.local_path_indices[0][3]]+=offset
                    initial[3]+=.3
                r,j=model.evaluate(initial);direction=.002*np.cos(np.arange(model.count));direction[frozen]=0;h=1e-4
                fd=(model.evaluate(initial+h*direction)[0]-model.evaluate(initial-h*direction)[0])/(2*h)
                error=float(np.linalg.norm(fd-j@direction)/max(np.linalg.norm(fd),1e-15))
                if error>4e-4:raise ValueError('Extended original-data derivative mismatch')
                fit=model.fit(initial,240,free_indices=free);p=np.asarray(fit['parameters'])
                np.testing.assert_array_equal(p[frozen],seed[frozen])
                fit.update(start=start,prior_cost=model.prior_cost(p),initial_directional_derivative_error=error,
                    prediction=subset_prediction(model,p,[parts[c]]),phase=phase_diagnostics(model,p,[parts[c]]),
                    information={str(s):path_information(model,p,component=s) for s in (1,3)})
                entry['trials'].append(fit);save()
                print(std,entry['name'],start,'cost',fit['cost'],'unused',fit['prediction']['testing'][0]['heldout_error'],
                    'gradient',fit['maximum_projected_gradient'],'bounds',fit['bound_indices'],flush=True)
            entry['best_index']=int(np.argmin([f['cost'] for f in entry['trials']]));save()
        group['validation_error']=sum(e['trials'][e['best_index']]['prediction']['validation'][0]['heldout_error'] for e in group['independent']);save()
    best=min(result['groups'],key=lambda g:g['validation_error']);std=best['jerk_std']
    result['validation_selected_jerk_std']=std;save()
    model=SmoothEchoCluster(bands,origins,(1,3),curve,shared_components=(1,),jerk_std=std)
    seed=np.zeros(model.count);paths=[]
    for c,e in enumerate(best['independent']):
        local_model=SmoothEchoCluster([bands[c]],origins,(1,3),curve,shared=False,jerk_std=std)
        p=np.asarray(e['trials'][e['best_index']]['parameters'])
        seed[8*c:8*c+8]=p[:8]
        for s,index in enumerate(model.local_path_indices[c]):seed[index]=p[local_model.local_path_indices[0][s]]
        paths.append(p[local_model.local_path_indices[0][1]])
    frozen=model.local_path_indices[3][2];free=np.setdiff1d(np.arange(model.count),frozen)
    for start,q in enumerate((np.median(paths,axis=0),paths[2])):
        initial=seed.copy();initial[model.local_path_indices[0][1]]=q
        fit=model.fit(initial,260,free_indices=free);p=np.asarray(fit['parameters'])
        fit.update(start=start,prior_cost=model.prior_cost(p),prediction=subset_prediction(model,p,parts),phase=phase_diagnostics(model,p,parts))
        result['common'].append(fit);save()
        print('common B',start,'cost',fit['cost'],'unused',sum(x['heldout_error'] for x in fit['prediction']['testing']),
            'gradient',fit['maximum_projected_gradient'],'bounds',fit['bound_indices'],flush=True)
    result['best_common_index']=int(np.argmin([f['cost'] for f in result['common']]))
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
