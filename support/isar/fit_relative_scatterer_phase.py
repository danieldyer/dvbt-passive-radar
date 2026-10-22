"""Test inter-scatterer constant-phase coherence on the corrected observations.

Same original rows, known waveform kernels, fixed-grid observed-cell static
profile, independent smooth B/D paths and positive brightness. Restrict only
the constant phase relation of B and D through their measured envelope delays.
This is a nested model test; successful prediction is necessary, not sufficient,
for treating those returns as shared physical scatterers or calibrating sites.
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
from fit_fixed_grid_phases import prepare_fixed
from fit_shared_echo_cluster import subset_prediction
from relative_scatterer_phase import RelativeScattererPhase
from report_measured_scatterer_phases import load_fields,serial
from smooth_echo_cluster import SmoothEchoCluster
from waveform_kernels import read_kernels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels',type=int,nargs='+',default=[0,1,2,3])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    if (out.exists() or len(set(args.channels))!=len(args.channels) or 2 not in args.channels or
            any(c not in range(5) for c in args.channels)):
        raise ValueError('Use a fresh output, distinct channels and reference Nine')
    root=Path('docs/data/isar_20261017');source=root/'longer-known-phases-6s/fits.json';previous=json.loads(source.read_text())
    if not previous['complete']:raise ValueError('Wait for independent source fits')
    first,last=previous['interval_s'];std=previous['validation_selected_jerk_std']
    group=next(g for g in previous['groups'] if g['jerk_std']==std)
    baseline_path=Path(previous['source']);baseline=json.loads(baseline_path.read_text())
    origins=baseline['selected_origins_samples']
    geometry=Path('docs/data/calibration_20261016_sites')
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    mean_bands,parts,_=prepare_fixed(fields,geometry,first,last,stationary=True)
    kernel=Path('build/isar-20261016/waveform-kernels-30s');_,records,samples=read_kernels(kernel/'response.kernels')
    with np.load(kernel/'rows.npz') as grids:
        bands=[known_band(mean_bands[c],grids[f'channel_{c}'],records[c],samples[c]) for c in args.channels]
    parts=[parts[c] for c in args.channels]
    curve=SplinePath(first,last,previous['knot_spacing_s'])
    base=SmoothEchoCluster(bands,origins,(1,3),curve,shared=False,jerk_std=std)
    seed=np.zeros(base.count);expected_cost=0.
    for c,original in enumerate(args.channels):
        fit=group['independent'][original]['trials'][group['independent'][original]['best_index']]
        p=np.array(fit['parameters']);local=SmoothEchoCluster([bands[c]],origins,(1,3),curve,shared=False,jerk_std=std)
        size=2*base.components
        seed[size*c:size*(c+1)]=p[:size]
        for s,index in enumerate(base.local_path_indices[c]):seed[index]=p[local.local_path_indices[0][s]]
        expected_cost+=fit['cost']
    residual,_=base.evaluate(seed);error=abs(residual@residual*base.energy-expected_cost)
    if error>1e-6:raise ValueError(f'Original independent objective changed: {error}')
    model=RelativeScattererPhase(base,1,3,reference_band=args.channels.index(2))
    frozen=base.local_path_indices[args.channels.index(3)][2] if 3 in args.channels else np.array([],int)
    out.mkdir(parents=True);target=out/'fits.json';start_time=time.monotonic()
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        channels=args.channels,interval_s=previous['interval_s'],source=str(source),
        original_independent_cost=expected_cost,maximum_original_objective_error=error,
        jerk_std=std,knot_spacing_s=previous['knot_spacing_s'],origins_samples=origins,
        frozen_base_indices=frozen,original_independent_parameters=seed,
        independent_prediction=subset_prediction(base,seed,parts),trials=[],
        hypothesis='Common intrinsic constant phase per B/D scatterer and one phase per band. Independent envelope origins and temporal paths remain measured. Only the constant relative phase is coupled to the envelope delay. No time-dependent phase alignment or antenna change.',
        prior_scope='Same white-jerk scale selected earlier by independent validation. Ten C nuisance trajectory remains fixed at its earlier training estimate. Source data have previously been inspected; unused prediction is an internal transfer control.',
        source_sha256={})
    files=[source,baseline_path,Path(__file__),kernel/'response.kernels',kernel/'rows.npz',*[Path('support/isar')/(n+'.py') for n in
        ('relative_scatterer_phase','test_relative_scatterer_phase','smooth_echo_cluster','flexible_echo_cluster',
         'component_phase','shared_echo_cluster','amplitude_pair','fixed_range_field','field','waveform_kernels')]]
    snapshots=out/'sources';snapshots.mkdir()
    for p in files:
        result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshots/p.name)
    def save():
        result['seconds']=time.monotonic()-start_time
        target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for start,phase in enumerate((-np.pi/2,0.,np.pi/2,np.pi)):
        initial=model.seed(seed,phase);r,j=model.evaluate(initial)
        direction=.0001*np.cos(np.arange(model.count));direction[np.isin(model.kept,frozen)]=0
        h=1e-4;finite=(model.evaluate(initial+h*direction)[0]-model.evaluate(initial-h*direction)[0])/(2*h)
        derivative_error=float(np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-15))
        if derivative_error>4e-4:raise ValueError('Original-data coupled phase/range derivative failed')
        print('start',start,'initial cost',r@r*base.energy,'derivative',derivative_error,flush=True)
        p,audit=model.fit(initial,frozen_base_indices=frozen,max_evaluations=240);mapped=model.mapping(p)
        np.testing.assert_array_equal(mapped[frozen],seed[frozen])
        residual,_=model.evaluate(p)
        fit=dict(start=start,reference_delay_seed_phase=phase,parameters=p,base_parameters=mapped,
            cost=residual@residual*base.energy,audit=audit,directional_derivative_error=derivative_error,
            prediction=subset_prediction(base,mapped,parts),prior_cost=base.prior_cost(mapped))
        result['trials'].append(fit);save()
        print('done',start,'cost',fit['cost'],'unused',sum(v['heldout_error'] for v in fit['prediction']['testing']),
            'gradient',audit['projected_gradient'],'bounds',audit['bound_indices'],flush=True)
    result['best_index']=int(np.argmin([f['cost'] for f in result['trials']]))
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
