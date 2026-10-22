"""Test two common nearby-return paths against independent-band controls.

Original rows, known pulse kernels and noise information are unchanged.
B/E share temporal paths plus the declared nominal interband geometric
increment; envelopes, constant phases and positive brightness remain per-band.
A/C/D remain independent nuisance echoes. This is not yet a physical siting
fit or a claim of coherent multiband reflectivity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from component_phase import component_measurements
from fit_shared_echo_cluster import subset_prediction
from nearby_echo_control import ROOT,load,independent_seed,phase_seed
from report_measured_scatterer_phases import serial
from smooth_echo_cluster import SmoothEchoCluster


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    if out.exists():raise ValueError('Use a fresh output directory')
    control=load();bands=control['bands'];models=control['independent_models']
    model=SmoothEchoCluster(bands,control['origins'],(1,3,4),control['curve'],
        shared_components=(1,4),jerk_std=control['jerk_std'])
    for c in range(5):
        for s in (1,4):model.lower[10*c+2*s],model.upper[10*c+2*s]=(-2.5,2.5)
    seed=independent_seed(model,control);frozen=model.local_path_indices[3][2]
    free=np.setdiff1d(np.arange(model.count),frozen)
    best=[d['trials'][d['best_index']] for d in control['saved']]
    paths=[]
    for c,(d,m) in enumerate(zip(control['saved'],models)):
        p=np.array(best[c]['parameters']);paths.append([p[m.local_path_indices[0][s]] for s in (1,4)])
    def local_paths(channel,trial):
        p=np.array(control['saved'][channel]['trials'][trial]['parameters']);m=models[channel]
        return np.array([p[m.local_path_indices[0][s]] for s in (1,4)])
    starts=[('ABC training winner',local_paths(4,control['saved'][4]['best_index'])),
        ('ABC alternative',local_paths(4,1-control['saved'][4]['best_index'])),
        ('Nine training winner',local_paths(2,control['saved'][2]['best_index'])),
        ('Median independent coefficients; initialization only',np.median(paths,axis=0))]
    out.mkdir(parents=True);target=out/'fits.json';started=time.monotonic()
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        interval_s=[control['curve'].epoch-control['curve'].scale,control['curve'].epoch+control['curve'].scale],
        origins_samples=control['origins'],jerk_std=control['jerk_std'],knot_spacing_s=control['knot_spacing_s'],
        shared_components=[1,4],independent_sources=[str(p) for p in control['sources']],
        replayed_independent_objectives=control['reproduced_objectives'],maximum_absolute_replayed_cost_error=control['maximum_absolute_cost_error'],
        independent_training_cost=sum(f['cost'] for f in best),
        independent_prior_cost=sum(f['prior_cost'] for f in best),
        independent_prediction={split:[f['prediction'][split][0] for f in best] for split in ('validation','testing')},
        policy='Same immutable original-row fixed-range observations and known kernels. Shared B/E white-jerk priors counted once per physical path, independent D once per band; do not compare MAP penalties of different dimensions as Bayes factors. Starts use training cost only. Unused prediction, signal-component cancellation and data-only information must validate the shared-path hypothesis.',
        phase_policy='Per-band constant phases remain free. Common temporal paths do not yet establish a shared intrinsic scattering phase or phase-centre coordinates.',
        conditional_nuisance='Only the earlier two Ten C nuisance path coefficients remain fixed; its envelope and phase are free. No blank Doppler mask.',
        trials=[],source_sha256={})
    files=[*control['sources'],Path(__file__),control['kernel']/'response.kernels',control['kernel']/'rows.npz',
        *[Path('support/isar')/(n+'.py') for n in ('nearby_echo_control','smooth_echo_cluster','flexible_echo_cluster',
            'component_phase','amplitude_pair','shared_echo_cluster','field','fixed_range_field','waveform_kernels')]]
    snapshots=out/'sources';snapshots.mkdir()
    for p in files:
        result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshots/p.name)
    def save():
        result['seconds']=time.monotonic()-started;target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for start,(name,shared) in enumerate(starts):
        p=seed.copy()
        for s,q in zip((1,4),shared):p[model.local_path_indices[0][s]]=q
        p=phase_seed(model,p,control)
        r,j=model.evaluate(p);direction=.0001*np.cos(np.arange(model.count));direction[frozen]=0;step=1e-4
        fd=(model.evaluate(p+step*direction)[0]-model.evaluate(p-step*direction)[0])/(2*step)
        error=float(np.linalg.norm(fd-j@direction)/max(np.linalg.norm(fd),1e-15))
        if error>4e-4:raise ValueError('Joint original-row derivative failed')
        print('start',start,name,'initial',r@r*model.energy,'derivative',error,flush=True)
        fit=model.fit(p,260,free_indices=free);q=np.array(fit['parameters'])
        np.testing.assert_array_equal(q[frozen],seed[frozen]);measured=[];pair_ratios=[]
        for c,(cluster,index) in enumerate(zip(model.clusters,model.indices)):
            records=component_measurements(cluster,q[index])
            pair=np.array([records[s]['frozen_component_prediction'][bands[c].training] for s in (1,4)])
            pair_ratios.append(float(np.sum(abs(pair)**2)/np.sum(abs(pair.sum(axis=0))**2)))
            measured.append({str(s):dict(time_s=records[s]['time_s'],physical_path_change_m=records[s]['physical_path_change_m'],
                complex_coefficient_real_imag=np.c_[records[s]['complex_coefficient'].real,records[s]['complex_coefficient'].imag],
                conditional_information=records[s]['conditional_information'],
                prediction_power=np.sum(abs(records[s]['frozen_component_prediction'])**2,axis=1)) for s in (1,4)})
        fit.update(start=start,name=name,directional_derivative_error=error,prior_cost=model.prior_cost(q),
            prediction=subset_prediction(model,q,control['parts']),measured=measured,
            local_B_E_component_to_sum_energy_ratio=pair_ratios)
        result['trials'].append(fit);save()
        print('done',start,'cost',fit['cost'],'unused',sum(v['heldout_error'] for v in fit['prediction']['testing']),
            'gradient',fit['maximum_projected_gradient'],'bounds',fit['bound_indices'],flush=True)
    result['best_index']=int(np.argmin([f['cost'] for f in result['trials']]))
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
