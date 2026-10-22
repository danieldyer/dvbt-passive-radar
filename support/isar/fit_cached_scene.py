"""Fit a fixed spatial scene with independently declared temporal holdouts.

Uses exact cached complex responses. No physical motion or phase centre is
refitted here. Test-time stationary nuisance is eliminated identically from
observations and frozen predictions, avoiding unstable nuisance extrapolation.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from assess_dense_scene import fit
from linear_observation import LinearObservation
from matched_phase import history
from fit_gpga import binned_metrics
from scene_dictionary import fingerprint
from stationary import Stationary


def selected_times(time, interval, duration):
    if duration < 0 or not np.isfinite(duration):
        raise ValueError('Holdout duration must be finite and nonnegative')
    return (np.arange(len(time))%2==0 if duration==0 else
        np.floor((time-interval[0])/duration).astype(int)%2==0)


def retain_observation(directory, label, observation):
    np.save(directory/(label+'-design.npy'),observation.design)
    np.savez(directory/(label+'-data.npz'),data=observation.data,
        orthogonal_energy=observation.orthogonal_energy,observations=observation.observations)
    return LinearObservation(np.load(directory/(label+'-design.npy'),mmap_mode='r'),
        observation.data,observation.orthogonal_energy,observation.observations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dictionary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bounds-m',type=float,nargs=4,help='Optional subset of cached physical points')
    parser.add_argument('--holdout-seconds',type=float,default=1.)
    parser.add_argument('--variants',nargs='+',choices=['shared','independent','registered'],default=['shared','independent','registered'])
    parser.add_argument('--statistics-only',action='store_true',help='Prepare exact train/test QR statistics without a dense scene fit')
    args=parser.parse_args()
    manifest=args.dictionary/'dictionary.json'
    source=json.loads(manifest.read_text())
    if not source['complete'] or len(source['channels'])!=5:
        raise ValueError('Wait for all exact response matrices to finish')
    all_points=np.array(source['points_body_m'])
    keep=np.ones(len(all_points),bool)
    if args.bounds_m:
        x0,x1,y0,y1=args.bounds_m
        if not x0<x1 or not y0<y1:
            raise ValueError('Invalid point selection bounds')
        keep=(all_points[:,0]>=x0)&(all_points[:,0]<=x1)&(all_points[:,1]>=y0)&(all_points[:,1]<=y1)
    indices=np.flatnonzero(keep)
    if not len(indices):
        raise ValueError('No spatial points selected')
    args.output.mkdir(parents=True,exist_ok=True)
    variants=[] if args.statistics_only else args.variants
    identity={'dictionary_manifest_sha256':fingerprint(manifest),'point_indices':indices.tolist(),
              'holdout_seconds':args.holdout_seconds,'variants':variants}
    result_path=args.output/'fit.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text())
        if previous['identity']!=identity:
            raise ValueError('Use a new output directory for a different test')
    result={'identity':identity,'dictionary':str(args.dictionary),'source_motion':source['source_motion'],
        'points_body_m':all_points[keep].tolist(),'holdout_seconds':args.holdout_seconds,'scope':__doc__,
        'phase_gauge':'Fixed existing channel gains; registered variant adds four constant phases with Ten as reference.',
        'limitation':'Motion and support decisions used this aperture. Only scene coefficients and optional channel phases are refitted on the new training subset.',
        'trials':[],'complete':False}
    train,test=[],[]
    started=time.perf_counter()
    for c in range(5):
        with np.load(args.dictionary/f'channel-{c}-observation.npz') as loaded:
            observation=dict(loaded)
        matrix=np.load(args.dictionary/f'channel-{c}-design.npy',mmap_mode='r')
        expected=source['channels'][c]
        if list(matrix.shape)!=expected['shape'] or fingerprint(args.dictionary/f'channel-{c}-design.npy')!=expected['design_sha256']:
            raise ValueError('Cached response matrix has changed')
        selected=selected_times(observation['time'],source['interval_s'],args.holdout_seconds)
        rows=np.repeat(selected,observation['data'].shape[1])
        for label,mask,container in [('train',rows,train),('test',~rows,test)]:
            nuisance=Stationary(observation['background'],mask)
            design=nuisance.remove(matrix[np.ix_(mask,indices)])
            data=nuisance.remove(observation['data'].ravel()[mask])
            reduced=LinearObservation.compress(design,data,nuisance.rank)
            container.append(retain_observation(args.output,f'channel-{c}-{label}',reduced))
            del design,data,reduced,nuisance
        del matrix,observation
        print(json.dumps({'channel':c,'points':len(indices),'stage':'Holdout observations ready','seconds':time.perf_counter()-started}),flush=True)
    result.update(statistics_complete=True,complete=not variants,seconds=time.perf_counter()-started)
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    shared_solution=None
    for variant in variants:
        trial=fit(train,test,independent=variant=='independent',register_phases=variant=='registered',shared_solution=shared_solution)
        trial['variant']=variant
        if variant=='shared':
            a=np.array(trial['channels'][0]['coefficient_real_imag'])
            shared_solution=(a[:,0]+1j*a[:,1],trial['channels'][0]['regularization'])
        for c in range(5):
            with np.load(args.dictionary/f'channel-{c}-observation.npz') as loaded:
                observation=dict(loaded)
            selected=selected_times(observation['time'],source['interval_s'],args.holdout_seconds)
            matrix=np.load(args.dictionary/f'channel-{c}-design.npy',mmap_mode='r')
            coefficient=np.array(trial['channels'][c]['coefficient_real_imag'])
            full=np.zeros(len(keep),complex)
            full[keep]=coefficient[:,0]+1j*coefficient[:,1]
            phase=1. if trial['phase_registration'] is None else np.exp(1j*np.radians(trial['phase_registration']['relative_phase_degrees'][c]))
            prediction=(phase*(matrix@full)).reshape(observation['data'].shape)
            test_rows=np.repeat(~selected,observation['data'].shape[1])
            background=Stationary(observation['background'],test_rows)
            y=background.remove(observation['data'].ravel()[test_rows]).reshape((~selected).sum(),-1)
            predicted=background.remove(prediction.ravel()[test_rows]).reshape(y.shape)
            direct_error=float(np.sum(abs(y-predicted)**2))
            relative=abs(direct_error/trial['channels'][c]['prediction_error']-1)
            if relative>2e-8:
                raise ValueError(f'Original-space prediction and QR score disagree: {relative:g}')
            z,weight=history(y,predicted)
            metrics=binned_metrics(observation['time'][~selected],z,weight,np.ones(len(weight),bool),np.ones(1,complex),1.)
            trial['channels'][c]['one_second_phase_metrics']=metrics
            trial['channels'][c]['prediction_regeneration_relative_error']=relative
            np.savez(args.output/f'{variant}-channel-{c}-prediction.npz',time=observation['time'],
                selected=selected,prediction=prediction)
            del matrix,observation
        result['trials'].append(trial)
        result['seconds']=time.perf_counter()-started
        result['complete']=len(result['trials'])==len(variants)
        result_path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'variant':variant,'heldout_explained_fraction':trial['heldout_explained_fraction'],
            'phase_rms_degrees':[c['one_second_phase_metrics']['phase_rms_degrees_by_point'][0] for c in trial['channels']],
            'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    main()
