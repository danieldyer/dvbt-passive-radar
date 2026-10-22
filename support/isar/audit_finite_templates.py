"""Test realised finite waveform templates against the ISAR mean-PSF model.

This is a fixed-coefficient operator experiment, not a fit to recorded target
returns. Signal amplitude, noise spectrum and oracle are held known. Snapshot
ensemble results do not establish long-aperture motion-estimator efficiency.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve

from finite_reference import (BINS, FS, NFFT, circular_profile, coefficient_planes,
                              covariance, echo, probes, range_weights)
from scene_dictionary import fingerprint
from symbol_moments import moments


def inner_inverse(matrix, vector):
    return solve(matrix, vector, assume_a='pos')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-channel',type=int,default=0,help='Zero uses every probe; positive values make a declared smoke control')
    args=parser.parse_args()
    if args.output.exists() or args.per_channel<0:
        raise ValueError('Use a new result path and nonnegative limit')
    selected=[];count=defaultdict(int);weights=defaultdict(list)
    for p in probes(args.input):
        if args.per_channel and count[p['channel']]>=args.per_channel:continue
        count[p['channel']]+=1
        p={k:(v.astype(complex if np.iscomplexobj(v) else float) if isinstance(v,np.ndarray) else v) for k,v in p.items()}
        selected.append(p)
        key=(p['channel'],int(p['time_s']>=100))
        weights[key].append(abs(p['response'])**2*abs(p['current'])**2/p['variance']/p['information'])
    means={k:np.mean(v,axis=0) for k,v in weights.items()}
    offsets=np.arange(-8,9);k=np.arange(BINS)-3408
    rows=[];gpu=[];started=time.monotonic()
    for index,p in enumerate(selected):
        h,v=p['response'],p['variance'];origin=p['reference_origin']-p['source_offset']
        previous_origin=p['reference_origin']-p['previous_source_offset']
        args0=(p['current'],p['previous'],h,v,origin)
        residual=p['measured']-h*p['current']*np.exp(-2j*np.pi*k*origin/NFFT)
        averaged=means[p['channel'],int(p['time_s']>=100)]
        current=abs(h)**2*abs(p['current'])**2/v/p['information']
        for delay in (16.125,80.375,393.125,800.125):
            planes=coefficient_planes(*args0,delay,offsets,guard=p['guard'])
            faithful=coefficient_planes(*args0,delay,offsets,guard=p['guard'],
                                       previous_fraction=previous_origin-np.floor(previous_origin))
            centre,tap,w=range_weights(delay)
            recorded=np.stack([p[name][(centre+offsets[:,None]+tap)%1024]@w for name in ('zero','one')])
            predicted=faithful@residual
            gpu.append({'channel':p['channel'],'time_s':p['time_s'],'delay_samples':delay,
                'maximum_absolute_error':float(abs(recorded-predicted).max()),
                'relative_error':float(np.linalg.norm(recorded-predicted)/max(np.linalg.norm(predicted),1e-30)),
                'previous_clock_linear_error':float(np.linalg.norm((faithful-planes).reshape(2*len(offsets),-1)) /
                    np.linalg.norm(planes))})
            pulse_current=circular_profile(current,delay,delay,offsets)
            pulse_mean=circular_profile(averaged,delay,delay,offsets)
            approximate_covariance=circular_profile(averaged,0,0,
                (offsets[:,None]-offsets).ravel()).reshape(len(offsets),len(offsets))/(NFFT*p['information'])
            for fd in (-100.,0.,100.):
                omega=2*np.pi*fd*NFFT/FS
                q=echo(p['current'],p['previous'],h,origin,delay,fd,guard=p['guard'])
                a=planes[0]-1j*omega*planes[1]
                signal=a@q;noise=covariance(a,v)
                full_information=float(np.vdot(q,q/v).real)
                profiled=float(np.vdot(signal,inner_inverse(noise,signal)).real)
                g0,g1,_,_=moments(omega);factor=g0-1j*omega*g1
                candidates={'current_circular':pulse_current*factor,'mean_circular':pulse_mean*factor}
                result={'channel':p['channel'],'time_s':p['time_s'],'delay_samples':delay,'doppler_hz':fd,
                    'raw_information':full_information,'profile_information':profiled,
                    'profile_retention':profiled/full_information,'models':{}}
                for name,model in candidates.items():
                    for covariance_name,cov in [('exact_covariance',noise),('mean_covariance',approximate_covariance*(1+omega**2/12))]:
                        coefficient=inner_inverse(cov,model)
                        numerator=np.vdot(coefficient,signal)
                        variance=float(np.vdot(coefficient,noise@coefficient).real)
                        information=abs(numerator)**2/variance
                        result['models'][name+'_'+covariance_name]={
                            'total_efficiency':float(information/full_information),
                            'template_efficiency_in_profiles':float(information/profiled),
                            'coherent_response_real_imag':[float(numerator.real),float(numerator.imag)],
                            'noise_variance':variance,
                            'relative_mean_error':float(np.linalg.norm(signal-model)/np.linalg.norm(signal))}
                if not -1e-10<=result['profile_retention']<=1+1e-8:
                    raise ValueError('Information increased under linear compression')
                rows.append(result)
        if index%10==0:print(json.dumps({'probes':index+1,'of':len(selected),'seconds':time.monotonic()-started}),flush=True)
    summaries=[]
    for channel in sorted(count):
        for delay in (16.125,80.375,393.125,800.125):
            for fd in (-100.,0.,100.):
                cases=[r for r in rows if (r['channel'],r['delay_samples'],r['doppler_hz'])==(channel,delay,fd)]
                raw=sum(r['raw_information'] for r in cases);compressed=sum(r['profile_information'] for r in cases)
                models={}
                for name in cases[0]['models']:
                    response=sum(complex(*r['models'][name]['coherent_response_real_imag']) for r in cases)
                    variance=sum(r['models'][name]['noise_variance'] for r in cases)
                    efficiency=abs(response)**2/variance/raw
                    models[name]={'snapshot_ensemble_loss_db':float(-10*np.log10(efficiency)),
                        'median_single_row_loss_db':float(np.median([-10*np.log10(r['models'][name]['total_efficiency']) for r in cases])),
                        'maximum_single_row_loss_db':float(max(-10*np.log10(r['models'][name]['total_efficiency']) for r in cases)),
                        'median_relative_mean_error':float(np.median([r['models'][name]['relative_mean_error'] for r in cases]))}
                summaries.append({'channel':channel,'delay_samples':delay,'doppler_hz':fd,'probes':len(cases),
                    'profile_loss_db':float(-10*np.log10(compressed/raw)),'models':models})
    result={'scope':__doc__,'complete':True,'input':str(args.input),'input_sha256':fingerprint(args.input),
        'source_sha256':{str(p):fingerprint(p) for p in [Path(__file__),Path(__file__).with_name('finite_reference.py'),Path(__file__).with_name('symbol_moments.py')]},
        'probe_count_by_channel':dict(count),'row_selection_limit':args.per_channel,
        'noise_scope':'Known frozen production per-carrier variance; independent proper-complex carrier noise. No claim of true recorded interference covariance.',
        'waveform_scope':'Production fractional-origin convention with exact Fourier retiming of previous symbol. GPU comparison also reproduces its linear previous-clock interpolation.',
        'efficiency_scope':'Finite occupied-FFT observation bound; excludes demodulation, response estimation, CP discarding, inter-row interference, binning and motion uncertainty. Snapshot ensemble shares only the declared synthetic echo amplitude; it is not a tracked physical CPI.',
        'gpu_comparison':gpu,'cases':rows,'summaries':summaries,'seconds':time.monotonic()-started,
        'accepted_calibration':False,'production_changed':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'seconds':result['seconds'],
        'maximum_gpu_absolute_error':max(r['maximum_absolute_error'] for r in gpu),
        'maximum_mean_model_loss_db':max(r['models']['mean_circular_mean_covariance']['snapshot_ensemble_loss_db'] for r in summaries)}))


if __name__=='__main__':main()
