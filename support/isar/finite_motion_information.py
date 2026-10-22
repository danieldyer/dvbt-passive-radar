"""Finite-waveform delay/rate information and local mean-template bias.

Frozen extraction coordinates follow the true constant path rate; their
derivatives are zero. Only the physical echo changes. Unknown complex target
amplitude is eliminated across each band/interval, before comparing Fisher
information. This conditional sparse-row experiment is not a calibration of
the recorded interference covariance or the ship's scattering model.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from compression_information import point_information, relative_information
from finite_reference import (FS, NFFT, coefficient_planes, covariance,
    circular_profile, probes, segment)
from scene_dictionary import fingerprint
from symbol_moments import moments


def echo_jacobian(current, previous, response, origin, delay, doppler, *,
                  guard=16, nfft=NFFT, rate=FS):
    k=np.arange(len(current))-(len(current)-1)//2
    samples=segment(current,previous,origin+delay,0,nfft,guard=guard,nfft=nfft)
    derivative=segment(current*(-2j*np.pi*k/nfft),previous*(-2j*np.pi*k/nfft),
                       origin+delay,0,nfft,guard=guard,nfft=nfft)
    time=(np.arange(nfft)-(nfft-1)/2)/rate
    phase=np.exp(2j*np.pi*doppler*time)
    spectrum=np.fft.fft(np.stack([samples,derivative,samples*(2j*np.pi*time)])*phase,axis=1)[:,k % nfft]*response
    return spectrum[0],spectrum[1],spectrum[2]


def local_bias(signal, mean, derivative):
    """First-order real parameter displacement, after complex amplitude fit."""
    alpha=np.vdot(mean,signal)/np.vdot(mean,mean).real
    residual=signal-alpha*mean
    jac=alpha*derivative
    jac-=mean[:,None]*(mean.conj()@jac)[None,:]/np.vdot(mean,mean).real
    information=2*np.real(jac.conj().T@jac)
    score=2*np.real(jac.conj().T@residual)
    return np.linalg.solve(information,score)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve existing results')
    groups=defaultdict(list)
    for row in probes(args.input):
        row={k:(v.astype(complex if np.iscomplexobj(v) else float) if isinstance(v,np.ndarray) else v) for k,v in row.items()}
        groups[row['channel'],int(row['time_s']>=100)].append(row)
    results=[];offsets=np.arange(-8,9)
    for (channel,interval),rows in groups.items():
        epoch=np.mean([r['time_s'] for r in rows]);rf=rows[0]['rf_hz'];carrier=2*np.pi*rf/FS
        spectral=np.mean([abs(r['response'])**2*abs(r['current'])**2/r['variance']/r['information'] for r in rows],axis=0)
        k=np.arange(len(spectral))-(len(spectral)-1)//2
        for centre,fd in [(80.375,10.),(393.125,0.),(393.125,10.),(393.125,100.),(800.125,100.)]:
            raw_signal=[];raw_derivative=[];compressed_signal=[];compressed_derivative=[];means=[];mean_derivatives=[]
            for r in rows:
                t=r['time_s']-epoch;delay=centre-fd*FS/rf*t;origin=r['reference_origin']-r['source_offset']
                q,qd,qf=echo_jacobian(r['current'],r['previous'],r['response'],origin,delay,fd,guard=r['guard'])
                derivative=np.column_stack((qd-1j*carrier*q,t*qd-rf/FS*qf-1j*carrier*t*q))
                planes=coefficient_planes(r['current'],r['previous'],r['response'],r['variance'],origin,delay,offsets,guard=r['guard'])
                omega=2*np.pi*fd*NFFT/FS;a=planes[0]-1j*omega*planes[1]
                lower=np.linalg.cholesky(covariance(a,r['variance']))
                white=solve_triangular(lower,np.column_stack((a@q,a@derivative)),lower=True)
                raw_signal.append(q/np.sqrt(r['variance']));raw_derivative.append(derivative/np.sqrt(r['variance'])[:,None])
                compressed_signal.append(white[:,0]);compressed_derivative.append(white[:,1:])
                pulse=circular_profile(spectral,delay,delay,offsets)
                pulse_d=circular_profile(spectral*(-2j*np.pi*k/NFFT),delay,delay,offsets)
                g0,g1,dg0,dg1=moments(omega)
                factor=g0-1j*omega*g1
                # Centre-Doppler combination is a frozen observation operator.
                factor_f=(dg0-1j*omega*dg1)*2*np.pi*NFFT/FS
                mean=pulse*factor;md=pulse_d*factor;mf=pulse*factor_f
                mj=np.column_stack((md-1j*carrier*mean,t*md-rf/FS*mf-1j*carrier*t*mean))
                white_mean=solve_triangular(lower,np.column_stack((mean,mj)),lower=True)
                means.append(white_mean[:,0]);mean_derivatives.append(white_mean[:,1:])
            raw=np.concatenate(raw_signal);dr=np.concatenate(raw_derivative)
            compressed=np.concatenate(compressed_signal);dc=np.concatenate(compressed_derivative)
            energy,full=point_information(raw,dr);_,reduced=point_information(compressed,dc)
            fraction=relative_information(full,reduced)
            bias=local_bias(compressed,np.concatenate(means),np.concatenate(mean_derivatives))
            result={'channel':channel,'interval':interval,'time_start_end_s':[rows[0]['time_s'],rows[-1]['time_s']],
                'rows':len(rows),'midpoint_delay_samples':centre,'doppler_hz':fd,
                'relative_information_eigenvalues':fraction.tolist(),'worst_information_loss_db':float(-10*np.log10(fraction[0])),
                'mean_template_first_order_bias_samples_and_samples_per_s':bias.tolist(),
                'mean_template_first_order_bias_metres_and_mps':(bias*299792458/FS).tolist(),
                'squared_bias_in_crlb_units_per_parameter_at_snr_db':{str(snr):float(10**(snr/10)/energy*bias@full@bias/2) for snr in (20,40)},
                'unknown_complex_amplitude_eliminated':True}
            results.append(result);print(json.dumps({k:v for k,v in result.items() if k in ['channel','interval','midpoint_delay_samples','doppler_hz','worst_information_loss_db','mean_template_first_order_bias_metres_and_mps']}),flush=True)
    result={'scope':__doc__,'complete':True,'input':str(args.input),'input_sha256':fingerprint(args.input),
        'implementation_sha256':{str(f):fingerprint(f) for f in [Path(__file__),Path(__file__).with_name('finite_reference.py'),Path(__file__).with_name('compression_information.py'),Path(__file__).with_name('symbol_moments.py')]},
        'parameterization':'Delay intercept in samples, delay rate in samples/s; fD=-rf*delay_rate/FS. Carrier phase and within-symbol Doppler derivatives included.',
        'bias_scope':'Local first-order approximation under exact compressed covariance; not a completed nonlinear recovery experiment. Does not include original 16-symbol averaging, phase-centre nuisance or unknown distributed scattering.',
        'cases':results,'production_changed':False,'accepted_calibration':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
