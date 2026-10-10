#!/usr/bin/env python3
"""Separate common response variation from per-carrier noise before bounding gain.

The conditional covariance model is diagonal physical carrier noise plus two
common modes in source-divided coordinates (gain/phase and timing). It is not
an unconditional certificate for an unknown scene or temporal covariance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sensitivity import BINS, K, NFFT, blocks


def collect(path):
    records=[];variances=[];responses=[];powers=[];mode_covariances=[];real_covariances=[];modes=None
    for b in blocks(path):
        sl=slice(b['first'],b['last'])
        good=np.asarray(b['valid'][sl],bool).all(axis=1)
        x=np.asarray(b['s'][sl][good],complex)*np.exp(-2j*np.pi*K*b['reference_origin']/NFFT)
        y=np.asarray(b['y'][sl][good],complex)
        h=np.asarray(b['h'],complex)
        r=y/x
        r-=r.mean(axis=0)
        if modes is None:modes,_=np.linalg.qr(np.column_stack((h,1j*K/NFFT*h)))
        a=r@modes.conj()
        innovation=(r-a@modes.T)*x
        d=np.mean(abs(np.diff(innovation,axis=0))**2,axis=0)/2
        p=np.mean(abs(x)**2,axis=0)
        covariance=a.T@a.conj()/len(a)
        # Remove the thermal contribution to the estimated common coefficients.
        covariance-=modes.conj().T@((d/p)[:,None]*modes)
        real_coefficients=np.column_stack((a.real,a.imag))
        real_covariance=real_coefficients.T@real_coefficients/len(a)
        thermal=modes.conj().T@((d/p)[:,None]*modes)
        real_covariance-=.5*np.block([[thermal.real,-thermal.imag],[thermal.imag,thermal.real]])
        eigenvalues,real_basis=np.linalg.eigh(real_covariance)
        real_covariance=(real_basis*np.maximum(eigenvalues,0))@real_basis.T
        eigenvectors,eigenbasis=np.linalg.eigh(covariance)
        covariance=(eigenbasis*np.maximum(eigenvectors,0))@eigenbasis.conj().T
        z=innovation/np.sqrt(d)
        records.append(dict(batch=b['index'],rows=len(x),
            gaussian_power_median_ratio=float(np.median(abs(z)**2)/np.log(2)),
            gaussian_fourth_moment_ratio=float(np.mean(abs(z)**4)/2),
            lag_correlations=[float(abs(np.sum(z[lag:]*z[:-lag].conj()))/np.sum(abs(z)**2)) for lag in (1,2,4,8)],
            carrier_lag_correlations=[float(abs(np.sum(z[:,lag:]*z[:,:-lag].conj()))/np.sum(abs(z)**2)) for lag in (1,2,4,8)],
            innovation_pseudocovariance_ratio=float(abs(np.sum(z*z))/np.sum(abs(z)**2)),
            common_variation_trace=float(np.trace(covariance).real),
            common_pseudocovariance_ratio=float(np.linalg.norm(a.T@a/len(a))/np.linalg.norm(a.T@a.conj()/len(a))),
            per_carrier_variance_median=float(np.median(d))))
        variances.append(d);responses.append(h);powers.append(p);mode_covariances.append(covariance)
        real_covariances.append(real_covariance)
    d=np.mean(variances,axis=0);h=np.mean(responses,axis=0);p=np.mean(powers,axis=0)
    return dict(records=records,d=d,h=h,p=p,modes=modes,modes_cov=np.mean(mode_covariances,axis=0),
                real_cov=np.mean(real_covariances,axis=0),per_batch_d=np.array(variances))


def bound(model, delay):
    d,h,p=model['d'],model['h'],model['p']
    modes=model['modes']
    covariance=model['modes_cov']
    phase=np.exp(-2j*np.pi*K*delay/NFFT)
    q=h*phase;w=p*phase
    inverse=p/d
    v=modes.conj().T@(inverse*q)
    gram=modes.conj().T@(inverse[:,None]*modes)
    # Woodbury form avoids inversion of a 6817 x 6817 matrix.
    values,basis=np.linalg.eigh(covariance)
    square_root=(basis*np.sqrt(np.maximum(values,0)))@basis.conj().T
    correction=np.linalg.solve(np.eye(2)+square_root@gram@square_root,square_root@v)
    information=float(np.sum(abs(q)**2*inverse)-(v.conj()@square_root@correction).real)
    white_information=float(np.sum(abs(q)**2*inverse))
    thermal_variance=float(np.sum(abs(w)**2/inverse))
    projection=modes.conj().T@w
    common_variance=float((projection.conj()@covariance@projection).real)
    current=abs(np.vdot(w,q))**2/(thermal_variance+common_variance)
    white_current=abs(np.vdot(w,q))**2/thermal_variance
    real_modes=np.block([[modes.real,-modes.imag],[modes.imag,modes.real]])
    real_q=np.column_stack((np.concatenate((q.real,q.imag)),np.concatenate((-q.imag,q.real))))
    real_w=np.column_stack((np.concatenate((w.real,w.imag)),np.concatenate((-w.imag,w.real))))
    real_inverse=np.tile(2*inverse,2)
    real_diagonal=1/real_inverse
    real_cov=model['real_cov']
    values,basis=np.linalg.eigh(real_cov)
    root=(basis*np.sqrt(np.maximum(values,0)))@basis.T
    gram=real_modes.T@(real_inverse[:,None]*real_modes)
    projection_q=real_modes.T@(real_inverse[:,None]*real_q)
    information_real=real_q.T@(real_inverse[:,None]*real_q)
    information_real-=projection_q.T@root@np.linalg.solve(np.eye(4)+root@gram@root,root@projection_q)
    projection_w=real_modes.T@real_w
    output_cov=real_w.T@(real_diagonal[:,None]*real_w)+projection_w.T@real_cov@projection_w
    signal_gain=real_w.T@real_q
    current_information=signal_gain.T@np.linalg.solve(output_cov,signal_gain)
    values,basis=np.linalg.eigh(current_information)
    inverse_root=(basis/np.sqrt(values))@basis.T
    phase_ratios=np.linalg.eigvalsh(inverse_root@information_real@inverse_root)
    return dict(delay_samples=delay,diagonal_noise_only_gain_db=float(10*np.log10(white_information/white_current)),
        two_mode_conditional_gain_db=float(10*np.log10(information/current)),
        common_to_thermal_output_variance=float(common_variance/thermal_variance),
        information_fraction_vs_thermal_only=float(information/white_information),
        improper_noise_gain_db_phase_min_max=(10*np.log10(phase_ratios)).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();models=[collect(p) for p in args.inputs]
    results=[]
    for path,model in zip(args.inputs,models):
        n=len(model['per_batch_d']);train=model['per_batch_d'][:n//2].mean(axis=0)
        test=model['per_batch_d'][n//2:].mean(axis=0)
        # Prediction with diagonal physical noise: compare held-out spectrum to
        # the training spectrum; this is not a full correlation-model test.
        ratio=test/train
        h,p=model['h'],model['p']
        current=abs(np.sum(h*p))**2/np.sum(test*p)
        optimum=np.sum(abs(h)**2*p/test)
        trained=np.sum(abs(h)**2*p/train)**2/np.sum(abs(h)**2*p*test/train**2)
        results.append(dict(source=str(path),records=model['records'],
            heldout_variance_ratio_quantiles=np.quantile(ratio,[.05,.5,.95]).tolist(),
            heldout_oracle_diagonal_gain_db=float(10*np.log10(optimum/current)),
            heldout_training_weights_gain_db=float(10*np.log10(trained/current)),
            bounds=[bound(model,d) for d in (0,1,4,8,16,64,128,256,512,800)]))
        args.output.with_name(path.stem+'-carrier-noise.npz').parent.mkdir(parents=True,exist_ok=True)
        np.savez(args.output.with_name(path.stem+'-carrier-noise.npz'),noise_variance=model['d'],
                 training_noise=train,heldout_noise=test,response=h)
    args.output.write_text(json.dumps(dict(model='white slow-time innovations; diagonal carrier noise plus gain/phase and timing modes',
        limitations='Common modes can include wanted reflections. Temporal whiteness and covariance structure require validation. No adaptive-response self-subtraction or finite-waveform coupling is included.',
        excerpts=results),indent=2)+'\n')


if __name__=='__main__':main()
