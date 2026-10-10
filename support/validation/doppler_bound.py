#!/usr/bin/env python3
"""Doppler noise profile and held-out local matched-filter information bounds."""
import argparse
import json
from pathlib import Path

import numpy as np
from sensitivity import BINS, DT, K, NFFT, map_at, window


def white_local_model(name, radius):
    n=8192;w=window(name,n)
    signal=np.fft.fft(w)/w.sum()
    covariance=np.fft.fft(w*w)*n/w.sum()**2
    offsets=np.arange(-radius,radius+1)
    q=signal[offsets%n]
    c=covariance[(offsets[:,None]-offsets)%n]
    return q,c


def white_local_bound(name, radius):
    q,c=white_local_model(name,radius)
    information=np.vdot(q,np.linalg.solve(c,q)).real
    return dict(window=name,cells=len(q),information_vs_untapered=float(information),
        gain_vs_single_cell_db=float(10*np.log10(information*c[radius,radius].real)))


def snr(weights, q, covariance):
    return abs(np.vdot(weights,q))**2/np.vdot(weights,covariance@weights).real


def regularized_weights(train, validation, q):
    """Choose shrinkage on validation only; the evaluation block is unseen."""
    _,white=white_local_model('hann',len(q)//2)
    correlation=white/white[0,0].real
    deviation=np.sqrt(train.diagonal().real)
    target=correlation*deviation[:,None]*deviation
    scale=np.trace(train).real/len(q)
    candidates=[]
    for alpha in (0.,.1,.25,.5,.75,1.):
        for ridge in (1e-5,1e-3,1e-2):
            covariance=(1-alpha)*train+alpha*target+ridge*scale*np.eye(len(q))
            weights=np.linalg.solve(covariance,q)
            candidates.append((snr(weights,q,validation),weights,alpha,ridge))
    _,weights,alpha,ridge=max(candidates,key=lambda item:item[0])
    return weights,alpha,ridge


def white_profile_control(windows=31):
    """Same FFT/window/PSD estimator, white noise in the occupied RF band."""
    rng=np.random.default_rng(71034);n=int(1/DT);profiles=[]
    for _ in range(windows):
        spectrum=(rng.standard_normal((n,BINS),dtype=np.float32)
                  +1j*rng.standard_normal((n,BINS),dtype=np.float32))/np.sqrt(2)
        packed=np.zeros((n,NFFT),np.complex64);packed[:,K.astype(int)%NFFT]=spectrum
        rows=np.fft.ifft(packed,axis=1)[:,:801]
        m,f=map_at(rows,1,remove_dc=True)
        keep=(abs(f)>5)&(abs(f)<=200)
        profiles.append(np.median(abs(m[keep,100:780])**2,axis=1)/np.log(2))
    profile=np.mean(profiles,axis=0);profile/=np.median(profile)
    return dict(nonoverlapping_windows=windows,
        noise_profile_db_quantiles=(10*np.log10(np.quantile(profile,[0,.05,.5,.95,1]))).tolist())


def measure(cache_path):
    a=np.load(cache_path);rows=a['after'];n=int(1/DT);radius=4
    psd=[];covariances=[];frequency=None
    centres=(10.,50.,100.,150.,190.)
    for start in range(0,len(rows)-n+1,n):
        m,f=map_at(rows[start:start+n],1,remove_dc=True)
        region=abs(f)<=200
        psd.append(np.median(abs(m[region,100:780])**2,axis=1)/np.log(2))
        frequency=f[region]
        covariance=[]
        for centre in centres:
            k=int(np.argmin(abs(f-centre)))
            z=m[k-radius:k+radius+1,100:780]
            covariance.append(z@z.conj().T/z.shape[1])
        covariances.append(covariance)
    psd=np.asarray(psd)
    first=np.mean(psd[:len(psd)//2],axis=0);second=np.mean(psd[len(psd)//2:],axis=0)
    first_split=len(covariances)//3;second_split=2*len(covariances)//3
    covariance_train=np.mean(covariances[:first_split],axis=0)
    covariance_validation=np.mean(covariances[first_split:second_split],axis=0)
    covariance_test=np.mean(covariances[second_split:],axis=0)
    q=np.zeros(2*radius+1,complex);q[radius-1:radius+2]=(-.5,1,-.5)
    bounds=[]
    for centre,train,validation,test in zip(centres,covariance_train,covariance_validation,covariance_test):
        # The ridge is stated and tiny; report oracle and held-out application
        # separately so an overfitted inverse cannot masquerade as sensitivity.
        ridge=1e-4*np.trace(train).real/len(train)
        weights=np.linalg.solve(train+ridge*np.eye(len(train)),q)
        trained_snr=snr(weights,q,test)
        regularized,alpha,selected_ridge=regularized_weights(train,validation,q)
        _,white=white_local_model('hann',radius)
        fixed=np.linalg.solve(white,q)
        optimum=np.vdot(q,np.linalg.solve(test,q)).real
        baseline=1/test[radius,radius].real
        correlation=test/np.sqrt(test.diagonal().real[:,None]*test.diagonal().real)
        bounds.append(dict(centre_hz=centre,
            heldout_oracle_local_gain_db=float(10*np.log10(optimum/baseline)),
            heldout_trained_weights_gain_db=float(10*np.log10(trained_snr/baseline)),
            heldout_regularized_weights_gain_db=float(10*np.log10(snr(regularized,q,test)/baseline)),
            heldout_known_window_weights_gain_db=float(10*np.log10(snr(fixed,q,test)/baseline)),
            validation_selected_shrinkage_fraction=alpha,
            validation_selected_ridge_fraction=selected_ridge,
            ridge_fraction=1e-4,condition_number=float(np.linalg.cond(train)),
            first_neighbour_noise_correlation=float(np.diag(correlation,1).real.mean()),
            second_neighbour_noise_correlation=float(np.diag(correlation,2).real.mean())))
    keep=abs(frequency)>5
    average=psd.mean(axis=0);scale=np.median(average[keep])
    normalised=average[keep]/scale
    result=dict(source=str(cache_path),cpi_s=n*DT,nonoverlapping_windows=len(psd),
        noise_profile_db_quantiles=(10*np.log10(np.quantile(normalised,[0,.05,.5,.95,1]))).tolist(),
        heldout_psd_ratio_quantiles=np.quantile(second[keep]/first[keep],[.05,.5,.95]).tolist(),
        covariance_windows_train_validation_test=[first_split,second_split-first_split,len(covariances)-second_split],
        local_bounds=bounds,
        interpretation='Noise covariance uses other range cells and disjoint times; returns and signal-dependent clutter may remain. Gains assume an on-bin constant-amplitude echo in the retained map; no trajectory or exact finite-waveform correction.')
    return result,dict(frequency=frequency,mean_psd=average,training_psd=first,testing_psd=second)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();results=[]
    for path in a.inputs:
        result,profile=measure(path);results.append(result)
        np.savez(a.output.with_name(path.stem+'-doppler-noise.npz'),**profile)
    theory=[white_local_bound(name,r) for name in ('hann','hamming') for r in (0,1,2,4,8,16)]
    a.output.write_text(json.dumps(dict(white_noise_bounds=theory,
        white_noise_profile_control=white_profile_control(),excerpts=results),indent=2)+'\n')


if __name__=='__main__':main()
