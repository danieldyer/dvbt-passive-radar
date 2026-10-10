#!/usr/bin/env python3
"""Independent observation-space matched-filter efficiency; no noise realizations or fitted gains."""
import argparse,json
from pathlib import Path
import numpy as np
from sensitivity import NFFT,BINS,FS,K,blocks


def shifted(source,previous,origin,delay,balance=False):
    coordinate=origin+delay
    integer=np.floor(coordinate).astype(int);fraction=coordinate-integer
    phase=np.exp(-2j*np.pi*K*fraction[:,None]/NFFT)
    if balance:
        source=source*np.where(abs(abs(source)**2-16/9)<1e-5,9/16,1)
        previous=previous*np.where(abs(abs(previous)**2-16/9)<1e-5,9/16,1)
    spectra=np.zeros((len(source)*2,NFFT),complex)
    spectra[:len(source),K%NFFT]=source*phase
    spectra[len(source):,K%NFFT]=previous*phase
    time=np.fft.ifft(spectra,axis=1);index=np.arange(NFFT)[None,:]-integer[:,None]
    # Previous useful symbol plus its cyclic prefix, continuous finite waveform.
    use_previous=index < -512
    index=np.where(use_previous,index+8704,index)%NFFT
    return np.where(use_previous,time[len(source):][np.arange(len(source))[:,None],index],
                    time[:len(source)][np.arange(len(source))[:,None],index])


def interpolated(source,previous,origin,delay):
    rows=len(source);fraction=origin-np.floor(origin);integer=np.floor(origin).astype(int)
    balance=lambda v:v*np.where(abs(abs(v)**2-16/9)<1e-5,9/16,1)
    spectra=np.zeros((2*rows,NFFT),complex);phase=np.exp(-2j*np.pi*K*fraction[:,None]/NFFT)
    spectra[:rows,K%NFFT]=balance(source)*phase;spectra[rows:,K%NFFT]=balance(previous)*phase
    time=np.fft.ifft(spectra,axis=1)
    j=np.arange(9184)[None,:]-928-integer[:,None];old=j < -512
    index=np.where(old,j+8704,j)%NFFT
    padded=np.zeros((rows,9216),complex)
    padded[:,:9184]=np.where(old,time[rows:][np.arange(rows)[:,None],index],time[:rows][np.arange(rows)[:,None],index])
    lags=np.r_[np.arange(929),np.arange(-64,0)];x=delay[:,None]-lags
    # Exact 1024-point periodic cardinal interpolant, including the crop and negative context.
    den=1024*np.sin(np.pi*x/1024)
    cardinal=np.divide(np.sin(np.pi*x),den,out=np.ones_like(x),where=abs(den)>1e-12)*np.exp(-1j*np.pi*x/1024)
    taps=np.zeros((rows,9216),complex);taps[:,lags%9216]=cardinal.conj()
    return np.fft.ifft(np.fft.fft(padded,axis=1)*np.fft.fft(taps,axis=1),axis=1)[:,928:928+NFFT]


def run(path,output):
    sources=[];previous=[];response=[];origins=[]
    for b in blocks(path):
        for row in range(max(1,b['first']),b['last'],37):
            if np.asarray(b['valid'][row-1:row+1],bool).all():
                sources.append(np.array(b['s'][row]));previous.append(np.array(b['s'][row-1]));
                response.append(np.array(b['h']));origins.append(b['reference_origin'])
        if len(sources)>=128:break
    s=np.array(sources[:128]);p=np.array(previous[:128]);h_measured=np.array(response[:128]);origin=np.array(origins[:128]);rows=len(s)
    if rows<64:raise ValueError('Insufficient independent complete reference symbols')
    dt=.001088;count=int(np.ceil(3/dt));fft=4*2880;df=1/(fft*dt)
    # Stratified temporal quadrature over the actual 3 s CPI, independent QAM symbols.
    time=(np.linspace(0,count-1,rows)-(count-1)/2)*dt
    within=(np.arange(NFFT)-4095.5)/FS
    h_shapes={'flat':np.ones_like(h_measured),'recorded':h_measured,
              'minimum_phase':np.broadcast_to(np.exp(.3*np.exp(-2j*np.pi*K/NFFT)+.14j*np.exp(-4j*np.pi*K/NFFT)),h_measured.shape)}
    noise={'white':np.ones(BINS),'smooth_20db':10**(2*(K/K.max())**2),
           'strong_narrowband':np.where(abs(K-2432)<=1,1e6,1.)}
    results=[]
    for name,h in h_shapes.items():
        for color,d in noise.items():
            for delay in (16.011,128.125,384.125,600.125,800.125):
                for fd in (-200.,-(1253.5)*df,-101.041,1.17,3.01,101.041,(1253.5)*df,200.):
                    fc=np.round(fd/df)*df;dc=np.round(delay*4)/4
                    truth_delay=delay-fd*FS/557625000*time
                    candidate_delay=dc-fc*FS/557625000*time
                    q_time=shifted(s,p,origin,truth_delay)*np.exp(2j*np.pi*fd*(time[:,None]+within))
                    w_time=interpolated(s,p,origin,candidate_delay)*np.exp(2j*np.pi*fc*time[:,None])*(1+2j*np.pi*fc*within)
                    q=np.fft.fft(q_time,axis=1)[:,K%NFFT]*h
                    w=np.fft.fft(w_time,axis=1)[:,K%NFFT]*h/d
                    # Independent estimation errors; true covariance is still used in the SNR denominator.
                    rng=np.random.default_rng(43059)
                    w*=np.exp(-.1*rng.standard_normal(BINS))[None,:]
                    eta=abs(np.vdot(w,q))**2/(np.sum(abs(w)**2*d)*np.sum(abs(q)**2/d))
                    # Additional conservative losses outside the retained FFT observations.
                    loss=-10*np.log10(eta)+10*np.log10(17/16)+10/np.log(10)*np.deg2rad(3)**2-10*np.log10(1-1/(np.pi*3)**2)
                    results.append(dict(channel=name,noise=color,delay_samples=delay,doppler_hz=fd,
                                        candidate_delay_samples=dc,candidate_doppler_hz=fc,
                                        observation_efficiency=float(eta),loss_including_cp_clock_db=float(loss)))
                    print(name,color,delay,fd,round(loss,5),flush=True)
    result=dict(source=str(path),rows=rows,cpi_s=count*dt,doppler_grid_hz=df,
                benchmark='constant path rate; 0.6-30 km, |fD|>=3/CPI Hz and <=200 Hz; proper independent Gaussian carrier noise',
                formula='eta=|w^H q|^2/[(w^H D w)(q^H D^-1 q)]',
                includes=['within-symbol Doppler moment','finite previous-symbol context','actual QAM/pilots','candidate range/Doppler grid','cropped finite-CAF range interpolation','10 percent RMS log carrier-variance estimation error',
                          'correct channel and diagonal noise shape','discarded guard interval mean-energy allowance','3 degree RMS phase allowance','conservative stationary suppression allowance'],
                excludes=['unidentified temporal/full covariance','unknown template mismatch','scene-dependent direct-path leakage',
                          'unknown front-end losses beyond existing regression mask','Keystone interpolation numerical error (separate GPU regression)'],
                worst_loss_db=max(r['loss_including_cp_clock_db'] for r in results),results=results)
    output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.input,a.output)
