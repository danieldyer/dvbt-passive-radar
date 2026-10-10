#!/usr/bin/env python3
"""Compare new matched-filter efficiency to the previous supported Hann operator.

The previous operator uses slow-decimation=1 (no FIR approximation), native
range/Doppler cells, circular current-symbol correlation, no migration, and no
carrier-noise weighting. Its best expected-SNR cell is selected in a generous
neighborhood without using noise realizations. This is a conditional model
comparison, not a measured target-SNR claim for the recording.
"""
import argparse,json
from pathlib import Path
import numpy as np
from sensitivity import K,NFFT,FS
from optimality_bound import shifted
from full_cpi_bound import select_cpi


def run(path,times,noise,new_results,output):
    s,p,h,origin,selected=select_cpi(path,times)
    new=json.loads(new_results.read_text())
    n=len(selected);time=selected-(selected[0]+selected[-1])/2
    window=.5-.5*np.cos(2*np.pi*np.arange(n)/n)
    within=(np.arange(NFFT)-4095.5)/FS
    df=1/(2880*.001088);half=int(np.floor(200/df))
    recorded=np.load(noise)['noise_variance'];results=[]
    for case in new['results']:
        delay,fd=case['delay_samples'],case['doppler_hz']
        d=recorded if case['noise']=='recorded' else np.ones(len(K))
        centre=int(np.round(fd/df));bins=np.unique(np.clip(np.arange(centre-5,centre+6),-half,half))
        frequencies=bins*df;ranges=np.arange(max(0,int(np.round(delay))-16),min(800,int(np.round(delay))+16)+1)
        spectral=np.zeros((len(bins),len(K)),complex);variance=np.zeros(len(bins));information=0
        temporal=window[None,:]*np.exp(-2j*np.pi*frequencies[:,None]*time)
        # Exact Hann-weighted constant projection of the previous operator.
        mean=np.sum(temporal,axis=1)/np.sum(window)
        temporal-=mean[:,None]*window
        for start in range(0,n,32):
            sl=slice(start,start+32);tt=time[sl]
            hh=h[sl] if case['channel']=='recorded' else np.ones_like(h[sl])
            waveform=shifted(s[sl],p[sl],origin[sl],delay-fd*FS/557625000*tt)*np.exp(2j*np.pi*fd*(tt[:,None]+within))
            q=np.fft.fft(waveform,axis=1)[:,K%NFFT]*hh
            reference=s[sl]*np.exp(-2j*np.pi*K*origin[sl,None]/NFFT)
            spectral+=temporal[:,sl]@(q*reference.conj())
            variance+=abs(temporal[:,sl])**2@(np.sum(abs(reference)**2*d,axis=1))
            information+=np.sum(abs(q)**2/d)
        responses=spectral@np.exp(2j*np.pi*K[:,None]*ranges[None,:]/NFFT)
        efficiencies=abs(responses)**2/(variance[:,None]*information)
        at=np.unravel_index(np.argmax(efficiencies),efficiencies.shape);eta=float(efficiencies[at])
        gain=10*np.log10(case['efficiency']/eta)+10*np.log10(1-1/(np.pi*3)**2)
        result=dict(**case,previous_best_efficiency=eta,previous_best_range_samples=int(ranges[at[1]]),
                    previous_best_doppler_hz=float(frequencies[at[0]]),conditional_snr_gain_db=float(gain))
        results.append(result);print(json.dumps(result),flush=True)
    output.write_text(json.dumps(dict(source=str(path),rows=n,cpi_s=n*.001088,
        previous='Exact supported Hann operator, slow-decimation=1; optimistic noiseless best native cell',
        new='Full-CPI matched-filter cases with independent 10 percent log-variance error',
        limitations='Specified diagonal covariance and template only. Shared guard/clock allowances cancel; the conservative new constant-projection allowance is retained. This is not a real-scene target SNR measurement.',
        results=results),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('times',type=Path)
    p.add_argument('noise',type=Path);p.add_argument('new_results',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.input,a.times,a.noise,a.new_results,a.output)
