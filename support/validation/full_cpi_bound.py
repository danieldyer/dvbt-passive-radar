#!/usr/bin/env python3
"""Full-symbol, full-CPI observation-space checks complement the stratified parameter sweep."""
import argparse,json
from pathlib import Path
import numpy as np
from sensitivity import blocks,K,NFFT,FS
from optimality_bound import shifted,interpolated


def select_cpi(path,times_path):
    selected=np.fromfile(times_path,'<f8');t0,t1=selected[0],selected[-1];sources=[];previous=[];hs=[];origins=[];times=[]
    for b in blocks(path):
        t=(b['origin']+b['coordinates']['sample']*b['decimation'])/b['rate']
        for r in range(max(1,b['first']),b['last']):
            if t0-1e-10<=t[r]<=t1+1e-10 and np.asarray(b['valid'][r-1:r+1],bool).all():
                sources.append(np.array(b['s'][r]));previous.append(np.array(b['s'][r-1]));hs.append(np.array(b['h']));origins.append(b['reference_origin']);times.append(t[r])
        if t[-1]>t1:break
    if len(times)!=len(selected) or not np.allclose(times,selected,atol=1e-10,rtol=0):raise ValueError('CPI coordinate mismatch')
    s,p,h,origin=map(np.asarray,(sources,previous,hs,origins))
    return s,p,h,origin,selected


def run(path,times_path,noise_path,output):
    s,p,h,origin,selected=select_cpi(path,times_path);t0,t1=selected[0],selected[-1]
    dt=.001088;df=1/(4*2880*dt);time=selected-(t0+t1)/2
    within=(np.arange(NFFT)-4095.5)/FS;recorded_noise=np.load(noise_path)['noise_variance'];jitter=np.exp(.1*np.random.default_rng(43059).standard_normal(len(K)))
    results=[]
    for delay,fd,color,channel in ((800.125,1253.5*df,'white','recorded'),(16.125,-1253.5*df,'white','flat'),(800.125,200.,'recorded','recorded'),(384.125,1.17,'white','recorded')):
        d=recorded_noise if color=='recorded' else np.ones(len(K));fc=np.round(fd/df)*df;dc=np.round(delay*4)/4
        correlation=0j;variance=information=0
        for start in range(0,len(s),32):
            sl=slice(start,start+32);hh=h[sl] if channel=='recorded' else np.ones_like(h[sl]);tt=time[sl]
            q_time=shifted(s[sl],p[sl],origin[sl],delay-fd*FS/557625000*tt)*np.exp(2j*np.pi*fd*(tt[:,None]+within))
            w_time=interpolated(s[sl],p[sl],origin[sl],dc-fc*FS/557625000*tt)*np.exp(2j*np.pi*fc*tt[:,None])*(1+2j*np.pi*fc*within)
            q=np.fft.fft(q_time,axis=1)[:,K%NFFT]*hh;w=np.fft.fft(w_time,axis=1)[:,K%NFFT]*hh/(d*jitter)
            correlation+=np.vdot(w,q);variance+=np.sum(abs(w)**2*d);information+=np.sum(abs(q)**2/d)
        eta=abs(correlation)**2/(variance*information)
        allowance=10*np.log10(17/16)+10/np.log(10)*np.deg2rad(3)**2-10*np.log10(1-1/(np.pi*3)**2)
        result=dict(delay_samples=delay,doppler_hz=fd,noise=color,channel=channel,efficiency=float(eta),loss_including_cp_clock_dc_db=float(-10*np.log10(eta)+allowance))
        results.append(result);print(json.dumps(result),flush=True)
    output.write_text(json.dumps(dict(source=str(path),rows=len(s),time_start_s=t0,time_end_s=t1,results=results),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('times',type=Path);p.add_argument('noise',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.input,a.times,a.noise,a.output)
