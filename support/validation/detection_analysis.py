#!/usr/bin/env python3
"""Measured background, reversible calibration and held-out local covariance checks."""
import argparse,json,struct
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sensitivity import blocks,NFFT,DT,map_at


def load(path):
    with path.open('rb') as f:
        magic,n,b=struct.unpack('<4sIQ',f.read(16));m=json.loads(f.read(n))
        if magic!=b'RDM1' or b!=m['ranges']*m['frequencies']*8:raise ValueError('Invalid map')
        a=np.frombuffer(f.read(b),'<c8').reshape(m['ranges'],m['frequencies'])
    return m,a


def covariance(m,a):
    f=m['frequency_start_hz']+np.arange(m['frequencies'])*m['frequency_step_hz']
    r=np.arange(m['ranges'])*m['range_step_m'];offs=np.arange(-4,5)*4
    q=np.sinc(offs*m['samples']/m['fft_size'])/np.sinc(offs/m['fft_size'])
    white=np.sinc((offs[:,None]-offs)*m['samples']/m['fft_size'])/np.sinc((offs[:,None]-offs)/m['fft_size'])
    results=[]
    for centre in (-190,-150,-100,-50,-10,10,50,100,150,190):
        k=np.argmin(abs(f-centre));z=a[::4,k+offs];ranges=r[::4]
        cs=[]
        for lo,hi in ((600,15000),(15000,23000),(23000,30000)):
            x=z[(ranges>=lo)&(ranges<hi)];energy=np.sum(abs(x)**2,axis=1)
            x=x[energy<np.quantile(energy,.9)];cs.append(x.T@x.conj()/len(x))
        train,val,test=cs;scale=np.trace(train).real/9
        target=white*np.sqrt(train.diagonal().real[:,None]*train.diagonal().real)
        best=None
        def snr(w,c):return abs(np.vdot(w,q))**2/np.vdot(w,c@w).real
        for alpha in (.25,.5,.75,1):
            w=np.linalg.solve((1-alpha)*train+alpha*target+1e-3*scale*np.eye(9),q)
            score=snr(w,val)
            if best is None or score>best[0]:best=(score,w,alpha)
        baseline=1/test[4,4].real;oracle=np.vdot(q,np.linalg.solve(test+1e-6*np.trace(test).real/9*np.eye(9),q)).real
        results.append(dict(doppler_hz=centre,local_oracle_gain_db=float(10*np.log10(oracle/baseline)),
                            heldout_gain_db=float(10*np.log10(snr(best[1],test)/baseline)),shrinkage=best[2]))
    return results


def run(prefixes,output,figures,baselines=None,uncancelled=None):
    output.parent.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    results=[];maps=[]
    for prefix in prefixes:
        m,a=load(prefix.with_suffix('.rdm'));_,z=load(prefix.with_name(prefix.name+'-normalized').with_suffix('.rdm'))
        f=m['frequency_start_hz']+np.arange(m['frequencies'])*m['frequency_step_hz'];r=np.arange(m['ranges'])*m['range_step_m'];keep=(abs(f)>=3/m['integration_s']);far=(r>=600)&(r<29000)
        bands=[]
        for matrix in (a,z):
            psd=np.median(abs(matrix[far][:,keep])**2,axis=0)/np.log(2);db=10*np.log10(psd/np.median(psd));bands.append(np.quantile(db,[0,.05,.5,.95,1]).tolist())
        results.append(dict(source=str(prefix),time_start_s=m['time_start_s'],time_end_s=m['time_end_s'],row_psd_quantiles_db=bands,
                            local_covariance=covariance(m,a)))
        maps.append((m,a,z,f,r))
    # Honest common scales: physical amplitude for raw; calibrated noise units for standardized.
    m,a,z,f,r=maps[0];step=4;power=lambda x:10*np.log10(np.maximum(abs(x[::step,::step].T)**2,1e-30))
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,matrix,title in zip(axes,(a,z),('Physical complex reflectivity','Noise-standardized detection statistic')):
        db=power(matrix);floor=np.median(db);im=ax.imshow(db,origin='lower',aspect='auto',extent=[0,r[-1]/1000,f[0],f[-1]],vmin=floor-10,vmax=floor+25,cmap='magma');ax.set(title=title,xlabel='Excess path [km]',ylabel='Doppler [Hz]');fig.colorbar(im,ax=ax,label='Power [dB; units differ as labelled]')
    fig.savefig(figures/'matched_detection.png',dpi=180);fig.savefig(figures/'matched_detection.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for matrix,label in ((a,'Corrected matched filter, physical units'),(z,'Reversible background calibration')):
        psd=np.median(abs(matrix[(r>=600)&(r<29000)])**2,axis=0)/np.log(2);db=10*np.log10(psd/np.median(psd[keep]));ax.plot(f[keep],db[keep],label=label,lw=.8)
    ax.set(xlabel='Doppler [Hz]',ylabel='Range-median noise power relative to median [dB]',ylim=(-4,5));ax.legend();fig.savefig(figures/'individual_frame_bands.png',dpi=180);fig.savefig(figures/'individual_frame_bands.pdf');plt.close(fig)
    comparisons=[]
    if baselines:
        if len(baselines)!=len(prefixes):raise ValueError('Supply one baseline for each excerpt')
        for prefix,(m,a,z,f,r) in zip(baselines,maps):
            b,old=load(prefix.with_suffix('.rdm'))
            if old.shape!=a.shape or abs(b['time_start_s']-m['time_start_s'])>1e-9:raise ValueError('Comparison CPI mismatch')
            far=(r>=600)&(r<29000);keep=abs(f)>=3/m['integration_s']
            p0=np.median(abs(old[far][:,keep])**2);p1=np.median(abs(a[far][:,keep])**2)
            comparisons.append(dict(baseline=str(prefix),after=str(prefixes[len(comparisons)]),
                                    median_background_reduction_db=float(10*np.log10(p0/p1))))
        m,a,z,f,r=maps[0];b,old=load(baselines[0].with_suffix('.rdm'));columns=[(old,'Minimum-phase cancellation only'),(a,'Frozen stationary prediction')]
        if uncancelled:
            u,raw=load(uncancelled.with_suffix('.rdm'))
            if raw.shape!=a.shape or abs(u['time_start_s']-m['time_start_s'])>1e-9:raise ValueError('Uncancelled CPI mismatch')
            columns.insert(0,(raw,'Uncancelled finite correlation'))
            far=(r>=600)&(r<29000);keep=abs(f)>=3/m['integration_s']
            comparisons[0]['total_reduction_from_uncancelled_db']=float(10*np.log10(np.median(abs(raw[far][:,keep])**2)/np.median(abs(a[far][:,keep])**2)))
        fig,axes=plt.subplots(1,len(columns),figsize=(5*len(columns),5),constrained_layout=True)
        for ax,(matrix,title) in zip(axes,columns):
            im=ax.imshow(power(matrix),origin='lower',aspect='auto',extent=[0,r[-1]/1000,f[0],f[-1]],vmin=-110,vmax=-75,cmap='magma')
            ax.set(title=title,xlabel='Excess path [km]',ylabel='Doppler [Hz]')
        fig.colorbar(im,ax=axes,label='Physical reflectivity power [dB; common scale]')
        fig.savefig(figures/'cancellation_comparison.png',dpi=180);fig.savefig(figures/'cancellation_comparison.pdf');plt.close(fig)
    output.write_text(json.dumps(dict(excerpts=results,stationary_comparisons=comparisons,
        limitations='Local covariance oracle is conditional on the chosen nine-cell template and held-out range regions. Normalization changes thresholds, not SNR. It cannot certify full observation-space optimum.'),indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('prefixes',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--figures',required=True,type=Path)
    p.add_argument('--baselines',nargs='+',type=Path);p.add_argument('--uncancelled',type=Path)
    a=p.parse_args();run(a.prefixes,a.output,a.figures,a.baselines,a.uncancelled)
