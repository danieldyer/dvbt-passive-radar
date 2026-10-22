"""Read-only clock metrology from archived CLCK and original PROF coordinates.

Allan statistics describe SDR timing relative to the received transmitter
ensemble, including measurement/channel errors. They are not an independent
measurement of the SDR TCXO. No data are retimed and no archive is rewritten.
"""
import argparse,json,struct,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.signal import welch


def read(path):
    rows={};segments=[];records=0
    with path.open('rb') as f:
        if f.read(8)!=b'DVBTLIV1':raise ValueError('Not a live capture')
        size,=struct.unpack('<I',f.read(4));header=json.loads(f.read(size));total=path.stat().st_size
        while f.tell()<total:
            prefix=f.read(16)
            if len(prefix)!=16:raise ValueError('Truncated archive record')
            tag,size,payload=struct.unpack('<4sIQ',prefix)
            if not 0<size<=1024*1024 or payload>64*1024*1024:raise ValueError('Invalid archive framing')
            meta=json.loads(f.read(size));records+=1
            if tag==b'CLCK':segments.extend(meta['segments'])
            elif tag==b'PROF':
                if len(meta['rows'][0])<8:raise ValueError('No original source timing coordinates')
                rows.setdefault(meta['channel'],[]).append(np.asarray(meta['rows'],float))
            f.seek(payload,1)
            if tag==b'DONE':
                if f.tell()!=total:raise ValueError('Trailing archive data')
                break
        else:raise ValueError('No complete DONE record')
    return header,np.asarray(segments),{c:np.concatenate(r) for c,r in rows.items()},records


def clock_at(segments,n):
    index=np.searchsorted(segments[:,0],n,side='right')-1
    if min(index)<0:raise ValueError('Clock coordinate precedes its history')
    selected=segments[index]
    return selected[:,1]+(n-selected[:,0])*(selected[:,2]-1),selected[:,2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    header,segments,rows,records=read(a.capture);fs=header['corrected_rate_hz'];d=header['decimation'];origin=header['origin']
    channels=[];series=[];summary=[]
    for c,r in sorted(rows.items()):
        n=origin+r[:,0]*d;offset,rate=clock_at(segments,n)
        guard=header['applied_channels'][c]['guard'];length=8192+8192//guard
        shift=r[:,0]-r[0,0]-np.arange(len(r))*length
        epoch=r[:,1];error=r[:,6]-shift
        phase=(offset-error*d*rate)/fs
        channels.append(c);series.append((epoch,phase))
        summary.append(dict(channel=c,id=header['signals'][c]['id'],rows=len(r),invalid_rows=int((r[:,5]==0).sum()),
            source_rate_error_ppm_percentiles=np.percentile(r[:,7]*1e6,[0,1,50,99,100]).tolist(),
            reconstructed_integer_window_shift_samples=[float(min(shift)),float(max(shift))]))
    # This grid is for metrology/plotting only. Original coordinates above
    # remain saved separately; it is never used to replace received samples.
    dt=(8192+8192//header['applied_channels'][0]['guard'])/8e6
    first=max(t[0] for t,x in series);last=min(t[-1] for t,x in series)
    t=np.arange(first,last,dt);phase=np.array([np.interp(t,q,x) for q,x in series])
    common=phase.mean(axis=0);n=t*fs;applied,_=clock_at(segments,n);applied/=fs
    centered=t-t.mean();trend=np.c_[np.ones(len(t)),centered]
    detrended=phase-(trend@np.linalg.lstsq(trend,phase.T,rcond=None)[0]).T
    common_detrended=detrended.mean(axis=0);applied_detrended=applied-trend@np.linalg.lstsq(trend,applied,rcond=None)[0]
    lags=np.unique(np.geomspace(1,len(t)//12,55).astype(int));allan=[]
    for m in lags:
        delta=phase[:,2*m:]-2*phase[:,m:-m]+phase[:,:-2*m];tau=m*dt
        avar=delta@delta.T/(2*tau*tau*delta.shape[1]);cross=(avar.sum()-np.trace(avar))/(len(channels)*(len(channels)-1))
        change=applied[2*m:]-2*applied[m:-m]+applied[:-2*m]
        allan.append(dict(tau_s=float(tau),variance_by_channel=np.diag(avar).tolist(),ensemble_variance=float(avar.mean()),
            mean_pair_cross_variance=float(cross),applied_variance=float(np.mean(change**2)/(2*tau*tau)),overlapping_terms=int(delta.shape[1])))
    frequency,power=welch(np.r_[common_detrended[None],applied_detrended[None],detrended],fs=1/dt,nperseg=min(32768,len(t)),axis=-1)
    np.savez_compressed(a.output/'coordinates.npz',segments=segments,time_s=t,relative_clock_phase_s=phase,applied_phase_s=applied,
        psd_frequency_hz=frequency,phase_psd_s2_per_hz=power,**{f'original_rows_{c}':r for c,r in rows.items()})
    span=max(c['rf_hz'] for c in header['applied_channels'])-min(c['rf_hz'] for c in header['applied_channels'])+6656250
    comparisons=[]
    difference=applied-common
    for duration in [.1,1.,3.,30.]:
        m=round(duration/dt);error=difference[m:]-difference[:-m]
        comparisons.append(dict(interval_s=m*dt,rms_difference_ns=float(np.std(error)*1e9),
            conservative_full_span_phase_rms_degrees=float(np.std(error)*360*span),
            interpretation='Applied map minus noisy transmitter-ensemble observations; not an actual clock-error or radar phase-loss measurement'))
    fig,ax=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    skip=max(1,len(t)//15000)
    for c,x in zip(channels,detrended):ax[0,0].plot(t[::skip],x[::skip]*1e9,lw=.5,alpha=.6,label=header['signals'][c]['id'])
    ax[0,0].plot(t[::skip],applied_detrended[::skip]*1e9,'k',lw=1,label='Applied common map');ax[0,0].legend(fontsize=8)
    ax[0,0].set(xlabel='Capture time (s)',ylabel='Time deviation after linear detrend (ns)',title='Relative clocks; includes propagation and pilot measurement errors')
    for c in range(len(channels)):
        ax[0,1].loglog(frequency[1:],power[2+c,1:],lw=.6,alpha=.5)
    ax[0,1].loglog(frequency[1:],power[0,1:],'C0',label='Observed equal-weight ensemble')
    ax[0,1].loglog(frequency[1:],power[1,1:],'k',label='Applied common map');ax[0,1].legend(fontsize=8)
    ax[0,1].set(xlabel='Temporal frequency (Hz)',ylabel='Time-error PSD (s²/Hz)',title='Observed and applied timing spectra')
    tau=np.array([r['tau_s'] for r in allan])
    for c in range(len(channels)):ax[1,0].loglog(tau,np.sqrt([r['variance_by_channel'][c] for r in allan]),alpha=.5)
    ax[1,0].loglog(tau,np.sqrt([r['ensemble_variance'] for r in allan]),label='Observed ensemble')
    ax[1,0].loglog(tau,np.sqrt([r['applied_variance'] for r in allan]),'k',label='Applied map');ax[1,0].legend(fontsize=8)
    ax[1,0].set(xlabel='Averaging time τ (s)',ylabel='Overlapping Allan deviation',title='Relative stability, not independently measured TCXO stability')
    ax[1,1].plot(segments[:,0]/fs,(segments[:,2]-1)*1e6,lw=.5)
    ax[1,1].set(xlabel='Capture time (s)',ylabel='Applied rate correction (ppm)',title='Actual published correction segments')
    fig.suptitle('Archived common-clock audit: no samples modified; linear detrending only in the first two plots')
    fig.savefig(a.output/'clock-metrology.png',dpi=150);plt.close(fig)
    result=dict(complete=True,scope=__doc__,capture=str(a.capture),coordinate_rate_hz=fs,source_rate_hz=header['source_nominal_rate_hz'],
        records=records,clock_segments=len(segments),channels=summary,interval_s=[first,last],allan=allan,applied_minus_observed_ensemble=comparisons,
        minimum_segment_interval_s=float(min(np.diff(segments[:,0]))/fs),median_segment_interval_s=float(np.median(np.diff(segments[:,0]))/fs),
        maximum_continuity_error_samples=float(max(abs(segments[1:,1]-segments[:-1,1]-(segments[1:,0]-segments[:-1,0])*(segments[:-1,2]-1)))),
        separate_tcxo_allan_variance_identifiable=False,seconds=time.monotonic()-started)
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['records','clock_segments','interval_s','median_segment_interval_s','maximum_continuity_error_samples','seconds']}),flush=True)


if __name__=='__main__':main()
