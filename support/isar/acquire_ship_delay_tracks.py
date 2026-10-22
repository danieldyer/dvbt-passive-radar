"""Acquire common delay/rate ridges before imposing a geographic ship patch.

Independent complex amplitudes are profiled for each channel. A constant
stationary profile is an orthogonal nuisance, including in the template norm;
there is no broad Doppler notch. Power interpolation only aligns the coarse
acquisition grids. It is never used for coherent phase or final imaging.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from motion import C,excess_path
from joint_ship_motion import JointMotion
from ship_coherent_movie import load_apertures


def stationary_glrt(data,information,time_s,frequencies):
    total=information.sum();phase=np.exp(-2j*np.pi*frequencies[:,None]*(time_s-time_s.mean())[None,:])
    constant=phase@information;norm=total-abs(constant)**2/total
    z=phase@(information[:,None]*data)-constant[:,None]*np.sum(information[:,None]*data,axis=0)[None]/total
    return np.divide(8192*abs(z)**2,norm[:,None],out=np.zeros_like(abs(z)),where=norm[:,None]>1e-8*total)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--aperture',type=Path,required=True)
    p.add_argument('--step-s',type=float,default=3.);p.add_argument('--seconds',type=float,default=3.);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training-field',type=Path,help='Use only the exact saved training-block identities')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);motion=JointMotion(a.motion);apertures,_=load_apertures(a.aperture)
    folds=[]
    for c,b in enumerate(apertures):
        selected=np.ones(len(b['time_s']),bool)
        if a.training_field:
            with np.load(a.training_field/f'channel-{c}/field.npz') as f:epochs=f['time'].copy()
            with np.load(a.training_field/f'channel-{c}/epochs.npz') as f:training=f['training'].copy()
            idx=np.clip(np.searchsorted(epochs,b['time_s']),0,len(epochs)-1);before=np.maximum(idx-1,0)
            idx=np.where(abs(epochs[before]-b['time_s'])<abs(epochs[idx]-b['time_s']),before,idx)
            selected=training[idx]&(b['time_s']>=epochs[0])&(b['time_s']<=epochs[-1])
        folds.append(selected)
    centres=np.arange(6.,599.,a.step_s);ranges=np.arange(13000.,16500.,C/16e6);rates=np.arange(-6.,6.001,.1);scores=[];predicted=[];started=time.monotonic()
    for index,mid in enumerate(centres):
        physical=motion.state(np.array([mid]));nominal=[excess_path(physical['position_ecef_m'],tx,motion.rx)[0][0] for tx in motion.tx]
        nominal_rate=np.sum(excess_path(physical['position_ecef_m'],motion.tx[3],motion.rx)[1]*physical['velocity_ecef_mps'],axis=1)[0]
        total=np.zeros((len(rates),len(ranges)));band_peaks=[]
        for c,b in enumerate(apertures):
            t=b['time_s'];use=(t>=mid-a.seconds/2)&(t<mid+a.seconds/2)&folds[c];t=t[use];info=b['information'][use]
            raw_ranges=(b['meta']['first_range_sample']+np.arange(b['moments'].shape[-1]))*C/8e6;keep=(raw_ranges>=12500)&(raw_ranges<=17000)
            raw=b['moments'][0,use][:,keep]*np.exp(-2j*np.pi*motion.rf[c]/C*b['carrier_reference_path_m'][use,None])
            power=stationary_glrt(raw,info,t,-motion.rf[c]/C*rates)
            aligned=interp1d(raw_ranges[keep],power,axis=1,bounds_error=True)(ranges+nominal[c]-nominal[3]);total+=aligned
            at=np.unravel_index(np.argmax(aligned),aligned.shape);band_peaks.append([float(ranges[at[1]]),float(rates[at[0]]),float(aligned[at])])
        scores.append(total.astype('<f4'));predicted.append([float(nominal[3]),float(nominal_rate)])
        if index%20==0:
            np.savez_compressed(a.output/'acquisition.npz',centres_s=centres[:index+1],ranges_m=ranges,rates_mps=rates,power=np.asarray(scores),predicted=np.asarray(predicted))
            print(json.dumps(dict(windows=index+1,total=len(centres),mid_s=mid,band_peaks=band_peaks,seconds=time.monotonic()-started)),flush=True)
    cube=np.asarray(scores);np.savez_compressed(a.output/'acquisition.npz',centres_s=centres,ranges_m=ranges,rates_mps=rates,power=cube,predicted=np.asarray(predicted))
    peaks=[]
    for t,z,prior in zip(centres,cube,predicted):
        at=np.unravel_index(np.argmax(z),z.shape);peaks.append(dict(mid_s=t,path_m=float(ranges[at[1]]),rate_mps=float(rates[at[0]]),power=float(z[at]),predicted_path_m=prior[0],predicted_rate_mps=prior[1]))
    report=dict(complete=True,scope=__doc__,motion=str(a.motion),peaks=peaks,seconds=time.monotonic()-started,training_field=str(a.training_field) if a.training_field else None,
        warning='Independent per-window acquisition maxima are not a continuous target track or phase estimate.')
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');plot(a.output,centres,ranges,rates,cube,np.asarray(predicted),peaks)


def plot(out,t,r,v,p,prior,peaks):
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(15,12),layout='constrained',sharex=True)
    axes[0].imshow(10*np.log10(np.maximum(p.max(axis=1).T,1e-20)),extent=[t[0],t[-1],r[0],r[-1]],origin='lower',aspect='auto',cmap='magma',vmin=5,vmax=55)
    axes[0].plot(t,prior[:,0],color='#00e5ff',lw=1,label='Current fitted AIS-centred path');axes[0].legend();axes[0].set(ylabel='Ten-equivalent excess path (m)')
    axes[1].imshow(10*np.log10(np.maximum(p.max(axis=2).T,1e-20)),extent=[t[0],t[-1],v[0],v[-1]],origin='lower',aspect='auto',cmap='magma',vmin=5,vmax=55)
    axes[1].plot(t,prior[:,1],color='#00e5ff',lw=1);axes[1].set(ylabel='Total path rate (m/s)')
    axes[2].plot(t,[x['path_m']-x['predicted_path_m'] for x in peaks],'.',ms=3);axes[2].axhline(0,color='grey');axes[2].set(xlabel='Capture time (s)',ylabel='Acquisition peak minus current path (m)')
    fig.suptitle('Five-band delay/rate acquisition, before geographic association\nExact epochs; static profile is a profiled nuisance with its template norm retained; no broad Doppler notch; fixed scale')
    fig.savefig(out/'delay-rate-acquisition.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
