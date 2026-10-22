"""Inspect actual delay/Doppler support without a ship-scattering model.

The known bin carrier deramp is reversed, then an exact-epoch nonuniform DFT
forms short rectangular-aperture maps. There is no zero-Doppler notch, target
motion correction, fitted phase or image normalisation. This checks target
association independently of the extended-scene likelihood's spatial grid.
"""
import argparse,json
from pathlib import Path
import numpy as np
from motion import C,excess_path
from joint_ship_motion import JointMotion
from ship_coherent_movie import load_apertures


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--aperture',type=Path,required=True)
    p.add_argument('--centres',type=float,nargs='+',default=[18.32,93.32,153.32,333.32,573.32]);p.add_argument('--seconds',type=float,default=3.)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    motion=JointMotion(a.motion);apertures,_=load_apertures(a.aperture);doppler=np.arange(-10.,10.001,.1);reports=[];values=[]
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(len(a.centres),5,figsize=(20,3.8*len(a.centres)),layout='constrained',sharex=True,sharey=True)
    for i,mid in enumerate(a.centres):
        state=motion.state(np.array([mid]));record=dict(mid_s=mid,bands=[])
        for c,b in enumerate(apertures):
            t=b['time_s'];select=(t>=mid-a.seconds/2)&(t<mid+a.seconds/2);t=t[select];information=b['information'][select]
            ranges=(b['meta']['first_range_sample']+np.arange(b['moments'].shape[2]))*C/8e6
            keep=(ranges>=10000)&(ranges<=20000);rng=ranges[keep]
            raw=b['moments'][0,select][:,keep]*np.exp(-2j*np.pi*motion.rf[c]/C*b['carrier_reference_path_m'][select,None])
            transform=np.exp(-2j*np.pi*doppler[:,None]*(t-mid)[None,:]);image=transform@(information[:,None]*raw)
            power=8192*abs(image)**2/information.sum();length,gradient=excess_path(state['position_ecef_m'],motion.tx[c],motion.rx)
            rate=np.sum(gradient*state['velocity_ecef_mps'],axis=1)[0];expected=-motion.rf[c]/C*rate;path=length[0]
            ax=axes[i,c];im=ax.imshow(10*np.log10(np.maximum(power,1e-30)),extent=[rng[0]/1000,rng[-1]/1000,doppler[0],doppler[-1]],origin='lower',aspect='auto',cmap='magma',vmin=0,vmax=55)
            ax.plot(path/1000,expected,'+',color='#00e5ff');ax.set(xlabel='Total excess path (km)',ylabel='Absolute Doppler (Hz)',title=['Seven','SBS','Nine','Ten','ABC'][c]+f' · {mid:.2f} s')
            region=(abs(rng-path)<=600)[None,:]&(abs(doppler-expected)<=2)[:,None]
            at=np.unravel_index(np.argmax(np.where(region,power,-np.inf)),power.shape)
            record['bands'].append(dict(channel=c,predicted_path_m=float(path),predicted_doppler_hz=float(expected),
                peak_near_prediction_path_m=float(rng[at[1]]),peak_near_prediction_doppler_hz=float(doppler[at[0]]),maximum_near_prediction_power_db=float(10*np.log10(power[at]))))
            np.savez_compressed(a.output/f'window-{mid:07.2f}-channel-{c}.npz',ranges_m=rng,doppler_hz=doppler,power=power,predicted_path_m=path,predicted_doppler_hz=expected)
        reports.append(record);print(json.dumps(record),flush=True)
    fig.colorbar(im,ax=list(axes.ravel()),shrink=.6,label='Matched power / conditional white-noise variance (dB)')
    fig.suptitle(f'Original retained C0: {a.seconds:g} s rectangular apertures, exact original epochs\nKnown carrier deramp reversed; no Doppler notch or motion correction; cyan crosses are current physical predictions; fixed colour scale')
    fig.savefig(a.output/'range-doppler-support.png',dpi=140);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,motion=str(a.motion),coherent_seconds=a.seconds,windows=reports),indent=2)+'\n')
if __name__=='__main__':main()
