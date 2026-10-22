"""Five-channel recorded-data images under one whole-capture physical fit."""
import argparse,json,time
from pathlib import Path
import numpy as np
from bin_ship_aperture import coherent_means
from channel_update import ChannelUpdates
from extract_full_aperture import load
from joint_ship_motion import JointMotion
from motion import enu_basis
from imaging import geometry_from_state
from combine_images import combine


def render(motion,mid,out,spacing):
    out.mkdir(parents=True,exist_ok=False);east=np.arange(-300.,300.01,spacing);north=np.arange(-400.,400.01,spacing);xx,yy=np.meshgrid(east,north)
    pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.),np.zeros(xx.size)]
    anchor=motion.state(np.array([mid]));a=motion.track.sample(np.array([motion.epoch+mid]),max_gap_s=400.);local=enu_basis(a['latitude'][0],a['longitude'][0])
    gauge=anchor['body_to_ecef'][0].T@local;worker=ChannelUpdates(pixels,out/'worker');num=np.zeros((5,len(pixels)),complex);information=np.zeros(5);started=time.monotonic()
    try:
        for c in range(5):
            raw=load(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'));meta=json.loads(raw['metadata_json']);index=np.flatnonzero((raw['time_s']>=mid-15)&(raw['time_s']<mid+15))
            for t,info,moments,_,_,reference in coherent_means(motion,c,raw,index[0],index[-1]+1,4):
                state=motion.state(t);state['body_to_ecef']=state['body_to_ecef']@gauge;state['frame_velocity']=state['frame_velocity']@gauge
                geo,_=geometry_from_state(state,motion.tx[c],motion.rx,motion.rf[c],meta['first_range_sample'],info)
                geo[:,9,1]=1.;geo[:,9,2]=0.;geo[:,5,3]=2*np.pi*motion.rf[c]/299792458.;geo[:,6,3]=c;geo[:,7,3]=1.;geo[:,8,3]=1.
                for begin in range(0,len(t),256):
                    z,w=worker.update(geo[begin:begin+256],moments[:,begin:begin+256]);num+=z;information+=w
            print(json.dumps(dict(mid=mid,channel=c,seconds=time.monotonic()-started)),flush=True)
    finally:worker.close()
    images=num/information[:,None];combined=np.einsum('c,cp->p',motion.gains.conj(),num)/np.sum(abs(motion.gains)**2*information)
    power=np.r_[8192*abs(num)**2/information[:,None],(8192*abs(combined)**2*np.sum(abs(motion.gains)**2*information))[None]].reshape(6,len(north),len(east))
    np.savez_compressed(out/'images.npz',east_m=east,north_m=north,individual=images.reshape(5,len(north),len(east)),information=information,
        combined=combined.reshape(len(north),len(east)),power=power)
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,13),layout='constrained',sharex=True,sharey=True)
    ais=(a['position_ecef_m'][0]-anchor['position_ecef_m'][0])@local
    for c,ax in enumerate(axes.ravel()):
        im=ax.imshow(10*np.log10(np.maximum(power[c],1e-30)),extent=[east[0],east[-1],north[0],north[-1]],origin='lower',cmap='magma',vmin=3,vmax=55,aspect='equal')
        ax.plot(ais[0],ais[1],'+',color='#00e5ff',ms=8);ax.set(title=['Seven','SBS','Nine','Ten','ABC','Static calibrated coherent sum'][c],xlabel='East (m)',ylabel='North (m)')
    fig.colorbar(im,ax=list(axes.ravel()),label='Matched power / conditional thermal variance (dB)',shrink=.8)
    fig.suptitle(f'Whole-capture common 2D motion · {mid-15:.2f}–{mid+15:.2f} s · north up\nOriginal C0/C1; validated four-symbol means; no notch, image normalization or frame phase fit')
    fig.savefig(out/'images.png',dpi=150);plt.close(fig)
    (out/'analysis.json').write_text(json.dumps(dict(complete=True,motion=str(motion.path),mid_s=mid,interval_s=[mid-15,mid+15],
        maximum_matched_power_db=[float(10*np.log10(p.max())) for p in power],seconds=time.monotonic()-started),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--centres',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--spacing-m',type=float,default=2.);a=p.parse_args()
    motion=JointMotion(a.motion)
    for mid in a.centres:
        out=a.output/f'window-{mid:07.2f}'
        if not (out/'analysis.json').exists():render(motion,mid,out,a.spacing_m)
if __name__=='__main__':main()
