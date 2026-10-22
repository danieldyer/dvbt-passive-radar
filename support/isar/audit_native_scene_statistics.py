"""Compare two original-data statistics at identical physical source points.

The scene likelihood correlates a known pulse across a finite whitened crop.
Native BP interpolates the original range-compressed stream at the moving
point. These filters need not be identical for a finite crop and a changing
pulse. This diagnostic uses identical original training epochs, motion and
carrier references, without nuisance projection or fitted phase alignment.
It is a model/operator comparison, not an assumption of identical noise.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from full_path_scene import SceneBand
from local_ship_motion import load_motion
from extract_full_aperture import load as load_original
from training_aperture import select
from channel_update import ChannelUpdates
from imaging import geometry_from_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--aperture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval',type=float,nargs=2,help='Optional interval within one declared source support')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    fit,reference,bands,saved=open_scene(a.motion);motion=load_motion(a.motion)
    if a.interval:
        subset=[]
        for band in bands:
            b=band.point;first,last=np.searchsorted(b.t,a.interval)
            view=b.waveform_window(first,last);windows=np.unique(band.window_index[first:last])
            if len(windows)!=1:raise ValueError('Comparison interval crosses a source-identity boundary')
            w=windows[0]
            points=band.points if band.points.ndim==2 else band.points[w,band.active_points[w]]
            subset.append(SceneBand(view,points))
        bands=subset
    if bands[0].points.ndim!=2:raise ValueError('Use one persistent-scene aperture or declare its interval')
    points=bands[0].points;apertures=[]
    # Original individual rows are essential: a pre-binned observation can
    # straddle a 16-row training-block boundary and cannot be split exactly.
    for c in range(5):
        raw=load_original(a.aperture/f'channel-{c}');raw['meta']=json.loads(raw['metadata_json'])
        raw['first_time_s']=raw['last_time_s']=raw['time_s']
        raw['information']=raw['information']*raw['valid']
        raw['carrier_reference_path_m']=np.zeros(len(raw['time_s']));apertures.append(raw)
    native=np.zeros((5,len(points)),complex);norm=np.zeros(5);cpu=[];cpu_norm=[]
    worker=ChannelUpdates(np.c_[points,np.zeros(len(points))],a.output/'worker')
    try:
        for c,(band,aperture) in enumerate(zip(bands,apertures)):
            b=band.point;h=band.design(np.zeros(b.b.shape[1]),c,False)
            cpu.append(np.einsum('trp,tr->p',h[b.training].conj(),b.field.y[b.training]))
            cpu_norm.append(np.sum(abs(h[b.training])**2,axis=(0,1)))
            chosen=select(aperture['first_time_s'],aperture['last_time_s'],b.coordinate['native_time_s'],b.coordinate['native_information'],b.training)
            index=np.flatnonzero(chosen&(aperture['information']>0))
            for first in range(0,len(index),256):
                at=index[first:first+256];state=motion.state(aperture['time_s'][at])
                geometry,details=geometry_from_state(state,motion.tx[c],motion.rx,motion.rf[c],aperture['meta']['first_range_sample'],aperture['information'][at])
                phase=2*np.pi*motion.rf[c]/299792458.*(details['path_m']-aperture['carrier_reference_path_m'][at])
                geometry[:,9,1]=np.cos(phase);geometry[:,9,2]=np.sin(phase)
                geometry[:,5,3]=2*np.pi*motion.rf[c]/299792458.;geometry[:,6,3]=c;geometry[:,7,3]=1.;geometry[:,8,3]=1.
                z,w=worker.update(geometry,aperture['moments'][:,at]);native+=8192*z;norm+=8192*w
            expected=8192*b.field.data['information'][b.training].sum()
            np.testing.assert_allclose(norm[c],expected,rtol=3e-7,atol=1e-25)
            print(json.dumps(dict(channel=c,native_bins=len(index),information_ratio=norm[c]/expected,seconds=time.monotonic()-started)),flush=True)
    finally:worker.close()
    cpu=np.asarray(cpu);cpu_norm=np.asarray(cpu_norm);phase=np.angle(cpu*native.conj())
    amplitude=np.divide(abs(cpu),abs(native),out=np.full(cpu.shape,np.nan),where=abs(native)>0)
    np.savez_compressed(a.output/'statistics.npz',points_body_m=points,crop_numerator=cpu,native_numerator=native,
        crop_information=cpu_norm,native_information=norm,phase_difference_radians=phase)
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(14,10),layout='constrained',sharex=True)
    for c,name in enumerate(['Seven','SBS','Nine','Ten','ABC']):
        axes[0].plot(20*np.log10(amplitude[c]),'.-',label=name,lw=.8)
        axes[1].plot(np.degrees(phase[c]),'.-',label=name,lw=.8)
        axes[2].plot(10*np.log10(cpu_norm[c]/norm[c]),'.-',label=name,lw=.8)
    for ax in axes:ax.legend(ncol=5,fontsize=8)
    axes[0].set(ylabel='Crop/native numerator magnitude (dB)')
    axes[1].set(ylabel='Crop minus native phase (degrees)')
    axes[2].set(ylabel='Crop/native template information (dB)',xlabel='Identical physical source index')
    fig.suptitle('Recorded samples: finite-crop known-pulse GLS versus native range interpolation\nSame physical motion and training epochs; no stationary projection or fitted alignment')
    fig.savefig(a.output/'native-vs-crop-statistics.png',dpi=145);plt.close(fig)
    result=dict(complete=True,scope=__doc__,motion=str(a.motion),original_epoch_information_checked=True,
        interval_s=a.interval or fit['interval_s'],
        phase_alignment_fitted=False,stationary_projection_applied=False,
        numerator_magnitude_ratio_db=(20*np.log10(amplitude)).tolist(),phase_difference_degrees=np.degrees(phase).tolist(),
        template_information_ratio_db=(10*np.log10(cpu_norm/norm[:,None])).tolist(),seconds=time.monotonic()-started)
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
