"""Reproduce the earlier ship image and isolate its angular-motion assumption.

This replays the original C0/C1 profiles, original rows, antenna coordinates,
common spline translation and coherent summation. Only the target-frame
orientation history changes in the declared controls. No image is rotated
after formation and no phase screen is fitted by this audit.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv,utc_seconds
from combine_images import combine
from extract_full_aperture import load
from imaging import geometry_from_state,motion_state,read_image,write_input,sites
from spline_motion import SplineMotion
from motion import body_basis,enu_basis


def fixed_frame(state,midpoint_state):
    return {**state,'body_to_ecef':np.broadcast_to(midpoint_state['body_to_ecef'][0],state['body_to_ecef'].shape),
        'frame_velocity':np.zeros_like(state['frame_velocity'])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--channels',type=int,nargs='+',default=list(range(5)));p.add_argument('--spacing-m',type=float,default=2.)
    p.add_argument('--variants',nargs='+',choices=['legacy','fixed_orientation','linear_orientation','reported_course'],default=['legacy','fixed_orientation','linear_orientation','reported_course'])
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    original=Path('build/isar-20261015/heading-five-142-172')
    description=json.loads((original/'image-description.json').read_text());model=description['motion_model'];curve=SplineMotion(model)
    tx,rx,_=sites(Path(model['geometry_directory']))
    epoch=utc_seconds(description['nominal_start_utc']);track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    def state(time):
        return curve.apply(time,motion_state(track,epoch+time,model),tx[3],rx)
    mid=float(np.mean(model['interval_s']));anchor=state(np.array([mid]))
    sample=np.linspace(142.001,171.999,1001);sample_state=state(sample)
    r=enu_basis(*np.radians([-37.81,144.98]));v=sample_state['body_to_ecef'][:,:,0]@r
    angle=np.unwrap(np.arctan2(v[:,0],v[:,1]));rate=np.polyfit(sample-mid,angle,1)
    x=np.arange(-350.,350.01,args.spacing_m);y=np.arange(-150.,150.01,args.spacing_m);xx,yy=np.meshgrid(x,y)
    pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.),np.zeros(xx.size)]
    variants=args.variants;values={v:[] for v in variants}
    result=dict(interval_s=model['interval_s'],original_source=str(original),original_motion=model,
        heading_sweep_degrees=np.degrees(angle[[0,-1]]).tolist(),heading_rate_extrema_degrees_per_s=np.degrees(np.gradient(angle,sample)).take([np.argmin(np.gradient(angle,sample)),np.argmax(np.gradient(angle,sample))]).tolist(),
        linear_heading_degrees=np.degrees(rate).tolist(),channels=[])
    for c in args.channels:
        tic=time.monotonic();a=load(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'))
        t=a['time_s'];keep=(t>=142.)&(t<172.);t=t[keep];raw=a['moments'][:,keep,:]
        meta=json.loads(a['metadata_json']);w=a['information'][keep]*a['valid'][keep];rf=meta['channel_configuration']['rf_hz']
        physical=state(t);record=dict(channel=c,variants={})
        for variant in variants:
            s=physical
            if variant=='fixed_orientation':s=fixed_frame(s,anchor)
            if variant=='reported_course':
                prior=motion_state(track,epoch+t,{**model,'attitude_prior':'reported-course-prior'})
                s=curve.apply(t,prior,tx[3],rx)
            if variant=='linear_orientation':
                heading=np.polyval(rate,t-mid)
                # Fixed tangent frame, with exactly one linear angular sweep.
                e=np.stack([np.sin(heading),np.cos(heading),np.zeros(len(t))],axis=1)
                port=np.stack([-np.cos(heading),np.sin(heading),np.zeros(len(t))],axis=1)
                up=np.broadcast_to([0.,0.,1.],e.shape);edot=rate[0]*np.stack([np.cos(heading),-np.sin(heading),np.zeros(len(t))],axis=1)
                pdot=rate[0]*np.stack([np.sin(heading),np.cos(heading),np.zeros(len(t))],axis=1)
                s={**s,'body_to_ecef':np.einsum('ij,njk->nik',r,np.stack([e,port,up],axis=2)),
                    'frame_velocity':np.einsum('ij,njk->nik',r,np.stack([edot,pdot,np.zeros_like(e)],axis=2))}
            g,_=geometry_from_state(s,tx[c],rx,rf,meta['first_range_sample'],w)
            saved=out/f'{variant}-channel-{c}.npz'
            source=out/f'{variant}-channel-{c}.bin';target=out/f'{variant}-channel-{c}-image.bin'
            if saved.exists():
                with np.load(saved) as existing:
                    np.testing.assert_array_equal(existing['along_m'],x);np.testing.assert_array_equal(existing['port_m'],y)
                    z=existing['complex_image'].ravel();information=existing['information'].ravel()
            else:
                write_input(source,g,raw,pixels,rf)
                subprocess.run(['build/release/support/isar/isar_backprojection',str(source),str(target)],check=True)
                v=read_image(target,len(pixels));z=(v[:,0]+1j*v[:,1])/v[:,2];information=v[:,2]
                np.savez(saved,along_m=x,port_m=y,complex_image=z.reshape(xx.shape),information=information.reshape(xx.shape))
                source.unlink()  # Only this newly written, consumed input packet.
            values[variant].append((z.reshape(xx.shape),information.reshape(xx.shape)))
            if variant=='legacy' and args.spacing_m==2.:
                old=np.load(original/f'channel-{c}-image.npz');expected=old['complex_image']
                record['replay_relative_complex_rms']=float(np.linalg.norm(z.reshape(xx.shape)-expected)/np.linalg.norm(expected))
                record['replay_peak_power_change_db']=float(20*np.log10(abs(z).max()/abs(expected).max()))
                if not np.array_equal(t,old['time_s']):raise ValueError('Replay has different native epochs')
            record['variants'][variant]={'maximum_power_db':float(10*np.log10((8192*information*abs(z)**2).max()))}
        record['seconds']=time.monotonic()-tic;result['channels'].append(record)
        (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(record,flush=True)
    if args.channels==list(range(5)):
        fig,axes=plt.subplots(6,len(variants),figsize=(22,16),layout='constrained',sharex=True,sharey=True)
        for j,variant in enumerate(variants):
            amplitude=np.asarray([e[0] for e in values[variant]]);information=np.asarray([e[1] for e in values[variant]])
            combined,total,_=combine(amplitude,information)
            panel=np.r_[8192*information*abs(amplitude)**2,(8192*total*abs(combined)**2)[None]]
            np.savez(out/f'{variant}-combined.npz',along_m=x,port_m=y,complex_image=combined,information=total)
            for c,ax in enumerate(axes[:,j]):
                im=ax.imshow(10*np.log10(np.maximum(panel[c],1e-30)),extent=[x[0],x[-1],y[0],y[-1]],origin='lower',aspect='equal',cmap='magma',vmin=5,vmax=50,interpolation='nearest')
                ax.set_title(['Seven','SBS','Nine','Ten','ABC','Complex sum'][c]+' · '+variant,fontsize=10)
                ax.set(xlabel='Forward coordinate (m)',ylabel='Port (m)')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Matched power / original conditional thermal variance (dB)',shrink=.8)
        fig.suptitle('Replay of the earlier ship-like image · identical 142–172 s native data\nOnly the orientation history changes; same centre motion, sites, range interpolation and fixed colour scale')
        fig.savefig(out/'01-controlled-replay.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
