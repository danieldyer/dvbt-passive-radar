"""Put the original ship replay and a native refit on the same geographic axes.

This is a display-only coordinate comparison. Powers retain their original
conditional noise units. No image is rephased, rescaled to its peak or filled
outside its formed support. Antenna/motion hypotheses are explicitly different.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv,utc_seconds
from imaging import motion_state,sites
from local_ship_motion import load_motion
from motion import enu_basis
from spline_motion import SplineMotion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--replay',type=Path,default=Path('docs/data/isar_20261017/ship-baseline-replay'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.native/'analysis.json').read_text());interval=meta['interval_s'];mid=float(np.mean(interval))
    if interval!=[142.,172.]:raise ValueError('This reference replay has exactly the 142–172 s aperture')
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    native=load_motion(a.motion);sample=track.sample(np.array([native.epoch+mid]),max_gap_s=400.)
    origin=sample['position_ecef_m'][0];local=enu_basis(sample['latitude'][0],sample['longitude'][0])
    old=json.loads((a.replay/'analysis.json').read_text())['original_motion'];curve=SplineMotion(old)
    tx,rx,_=sites(Path(old['geometry_directory']));epoch=utc_seconds(old['nominal_start_utc'])
    state=curve.apply(np.array([mid]),motion_state(track,epoch+np.array([mid]),old),tx[3],rx)
    old_values=[]
    for c in range(5):
        z=np.load(a.replay/f'legacy-channel-{c}.npz');old_values.append(8192*z['information']*abs(z['complex_image'])**2)
    z=np.load(a.replay/'legacy-combined.npz');old_values.append(8192*z['information']*abs(z['complex_image'])**2)
    xx,yy=np.meshgrid(z['along_m'],z['port_m']);points=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.)]
    old_position=state['position_ecef_m'][0]+points@state['body_to_ecef'][0].T
    old_xy=((old_position-origin)@local)[:,:2].reshape(*xx.shape,2)
    new=np.load(a.native/'images.npz');new_state=native.state(np.array([mid]))
    shift=(new_state['position_ecef_m'][0]-origin)@local
    x,y=np.meshgrid(new['east_m']+shift[0],new['north_m']+shift[1]);new_xy=np.stack([x,y],axis=-1)
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(6,2,figsize=(12,20),layout='constrained',sharex=True,sharey=True)
    names=['Seven','SBS','Nine','Ten','ABC','Complex coherent sum']
    for column,(xy,power,label) in enumerate([(old_xy,old_values,'Original exact replay'),(new_xy,new['power'],'New native physical refit')]):
        for c,ax in enumerate(axes[:,column]):
            im=ax.pcolormesh(xy[:,:,0],xy[:,:,1],10*np.log10(np.maximum(power[c],1e-30)),shading='nearest',cmap='magma',vmin=5,vmax=50,rasterized=True)
            ax.plot(0,0,'+',color='#00e5ff');ax.set(title=f'{names[c]} · {label}',xlabel='East from AIS antenna (m)',ylabel='North from AIS antenna (m)',xlim=[-200,200],ylim=[-250,250],aspect='equal')
    fig.colorbar(im,ax=list(axes.ravel()),shrink=.6,label='Matched power / conditional thermal variance (dB); same absolute scale')
    fig.suptitle('Same 142–172 s recording · north up, east right · no peak normalization\nDifferent explicit motion/site/calibration hypotheses; cyan is the interpolated AIS antenna. Blank corners were not formed.')
    fig.savefig(a.output/'same-aperture-geographic-comparison.png',dpi=150);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,interval_s=interval,native=str(a.native),motion=str(a.motion),replay=str(a.replay),scale_db=[5,50],native_centre_offset_from_ais_enu_m=shift.tolist()),indent=2)+'\n')


if __name__=='__main__':main()
