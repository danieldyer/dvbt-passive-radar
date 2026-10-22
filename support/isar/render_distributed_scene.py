"""North-up coefficient images from original-data inverse scene estimates.

These are regularized reflectivity estimates, not raw matched-filter power
or calibrated target SNR. Independent-band images retain their own complex
solutions; the joint image is estimated coherently from all five bands.
No observed-cell normalization or image rotation is applied.
"""
import argparse,json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from radar_path_motion import RadarPathMotion
from motion import enu_basis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--joint',type=Path,required=True)
    p.add_argument('--independent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scale-db',type=float,nargs=2,default=[-125.,-80.]);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    description=json.loads((a.joint/'analysis.json').read_text());independent=json.loads((a.independent/'analysis.json').read_text())
    truth=description.get('synthetic_control')
    if truth is None and description.get('geometry'):
        truth=json.loads((Path(description['geometry'])/'analysis.json').read_text()).get('synthetic_control')
    motion=RadarPathMotion(Path(description['motion']));mid=float(np.mean(motion.interval));state=motion.state(np.array([mid]))
    sample=motion.track.sample(np.array([motion.epoch+mid]),max_gap_s=400.);local=enu_basis(sample['latitude'][0],sample['longitude'][0])
    with np.load(a.joint/'scene.npz') as f:points=f['points_body_m'];joint=f['coefficient']
    with np.load(a.independent/'scene.npz') as f:values=[f[f'coefficient_{c}'] for c in range(5)]+[joint]
    if any(value.shape!=(len(points),) for value in values):raise ValueError('All coefficient images need the same physical basis')
    x=np.unique(points[:,0]);y=np.unique(points[:,1])
    if len(x)*len(y)!=len(points):raise ValueError('This reference renderer requires a rectangular physical basis')
    np.testing.assert_allclose(points[:len(x),0],x,rtol=0,atol=0)
    position=points@state['body_to_ecef'][0].T@local
    antenna=(sample['position_ecef_m'][0]-state['position_ecef_m'][0])@local
    east=position[:,0].reshape(len(y),len(x));north=position[:,1].reshape(len(y),len(x))
    fig,axes=plt.subplots(2,3,figsize=(15,11),layout='constrained',sharex=True,sharey=True)
    for c,(ax,value) in enumerate(zip(axes.ravel(),values)):
        power=10*np.log10(np.maximum(abs(value.reshape(len(y),len(x)))**2,1e-300))
        image=ax.pcolormesh(east,north,power,shading='nearest',cmap='magma',vmin=a.scale_db[0],vmax=a.scale_db[1])
        ax.plot(antenna[0],antenna[1],'+',color='#00e5ff',markersize=9)
        if truth:
            locations=np.asarray(truth['points_body_m'])@state['body_to_ecef'][0].T@local
            ax.plot(locations[:,0],locations[:,1],'+',color='#00ff00',markersize=7)
        ax.set(aspect='equal',xlabel='East from fitted body origin (m)',ylabel='North from fitted body origin (m)',
            title=['Seven: independent','SBS: independent','Nine: independent','Ten: independent','ABC: independent','All five: shared complex scene'][c])
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Inverse coefficient power (dB relative to model units; not SNR)',shrink=.75)
    kind='SYNTHETIC, given motion/calibration; green: true positions' if truth else 'RECORDED DATA; no accepted hull reconstruction'
    prior='Independent-band ridge; joint complex-Laplace prior' if description.get('prior') else 'Training-estimated ridge priors'
    fig.suptitle(f"{kind}\n{motion.interval[0]:.2f}–{motion.interval[1]:.2f} s · {description.get('body_frame_hypothesis','upstream-motion')}\n{prior}; cyan: AIS antenna; physical north/east coordinates; fixed colour scale")
    fig.savefig(a.output/'inverse-scene-north-up.png',dpi=155);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(dict(scope=__doc__,joint=str(a.joint),independent=str(a.independent),
        scale_db=a.scale_db,projection='Exact physical body frame at CPI midpoint, east/north in local tangent plane',
        joint_outer_explained_fraction=description['outer']['explained_fraction'],independent_outer_explained_fraction=independent['outer_explained_fraction'],
        image_is_accepted_focused_ship=False),indent=2)+'\n')


if __name__=='__main__':main()
