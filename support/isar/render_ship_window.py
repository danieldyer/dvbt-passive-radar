"""Native original-symbol verification of a recorded-data ship window fit."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np

from ais import read_csv,utc_seconds
from combine_images import combine
from extract_full_aperture import load
from imaging import sites,motion_state,geometry_from_state,write_input,read_image
from motion import enu_basis
from rigid_patch import RigidPatch
from spline_motion import SplineMotion,basis
from ship_site_signature import assigned_rays


class WindowMotion:
    def __init__(self,result):
        self.result=result;self.model=result['source_model'];self.curve=SplineMotion(self.model)
        self.first,self.last=result['interval_s'];self.mid=(self.first+self.last)/2
        self.epoch=utc_seconds(self.model['nominal_start_utc'])
        self.track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
        self.ref_tx,self.ref_rx,_=sites(Path(self.model['geometry_directory']))
        self.tx,self.rx=assigned_rays(Path('docs/data/calibration_20261016_sites'),result['assignment'])
        a=self.track.sample(np.array([self.epoch+self.mid]),max_gap_s=400.)
        self.local=enu_basis(a['latitude'][0],a['longitude'][0]);anchor=self.base(np.array([self.mid]))
        self.gauge=anchor['body_to_ecef'][0].T@self.local
        self.anchor=anchor['position_ecef_m'][0]

    def base(self,t):
        return self.curve.apply(t,motion_state(self.track,self.epoch+np.asarray(t),self.model),self.ref_tx[3],self.ref_rx)

    def state(self,t,*,refined=True):
        s=self.base(t)
        kin=RigidPatch(t,self.mid,s['position_ecef_m'],s['velocity_ecef_mps'],self.local,self.tx[3],self.rx,
            reference_frame=s['body_to_ecef']@self.gauge,reference_frame_rate=s['frame_velocity']@self.gauge)
        p=np.asarray(self.result['path_parameters_m']) if refined else np.zeros(len(self.result['path_parameters_m']))
        path=basis(t,[self.first,self.last],self.result['path_knot_seconds'],fixed_midpoint=True)@p
        rate=basis(t,[self.first,self.last],self.result['path_knot_seconds'],derivative=1)@p
        return dict(position_ecef_m=kin.position+path[:,None]*kin.direction,
            velocity_ecef_mps=kin.velocity+rate[:,None]*kin.direction+path[:,None]*kin.direction_rate,
            body_to_ecef=kin.frame,frame_velocity=kin.frame_rate)


def render(result,out,spacing=2.,*,refined=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);motion=WindowMotion(result)
    east=np.arange(-300.,300.01,spacing);north=np.arange(-400.,400.01,spacing);xx,yy=np.meshgrid(east,north)
    pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.),np.zeros(xx.size)];images=[];information=[]
    for c in range(5):
        started=time.monotonic();a=load(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'))
        meta=json.loads(a['metadata_json']);keep=(a['time_s']>=motion.first)&(a['time_s']<motion.last);t=a['time_s'][keep]
        s=motion.state(t,refined=refined);rf=meta['channel_configuration']['rf_hz'];g,_=geometry_from_state(s,motion.tx[c],motion.rx,rf,meta['first_range_sample'],a['information'][keep]*a['valid'][keep])
        source=out/f'input-{c}.bin';target=out/f'image-{c}.bin';write_input(source,g,a['moments'][:,keep],pixels,rf)
        subprocess.run(['build/release/support/isar/isar_backprojection',str(source),str(target)],check=True)
        raw=read_image(target,len(pixels));image=(raw[:,0]+1j*raw[:,1])/raw[:,2]
        images.append(image.reshape(xx.shape));information.append(raw[:,2].reshape(xx.shape));source.unlink()
        print(json.dumps(dict(mid=motion.mid,channel=c,seconds=time.monotonic()-started)),flush=True)
    images=np.asarray(images);information=np.asarray(information)
    gain_source=Path('docs/data/isar_20261017/coherent-ship-hull-north-up/analysis.json')
    calibration=json.loads(gain_source.read_text());gains=np.asarray([complex(*z) for z in calibration['gains_real_imag']])
    z,info,eff=combine(images,information,gains);power=np.r_[8192*information*abs(images)**2,(8192*info*abs(z)**2)[None]]
    np.savez(out/'images.npz',east_m=east,north_m=north,individual=images,information=information,combined=z,combined_information=info,
        coherent_to_independent_energy=eff,power=power)
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,12),layout='constrained',sharex=True,sharey=True)
    ais=motion.track.sample(np.array([motion.epoch+motion.mid]),max_gap_s=400.)['position_ecef_m'][0]
    ais_en=(ais-motion.anchor)@motion.local
    for c,ax in enumerate(axes.ravel()):
        im=ax.imshow(10*np.log10(np.maximum(power[c],1e-30)),extent=[east[0],east[-1],north[0],north[-1]],origin='lower',cmap='magma',vmin=3,vmax=55,aspect='equal')
        ax.plot(ais_en[0],ais_en[1],'+',color='#00e5ff',ms=8);ax.set_title(['Seven','SBS','Nine','Ten','ABC','Coherent sum; fixed flash calibration'][c])
        ax.set(xlabel='East (m)',ylabel='North (m)')
    fig.colorbar(im,ax=list(axes.ravel()),label='Matched power / conditional thermal variance (dB)',shrink=.8)
    fig.suptitle(f'Native original-symbol BP · {motion.first:.2f}–{motion.last:.2f} s · north up\nShared physical path correction; cyan = AIS antenna; full unmasked viewport')
    fig.savefig(out/'images.png',dpi=150);plt.close(fig)
    description=dict(interval_s=[motion.first,motion.last],calibration_source=str(gain_source),refined=refined,complete=True,
        maximum_matched_power_db=[float(10*np.log10(p.max())) for p in power],phase_centres_calibrated=False)
    (out/'analysis.json').write_text(json.dumps(description,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fits',type=Path,required=True);p.add_argument('--centres',nargs='+',type=float,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--spacing-m',type=float,default=2.);p.add_argument('--unrefined',action='store_true');a=p.parse_args()
    for mid in a.centres:
        target=a.output/f'window-{mid:07.2f}'
        if (target/'analysis.json').exists():continue
        result=json.loads((a.fits/f'window-{mid:07.2f}'/'analysis.json').read_text());render(result,target,a.spacing_m,refined=not a.unrefined)

if __name__=='__main__':main()
