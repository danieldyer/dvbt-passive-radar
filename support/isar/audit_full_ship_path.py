"""Excluded-epoch phase and native-state controls for a continuous radio path."""
import argparse,json
from pathlib import Path
import numpy as np
from full_ship_field import load
from full_path_point import PointBand
from joint_ship_motion import JointMotion
from local_ship_motion import load_motion
from motion import excess_path
from rigid_patch import profile_static
from gpga import Isolation
from coherent_amplitude_glrt import statistic
from full_path_scene import SceneBand


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);d=json.loads(a.motion.read_text());motion=JointMotion(Path(d['motion']));native=load_motion(a.motion)
    fields,coordinates,_=load(Path(d['field']));saved=np.load(a.motion.parent/'trajectory.npz');point=np.asarray(d['point_body_m'])
    points=np.asarray(d.get('scene_points_body_m',[point]));phi=np.asarray(d.get('scene_point_phase_radians',[0.]));gains=np.array([complex(*z) for z in d['physical_gains_real_imag']])
    interval=d['interval_s'];arrays={};reports=[];centres=np.arange(interval[0]+15,interval[1]-15+.001,15.);scores=[]
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(15,12),layout='constrained',sharex=True)
    names=['Seven','SBS','Nine','Ten','ABC']
    for c,(field,coordinate) in enumerate(zip(fields,coordinates)):
        if interval!=motion.interval:
            original=field;field=field.windowed(*interval);indices=np.searchsorted(original.time,field.time)
            coordinate={k:coordinate[k].reshape(-1,16)[indices].ravel() if k in ['native_time_s','native_information'] else coordinate[k][indices] for k in coordinate if k in ['native_time_s','native_information','training','stationary_groups']}
        b=PointBand(field,coordinate,motion,point,np.asarray(d['path_knots_s']),np.asarray(d['path_coefficients_m']),d.get('anchor_path_time_s',162.),d['amplitude_spacing_s'],d.get('orientation','existing'),angular_rate_offset_degps=d.get('angular_rate_offset_degps',0.),angular_acceleration_offset_degps2=d.get('angular_acceleration_offset_degps2',0.))
        band=SceneBand(b,points,d.get('scene_window_edges_s'),d['amplitude_spacing_s'],d.get('scene_active_points'))
        h=band.design(np.zeros(b.b.shape[1]),c,False)
        amp=saved[f'amplitude_{c}'];amp=amp[:,None] if amp.ndim==1 else amp
        template=np.sum(h*(amp*np.exp(1j*(band.phase_at_time(phi)+np.angle(gains[c]))))[:,None,:],axis=-1)
        model=np.zeros_like(template);data=np.zeros_like(template);z=np.zeros(len(b.t),complex);info=np.zeros(len(b.t))
        for mask in [b.training,~b.training]:
            model[mask]=profile_static(template,field.root_information,mask,b.groups)[mask];data[mask]=profile_static(field.y,field.root_information,mask,b.groups)[mask]
        info=np.sum(abs(model)**2,axis=1);product=np.sum(model.conj()*data,axis=1)
        z=np.divide(product,info,out=np.zeros_like(product),where=info>0)
        smooth,_=Isolation(b.t,info,b.training&(info>0),.6).apply(z[:,None])
        axes[0].plot(b.t,np.degrees(np.angle(smooth[:,0])),label=names[c],lw=.8)
        support=[];coherence=[];score=[];signed_noise_corrected=[]
        for t in centres:
            mask=(abs(b.t-t)<15)&~b.training&(info>0);total=info[mask].sum();energy=np.sum(info[mask]*abs(z[mask])**2);sumz=product[mask].sum()
            coh=float(abs(sumz)**2/(total*energy)) if total*energy>0 else None
            coherence.append(coh);score.append(sumz/np.sqrt(total) if total>0 else 0j);support.append(float(total))
            denominator=total*(energy-mask.sum());signed_noise_corrected.append(float((abs(sumz)**2-total)/denominator) if denominator>0 else None)
        scores.append(score)
        axes[1].plot(centres,coherence,'.-',label=names[c]);axes[2].plot(b.t,10*np.log10(np.maximum(info,1e-30)),label=names[c],lw=.7)
        # A different evaluator must describe the same point and velocity.
        t=b.t[::97];s=native.state(t);position=s['position_ecef_m']+np.einsum('tij,j->ti',s['body_to_ecef'],point)
        velocity=s['velocity_ecef_mps']+np.einsum('tij,j->ti',s['frame_velocity'],point)
        path,gradient=excess_path(position,native.tx[3],native.rx);rate=np.sum(gradient*velocity,axis=1)
        target=native.path_curve(t);target_rate=native.path_curve(t,1)
        path_error=float(max(abs(path-target)));rate_error=float(max(abs(rate-target_rate)))
        if path_error>1e-6 or rate_error>1e-8:raise AssertionError('Native state differs from the fitted physical carrier path')
        for key,value in dict(time_s=b.t,value=z,information=info,training=b.training,model=model,data=data).items():arrays[f'{key}_{c}']=value
        reports.append(dict(channel=c,coherence=coherence,conditional_noise_corrected_coherence=signed_noise_corrected,window_information=support,
            maximum_native_path_error_m=path_error,maximum_native_rate_error_mps=rate_error))
        print(json.dumps({k:v for k,v in reports[-1].items() if not isinstance(v,list)}),flush=True)
    score=np.asarray(scores);glrt,_=statistic(score);upper=np.sum(abs(score)**2,axis=0);ratio=np.divide(glrt,upper,out=np.zeros_like(glrt),where=upper>0)
    np.savez_compressed(a.output/'histories.npz',**arrays)
    axes[0].set(ylabel='Matched residual phase (degrees)',ylim=[-180,180],title='Training-only 0.6 Hz display smoothing; no correction applied');axes[0].legend(ncol=5)
    axes[1].set(ylabel='Excluded temporal coherent fraction',ylim=[0,1.02],title='30 s windows, 15 s centres; includes noise and scattering-model mismatch')
    axes[1].plot(centres,ratio,'k--',label='Cross-band GLRT / free-band bound');axes[1].legend(ncol=3,fontsize=8)
    axes[2].set(ylabel='Predicted matched information (dB)',xlabel='Capture time (s)',title='Weak support remains visible; conditional thermal units, not certified interference SNR')
    scene_label=(f'{len(points)} aspect supports, {points.shape[1]} points each' if points.ndim==3 else f'{len(points)} persistent scatterers')
    fig.suptitle(f'Whole-record shared physical path · {scene_label} · constant band/scatterer phases\nNative geometric delay/rate checked independently; excluded data were not used to estimate this model')
    fig.savefig(a.output/'full-path-phase-validation.png',dpi=140);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,motion=str(a.motion),scope=__doc__,centres_s=centres.tolist(),channels=reports,
        excluded_cross_band_glrt_ratio=ratio.tolist(),geographic_localization_proven=False),indent=2)+'\n')

if __name__=='__main__':main()
