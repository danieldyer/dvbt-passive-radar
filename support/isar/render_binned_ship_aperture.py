"""Form complete physical ship images for arbitrary supported apertures."""
import argparse,json,time
from pathlib import Path
import numpy as np
from joint_ship_motion import JointMotion
from local_ship_motion import load_motion
from ship_coherent_movie import load_apertures
from channel_update import ChannelUpdates
from imaging import geometry_from_state
from motion import enu_basis
from coherent_amplitude_glrt import statistic


def form(motion,apertures,first,last,out,spacing):
    out.mkdir(parents=True,exist_ok=False);mid=(first+last)/2;s=motion.state(np.array([mid]));ais=motion.track.sample(np.array([motion.epoch+mid]),max_gap_s=400.)
    local=enu_basis(ais['latitude'][0],ais['longitude'][0]);gauge=s['body_to_ecef'][0].T@local
    east=np.arange(-300.,300.01,spacing);north=np.arange(-400.,400.01,spacing);xx,yy=np.meshgrid(east,north);pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.),np.zeros(xx.size)]
    num=np.zeros((5,len(pixels)),complex);information=np.zeros(5);started=time.monotonic();worker=ChannelUpdates(pixels,out/'worker')
    try:
        for c,a in enumerate(apertures):
            selected=np.flatnonzero((a['first_time_s']>=first)&(a['last_time_s']<=last)&(a['information']>0))
            for begin in range(0,len(selected),256):
                idx=selected[begin:begin+256];t=a['time_s'][idx];state=motion.state(t);state['body_to_ecef']=state['body_to_ecef']@gauge;state['frame_velocity']=state['frame_velocity']@gauge
                geo,details=geometry_from_state(state,motion.tx[c],motion.rx,motion.rf[c],a['meta']['first_range_sample'],a['information'][idx])
                phase=2*np.pi*motion.rf[c]/299792458.*(details['path_m']-a['carrier_reference_path_m'][idx]);geo[:,9,1]=np.cos(phase);geo[:,9,2]=np.sin(phase)
                geo[:,5,3]=2*np.pi*motion.rf[c]/299792458.;geo[:,6,3]=c;geo[:,7,3]=1.;geo[:,8,3]=1.
                z,w=worker.update(geo,a['moments'][:,idx]);num+=z;information+=w
            print(json.dumps(dict(interval_s=[first,last],channel=c,seconds=time.monotonic()-started)),flush=True)
    finally:worker.close()
    images=num/information[:,None];total=np.sum(abs(motion.gains)**2*information);combined=np.einsum('c,cp->p',motion.gains.conj(),num)/total
    power=np.r_[8192*abs(num)**2/information[:,None],(8192*abs(combined)**2*total)[None]].reshape(6,len(north),len(east))
    whitened=np.sqrt(8192)*num/np.sqrt(information[:,None])*np.exp(-1j*np.angle(motion.gains))[:,None]
    glrt,reflector_phase=statistic(whitened);glrt=glrt.reshape(len(north),len(east));bound=np.sum(power[:5],axis=0)
    np.savez_compressed(out/'images.npz',east_m=east,north_m=north,individual=images.reshape(5,len(north),len(east)),information=information,combined=combined.reshape(len(north),len(east)),power=power,
        coherent_positive_amplitude_glrt=glrt,glrt_reflector_phase_radians=reflector_phase.reshape(glrt.shape),independent_amplitude_bound=bound)
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,13),layout='constrained',sharex=True,sharey=True);offset=(ais['position_ecef_m'][0]-s['position_ecef_m'][0])@local
    for c,ax in enumerate(axes.ravel()):
        im=ax.imshow(10*np.log10(np.maximum(power[c],1e-30)),extent=[east[0],east[-1],north[0],north[-1]],origin='lower',cmap='magma',vmin=3,vmax=55,aspect='equal')
        ax.plot(offset[0],offset[1],'+',color='#00e5ff');ax.set(title=['Seven','SBS','Nine','Ten','ABC','Static-calibration coherent sum'][c],xlabel='East (m)',ylabel='North (m)')
    fig.colorbar(im,ax=list(axes.ravel()),label='Matched power / conditional thermal variance (dB)',shrink=.8)
    label='Local physical motion; whole-capture band calibration' if hasattr(motion,'refinement') else 'Whole-capture physical motion and band calibration'
    if hasattr(motion,'path_fit'):
        points=np.asarray(motion.path_fit.get('scene_points_body_m',[motion.path_fit['point_body_m']]))
        if points.ndim==3:
            counts=np.asarray(motion.path_fit.get('scene_active_points',np.ones(points.shape[:2],bool))).sum(axis=1)
            source_count=str(counts[0]) if np.all(counts==counts[0]) else f'{counts.min()}–{counts.max()}'
            scene_label=f'{len(points)} aspect supports, {source_count} sources per support'
        else:scene_label=f'{len(points)} points'
        label=f'Whole-record physical path; {scene_label}; constant band phases'
        if motion.path_fit.get('initializer_only'):label+='; motion initializer'
    if hasattr(motion,'refinement') and 'physical_gains_real_imag' in motion.refinement:label='Local physical motion and aperture-wide constant band calibration; not full-capture calibrated'
    if 'stationary_projection' in apertures[0]:label+='; whole-record stationary projection (original norm retained)'
    if 'training_epoch_selection' in apertures[0]:label+='; training epochs only'
    from textwrap import fill
    fig.suptitle(f'Complete {last-first:.2f} s aperture · {first:.2f}–{last:.2f} s · north up\n'
        +fill(label,width=105)+'\nNo time-varying band phase or notch; geographic and extended-scene validation pending',fontsize=11)
    fig.savefig(out/'images.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,13),layout='constrained',sharex=True,sharey=True)
    for ax,p,title in zip(axes.ravel()[:3],[power[5],glrt,bound],['Fixed-spectrum complex sum','Coherent GLRT: unknown positive band brightness','Independent-band upper bound: arbitrary band phases']):
        im=ax.imshow(10*np.log10(np.maximum(p,1e-30)),extent=[east[0],east[-1],north[0],north[-1]],origin='lower',cmap='magma',vmin=3,vmax=55,aspect='equal');ax.set(title=title,xlabel='East (m)',ylabel='North (m)')
        ax.plot(offset[0],offset[1],'+',color='#00e5ff')
    fig.colorbar(im,ax=list(axes.ravel()[:3]),label='Conditional Gaussian matched-power statistic (dB)',shrink=.65)
    ax=axes[1,1];eff=np.divide(glrt,bound,out=np.zeros_like(glrt),where=bound>0)
    im=ax.imshow(eff,extent=[east[0],east[-1],north[0],north[-1]],origin='lower',cmap='viridis',vmin=0,vmax=1,aspect='equal');ax.set(title='Coherent GLRT / independent-band bound',xlabel='East (m)',ylabel='North (m)');fig.colorbar(im,ax=ax,shrink=.6)
    fig.suptitle(f'{first:.2f}–{last:.2f} s · exact native physical BP; same maps and fixed colour scale\nGLRT profiles one common reflector phase and positive magnitudes; calibrated band phases stay fixed')
    fig.savefig(out/'coherent-statistics.png',dpi=150);plt.close(fig)
    record=dict(complete=True,interval_s=[first,last],motion=str(motion.path),spacing_m=spacing,seconds=time.monotonic()-started,
        maximum_matched_power_db=[float(10*np.log10(p.max())) for p in power])
    if 'stationary_projection' in apertures[0]:
        fraction=information/np.array([a['information'].sum() for a in apertures])
        record.update(stationary_projection=apertures[0]['stationary_projection'],original_normalization_retained=True,
            conditional_maximum_omitted_schur_information_change_db=(-10*np.log10(1-fraction)).tolist())
    if 'training_epoch_selection' in apertures[0]:record['training_epoch_selection']=[a['training_epoch_selection'] for a in apertures]
    (out/'analysis.json').write_text(json.dumps(record,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--aperture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--intervals',nargs='+',type=float,required=True);p.add_argument('--spacing-m',type=float,default=1.)
    p.add_argument('--stationary-projection',action='store_true',help='Reversible whole-record constant-profile nuisance comparison; original normalization retained')
    p.add_argument('--training-field',type=Path,help='Restrict candidate images to the original training waveform blocks and known-template support')
    a=p.parse_args();motion=load_motion(a.motion);apertures,_=load_apertures(a.aperture)
    if a.stationary_projection:
        from projected_aperture import apply
        apertures=apply(apertures,motion.rf)
    if a.training_field:
        from training_aperture import apply
        apertures=apply(apertures,a.training_field)
    if len(a.intervals)%2:raise ValueError('Supply first/last pairs')
    for first,last in zip(a.intervals[::2],a.intervals[1::2]):
        out=a.output/f'aperture-{first:07.2f}-{last:07.2f}'
        if not (out/'analysis.json').exists():form(motion,apertures,first,last,out,a.spacing_m)
if __name__=='__main__':main()
