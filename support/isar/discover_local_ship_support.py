"""Training-only, range-separated scattering supports for each coherent interval.

Candidate ranking uses the calibrated coherent likelihood, not an image
normalization. The AIS neighbourhood supplies a broad acquisition gate only.
Positions remain conditional on the current geometry/motion. Low-support
candidates are reported; they are not certified detections or hull points.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
from local_ship_motion import load_motion
from motion import C,enu_basis,excess_path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--images',type=Path,required=True);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--points',type=int,default=3);p.add_argument('--range-separation-m',type=float,default=45.);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    motion=load_motion(a.motion);scenes=[];edges=[];phases=[];reports=[]
    roots=sorted(a.images.glob('aperture-*'));at=np.array([157.]);ref=motion.state(at);sample=motion.track.sample(motion.epoch+at,max_gap_s=400.);reference_gauge=ref['body_to_ecef'][0].T@enu_basis(sample['latitude'][0],sample['longitude'][0])
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(4,5,figsize=(20,17),layout='constrained',sharex=True,sharey=True)
    for w,root in enumerate(roots):
        d=json.loads((root/'analysis.json').read_text())
        if 'training_epoch_selection' not in d:raise ValueError('Scatterer discovery must not use excluded epochs')
        first,last=d['interval_s'];mid=(first+last)/2;s=motion.state(np.array([mid]));ais=motion.track.sample(np.array([motion.epoch+mid]),max_gap_s=400.)
        local=enu_basis(ais['latitude'][0],ais['longitude'][0]);gauge=s['body_to_ecef'][0].T@local
        z=np.load(root/'images.npz');east,north=z['east_m'],z['north_m'];xx,yy=np.meshgrid(east,north)
        pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.)];body=pixels@gauge.T
        position=s['position_ecef_m'][0]+body@s['body_to_ecef'][0].T
        path=excess_path(position,motion.tx[3],motion.rx)[0].reshape(xx.shape)
        predicted=excess_path(ais['position_ecef_m'],motion.tx[3],motion.rx)[0][0]
        power=z['coherent_positive_amplitude_glrt'];upper=z['independent_amplitude_bound'];ratio=np.divide(power,upper,out=np.zeros_like(power),where=upper>0)
        offset=(ais['position_ecef_m'][0]-s['position_ecef_m'][0])@local
        # This deliberately selects the earlier AIS-neighbourhood branch,
        # whose good image is the requested baseline. The stronger delayed
        # branch remains a separate association/control, not a forced hull.
        gate=(abs(path-predicted)<275)&((xx-offset[0])**2+(yy-offset[1])**2<350**2)
        maxima=(power==maximum_filter(power,size=5,mode='constant'))&gate
        candidates=np.flatnonzero(maxima);order=candidates[np.argsort(power.ravel()[candidates])[::-1]];chosen=[]
        for index in order:
            if all(abs(path.ravel()[index]-path.ravel()[j])>=a.range_separation_m for j in chosen):chosen.append(int(index))
            if len(chosen)==a.points:break
        if len(chosen)!=a.points:raise ValueError(f'{root.name}: insufficient distinct range supports; do not duplicate unresolved points')
        points=body[chosen];phase=z['glrt_reflector_phase_radians'].ravel()[chosen]
        if w==0:edges.append(first)
        elif abs(first-edges[-1])>1e-6:raise ValueError('Scattering supports must tile the observations without overlap/gaps')
        edges.append(last);scenes.append(points@reference_gauge);phases.extend(phase)
        quality=[]
        for index in chosen:
            perband=z['power'][:5].reshape(5,-1)[:,index]
            quality.append(dict(east_m=float(xx.ravel()[index]),north_m=float(yy.ravel()[index]),excess_path_m=float(path.ravel()[index]),ais_path_residual_m=float(path.ravel()[index]-predicted),
                coherent_statistic_db=float(10*np.log10(power.ravel()[index])),band_matched_power_db=(10*np.log10(perband)).tolist(),
                common_phase_fraction=float(ratio.ravel()[index]),bands_above_12db=int(np.count_nonzero(perband>10**1.2))))
        reports.append(dict(interval_s=[first,last],candidates=quality));ax=axes.ravel()[w]
        ax.imshow(10*np.log10(np.maximum(power,1e-30)),extent=[east[0],east[-1],north[0],north[-1]],origin='lower',cmap='magma',vmin=3,vmax=55,aspect='equal')
        ax.plot(offset[0],offset[1],'+',color='#00e5ff')
        for j,index in enumerate(chosen):
            x,y=xx.ravel()[index],yy.ravel()[index];ax.plot(x,y,'o',mfc='none',mec='#00e5ff');ax.annotate(str(j+1),(x,y),color='#00e5ff',fontsize=8)
        ax.set(title=f'{first:.1f}–{last:.1f} s',xlabel='East (m)',ylabel='North (m)')
    phases=np.asarray(phases);beta=np.angle(motion.gains)+phases[0]
    output=dict(complete=True,scope=__doc__,motion=str(a.motion),mid_s=157.,points=np.asarray(scenes).tolist(),scene_window_edges_s=edges,
        phase_parameters=np.r_[np.zeros(4),phases].tolist(),band_phase_initial_radians=beta.tolist(),candidate_quality=reports,
        range_separation_m=a.range_separation_m,source_images=str(a.images),all_candidates_are_training_only=True,accepted_as_hull=False)
    (a.output/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    fig.suptitle('Training-only candidate supports through the complete recording; one fixed colour scale\nCyan circles: range-separated local likelihood maxima; cyan cross: AIS prior; candidates are not certified hull points')
    fig.savefig(a.output/'selected-scattering-supports.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(windows=len(scenes),points_per_window=a.points,phase_groups=len(phases))),flush=True)

if __name__=='__main__':main()
