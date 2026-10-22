"""Prepare a physical search patch around an acquired delay branch.

The fixed observer is a declared reversible range/phase coordinate choice.
It does not force the target to the AIS antenna or shift observed ranges.
Pixels span a bistatic path-offset coordinate and transverse coordinate;
the source archive and final native image keep their full original support.
"""
import argparse,json
from pathlib import Path
import numpy as np
from dense_patch_image import DenseImages
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from motion import C,excess_path
from ship_window import Dataset
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--centre',type=float,default=157.)
    p.add_argument('--path-offset-m',type=float,default=450.);p.add_argument('--spacing-m',type=float,default=5.);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();directory=a.output/f'window-{a.centre:07.2f}';directory.mkdir(parents=True,exist_ok=False);motion=JointMotion(a.motion);r=motion.result;dataset=Dataset()
    first,last=a.centre-15,a.centre+15;anchor=dataset.state(np.array([a.centre]),r['base_model']);sample=dataset.track.sample(np.array([dataset.epoch+a.centre]),max_gap_s=400.)
    from motion import enu_basis
    local=enu_basis(sample['latitude'][0],sample['longitude'][0]);_,gradient=excess_path(anchor['position_ecef_m'],dataset.tx[3],dataset.rx)
    g=gradient[0];up=local[:,2];g-=up*(g@up);q=g/(g@g);cross=np.cross(up,q);cross/=np.linalg.norm(cross)
    observer=anchor['position_ecef_m'][0]+q*(a.path_offset_m/2)
    window=dataset.window(first,last,model=r['base_model'],observer_ecef_m=observer)
    u=np.arange(a.path_offset_m-200,a.path_offset_m+200+.01,a.spacing_m);v=np.arange(-500.,500.01,a.spacing_m);uu,vv=np.meshgrid(u,v)
    pixels=np.c_[uu.ravel()[:,None]*(q@local)[None,:2]+vv.ravel()[:,None]*(cross@local)[None,:2],np.full(uu.size,20.)]
    context={}
    for c,b in enumerate(window['fast']):
        f=b.field;kin=b.kinematics
        for name,value in dict(time=f.time,information=f.data['information'],position=kin.position,velocity=kin.velocity,
                frame=kin.frame,frame_rate=kin.frame_rate,direction=kin.direction,direction_rate=kin.direction_rate,
                observer=f.data['centre_path_m'],carrier=f.data.get('carrier_reference_path_m',f.data['centre_path_m']),
                tx=b.tx,rx=b.rx,rf_hz=f.metadata['rf_hz'],training=b.training).items():context[f'{name}_{c}']=value
    np.savez(directory/'geometry-context.npz',**context)
    result=dict(complete=False,mid_s=a.centre,interval_s=[first,last],source_model=r['base_model'],path_knot_seconds=3.,path_parameters_m=[],
        fixed_observer_ecef_m=observer,association='Acquired excess-path branch; not a hull mask or assertion of AIS identity',
        association_origin_east_north_m=[0.,0.],forward_limits_m=[u[0],u[-1]],axis_labels=['Bistatic path offset coordinate (m)','Transverse coordinate (m)'],
        path_offset_m=a.path_offset_m,spacing_m=a.spacing_m,source_motion=str(a.motion),training_epoch_s=3.32)
    (directory/'analysis.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    w=Window(directory,r['interval_s'],r['path_spacing_seconds'],r['cross_spacing_seconds']);images={}
    for training in [True,False]:
        label='training' if training else 'unused';worker=DenseImages(window['fast'],pixels,directory/label,training=training)
        worker.close();worker=DenseImages.reopen(directory/label/'model.bin',derivatives=8);worker.training=training
        try:
            geometry,_,_=w.geometry(np.asarray(r['parameters']),worker,motion.gauge_point,r['transmitter_groups'],motion.receiver_local,antennas=bool(r['parameter_counts']['antenna']))
            image=worker.evaluate_geometry(geometry,derivatives=False)[:,0];images[label]=image
        finally:worker.close()
        np.savez_compressed(directory/(label+'-images.npz'),pixels=pixels,forward_m=u,port_m=v,before=image,after=image)
    result['complete']=True;(directory/'analysis.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(20,9),layout='constrained')
    for row,label in enumerate(['training','unused']):
        for c in range(5):
            info=8192*np.sum(context[f'information_{c}'][w.training[c] if label=='training' else ~w.training[c]])
            power=abs(images[label][c])**2/info
            im=axes[row,c].imshow(10*np.log10(np.maximum(power,1e-30)).reshape(v.size,u.size),extent=[u[0],u[-1],v[0],v[-1]],origin='lower',cmap='magma',vmin=0,vmax=55,aspect='auto')
            axes[row,c].set(title=['Seven','SBS','Nine','Ten','ABC'][c]+' · '+label,xlabel='Bistatic path-offset coordinate (m)',ylabel='Transverse coordinate (m)')
    fig.colorbar(im,ax=list(axes.ravel()),shrink=.7,label='Matched power / conditional thermal variance (dB)');fig.suptitle(f'{first:.2f}–{last:.2f} s · acquired delay branch before motion refinement\nExpanded physical association; same original data, fixed surveyed sites, no fitted per-band phase histories')
    fig.savefig(directory/'branch-acquisition.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
