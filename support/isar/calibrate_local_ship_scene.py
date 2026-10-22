"""Compare shared-scene and independent-band responses after a motion fit.

Any estimated band phase is constant over this aperture and shared by all
selected physical locations. It is not a time-varying phase screen or a
full-recording calibration. Unused complex data assess transfer unchanged.
"""
import argparse,json
from pathlib import Path
import numpy as np
from audit_ship_exact_aperture import state
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from rigid_patch import RigidPatch,PatchBand,profile_static
from ship_window import Dataset
from fit_ship_extended_scene import report_fit
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--windows',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.motion.read_text());base=JointMotion(Path(r['motion']));description=base.result;mid=r['mid_s']
    win=Window(a.windows/f'window-{mid:07.2f}',description['interval_s'],description['path_spacing_seconds'],description['cross_spacing_seconds'])
    dataset=Dataset();exact=dataset.window_from_description(win.result);points=np.asarray(r['points']);parameters=np.asarray(r['parameters'])
    designs=[];data=[];selected=[]
    for c,old in enumerate(exact['fast']):
        s=state(win,np.asarray(description['parameters']),c);kin=RigidPatch(win.time[c],mid,s['position'],s['velocity'],exact['local_to_ecef'],base.tx[3],base.rx,s['frame'],s['frame_rate'])
        band=PatchBand(old.field,kin,points,old.training,base.tx[c],base.rx);h,_=band.design(parameters);y=band.field.y;root=band.field.root_information
        design=np.zeros_like(h);value=np.zeros_like(y)
        for mask in [band.training,~band.training]:
            design[mask]=profile_static(h,root,mask)[mask];value[mask]=profile_static(y,root,mask)[mask]
        designs.append(design.reshape(-1,len(points)));data.append(value.ravel());selected.append(np.repeat(band.training,len(band.field.offsets)))
    fit=report_fit(designs,data,list(range(len(points))),selected)
    g=np.asarray(fit['phase_coherent']['gain_real_imag']);g=g[:,0]+1j*g[:,1];g*=abs(base.gains)
    r.update(scene_response_comparison=fit,physical_gains_real_imag=np.c_[g.real,g.imag],
        calibration_scope='Training-only constant band phases for this aperture; shared scatterer phases and independent positive magnitudes. Not a full-capture calibration.',
        source_motion_fit=str(a.motion))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(serial(r),indent=2)+'\n')
    print(json.dumps(serial(fit['unused'])),flush=True)
if __name__=='__main__':main()
