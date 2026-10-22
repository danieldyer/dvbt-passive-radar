"""Refine physical point positions and motion with shared multiband phases.

One constant phase per band and one phase per physical scatterer explain all
training epochs. Positive scatterer magnitudes may differ across bands. Exact
waveform variable projection profiles those magnitudes; no pixel rephasing,
image rotation or validation samples enter the optimizer.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from audit_ship_exact_aperture import state
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from rigid_patch import RigidPatch,PatchBand
from phase_rigid_patch import PhasePatch
from ship_window import Dataset
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--windows',type=Path,required=True);p.add_argument('--iterations',type=int,default=120);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);old=json.loads(a.motion.read_text());motion=JointMotion(Path(old['motion']));r=motion.result;mid=old['mid_s']
    win=Window(a.windows/f'window-{mid:07.2f}',r['interval_s'],r['path_spacing_seconds'],r['cross_spacing_seconds'])
    dataset=Dataset();window=dataset.window_from_description(win.result);points=np.asarray(old['points']);bands=[]
    for c,base in enumerate(window['fast']):
        s=state(win,np.asarray(r['parameters']),c);kin=RigidPatch(win.time[c],mid,s['position'],s['velocity'],window['local_to_ecef'],motion.tx[3],motion.rx,s['frame'],s['frame_rate'])
        bands.append(PatchBand(base.field,kin,points,base.training,motion.tx[c],motion.rx))
    model=PhasePatch(bands,np.asarray(old['parameters']));start=time.monotonic()
    low=np.r_[[-.8,-.08,-.008,-.2,-.02],np.full(model.size-5,-3.)]
    high=np.r_[[.8,.08,.008,.2,.02],np.full(model.size-5,3.)]
    before=model.validation(model.initial);parameters,fit=model.fit(bounds=(low,high),max_iterations=a.iterations)
    result=dict(old);result.update(source_motion_fit=str(a.motion),coherent_scene_refinement=fit,parameters=fit['motion_parameters'],
        points=fit['points_east_north_up_m'],unused_before_shared_phase=before,unused_after_shared_phase=fit['validation'],
        calibration_scope='Training-only constant band phases shared by all refined scatterers, positive band-dependent point magnitudes. Local aperture only.',
        seconds=time.monotonic()-start,numerical_convergence=fit['success'],accepted_as_global_motion=False)
    gains=np.asarray([complex(*z) for z in fit['gain_phase_real_imag']])*abs(motion.gains)
    result['physical_gains_real_imag']=np.c_[gains.real,gains.imag]
    (a.output/'analysis.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    print(json.dumps(serial(dict(success=fit['success'],gradient=fit['maximum_scaled_gradient'],parameters=fit['motion_parameters'],
        unused_before=sum(v['unused_rss'] for v in before),unused_after=sum(v['unused_rss'] for v in fit['validation']),seconds=result['seconds']))),flush=True)
if __name__=='__main__':main()
