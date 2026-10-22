"""Measure the phase effect of the inherited transverse trajectory.

The measured reference-channel path/rate are retained by the exact physical
path lift. Only the inherited transverse-displacement hypothesis changes.
No observations are rephased and no new trajectory is adopted by this audit.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from full_path_point import PointBand
from bistatic_path_lift import lift
from motion import excess_path


def paths(band):
    s=lift(band.anchor,band.velocity,band.direction,band.direction_rate,band.base_path,band.base_rate,band.motion.tx[3],band.motion.rx)
    return np.array([excess_path(s['position_ecef_m'],tx,band.motion.rx)[0] for tx in band.motion.tx])


def alternate(band,fit,scale):
    motion=copy.copy(band.motion);motion.cross=motion.cross*scale
    return PointBand(band.field,band.coordinate,motion,band.reference_point,fit['path_knots_s'],fit['path_coefficients_m'],
        fit.get('anchor_path_time_s',162.),fit['amplitude_spacing_s'],fit.get('orientation','existing'),
        angular_rate_offset_degps=fit.get('angular_rate_offset_degps',0.),angular_acceleration_offset_degps2=fit.get('angular_acceleration_offset_degps2',0.))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--scales',type=float,nargs='+',default=[0.,.5,1.]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);fit,motion,scene,_=open_scene(a.motion)
    b=scene[3].point;old=paths(b);reports=[];arrays=dict(time_s=b.t)
    for scale in a.scales:
        value=alternate(b,fit,scale);new=paths(value);delta=new-old
        phase=delta*np.array([f.field.k for f in [s.point for s in scene]])[:,None]
        relative=phase-phase[:,len(b.t)//2,None]
        reports.append(dict(transverse_scale=scale,phase_excursion_degrees=np.degrees(np.ptp(phase,axis=1)).tolist(),
            maximum_reference_path_error_m=float(max(abs(delta[3])))))
        arrays[f'path_delta_{scale:g}']=delta;arrays[f'phase_delta_{scale:g}']=relative
    result=dict(scope=__doc__,motion=str(a.motion),inherited_cross_coefficients=len(motion.cross),
        maximum_inherited_cross_coefficient_m=float(max(abs(motion.cross),default=0)),hypotheses=reports)
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(a.output/'histories.npz',**arrays)
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(5,1,figsize=(12,11),layout='constrained',sharex=True)
    for c,ax in enumerate(axes):
        for scale in a.scales:ax.plot(b.t,np.degrees(arrays[f'phase_delta_{scale:g}'][c]),label=f'Transverse scale {scale:g}')
        ax.set(ylabel=['Seven','SBS','Nine','Ten','ABC'][c]+' phase (deg)');ax.legend(fontsize=8)
    axes[-1].set_xlabel('Capture time (s)');fig.suptitle('Inherited transverse-motion phase effect at one physical reference point\nTen path/rate held exact; one constant phase origin removed for display only; no correction applied')
    fig.savefig(a.output/'transverse-phase-effect.png',dpi=145);plt.close(fig);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
