"""Recorded-data gradient control for joint fading/source/motion refinement."""
import argparse,json
from pathlib import Path
import numpy as np
from full_ship_field import load
from full_path_point import PointBand
from full_path_scene import SceneBand,RefinedScenePath
from joint_ship_motion import JointMotion
from scene_phase_coordinates import transform
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.motion.read_text());motion=JointMotion(Path(d['motion']));motion.interval=d['interval_s'];fields,coordinates,_=load(d['field']);bands=[]
    points=np.asarray(d['scene_points_body_m'])
    for full,coordinate in zip(fields,coordinates):
        f=full.windowed(*motion.interval);indices=np.searchsorted(full.time,f.time)
        co={k:coordinate[k].reshape(-1,16)[indices].ravel() for k in ['native_time_s','native_information']};co['training']=coordinate['training'][indices]
        if 'stationary_groups' in coordinate:co['stationary_groups']=coordinate['stationary_groups'][indices]
        b=PointBand(f,co,motion,d['point_body_m'],np.asarray(d['path_knots_s']),np.asarray(d['path_coefficients_m']),d['anchor_path_time_s'],d['amplitude_spacing_s'],d['orientation'])
        bands.append(SceneBand(b,points,amplitude_spacing=d['amplitude_spacing_s']))
    likelihood=RefinedScenePath(bands);n=likelihood.n
    x=np.r_[np.zeros(n),np.angle([complex(*z) for z in d['physical_gains_real_imag']]),np.asarray(d['scene_point_phase_radians'])[1:],np.zeros(2*likelihood.p)]
    matrix=transform(likelihood,len(x));q=np.linalg.solve(matrix,x);cost,g=likelihood.objective(x);rng=np.random.default_rng(12716);checks=[]
    for _ in range(4):
        direction=rng.normal(size=len(x));direction/=np.linalg.norm(direction);step=1e-4
        plus=likelihood.objective(matrix@(q+step*direction))[0];minus=likelihood.objective(matrix@(q-step*direction))[0]
        numerical=(plus-minus)/(2*step);analytic=g@matrix@direction
        checks.append(dict(numerical=float(numerical),analytic=float(analytic),error=float(abs(numerical-analytic))))
    maximum=max(c['error'] for c in checks)
    output=dict(complete=True,scope=__doc__,motion=str(a.motion),initial_objective=cost,checks=checks,maximum_directional_cost_error=maximum,passed=maximum<1e-7)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(serial(output),indent=2)+'\n');print(json.dumps(serial(output)),flush=True)
    if not output['passed']:raise AssertionError('Joint fading/source/path derivative did not match the recorded likelihood')


if __name__=='__main__':main()
