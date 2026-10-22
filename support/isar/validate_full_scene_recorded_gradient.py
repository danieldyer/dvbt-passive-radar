"""Check coupled whole-record scene gradients on retained recorded epochs."""
import argparse,json
from pathlib import Path
import numpy as np
from full_ship_field import load
from full_path_point import PointBand
from full_path_scene import SceneBand,ScenePath,RefinedScenePath
from recovery_experiment import thin_field
from joint_ship_motion import JointMotion
from motion import enu_basis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refine-points',action='store_true')
    p.add_argument('--amplitude-spacing-s',type=float)
    a=p.parse_args();d=json.loads(a.reference.read_text());scene=json.loads(a.scene.read_text());motion=JointMotion(Path(d['motion']));fields,coordinates,_=load(Path(d['field']))
    at=np.array([scene['mid_s']]);s=motion.state(at);sample=motion.track.sample(motion.epoch+at,max_gap_s=400.);g=s['body_to_ecef'][0].T@enu_basis(sample['latitude'][0],sample['longitude'][0]);points=np.asarray(scene['points'])@g.T
    bands=[]
    for f,c in zip(fields,coordinates):
        bins=np.arange(0,len(f.time),7);selected=thin_field(f,7);coordinate={}
        for key in ['native_time_s','native_information']:coordinate[key]=c[key].reshape(-1,16)[bins].ravel()
        coordinate['training']=c['training'][bins]
        if 'stationary_groups' in c:coordinate['stationary_groups']=c['stationary_groups'][bins]
        spacing=d['amplitude_spacing_s'] if a.amplitude_spacing_s is None else a.amplitude_spacing_s
        b=PointBand(selected,coordinate,motion,d['point_body_m'],np.asarray(d['path_knots_s']),np.asarray(d['path_coefficients_m']),162.,spacing,'reported-course')
        bands.append(SceneBand(b,points,scene.get('scene_window_edges_s'),spacing))
    likelihood=(RefinedScenePath if a.refine_points else ScenePath)(bands);end=likelihood.n+5+likelihood.p-1
    parameters=np.zeros(end+(2*likelihood.p if a.refine_points else 0))
    parameters[likelihood.n:likelihood.n+5]=np.angle([complex(*z) for z in d['physical_gains_real_imag']]);phi=np.asarray(scene['phase_parameters'])[4:];parameters[likelihood.n+5:end]=phi[1:]-phi[0]
    value,gradient=likelihood.objective(parameters);rng=np.random.default_rng(20261018);reports=[]
    for j in range(4):
        direction=rng.normal(size=len(parameters));direction/=np.linalg.norm(direction);step=1e-4
        plus=likelihood.objective(parameters+step*direction)[0];minus=likelihood.objective(parameters-step*direction)[0]
        measured=(plus-minus)/(2*step);predicted=float(gradient@direction);error=abs(measured-predicted)
        reports.append(dict(direction=j,finite_difference=measured,analytic=predicted,absolute_error=error));print(json.dumps(reports[-1]),flush=True)
        if error>2e-5:raise AssertionError('Recorded whole-scene motion/phase gradient failed')
    a.output.write_text(json.dumps(dict(complete=True,scope=__doc__,stride=7,objective=value,checks=reports),indent=2)+'\n')

if __name__=='__main__':main()
