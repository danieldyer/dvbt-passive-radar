"""Exact physical local refinement around a retained whole-capture model."""
import json
from pathlib import Path
import numpy as np
from joint_ship_motion import JointMotion
from rigid_patch import RigidPatch
from motion import enu_basis


class LocalShipMotion(JointMotion):
    def __init__(self,path):
        description=json.loads(Path(path).read_text());super().__init__(Path(description['motion']))
        self.path=Path(path);self.refinement=description;self.local_parameters=np.asarray(description['parameters'])
        self.mid=description['mid_s'];self.interval=[self.mid-15,self.mid+15]
        a=self.track.sample(np.array([self.epoch+self.mid]),max_gap_s=400.)
        self.refinement_local=enu_basis(a['latitude'][0],a['longitude'][0])
        if 'physical_gains_real_imag' in description:self.gains=np.asarray([complex(*z) for z in description['physical_gains_real_imag']])

    def state(self,t):
        t=np.atleast_1d(t)
        if np.any(t<self.interval[0]) or np.any(t>self.interval[1]):raise ValueError('Local motion extrapolation outside its fitted aperture is forbidden')
        s=super().state_at_basis(t,self.refinement_local)
        kin=RigidPatch(t,self.mid,s['position_ecef_m'],s['velocity_ecef_mps'],self.refinement_local,self.tx[3],self.rx,
            s['body_to_ecef'],s['frame_velocity'])
        position,velocity=kin.point_state(self.local_parameters,[0.,0.,0.]);frame,rate=kin.orientation(self.local_parameters)
        return dict(position_ecef_m=position,velocity_ecef_mps=velocity,body_to_ecef=frame,frame_velocity=rate)


def load_motion(path):
    description=json.loads(Path(path).read_text())
    if description.get('model_kind')=='native-continuous-ship':
        from continuous_ship_motion import ContinuousShipMotion
        return ContinuousShipMotion(path)
    if description.get('model_kind') in ['whole-capture-radar-point-path','whole-capture-radar-scene-path']:
        from radar_path_motion import RadarPathMotion
        return RadarPathMotion(path)
    return LocalShipMotion(path) if 'source_scene' in description else JointMotion(path)
