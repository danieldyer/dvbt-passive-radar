"""Evaluate an explicitly fitted physical point-path model for native BP."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from joint_ship_motion import JointMotion
from rigid_patch import RigidPatch
from bistatic_path_lift import lift
from declared_ship_orientation import state as orientation_state,rotate_frame


class RadarPathMotion(JointMotion):
    def __init__(self,path):
        d=json.loads(Path(path).read_text());super().__init__(Path(d['motion']))
        self.path=Path(path);self.path_fit=d;self.point=np.asarray(d['point_body_m']);self.interval=d['interval_s']
        self.path_curve=BSpline(d['path_knots_s'],d['path_coefficients_m'],3,extrapolate=False)
        self.gains=np.asarray([complex(*z) for z in d['physical_gains_real_imag']])

    def state(self,t):
        t=np.atleast_1d(t)
        if np.any(t<self.interval[0]) or np.any(t>self.interval[1]):raise ValueError('Whole-record radio path cannot be extrapolated')
        # Evaluate the reference without recursively calling this wrapper.
        from types import SimpleNamespace
        proxy=SimpleNamespace(state=super().state,epoch=self.epoch,track=self.track)
        s=orientation_state(proxy,t,self.path_fit.get('orientation','existing'))
        offset=np.einsum('tij,j->ti',s['body_to_ecef'],self.point);offset_rate=np.einsum('tij,j->ti',s['frame_velocity'],self.point)
        anchor=s['position_ecef_m']+offset;velocity=s['velocity_ecef_mps']+offset_rate
        kin=RigidPatch(t,162.,anchor,velocity,self.local,self.tx[3],self.rx)
        physical=lift(anchor,velocity,kin.direction,kin.direction_rate,self.path_curve(t),self.path_curve(t,1),self.tx[3],self.rx)
        s=rotate_frame(s,t,self.path_fit.get('anchor_path_time_s',162.),self.path_fit.get('angular_rate_offset_degps',0.),self.path_fit.get('angular_acceleration_offset_degps2',0.))
        offset=np.einsum('tij,j->ti',s['body_to_ecef'],self.point);offset_rate=np.einsum('tij,j->ti',s['frame_velocity'],self.point)
        return dict(position_ecef_m=physical['position_ecef_m']-offset,velocity_ecef_mps=physical['velocity_ecef_mps']-offset_rate,
            body_to_ecef=s['body_to_ecef'],frame_velocity=s['frame_velocity'])
