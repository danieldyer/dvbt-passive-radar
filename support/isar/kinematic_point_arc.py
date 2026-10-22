"""Physical E/N motion corrections to a coarse AIS point trajectory.

The fitted point has constant height and no fitted yaw, pitch or roll. Both
horizontal components are free: AIS does not fix the tangential motion. Velocity,
acceleration and white-jerk priors standardize the spline coordinates, so the
existing geometry likelihood counts the Gaussian prior exactly once.
"""
import numpy as np
from scipy.linalg import solve_triangular

from ais_point_arc import AisPointGeometry
from cluster_phase import SplinePath
from smooth_echo_cluster import jerk_factor


class KinematicPointGeometry(AisPointGeometry):
    def __init__(self,*args,half_width_s=2.,spacing_s=1.,velocity_sigma_mps=.5,
                 acceleration_sigma_mps2=.1,jerk_std=.05,**kwargs):
        super().__init__(*args,follow_course_change=False,**kwargs)
        scales=np.array([half_width_s,spacing_s,velocity_sigma_mps,acceleration_sigma_mps2,jerk_std])
        if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Positive finite kinematic scales required')
        self.curve=SplinePath(-half_width_s,half_width_s,spacing_s)
        velocity=self.curve([0.],1)[0]/velocity_sigma_mps
        acceleration=self.curve([0.],2)[0]/acceleration_sigma_mps2
        jerk=jerk_factor(self.curve)/jerk_std
        precision=np.outer(velocity,velocity)+np.outer(acceleration,acceleration)+jerk.T@jerk
        self.correction_factor=solve_triangular(np.linalg.cholesky(precision).T,np.eye(self.curve.degree),lower=False)
        self.count=22+2*self.curve.degree
        self.lower=np.r_[self.lower,np.full(2*self.curve.degree,-3.)]
        self.upper=np.r_[self.upper,np.full(2*self.curve.degree,3.)]
        self.motion_prior=dict(velocity_sigma_mps=velocity_sigma_mps,acceleration_sigma_mps2=acceleration_sigma_mps2,
            jerk_std_mps_5_over_2=jerk_std,spacing_s=spacing_s,midpoint_position_correction_m=0.)

    def correction_coefficients(self,parameters):
        return self.correction_factor@np.asarray(parameters[22:]).reshape(2,self.curve.degree).T

    def trajectory(self,parameters,time):
        p,v,a=super().trajectory(np.asarray(parameters)[:22],time)
        coefficients=self.correction_coefficients(parameters)
        time=np.atleast_1d(time)
        output=[]
        for derivative,value in enumerate((p,v,a)):
            correction=self.curve(time,derivative)@coefficients
            output.append(value+np.column_stack([correction,np.zeros(len(time))]))
        return tuple(output)
