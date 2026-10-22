import unittest
import numpy as np
from check_joint_ais_consistency import fit_lever_delay


class AisConsistencyTests(unittest.TestCase):
    def test_recovers_rigid_lever_and_delay_in_rotated_world_coordinates(self):
        c,s=np.cos(.4),np.sin(.4)
        local=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
        origin=np.array([-4e6,3e6,-3e6])
        def state(t):
            t=np.asarray(t);angle=.002*t+.000005*t*t;cosine,sine=np.cos(angle),np.sin(angle)
            body=np.zeros((len(t),3,3));body[:,0,0]=body[:,1,1]=cosine
            body[:,1,0]=sine;body[:,0,1]=-sine;body[:,2,2]=1.
            p=np.c_[4*t+.001*t*t+.000003*t**3,.02*t*t-.00002*t**3,np.zeros(len(t))]
            return dict(position_ecef_m=origin+p@local.T,body_to_ecef=local@body)
        time=np.array([20.,60.,110.,160.,230.,280.]);delay=2.35;lever=np.array([30.,-55.,0.])
        truth=state(time-delay);observed=truth['position_ecef_m']+truth['body_to_ecef']@lever
        result=fit_lever_delay(state,time,observed,local,limit_s=5.)
        self.assertAlmostEqual(result['delay_s'],delay,places=5)
        np.testing.assert_allclose(result['lever_reference_axes_m'],lever[:2],rtol=0,atol=1e-4)
        self.assertLess(result['rms_horizontal_error_m'],1e-5)
        self.assertFalse(result['delay_at_search_boundary'])


if __name__=='__main__':unittest.main()
