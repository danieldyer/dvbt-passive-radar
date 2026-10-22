import unittest
import numpy as np
from bistatic_path_lift import lift
from motion import excess_path
from rigid_patch import RigidPatch


class PhysicalPathLiftTests(unittest.TestCase):
    def test_path_rate_and_jacobians_describe_the_same_physical_motion(self):
        tx=np.array([30000.,2000.,700.]);rx=np.array([0.,0.,20.]);t=np.linspace(-50,50,1001)
        def state(t):
            anchor=np.array([-5000.,-7000.,25.])+t[:,None]*np.array([-2.,4.,0.]);velocity=np.broadcast_to([-2.,4.,0.],anchor.shape)
            kin=RigidPatch(t,0.,anchor,velocity,np.eye(3),tx,rx)
            length,gradient=excess_path(anchor,tx,rx);rate=np.sum(gradient*velocity,axis=1)
            return anchor,velocity,kin.direction,kin.direction_rate,length+450+.2*np.sin(.1*t),rate+.02*np.cos(.1*t)
        args=state(t);actual=lift(*args,tx,rx);path,gradient=excess_path(actual['position_ecef_m'],tx,rx)
        np.testing.assert_allclose(path,args[4],atol=1e-7,rtol=0);np.testing.assert_allclose(np.sum(gradient*actual['velocity_ecef_mps'],axis=1),args[5],atol=1e-12,rtol=0)
        dt=.001;finite=(lift(*state(t+dt),tx,rx)['position_ecef_m']-lift(*state(t-dt),tx,rx)['position_ecef_m'])/(2*dt)
        np.testing.assert_allclose(finite,actual['velocity_ecef_mps'],atol=2e-8,rtol=0)
        for value,key in [(4,'position_path_jacobian'),(5,'velocity_rate_jacobian')]:
            plus=list(args);minus=list(args);plus[value]=plus[value]+.001;minus[value]=minus[value]-.001
            target='position_ecef_m' if value==4 else 'velocity_ecef_mps'
            numeric=(lift(*plus,tx,rx)[target]-lift(*minus,tx,rx)[target])/.002
            np.testing.assert_allclose(numeric,actual[key],atol=3e-8,rtol=0)
        plus=list(args);minus=list(args);plus[4]=plus[4]+.001;minus[4]=minus[4]-.001
        numeric=(lift(*plus,tx,rx)['velocity_ecef_mps']-lift(*minus,tx,rx)['velocity_ecef_mps'])/.002
        np.testing.assert_allclose(numeric,actual['velocity_path_jacobian'],atol=3e-10,rtol=0)
if __name__=='__main__':unittest.main()
