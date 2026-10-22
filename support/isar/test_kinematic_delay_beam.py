import unittest
import numpy as np
from kinematic_delay_beam import track


class KinematicAcquisitionTests(unittest.TestCase):
    def test_distracting_bright_cells_cannot_make_range_jump_independently_of_doppler(self):
        t=np.arange(0,30.1,3.);r=np.arange(0.,101.,2.);v=np.arange(-2.,2.01,.1)
        truth=30+.4*t+.005*t*t;velocity=.4+.01*t
        power=np.array([100*np.exp(-.5*((r[None]-x)/2)**2-.5*((v[:,None]-y)/.12)**2) for x,y in zip(truth,velocity)])
        power[4,:,40]+=150;gate=np.c_[truth-20,truth+20]
        result=track(t,r,v,power,gate,beam=1024,range_resolution_m=.5,rate_resolution_mps=.025)
        self.assertLess(result['maximum_path_rate_consistency_error_m'],1e-12)
        self.assertLess(result['maximum_interval_acceleration_mps2'],.151)
        self.assertLess(max(abs(result['path_m']-truth)),3.)
        self.assertLess(max(abs(result['rate_mps']-velocity)),.15)
if __name__=='__main__':unittest.main()
