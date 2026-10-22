import unittest
import numpy as np
from acquire_ship_delay_tracks import stationary_glrt


class StationaryGlrtTests(unittest.TestCase):
    def test_close_to_dc_echo_uses_projected_template_information(self):
        t=np.linspace(0,3,301);w=1+.1*np.sin(t);f=np.array([0.,.01,.7]);z=np.exp(2j*np.pi*.01*t)[:,None]*(1+.2j)+20j
        p=stationary_glrt(z,w,t,f)
        phase=np.exp(2j*np.pi*.01*t);s=phase-np.sum(w*phase)/w.sum();info=np.sum(w*abs(s)**2)
        self.assertEqual(p[0,0],0.);self.assertAlmostEqual(p[1,0],8192*info*abs(1+.2j)**2,places=6)
        np.testing.assert_allclose(stationary_glrt(z+1000,w,t,f),p,rtol=1e-8,atol=1e-7)


if __name__=='__main__':unittest.main()
