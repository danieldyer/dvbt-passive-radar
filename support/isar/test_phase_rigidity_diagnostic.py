import unittest
import numpy as np
from phase_rigidity_diagnostic import velocity_modes


class TestVelocityModes(unittest.TestCase):
    def test_path_and_spin_units_and_unresolved_mode(self):
        direction=np.linspace(.99,1.01,17);lever=np.linspace(-180.,240.,17)
        truth=np.array([.37,np.radians(.04)]);value=np.c_[direction,lever]@truth
        fitted=velocity_modes(direction,lever,value,np.linspace(1,3,17))
        np.testing.assert_allclose([fitted['path_rate_mps'],fitted['angular_rate_radps']],truth,atol=1e-15)
        self.assertLess(fitted['rigid_velocity_rss'],1e-28)
        self.assertGreater(fitted['common_path_rss'],.1)
        self.assertIsNone(velocity_modes(direction,direction*200,value,np.ones(17)))


if __name__=='__main__':unittest.main()
