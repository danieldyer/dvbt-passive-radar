import unittest
import numpy as np
from native_scene_histories import template_support
from stationary import Stationary


class NativeSceneHistoriesTests(unittest.TestCase):
    def test_known_brightness_mismatch_and_stationary_projection(self):
        time=np.arange(4.)
        unit=np.array([1.,-1.,1.,-1.],complex)[:,None]
        signal=unit*np.array([1.,1.,3.,3.])[:,None]
        static=Stationary(np.ones((4,1),complex),np.ones(4,bool))
        result=template_support(time,unit,signal,static)
        self.assertAlmostEqual(result['constant_strength_template_efficiency_after_stationary'],.8)
        self.assertAlmostEqual(result['target_information_retention_after_stationary'],1.)
        same=template_support(time,unit,unit*(2+3j),static)
        self.assertAlmostEqual(same['constant_strength_template_efficiency_after_stationary'],1.)
        shifted=template_support(time+100.,unit,signal,static)
        self.assertAlmostEqual(shifted['rectangular_moment_equivalent_aperture_s'],result['rectangular_moment_equivalent_aperture_s'])
        self.assertFalse(template_support(time,unit,signal*0.,static)['supported'])


if __name__=='__main__':unittest.main()
