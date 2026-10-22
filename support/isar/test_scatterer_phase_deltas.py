import unittest

import numpy as np

from measured_phase import anchored_history
from scatterer_phase_deltas import compare_pair


class ScattererPhaseDeltaTests(unittest.TestCase):
    def test_unequal_rf_and_arbitrary_constant_phases_preserve_rigid_point_difference(self):
        time=np.arange(161.)*.025;paths=[];truth=.013*time**2-.08*time
        for candidate in range(2):
            bands=[]
            for c,rf in enumerate((177.5e6,184.5e6,191.625e6,219.5e6,226.5e6)):
                k=2*np.pi*rf/299792458.
                common=-1.2*time+.03*time**2+c*.003*time
                true_path=common+candidate*truth
                old=1000-c*.025*time
                z=np.exp(-1j*k*(true_path-old)+1j*(2.1*c-1.8*candidate))
                history=anchored_history(time,z,np.full(len(time),1e-5),k,old,np.zeros(len(time)),60)
                bands.append(history['measured_path_m'])
            paths.append(bands)
        result=compare_pair(paths,time,np.ones((2,5,len(time))))
        expected=truth-truth[result['anchor_index']]
        np.testing.assert_allclose(result['pair_path_change_m'],np.broadcast_to(expected,(5,len(time))),atol=5e-13)
        np.testing.assert_allclose(result['double_difference_m'],0,atol=5e-13)

    def test_missing_phases_remain_missing_and_a_different_slope_is_never_aligned(self):
        time=np.arange(81.)*.05;paths=np.zeros((2,5,len(time)));power=np.ones_like(paths)
        paths[1,4]=.12*time;paths[0,0,:5]=np.nan;power[:,:,40]=3
        result=compare_pair(paths,time,power)
        self.assertEqual(result['anchor_index'],40)
        self.assertTrue(np.isnan(result['double_difference_m'][0,:5]).all())
        np.testing.assert_allclose(result['double_difference_m'][4],.12*(time-2),atol=1e-16)
        self.assertFalse(result['common_support'][:5].any())
        with self.assertRaises(ValueError):compare_pair(np.full_like(paths,np.nan),time,power)


if __name__=='__main__':unittest.main()
