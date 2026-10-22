import unittest

import numpy as np

from report_multiscatterer_phase_selections import relative_point_paths


def histories(time,values):
    return dict(time_s=time,channels=[dict(measured_path_change_m=p) for p in values])


class MultipleScattererPhaseReportTests(unittest.TestCase):
    def test_only_constants_removed_and_interband_motion_differences_preserved(self):
        time=np.linspace(163.2,164.8,33);motion=-1.1*(time-164.)+.2*(time-164.)**2
        rate=np.arange(5)*.03;offset=np.arange(5)*2.
        first=np.array([motion+r*(time-164.)+o for r,o in zip(rate,offset)])
        second=np.tile(motion,(5,1))-offset[:,None]
        result=relative_point_paths(histories(time,first),histories(time,second))
        expected=rate[:,None]*(time-result['anchor_time_s'])
        np.testing.assert_allclose(result['relative_path_change_m'],expected,atol=3e-15)

    def test_missing_band_and_phase_gaps_are_never_filled_by_other_bands(self):
        time=np.linspace(1.,3.,9);first=np.tile(time,(5,1));second=np.zeros_like(first)
        first[0]=np.nan;first[1,:2]=np.nan;first[2,-2:]=np.nan
        result=relative_point_paths(histories(time,first),histories(time,second))
        self.assertFalse(result['connected_channels'][0]);self.assertTrue(np.isnan(result['relative_path_change_m'][0]).all())
        self.assertTrue(np.isnan(result['relative_path_change_m'][1,:2]).all())
        self.assertTrue(np.isnan(result['relative_path_change_m'][2,-2:]).all())
        np.testing.assert_allclose(result['relative_path_change_m'][3],time-result['anchor_time_s'])


if __name__=='__main__':unittest.main()
