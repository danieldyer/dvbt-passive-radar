import unittest
from types import SimpleNamespace

import numpy as np

from report_fixed_grid_guided_return import phase_comparison


class PhaseComparisonTests(unittest.TestCase):
    def scene(self,last):
        time=np.arange(.005,last,.01);path=-.8*time+.01*time*time
        amplitude=.2+5*np.exp(-((time-last/2)/.2)**2);bands=[];records=[]
        for k in (3.7,4.1,4.7):
            bands.append(SimpleNamespace(field=SimpleNamespace(k=k)))
            records.append(dict(time_s=time,physical_path_change_m=path,
                complex_coefficient=amplitude*np.exp(-1j*k*path),conditional_information=np.full(len(time),100.)))
        return bands,records

    def test_two_second_window_has_a_magnitude_selected_interior_origin(self):
        bands,records=self.scene(2.)
        result=phase_comparison(records,bands,0.,2.)
        self.assertGreater(result['anchor_time_s'],.75)
        self.assertLess(result['anchor_time_s'],1.25)
        grid=result['time_s'];anchor=result['anchor_time_s'];expected=-.8*(grid-anchor)+.01*(grid*grid-anchor*anchor)
        for channel in result['channels']:
            value=channel['measured_path_change_m'];valid=np.isfinite(value)
            self.assertGreater(valid.sum(),.9*len(grid))
            np.testing.assert_allclose(value[valid],expected[valid],atol=4e-4)

    def test_six_second_origin_still_uses_the_original_one_second_margins(self):
        bands,records=self.scene(6.);result=phase_comparison(records,bands,0.,6.)
        time=result['time_s'];power=np.min([c['power_to_noise'] for c in result['channels']],axis=0)
        selected=np.argmax(np.where((time>=1)&(time<=5),power,-np.inf))
        self.assertEqual(result['anchor_time_s'],time[selected])


if __name__=='__main__':unittest.main()
