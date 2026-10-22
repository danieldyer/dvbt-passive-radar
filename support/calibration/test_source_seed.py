import unittest

import numpy as np

from phase_history import channel_seeds
from seed_retention import retention


class SharedSeedTests(unittest.TestCase):
    def test_one_selected_track_supplies_all_physical_deramps_without_duplicate_measurements(self):
        time = np.arange(20)+.5
        rate = 12.+.1*time
        track = np.c_[time, 1200+12*time+.05*time*time, rate, np.full(20,20.)]
        frequencies = [177.5e6,184.5e6,191.625e6,219.5e6,226.5e6]
        seeds = channel_seeds(track, 0., 20., 2, frequencies)
        self.assertEqual(sum(s['coarse_points'] for s in seeds), len(track))
        self.assertEqual([len(s['coarse_observations']) for s in seeds], [0,0,20,0,0])
        for seed, frequency in zip(seeds, frequencies):
            np.testing.assert_allclose(seed['path'], seeds[2]['path'], atol=0)
            self.assertEqual(seed['rf_hz'], frequency)
        seeds[0]['path'][0] += 1
        self.assertNotEqual(seeds[0]['path'][0],seeds[1]['path'][0])

    def test_wrong_deramp_can_exclude_echo_while_shared_seed_retains_it(self):
        time = np.arange(12000)*.001+.0005
        weight = np.ones(len(time))
        correct = retention(time,weight,np.zeros(len(time)),0.,12.)
        wrong = retention(time,weight,2*np.pi*10*time,0.,12.)
        self.assertAlmostEqual(correct['binning_and_gate_retained_fraction'],1.,places=11)
        self.assertLess(wrong['binning_and_gate_retained_fraction'],1e-10)
        self.assertGreater(wrong['binning_retained_fraction'],.8)


if __name__ == '__main__':
    unittest.main()
