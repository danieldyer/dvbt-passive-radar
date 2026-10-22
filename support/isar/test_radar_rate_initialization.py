import unittest

import numpy as np

from full_motion import Band
from radar_rate_initialization import rate_seed
from spline_motion import SplineMotion, basis
from test_rigid_motion import fixture


def experiment():
    fields = [fixture(c) for c in range(5)]
    time = fields[0].data['row_time']
    interval = [time[0], time[-1]]
    model = {'interval_s': interval, 'path_spacing_seconds': 6., 'heading_spacing_seconds': 8.,
             'anchor_path': False}
    path = basis(time, interval, 6.)
    count = path.shape[1]+basis(time, interval, 8.).shape[1]
    u = time-np.mean(interval)
    truth = np.r_[np.linalg.lstsq(path, .6*u+.01*u*u, rcond=None)[0], np.zeros(count-path.shape[1])]
    initial = np.zeros(count)
    model['parameters'] = initial
    bands = []
    for c, field in enumerate(fields):
        d = field.data
        prior = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
            'body_to_ecef': d['body_to_ecef'],
            'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
        selected = np.floor(field.time/.5).astype(int)%2 == 0
        band = Band(field, prior, fields[3].data['transmitter_ecef_m'], d['receiver_ecef_m'],
                    [[0., 0., 20.]], model, selected)
        target = band.design(truth)[0][:, 0]*np.exp(.4j*c)
        clutter = band.nuisance.design[:, band.nuisance.design.shape[1]//2]*(25+13j)
        field.y = (target+clutter).reshape(field.y.shape)
        band.clean = band.nuisance.remove(field.y.ravel()[band.selected])
        band.energy = float(np.vdot(band.clean, band.clean).real)
        bands.append(band)
    return bands, initial, truth


class RadarRateInitializationTests(unittest.TestCase):
    def test_common_physical_rate_from_five_bands_in_strong_stationary_clutter(self):
        bands, initial, truth = experiment()
        seeded, details = rate_seed(bands, initial)
        model = bands[0].model
        time = np.linspace(*model['interval_s'], 301)
        actual = SplineMotion({**model, 'parameters': seeded})
        expected = SplineMotion({**model, 'parameters': truth})
        error = actual.components(time, 1)[:, 0]-expected.components(time, 1)[:, 0]
        self.assertLess(np.sqrt(np.mean(error**2)), .03)
        self.assertLess(abs(actual.components([np.mean(time)])[0, 0]), 1e-12)
        self.assertLessEqual(np.max(abs(actual.components(time, 2)[:, 0])), .15+1e-10)
        self.assertTrue(all(n>0 for n in details['valid_training_pairs_by_band']))

    def test_heldout_samples_do_not_change_initialization(self):
        bands, initial, _ = experiment()
        first, details = rate_seed(bands, initial)
        rng = np.random.default_rng(891)
        for band in bands:
            keep = ~band.selected_time
            band.field.y[keep] = 1e8*(rng.normal(size=band.field.y[keep].shape)+1j*rng.normal(size=band.field.y[keep].shape))
            band.clean = band.nuisance.remove(band.field.y.ravel()[band.selected])
        second, repeated = rate_seed(bands, initial)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(details, repeated)


if __name__ == '__main__':
    unittest.main()
