import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]/'calibration'))
from range_fit import Field, Pulse
from integrated_moving_road import IntegratedMovingField
from integrated_stationary_road import RecordedBins


def fixture():
    rng = np.random.default_rng(849203)
    time = (np.arange(1000)+.37)*.001
    time = time[rng.random(len(time)) > .09]
    weights = rng.uniform(.3, 2., len(time))
    index = np.floor(time/.02).astype(int)
    mean = np.bincount(index, weights=weights*time)/np.bincount(index, weights=weights)
    seed = {'epoch': .5, 'duration_s': 1., 'scale': 1., 'path': [1000., 30., 2.], 'rf_hz': 219.5e6}
    offsets = np.arange(-2, 3)
    base = Field({'time': mean, 'offset_samples': offsets,
                  'value': np.zeros((len(mean), 5), complex), 'variance': np.ones((len(mean), 5))},
                 Pulse(np.sinc(.8*np.arange(-64, 65))), seed, bandwidth_hz=8.)
    bins = RecordedBins(base, time, weights)
    return IntegratedMovingField(base, bins), weights, index


class IntegratedMovingTests(unittest.TestCase):
    def test_irregular_rows_match_independent_complex_average_and_keep_gain(self):
        field, weights, index = fixture()
        t = field.time-.5
        path = 1002.+33*t+2.7*t*t
        origin, amplitude = 1.3, .7-.4j
        result, _ = field.template(path, range_bias=origin)
        means = []
        for i in range(field.bins.count):
            choose = index == i
            seed = 1000.+30*t[choose]+2*t[choose]**2
            p = path[choose]
            row = np.exp(-1j*field.k*(p-seed+1000.))[:, None]
            row = row*field.original.pulse(field.offsets[None, :]-(p+origin-seed)[:, None]*8e6/299792458)
            means.append(np.average(row, weights=weights[choose], axis=0))
        expected = field.original.apply(np.asarray(means)).ravel()
        np.testing.assert_allclose(result, expected, atol=2e-12, rtol=2e-12)
        recovered = np.vdot(result, amplitude*expected)/np.vdot(result, result)
        self.assertAlmostEqual(abs(recovered-amplitude), 0., delta=2e-12)
        midpoint_path = 1002.+33*(field.original.time-.5)+2.7*(field.original.time-.5)**2
        midpoint = field.original.template(midpoint_path, range_bias=origin)[0]
        self.assertGreater(np.linalg.norm(midpoint-result)/np.linalg.norm(result), 1e-4)

    def test_physical_path_and_envelope_origin_derivatives(self):
        field, _, _ = fixture()
        t = field.time-.5
        basis = np.column_stack((np.ones(len(t)), t, t*t))
        coefficients = np.array([1002., 33., 2.7])
        _, derivative = field.template(basis@coefficients, basis, 1.3, True)
        for i in range(4):
            delta = 1e-5
            plus, minus = coefficients.copy(), coefficients.copy()
            if i < 3:
                plus[i] += delta; minus[i] -= delta
            a = field.template(basis@plus, range_bias=1.3+(delta if i == 3 else 0.))[0]
            b = field.template(basis@minus, range_bias=1.3-(delta if i == 3 else 0.))[0]
            numeric = (a-b)/(2*delta)
            np.testing.assert_allclose(derivative[:, i], numeric, atol=3e-7, rtol=3e-6)

    def test_slow_time_gate_and_measurements_are_unchanged(self):
        field, _, _ = fixture()
        self.assertIs(field.y, field.original.y)
        phase = np.exp(-1j*field.k*1000.)
        expected = field.original.apply(np.broadcast_to(
            phase*field.original.pulse(field.offsets), (field.bins.count, len(field.offsets)))).ravel()
        result, _ = field.template(field.seed_path)
        # A target following the deramp has no within-bin phase or range change.
        np.testing.assert_allclose(result, expected, atol=1e-11, rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
