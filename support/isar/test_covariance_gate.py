import unittest

import numpy as np

from covariance_gate import whiten_first_field
from road_gate_information import full_field
from range_fit import Field
from test_integrated_moving_road import fixture


class CovarianceGateTests(unittest.TestCase):
    def test_bandlimited_complex_signals_retain_full_information_and_white_noise(self):
        initial, _, _ = fixture()
        data = dict(initial.original.data)
        time = data['time']
        variance = np.exp(3*np.sin(2*np.pi*time*7))
        data['variance'] = np.broadcast_to(variance[:, None], data['variance'].shape).copy()
        base = Field(data, initial.original.pulse, initial.seed, bandwidth_hz=8.)
        corrected, full = whiten_first_field(base), full_field(base)
        np.testing.assert_allclose((corrected.q*variance)@corrected.q.conj().T,
                                   np.eye(len(corrected.q)), atol=2e-12)
        rng = np.random.default_rng(971823)
        harmonics = np.arange(len(base.q))-len(base.q)//2
        basis = np.exp(2j*np.pi*(time-.5)[:, None]*harmonics)
        coefficients = rng.normal(size=(len(harmonics), len(base.offsets)))+1j*rng.normal(size=(len(harmonics), len(base.offsets)))
        signal = basis@coefficients
        raw = full.apply(signal); retained = corrected.apply(signal)
        np.testing.assert_allclose(np.vdot(retained, retained), np.vdot(raw, raw), rtol=2e-13)
        legacy = base.apply(signal)
        self.assertLess(np.vdot(legacy, legacy).real, .9*np.vdot(raw, raw).real)

    def test_selection_and_observation_values_are_preserved(self):
        initial, _, _ = fixture()
        base = initial.original
        selected = np.arange(len(base.time))%3 != 0
        base = Field(base.data, base.pulse, base.seed, bandwidth_hz=3., selected=selected)
        result = whiten_first_field(base)
        np.testing.assert_array_equal(result.q[:, ~selected], 0.)
        self.assertIs(result.data, base.data)
        self.assertIs(result.whiten, base.whiten)
        self.assertEqual(result.y.shape, base.y.shape)


if __name__ == '__main__':
    unittest.main()
