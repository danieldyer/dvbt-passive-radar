"""Temporal model comparison must distinguish amplitude, phase and band changes."""
from types import SimpleNamespace
import unittest

import numpy as np

from stationary import Stationary
from temporal_scene import TemporalScene, block_design


def experiment(*, varying_phase=False, different_bands=False, band_magnitudes=False):
    time = np.arange(80)*.1
    rows, points = 4, 2
    t, r = np.repeat(time, rows), np.tile(np.arange(rows), len(time))
    train_time = np.arange(len(time))%2 == 0
    train = np.repeat(train_time, rows)
    gains = np.array([.8*np.exp(.3j), 1., 1.3*np.exp(-.7j)])
    phase = np.exp(1j*np.array([.4, -1.1]))
    alpha = np.array([[1., .3], [.2, 1.1], [1.3, .7], [.5, .5]])*phase
    if varying_phase:
        alpha *= np.exp(1j*np.array([[.6, -.5], [-.8, .7], [.9, 1.1], [-.6, -.7]]))
    block = np.floor((t+1e-10)/2).astype(int)
    bands, designs = [], []
    for c, gain in enumerate(gains):
        a = np.column_stack([np.exp(-.5*(r-3*p)**2)*np.exp(1j*(1.3+.6*p+.03*c)*t) for p in range(points)])
        background = np.c_[np.exp(.05j*r), np.exp(-.2j*r)*(1+.2*np.sin(.1*t))]
        amplitudes = alpha[block].copy()
        if different_bands:
            amplitudes[:, 1] *= np.exp(.7j*c)
        if band_magnitudes:
            amplitudes *= np.array([.4+.5*c, 1.5-.4*c])
        y = gain*np.sum(a*amplitudes, axis=1)+background@np.array([120+80j, -80+130j])
        nuisance = Stationary(background, train)
        field = SimpleNamespace(time=time, y=y.reshape(len(time), rows))
        bands.append(SimpleNamespace(field=field, selected=train, nuisance=nuisance, clean=nuisance.remove(y[train])))
        designs.append(a)
    return TemporalScene(bands, designs, gains, [0, 8]), phase


class TemporalSceneTests(unittest.TestCase):
    def test_block_operator_and_partition_reject_invalid_rows(self):
        a = np.array([[1+2j, 3], [4, 5j], [6, 7]])
        result = block_design(a, np.array([0, 1, 0]), 2)
        np.testing.assert_array_equal(result, [[1+2j, 3, 0, 0], [0, 0, 4, 5j], [6, 7, 0, 0]])
        with self.assertRaises(ValueError):
            block_design(a, np.array([0, 2, 0]), 2)

    def test_positive_amplitude_changes_do_not_require_phase_changes(self):
        model, phase = experiment()
        constant = model.fit(8.)
        amplitude = model.fit(2., phase_reference=phase)
        complex_response = model.fit(2.)
        self.assertGreater(constant['validation_error'], 1.)
        for result in (amplitude, complex_response):
            self.assertLess(result['validation_error'], 1e-16)
            self.assertLess(result['maximum_linear_stationarity_residual'], 1e-9)
            self.assertAlmostEqual(result['heldout_explained_energy'], result['stationary_only_validation_energy'], places=9)
        self.assertAlmostEqual(amplitude['matched_energy'], complex_response['matched_energy'], places=9)

    def test_phase_and_frequency_changes_are_not_hidden_by_amplitude_fit(self):
        model, phase = experiment(varying_phase=True)
        amplitude, complex_response = model.fit(2., phase_reference=phase), model.fit(2.)
        self.assertGreater(amplitude['validation_error'], 10.)
        self.assertLess(complex_response['validation_error'], 1e-16)
        self.assertGreater(max(c['prediction_phase_rms_degrees'] for c in amplitude['channels']), 10.)
        self.assertLess(max(c['prediction_phase_rms_degrees'] for c in complex_response['channels']), 1e-6)
        model, _ = experiment(different_bands=True)
        shared, independent = model.fit(2.), model.fit(2., independent_channels=True)
        self.assertGreater(shared['validation_error'], 10.)
        self.assertLess(independent['validation_error'], 1e-16)
        magnitude_only = model.fit(2., independent_channels=True, phase_reference=phase)
        self.assertGreater(magnitude_only['validation_error'], 10.)

    def test_frequency_dependent_strengths_preserve_shared_phases(self):
        model, phase = experiment(band_magnitudes=True)
        shared = model.fit(2., phase_reference=phase)
        separate = model.fit(2., phase_reference=phase, independent_channels=True)
        self.assertGreater(shared['validation_error'], 1.)
        self.assertLess(separate['validation_error'], 1e-16)
        for channel in separate['coefficients_real_imag']:
            a = np.array(channel)
            coefficients = a[:, :, 0]+1j*a[:, :, 1]
            np.testing.assert_allclose(coefficients/abs(coefficients), np.broadcast_to(phase, coefficients.shape), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
