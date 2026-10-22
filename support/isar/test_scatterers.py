import unittest

import numpy as np

from scatterers import normal, independent, shared, prediction


class SharedScattererTests(unittest.TestCase):
    def test_full_complex_model_recovers_fixed_band_response(self):
        random = np.random.default_rng(751)
        amplitude = np.array([1+.4j, .8-.7j, -.6+.1j])
        gains = np.array([.4+.7j, -.5+.2j, 1.3-1.2j, 1., -.3-.4j])
        train, test = [], []
        for gain in gains:
            # Nonorthogonal complex PSFs must be unmixed before comparing bands.
            design = random.normal(size=(160, 3))+1j*random.normal(size=(160, 3))
            design[:, 1] += .7*design[:, 0]
            data = design@(gain*amplitude)
            train.append(normal(design[:80], data[:80]));test.append(normal(design[80:], data[80:]))
        fit = shared(train)
        np.testing.assert_allclose(fit['gains'], gains, atol=2e-6)
        np.testing.assert_allclose(fit['amplitude'], amplitude, atol=2e-6)
        result = prediction(test, fit['gains'][:, None]*fit['amplitude'])
        self.assertLess(abs(result['efficiency_loss_db']), 1e-8)
        self.assertLess(abs(result['prediction_error']), 1e-7)
        np.testing.assert_allclose(independent(train), gains[:, None]*amplitude, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
