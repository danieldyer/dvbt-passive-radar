import unittest

import numpy as np

from coherence import efficiency


class CoherenceTests(unittest.TestCase):
    def test_complex_gain_and_weighted_cauchy_bound(self):
        random = np.random.default_rng(81)
        template = random.normal(size=(5, 7))+1j*random.normal(size=(5, 7))
        information = np.exp(random.normal(size=(5, 7)))
        np.testing.assert_allclose(efficiency(template, (2-3j)*template, information), 1)
        changed = template*np.exp(1j*random.normal(size=(5, 7)))
        result = efficiency(template, changed, information)
        self.assertTrue(np.all((result >= 0)&(result <= 1)))

    def test_zero_signal_does_not_claim_coherence(self):
        with self.assertRaises(ValueError):
            efficiency(np.zeros((5, 1), complex), np.ones((5, 1)), np.ones((5, 1)))


if __name__ == '__main__':
    unittest.main()
