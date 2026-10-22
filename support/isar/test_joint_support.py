import unittest

import numpy as np

from joint_support import select
from linear_observation import LinearObservation


class JointSupportTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1532)

    def complex(self, shape):
        return self.rng.normal(size=shape)+1j*self.rng.normal(size=shape)

    @staticmethod
    def error(a, y, support):
        matrix = a[:, support]
        coefficient = np.linalg.lstsq(matrix, y, rcond=1e-10)[0]
        return float(np.linalg.norm(y-matrix@coefficient)**2)

    def test_every_step_matches_exhaustive_complex_least_squares(self):
        designs = [self.complex((40+c, 17)) for c in range(5)]
        data = [self.complex((len(a),)) for a in designs]
        support = []
        energy = sum(np.linalg.norm(y)**2 for y in data)
        for actual, detail in select(designs, data, 10):
            choices = [j for j in range(17) if j not in support]
            errors = [sum(self.error(a, y, support+[j]) for a, y in zip(designs, data)) for j in choices]
            best = int(np.argmin(errors))
            self.assertEqual(actual, support+[choices[best]])
            self.assertAlmostEqual(energy-errors[best], detail['actual_energy_reduction'], places=10)
            support, energy = actual, errors[best]

    def test_column_units_band_phase_and_qr_do_not_change_selection(self):
        designs = [self.complex((93, 23)) for _ in range(5)]
        data = [self.complex((len(a),)) for a in designs]
        expected = list(select(designs, data, 12))[-1][0]
        factors = [np.exp(self.complex((23,))) for _ in designs]
        scaled = [a*f for a, f in zip(designs, factors)]
        rotated = [y*np.exp(.5j*c) for c, y in enumerate(data)]
        self.assertEqual(list(select(scaled, rotated, 12))[-1][0], expected)
        compressed = [LinearObservation.compress(a, y) for a, y in zip(designs, data)]
        self.assertEqual(list(select([v.design for v in compressed], [v.data for v in compressed], 12))[-1][0], expected)

    def test_common_support_allows_different_complex_responses_and_dependent_bands(self):
        designs = [self.complex((180, 31)) for _ in range(5)]
        # The two target points coincide only in the first band's dictionary.
        designs[0][:, 11] = designs[0][:, 3]
        truth = [3, 11, 22]
        coefficients = self.complex((5, 3))
        data = [a[:, truth]@b+1e-4*self.complex((len(a),)) for a, b in zip(designs, coefficients)]
        steps = list(select(designs, data, 3))
        self.assertEqual(set(steps[-1][0]), set(truth))
        self.assertEqual(steps[-1][1]['band_ranks'], [2, 3, 3, 3, 3])


if __name__ == '__main__':
    unittest.main()
