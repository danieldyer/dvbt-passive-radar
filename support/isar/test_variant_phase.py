import unittest

import numpy as np

from variant_phase import VariantPhase


class VariantPhaseTests(unittest.TestCase):
    def fixture(self, noise=0.):
        rng = np.random.default_rng(50724)
        points, variants, bands = 5, 3, 3
        designs = [rng.normal(size=(90, points*variants))+1j*rng.normal(size=(90, points*variants)) for _ in range(bands)]
        chosen = np.array([2, 0, 1, 2, 1])
        phase = rng.uniform(-np.pi, np.pi, points)
        band = np.array([.7, -.4, 0.])
        truth = rng.uniform(.7, 2., (bands, points))*np.exp(1j*(phase+band[:, None]))
        columns = np.arange(points)*variants+chosen
        observations = [a[:, columns]@c+noise*(rng.normal(size=90)+1j*rng.normal(size=90)) for a, c in zip(designs, truth)]
        return VariantPhase(designs, observations, np.ones((points, variants), bool), reference=2), chosen, truth, band

    def test_known_variants_and_band_phase_recovered_from_wrong_initial_height(self):
        model, chosen, truth, band = self.fixture()
        coefficients, result = model.fit([(np.zeros(5, int), anchor) for anchor in range(3)])
        self.assertTrue(result['best']['converged'])
        self.assertLess(result['best']['objective'], 1e-15)
        np.testing.assert_equal(result['best']['variants'], chosen)
        np.testing.assert_allclose(coefficients, truth, atol=2e-7)
        np.testing.assert_allclose(np.exp(1j*np.array(result['best']['band_phase_radians'])), np.exp(1j*band), atol=1e-7)

    def test_coordinate_updates_monotone_in_original_sample_likelihood(self):
        model, _, _, _ = self.fixture(noise=.04)
        variants = np.zeros(5, int)
        parameters = model.model(variants).initial(0)
        initial = model.model(variants).evaluate(parameters)['value']
        variants, parameters, detail = model.coordinate(variants, parameters, sweeps=8)
        coefficients = model.model(variants).evaluate(parameters)['coefficients']
        raw = sum(np.linalg.norm(y-a[:, model.columns(variants)]@c)**2
            for a, y, c in zip(model.designs, model.observations, coefficients))/model.energy
        self.assertAlmostEqual(raw, detail['objective'][-1], places=12)
        self.assertLess(raw, initial)
        self.assertTrue((np.diff(detail['objective']) <= 1e-12).all())

    def test_ineligible_variants_are_never_selected(self):
        model, chosen, _, _ = self.fixture()
        model.allowed[0, chosen[0]] = False
        coefficients, result = model.fit([(np.zeros(5, int), 0)])
        self.assertNotEqual(result['best']['variants'][0], chosen[0])
        self.assertGreater(result['best']['objective'], .05)
        with self.assertRaises(ValueError):
            model.columns(chosen)


if __name__ == '__main__':
    unittest.main()
