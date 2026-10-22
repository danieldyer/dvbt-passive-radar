import unittest

import numpy as np

from antenna_motion import Antennas
from stationary_sufficient_road import SufficientProjection, UnionField
from test_shared_target_geometry import road_fixture


class SufficientStationaryTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(572042)
        nt, nr, nc, nq = 23, 3, 5, 4
        complex_normal = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
        temporal = np.linalg.qr(complex_normal((nt, nq)))[0].conj().T
        dictionary = complex_normal((nt, nr, nc))
        return rng, temporal, dictionary, SufficientProjection(temporal, dictionary)

    def test_noise_covariance_and_sufficiency_equal_full_gaussian_linear_model(self):
        rng, temporal, dictionary, operator = self.fixture()
        nt, nr, nc = dictionary.shape
        transform = operator.apply(np.eye(nt*nr).reshape(nt, nr, -1))
        np.testing.assert_allclose(transform@transform.conj().T, np.eye(len(transform)), atol=3e-15)
        np.testing.assert_allclose(operator.adjoint(operator.apply(dictionary)), dictionary, atol=1e-14)
        # All moving columns lie in the original Fourier subspace, while the
        # stationary columns extend outside it. Verify the complete linear
        # estimator and its conditional Fisher matrix against dense full data.
        envelope = rng.normal(size=(len(temporal), nr, 2))+1j*rng.normal(size=(len(temporal), nr, 2))
        moving = np.einsum('tq,qrk->trk', temporal.conj().T, envelope).reshape(nt*nr, 2)
        design = np.column_stack((moving, dictionary.reshape(nt*nr, nc)))
        compressed = transform@design
        np.testing.assert_allclose(compressed.conj().T@compressed, design.conj().T@design, atol=2e-13)
        y = rng.normal(size=nt*nr)+1j*rng.normal(size=nt*nr)
        np.testing.assert_allclose(np.linalg.lstsq(compressed, transform@y, rcond=None)[0],
            np.linalg.lstsq(design, y, rcond=None)[0], atol=3e-15)
        full_covariance = np.linalg.inv(design.conj().T@design)
        compressed_covariance = np.linalg.inv(compressed.conj().T@compressed)
        np.testing.assert_allclose(full_covariance[:2, :2], compressed_covariance[:2, :2], atol=1e-15)

    def test_redundant_stationary_columns_add_no_noise_coordinates(self):
        rng, temporal, dictionary, _ = self.fixture()
        weights = rng.normal(size=(len(temporal), dictionary.shape[1], 2))
        contained = np.einsum('tq,qrk->trk', temporal.conj().T, weights)
        operator = SufficientProjection(temporal, contained)
        self.assertEqual(operator.extra.shape[1], 0)
        np.testing.assert_allclose(operator.adjoint(operator.apply(contained)), contained, atol=2e-15)

    def test_nearly_contained_clutter_keeps_the_added_noise_statistics_orthogonal(self):
        rng, temporal, dictionary, _ = self.fixture()
        weights = rng.normal(size=(len(temporal), dictionary.shape[1], dictionary.shape[2]))
        nearly_contained = np.einsum('tq,qrk->trk', temporal.conj().T, weights)+1e-8*dictionary
        operator = SufficientProjection(temporal, nearly_contained)
        nt, nr, _ = dictionary.shape
        transform = operator.apply(np.eye(nt*nr).reshape(nt, nr, -1))
        np.testing.assert_allclose(transform@transform.conj().T, np.eye(len(transform)), atol=2e-12)
        np.testing.assert_allclose(operator.adjoint(operator.apply(nearly_contained)), nearly_contained, atol=2e-12)

    def test_nearly_dependent_clutter_does_not_reintroduce_correlated_noise(self):
        _, temporal, dictionary, _ = self.fixture()
        dependent = dictionary[:, :, :1]+1e-9*dictionary
        operator = SufficientProjection(temporal, dependent, cutoff=1e-11)
        nt, nr, _ = dictionary.shape
        transform = operator.apply(np.eye(nt*nr).reshape(nt, nr, -1))
        np.testing.assert_allclose(transform@transform.conj().T, np.eye(len(transform)), atol=2e-12)
        np.testing.assert_allclose(operator.adjoint(operator.apply(dependent)), dependent, atol=2e-12)

    def test_original_noise_weights_and_physical_template_derivatives_are_preserved(self):
        road, truth = road_fixture(Antennas(np.eye(3), [0]))
        original = road.training[0]
        rng = np.random.default_rng(752512)
        binned = rng.normal(size=original.data['value'].shape+(3,))+1j*rng.normal(size=original.data['value'].shape+(3,))
        union = UnionField(original, binned)
        x = rng.normal(size=original.data['value'].shape)+1j*rng.normal(size=original.data['value'].shape)
        count = len(original.y)
        np.testing.assert_allclose(union.apply(x)[:count], original.apply(x).ravel(), atol=5e-14)
        antenna = np.zeros(road.antennas.count)
        u, du = road.template(truth, antenna, 0, union)
        before, before_derivative = road.template(truth, antenna, 0, original)
        np.testing.assert_allclose(u[:count], before, atol=5e-14)
        np.testing.assert_allclose(du[:count], before_derivative, atol=1e-10)
        for index in (0, road.nmotion-1, road.nmotion, len(du.T)-1):
            p = np.r_[truth, antenna]
            step = np.eye(len(p))[index]*2e-5
            a, b = [road.template(v[:road.nmotion], v[road.nmotion:], 0, union)[0] for v in (p+step, p-step)]
            np.testing.assert_allclose((a-b)/4e-5, du[:, index], atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
