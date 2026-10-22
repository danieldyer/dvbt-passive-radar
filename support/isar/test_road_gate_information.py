import unittest

import numpy as np

from road_gate_information import SelectedWhitening, full_field
from test_integrated_moving_road import fixture


class RoadGateInformationTests(unittest.TestCase):
    def test_selected_diagonal_whitening_matches_explicit_covariance(self):
        selected = np.array([True, False, True, True, False])
        variance = np.array([2., .5, 3., .7, 1.1])
        operator = SelectedWhitening(selected, variance)
        dense = np.diag(1/np.sqrt(variance))[selected]
        rng = np.random.default_rng(80184)
        value = rng.normal(size=(5, 4))+1j*rng.normal(size=(5, 4))
        np.testing.assert_allclose(operator@value, dense@value, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(dense@np.diag(variance)@dense.T, np.eye(3), atol=1e-14)
        with self.assertRaises(ValueError):
            SelectedWhitening(selected, np.zeros(5))

    def test_full_observations_project_exactly_to_existing_gate(self):
        model, _, _ = fixture()
        base = model.original
        rng = np.random.default_rng(19404)
        base.data['value'] = rng.normal(size=base.data['value'].shape)+1j*rng.normal(size=base.data['value'].shape)
        full = full_field(base)
        variance = base.data['variance'][:, len(base.offsets)//2]
        projection = base.q[:, base.selected]*np.sqrt(variance[base.selected])[None, :]
        np.testing.assert_allclose(projection@projection.conj().T, np.eye(len(projection)), atol=1e-12)
        white = full.y.reshape(len(full.q), len(base.offsets))
        np.testing.assert_allclose(projection@white, base.apply(base.data['value']), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
