import unittest
import numpy as np
from full_scene_information import information, coordinates, aspect_information
from full_path_scene import SceneBand, RefinedScenePath,ScenePath
from rigid_patch import profile_static
from test_full_path_point import problem


class FullSceneInformationTests(unittest.TestCase):
    def test_single_aspect_information_matches_unpartitioned_operator(self):
        bands = []
        for c in range(2):
            b = problem()
            points = np.array([[[0., 0., 0.], [29., -21., 10.]]])
            band = SceneBand(b, points, [b.t[0], np.nextafter(b.t[-1], np.inf)], 4.)
            h = band.design(np.zeros(b.b.shape[1]), c, False)
            b.field.y = np.sum(h*np.exp(1j*np.array([.2, -.7]))[None, None, :], axis=2)
            for train in [True, False]:
                mask = b.training if train else ~b.training
                b.data[train] = profile_static(b.field.y, b.field.root_information, mask, b.groups)*mask[:, None]
                b.energy[train] = np.vdot(b.data[train], b.data[train]).real
            bands.append(band)
        model = ScenePath(bands)
        x = np.r_[np.zeros(model.n), .21, .21, 0., 0., 0., -.91]
        expected = information(model, x)[0]
        actual, records = aspect_information(model, x)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-7)
        self.assertEqual(len(records), 1)

    def test_projected_real_jacobian_reproduces_original_gradient(self):
        points = np.array([[0., 0., 0.], [29., -21., 10.]])
        bands = []
        for c in range(2):
            b = problem()
            band = SceneBand(b, points)
            h = band.design(np.zeros(b.b.shape[1]), c, False)
            amp = b.amplitude@np.c_[1+.1*np.cos(b.amplitude_nodes), .8+.1*np.sin(b.amplitude_nodes)]
            b.field.y = np.sum(h*(amp*np.exp(1j*np.array([.2+c*.1, -.7+c*.1])))[:, None, :], axis=2)
            for training in [True, False]:
                mask = b.training if training else ~b.training
                data = profile_static(b.field.y, b.field.root_information, mask, b.groups)*mask[:, None]
                b.data[training] = data
                b.energy[training] = np.vdot(data, data).real
            bands.append(band)
        likelihood = RefinedScenePath(bands)
        x = np.r_[np.linspace(-.001, .001, likelihood.n), .21, .31, 0., 0., 0., -.91, .01, -.01, -.01, .01]
        _, expected = likelihood.objective(x)
        fisher, actual, records = information(likelihood, x)
        np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=2e-6)
        self.assertGreater(np.linalg.norm(fisher), 0.)
        self.assertGreaterEqual(np.linalg.eigvalsh(fisher).min(), -1e-7*np.max(abs(fisher)))
        self.assertEqual(len(records), 2)

    def test_preconditioning_preserves_all_modes_and_original_constraints(self):
        rng = np.random.default_rng(327)
        a = rng.normal(size=(15, 8))
        fisher = a.T@a
        fisher[0] = 0
        fisher[:, 0] = 0
        pivot = np.eye(8)
        pivot[3, 1] = .7
        matrix, report = coordinates(fisher, pivot, 100)
        x = rng.normal(size=8)
        q = np.linalg.solve(matrix, x)
        np.testing.assert_allclose(matrix@q, x, rtol=1e-13, atol=1e-13)
        bound = rng.normal(size=(12, 8))
        np.testing.assert_allclose((bound@matrix)@q, bound@x, rtol=1e-12, atol=1e-12)
        self.assertEqual(report['optimizer_condition_cap'], 1e4)


if __name__ == '__main__':
    unittest.main()
