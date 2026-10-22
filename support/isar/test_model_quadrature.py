import json
import unittest

import numpy as np

from field import Field, average
from model_quadrature import gaussian_nodes, quadrature
from test_rigid_motion import fixture


class ModelQuadratureTests(unittest.TestCase):
    def test_moments_of_nonuniform_discrete_measure(self):
        rng = np.random.default_rng(45712)
        x = np.sort(rng.uniform(-1., 1., 16))
        for count in (1, 2, 4, 8):
            for weight in (rng.random(16), rng.random(16)*(np.arange(16)%3 != 0)):
                nodes, mass = gaussian_nodes(x, weight, count)
                self.assertTrue(np.all(mass >= 0))
                self.assertGreaterEqual(nodes.min(), x.min()-1e-14)
                self.assertLessEqual(nodes.max(), x.max()+1e-14)
                for degree in range(2*count):
                    self.assertAlmostEqual(mass@nodes**degree, weight@x**degree, delta=3e-13)

    def test_erased_and_rank_deficient_bins_preserve_information(self):
        time = 151.+np.arange(16)*.001088
        for weight in (np.zeros(16), np.eye(16)[3], np.eye(16)[2]+3*np.eye(16)[12]):
            nodes, mass = gaussian_nodes(time, weight, 4)
            self.assertTrue(np.isfinite(nodes).all())
            self.assertAlmostEqual(mass.sum(), weight.sum(), delta=1e-14)
            for degree in range(8):
                self.assertAlmostEqual(mass@(nodes-151.)**degree, weight@(time-151.)**degree, delta=5e-14)

    def test_original_data_and_gap_mask_remain_identical(self):
        original = fixture()
        data = dict(original.data)
        rng = np.random.default_rng(612)
        data['value'] = rng.normal(size=data['value'].shape)+1j*rng.normal(size=data['value'].shape)
        original = Field(data, original.pulse.samples)
        for count in (2, 4):
            reduced = quadrature(original, count)
            np.testing.assert_array_equal(reduced.y, original.y)
            np.testing.assert_array_equal(reduced.root_information, original.root_information)
            self.assertTrue(np.isfinite(reduced.data['centre_ecef_m']).all())

    def test_point_template_and_geometric_derivative_converge(self):
        original = fixture()
        data = dict(original.data)
        data['value'], data['information'] = average(original.data['value'], original.data['information'], 4)
        data['time'], _ = average(original.time, original.data['information'], 4)
        data['metadata_json'] = json.dumps({**original.metadata, 'block_rows': 4*original.metadata['block_rows']})
        original = Field(data, original.pulse.samples)
        point = [110., -12., 20.]
        exact, derivative = original.point(point, derivatives=True)
        errors, derivative_errors = [], []
        for count in (1, 2, 4):
            reduced = quadrature(original, count)
            value, jacobian = reduced.point(point, derivatives=True)
            errors.append(np.linalg.norm(value-exact)/np.linalg.norm(exact))
            derivative_errors.append(np.linalg.norm(jacobian-derivative)/np.linalg.norm(derivative))
        self.assertLess(errors[1], errors[0]/100)
        self.assertLess(errors[2], 1e-7)
        self.assertLess(derivative_errors[1], derivative_errors[0]/100)
        self.assertLess(derivative_errors[2], 1e-7)


if __name__ == '__main__':
    unittest.main()
