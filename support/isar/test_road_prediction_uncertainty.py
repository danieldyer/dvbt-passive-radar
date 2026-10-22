import unittest

import numpy as np

from road_prediction_uncertainty import prediction_noise_cost
import test_stationary_road_control as stationary_tests


class PredictionUncertaintyTests(unittest.TestCase):
    def test_covariance_trace_agrees_with_independent_dense_linear_estimator(self):
        model, truth = stationary_tests.StationaryRoadTests().fixture()
        train, test = model.training[0], model.validation[0]
        antenna = np.zeros(model.antennas.count)
        u, _ = model.raw_template(truth, antenna, 0, train)
        v, _ = model.raw_template(truth, antenna, 0, test)
        expected = prediction_noise_cost(train, test, u, v)
        design = np.column_stack((u, train.stationary))
        target = np.column_stack((v, test.stationary))
        transfer = target@np.linalg.pinv(design, rcond=1e-10)
        self.assertAlmostEqual(expected['total_coefficient_prediction_variance'],
            np.linalg.norm(transfer)**2, delta=2e-12)
        rng = np.random.default_rng(927524)
        n = 8000
        noise = (rng.normal(size=(len(u), n))+1j*rng.normal(size=(len(u), n)))/np.sqrt(2)
        actual = np.mean(np.sum(abs(transfer@noise)**2, axis=0))
        self.assertLess(abs(actual/expected['total_coefficient_prediction_variance']-1), .025)


if __name__ == '__main__':
    unittest.main()
