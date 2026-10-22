import unittest

import numpy as np

from stationary import Stationary, matched_projection


class ProjectedImageTests(unittest.TestCase):
    def test_matches_joint_complex_nuisance_likelihood(self):
        rng = np.random.default_rng(863)
        background = rng.normal(size=(130, 5))+1j*rng.normal(size=(130, 5))
        target = rng.normal(size=(130, 3))+1j*rng.normal(size=(130, 3))
        # A nearly stationary candidate must use its reduced information.
        target[:, 1] = background[:, 0]+.01*target[:, 1]
        data = background@(rng.normal(size=5)+1j*rng.normal(size=5))*100+target[:, 1]*(.7+.3j)
        nuisance = Stationary(background, np.ones(len(data), bool))
        numerator, information = matched_projection(target, data, nuisance)
        for j in range(3):
            joint = np.c_[background, target[:, j]]
            coefficient = np.linalg.lstsq(joint, data, rcond=None)[0]
            residual = data-joint@coefficient
            null = nuisance.remove(data)
            improvement = np.vdot(null, null).real-np.vdot(residual, residual).real
            self.assertAlmostEqual(abs(numerator[j])**2/information[j], improvement, delta=1e-9)
            self.assertAlmostEqual(abs(numerator[j]/information[j]-coefficient[-1]), 0., delta=1e-8)


if __name__ == '__main__':
    unittest.main()
