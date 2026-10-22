import unittest

import numpy as np

from observed_finite_scene import predict


class ObservedSceneTests(unittest.TestCase):
    def test_conditional_prediction_matches_full_augmented_gaussian_solve(self):
        rng=np.random.default_rng(912)
        def complex_normal(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
        a=complex_normal((47,3));s=complex_normal((47,4));b=complex_normal((47,6))*.3;y=complex_normal(47)
        selected=np.arange(47)%2==0;design=np.column_stack((a,s,b));training=design[selected]
        penalty=np.diag(np.r_[np.zeros(7),np.ones(6)])
        coefficients=np.linalg.solve(training.conj().T@training+penalty,training.conj().T@y[selected])
        expected=np.linalg.norm(y[~selected]-design[~selected]@coefficients)**2
        actual=predict(y,a,s,b,selected,finite_covariance=True)
        self.assertAlmostEqual(actual['prediction_error'],expected,places=10)

    def test_noiseless_complex_amplitude_and_no_stationary_basis(self):
        rng=np.random.default_rng(61);a=rng.normal(size=(37,3))+1j*rng.normal(size=(37,3))
        coefficient=np.array([2+1j,-3+.5j,1-2j]);selected=np.arange(37)%2==0
        actual=predict(a@coefficient,a,np.zeros((37,0),complex),np.zeros((37,2),complex),selected,finite_covariance=False)
        self.assertLess(actual['prediction_error'],1e-24)


if __name__=='__main__':unittest.main()
