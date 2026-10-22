import unittest

import numpy as np

from cancellation_information import linear_risk, marginal_root, rank_one_whiten


class CancellationInformationTests(unittest.TestCase):
    def test_rank_one_whitening_matches_independent_dense_covariance(self):
        rng=np.random.default_rng(319)
        q=rng.normal(size=(19,7,3))+1j*rng.normal(size=(19,7,3))
        z=rng.normal(size=(19,7))+1j*rng.normal(size=(19,7))
        white=rank_one_whiten(q,z)
        expected=np.zeros((3,3),complex)
        for k in range(7):
            covariance=np.eye(19)+np.outer(z[:,k],z[:,k].conj())
            expected+=q[:,k].conj().T@np.linalg.solve(covariance,q[:,k])
        root=marginal_root(q,z,carrier_block=2)
        np.testing.assert_allclose(root.conj().T@root,expected,rtol=2e-13,atol=2e-12)
        np.testing.assert_allclose(np.einsum('rki,rkj->ij',white.conj(),white),expected,rtol=2e-13,atol=2e-12)
        np.testing.assert_array_equal(rank_one_whiten(q,np.zeros_like(z)),q)

    def test_actual_filter_risk_and_optimal_bound(self):
        rng=np.random.default_rng(87)
        a=rng.normal(size=(31,4))+1j*rng.normal(size=(31,4))
        z=2*(rng.normal(size=(31,6))+1j*rng.normal(size=(31,6)))
        covariance=np.eye(31)+z@z.conj().T
        lower=np.linalg.cholesky(covariance)
        raw=np.linalg.qr(np.linalg.solve(lower,a),mode='r')
        self.assertAlmostEqual(linear_risk(raw,a,lower,optimum=True),1.,places=12)
        inverse=np.linalg.solve(a.conj().T@a,a.conj().T)
        truth=np.trace((raw.conj().T@raw)@(inverse@covariance@inverse.conj().T)).real/4
        self.assertAlmostEqual(linear_risk(raw,a,lower,optimum=False),truth,places=11)
        self.assertGreater(truth,1.)


if __name__=='__main__':
    unittest.main()
