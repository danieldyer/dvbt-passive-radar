import unittest

import numpy as np

from local_likelihood import compress


class LocalLikelihoodTests(unittest.TestCase):
    def test_profiled_quadratic_and_score_match_direct_elimination(self):
        rng=np.random.default_rng(7932);j=rng.normal(size=(113,9));r=rng.normal(size=113)
        j[:,8]=j[:,7]*3
        selected=np.array([0,2,4]);nuisance=np.delete(j,selected,axis=1)
        c=compress(r,j,selected)
        for _ in range(20):
            delta=rng.normal(size=3)
            full=r+j[:,selected]@delta
            coefficients=np.linalg.lstsq(nuisance,full,rcond=1e-10)[0]
            direct=full-nuisance@coefficients
            compact=c['offset']+c['matrix']@delta
            self.assertAlmostEqual(direct@direct,c['constant']+compact@compact,places=10)
        profiled=r-nuisance@np.linalg.lstsq(nuisance,r,rcond=1e-10)[0]
        np.testing.assert_allclose(c['matrix'].T@c['offset'],j[:,selected].T@profiled,atol=2e-13)

    def test_rank_deficiency_is_reported_without_inventing_information(self):
        t=np.linspace(-1,1,51);j=np.column_stack([np.ones(51),t,t*2]);r=np.sin(t)
        c=compress(r,j,[1,2])
        self.assertEqual(c['target_rank'],1)
        np.testing.assert_allclose(c['matrix']@np.array([2.,-1.]),0.,atol=2e-14)


if __name__=='__main__':unittest.main()
