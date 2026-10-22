import itertools
import unittest

import numpy as np

from integer_least_squares import nearest_integers,SearchLimitError,reduce_basis


class IntegerLeastSquaresTests(unittest.TestCase):
    def test_matches_brute_force_with_correlations(self):
        rng=np.random.default_rng(792)
        for dimension in (2,3):
            points=np.array(list(itertools.product(range(-5,6),repeat=dimension)))
            for _ in range(8):
                a=rng.uniform(-.5,.5,dimension);b=rng.normal(size=(dimension,dimension))
                metric=b@b.T+.2*np.eye(dimension)
                delta=points-a;cost=np.einsum('ni,ij,nj->n',delta,np.linalg.inv(metric),delta)
                expected=np.sort(cost)[:17];result=nearest_integers(a,metric,17)
                np.testing.assert_allclose([r['cost'] for r in result['candidates']],expected,rtol=2e-12,atol=2e-12)
                self.assertTrue(result['complete'])

    def test_correlated_ambiguities_are_not_componentwise_rounding(self):
        a=np.array([.49,-.49]);q=np.array([[1.,.99],[.99,1.]])
        result=nearest_integers(a,q,4)
        self.assertNotEqual(result['candidates'][0]['integer'],np.rint(a).astype(int).tolist())
        rounded=a@np.linalg.solve(q,a)
        self.assertLess(result['candidates'][0]['cost'],rounded/50)

    def test_integer_shift_and_unimodular_coordinate_invariance(self):
        a=np.array([.27,-.31,.22]);q=np.array([[2.,1.,.2],[1.,1.2,.3],[.2,.3,.7]])
        transform=np.array([[1,3,-2],[0,1,4],[0,0,1]])
        inv=np.linalg.inv(transform);shift=np.array([57,-18,130])
        result=nearest_integers(a,q,21)['candidates']
        shifted=nearest_integers(a+shift,q,21)['candidates']
        changed=nearest_integers(inv@a,inv@q@inv.T,21)['candidates']
        np.testing.assert_allclose([r['cost'] for r in result],[r['cost'] for r in changed],rtol=2e-11,atol=2e-11)
        np.testing.assert_allclose([r['cost'] for r in result],[r['cost'] for r in shifted],atol=2e-12)
        for plain,other,moved in zip(result,changed,shifted):
            np.testing.assert_array_equal(plain['integer'],transform@other['integer'])
            np.testing.assert_array_equal(np.array(plain['integer'])+shift,moved['integer'])

    def test_reduction_is_unimodular_and_preserves_lattice(self):
        b=np.array([[1.,38.,163.],[0.,.1,5.],[0.,0.,.01]])
        transform,q,r,_=reduce_basis(b)
        z=np.asarray(transform,float)
        self.assertAlmostEqual(abs(np.linalg.det(z)),1.,places=8)
        np.testing.assert_allclose(b@z,q@r,atol=1e-12)

    def test_scalar_ties_validation_and_explicit_search_limit(self):
        result=nearest_integers([.5],[[2.]],4)
        np.testing.assert_allclose([r['cost'] for r in result['candidates']],[.125,.125,1.125,1.125])
        for metric in ([[0.]], [[-1.]], [[np.nan]]):
            with self.assertRaises(ValueError):nearest_integers([0.],metric)
        with self.assertRaises(ValueError):nearest_integers([0.,0.],[[1.,1.],[0.,1.]])
        with self.assertRaises(SearchLimitError):nearest_integers([.1,.2],np.eye(2),20,max_nodes=1)


if __name__=='__main__':unittest.main()
