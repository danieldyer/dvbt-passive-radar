import unittest
import numpy as np
from range_nuisance import RangeNuisance


class RangeNuisanceTests(unittest.TestCase):
    def test_profiled_complex_history_equals_orthogonal_projection(self):
        rng=np.random.default_rng(14)
        design=rng.normal(size=(11,9,3))+1j*rng.normal(size=(11,9,3))
        y=rng.normal(size=(11,9,2))+1j*rng.normal(size=(11,9,2))
        metric=RangeNuisance(design)
        actual=metric.apply(y.reshape(99,2)).reshape(y.shape)
        expected=np.stack([v-a@np.linalg.lstsq(a,v,rcond=None)[0] for a,v in zip(design,y)])
        np.testing.assert_allclose(actual,expected,atol=2e-14)
        np.testing.assert_allclose(metric.apply(actual.reshape(99,2)),actual.reshape(99,2),atol=2e-14)
        np.testing.assert_allclose(metric.apply(design.reshape(99,3)),0,atol=2e-14)

    def test_complex_adjoint_and_multiple_derivative_columns(self):
        rng=np.random.default_rng(26)
        metric=RangeNuisance(rng.normal(size=(7,8,2))+1j*rng.normal(size=(7,8,2)))
        a=rng.normal(size=(56,1,6))+1j*rng.normal(size=(56,1,6))
        b=rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
        np.testing.assert_allclose(np.vdot(a,metric.apply(b)),np.vdot(metric.apply(a),b),atol=2e-13)


if __name__=='__main__':unittest.main()
