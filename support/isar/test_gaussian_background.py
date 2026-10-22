import unittest

import numpy as np

from amplitude_arc import AmplitudeArc
from amplitude_pair import AmplitudePair
from gaussian_background import GaussianBackground,predictive_factors,set_background
from test_amplitude_pair import fixture


class GaussianBackgroundTests(unittest.TestCase):
    def test_whitening_and_posterior_match_augmented_least_squares(self):
        rng=np.random.default_rng(9701);z=(rng.normal(size=(17,4))+1j*rng.normal(size=(17,4)))*.7
        selected=np.arange(17)%3!=1;n=sum(selected)
        operator=GaussianBackground(z,selected);root=operator.remove(np.eye(n))
        covariance=np.eye(n)+z[selected]@z[selected].conj().T
        np.testing.assert_allclose(root@covariance@root,np.eye(n),atol=4e-15)
        y=rng.normal(size=(n,3))+1j*rng.normal(size=(n,3))
        reference=np.linalg.lstsq(np.r_[z[selected],np.eye(4)],np.r_[y,np.zeros((4,3))],rcond=None)[0]
        beta=operator.solve(y);np.testing.assert_allclose(beta,reference,atol=2e-15)
        residual=y-z[selected]@beta;white=operator.remove(y)
        np.testing.assert_allclose(white.conj().T@white,residual.conj().T@residual+beta.conj().T@beta,atol=2e-14)
        empty=GaussianBackground(np.empty((17,0),complex),selected)
        np.testing.assert_array_equal(empty.remove(y),y)
        self.assertEqual(empty.solve(y).shape,(0,3))

    def test_exact_pair_derivative_under_non_idempotent_precision(self):
        old,truth=fixture();base=old.base;rng=np.random.default_rng(912)
        factors=(rng.normal(size=(base.field.y.size,4))+1j*rng.normal(size=(base.field.y.size,4)))*.15
        # Retain a nonzero misfit: it exercises the profiled brightness
        # derivative and the second application of the precision root.
        model=AmplitudePair(set_background(AmplitudeArc(base.field,base.training,base.knots),factors))
        p=truth+np.sin(np.arange(8)+.3)*.02;value=model.evaluate(p)
        direction=np.cos(np.arange(8))*.1;step=1e-5
        finite=(model.evaluate(p+step*direction)['residual']-model.evaluate(p-step*direction)['residual'])/(2*step)
        self.assertLess(np.linalg.norm(finite-value['jacobian']@direction)/np.linalg.norm(finite),2e-6)
        expected=model.base.nuisance.remove(base.field.y.ravel()[base.selected]-value['design'][base.selected]@value['amplitude'])
        np.testing.assert_allclose(value['residual']*np.sqrt(model.energy),np.r_[expected.real,expected.imag],atol=4e-13)

    def test_weighted_predictive_covariance(self):
        rng=np.random.default_rng(32);v=rng.normal(size=(9,4))+1j*rng.normal(size=(9,4));w=np.arange(1,10,dtype=float);w/=w.sum()
        mean,factor=predictive_factors(v,w)
        expected=sum(weight*np.outer(row-mean,(row-mean).conj()) for weight,row in zip(w,v))
        expected*=(1+w@w)/(1-w@w)
        np.testing.assert_allclose(factor@factor.conj().T,expected,atol=1e-15)
        np.testing.assert_allclose(mean,w@v,atol=1e-15)


if __name__=='__main__':unittest.main()
