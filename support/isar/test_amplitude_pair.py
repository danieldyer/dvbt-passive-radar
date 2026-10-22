import unittest

import numpy as np

from amplitude_arc import AmplitudeArc
from amplitude_pair import AmplitudePair
from test_amplitude_arc import problem


def fixture():
    base,_=problem();truth=np.array([-.2,.4,.25,-.08,.5,-.9,-.5,.13])
    strengths=np.array([[.2,1.4,.1,.8,.3],[.5,.2,1.1,.4,.7]])
    signal=np.zeros(base.field.y.size,complex)
    for p,a in zip(truth.reshape(2,4),strengths):
        raw,_=base.response(p)
        signal+=base.reduced(raw[:,:,None]*base.amplitude_basis[:,None,:])@a
    rng=np.random.default_rng(4012)
    static=(rng.normal(size=base.design.shape[1])+1j*rng.normal(size=base.design.shape[1]))*.2
    base.field.y=(signal+base.design@static).reshape(base.field.y.shape)
    return AmplitudePair(AmplitudeArc(base.field,base.training,base.knots)),truth


class AmplitudePairTests(unittest.TestCase):
    def test_exact_residual_derivative_and_identifiable_two_echo_recovery(self):
        model,truth=fixture();x=truth+np.sin(np.arange(8)+.3)*.02
        value=model.evaluate(x);direction=np.cos(np.arange(8))*.1;step=2e-5
        finite=(model.evaluate(x+step*direction)['residual']-model.evaluate(x-step*direction)['residual'])/(2*step)
        self.assertLess(np.linalg.norm(finite-value['jacobian']@direction)/np.linalg.norm(finite),1e-6)
        fitted=model.fit(x,(-1,1))
        np.testing.assert_allclose(fitted['parameters'],truth,atol=2e-7,rtol=0)
        self.assertEqual(fitted['information_rank'],8)
        self.assertLess(fitted['objective'],1e-18)
        prediction=model.prediction(np.asarray(fitted['parameters']),~model.base.training)
        self.assertGreater(prediction['heldout_explained_fraction'],1-1e-12)

    def test_frozen_prediction_and_exchange_symmetry(self):
        model,truth=fixture();x=truth+np.sin(np.arange(8))*.003
        a=model.evaluate(x);b=model.evaluate(np.r_[x[4:],x[:4]])
        np.testing.assert_allclose(a['residual'],b['residual'],atol=2e-13,rtol=0)
        original=model.prediction(truth,~model.base.training)
        model.base.field.y[~model.base.training]*=np.exp(.7j)
        other=AmplitudePair(AmplitudeArc(model.base.field,model.base.training,model.base.knots))
        np.testing.assert_array_equal(a['residual'],other.evaluate(x)['residual'])
        self.assertGreater(other.prediction(truth,~model.base.training)['heldout_error'],original['heldout_error']+1)


if __name__=='__main__':unittest.main()
