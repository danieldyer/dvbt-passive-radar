import unittest

import numpy as np

from amplitude_arc import AmplitudeArc
from amplitude_pair import AmplitudePair
from field import Field
from fit_reference_background_pairs import build_model
from report_background_extrapolation import original_values
from test_amplitude_arc import problem


def context_fixture():
    old,_=problem();data=dict(old.field.data)
    data['offset_samples']=np.arange(-8,9);data['value']=np.ones((len(old.field.time),17),complex)
    wide=Field(data,old.field.pulse.samples)
    data=dict(data);data['offset_samples']=np.arange(-2,3);data['value']=data['value'][:,6:11]
    narrow=AmplitudeArc(Field(data,old.field.pulse.samples),old.training,old.knots)
    model,_=build_model(wide,old.training,old.knots,None,'local',narrow.delays)
    return model,narrow


class BackgroundContextTests(unittest.TestCase):
    def test_context_resolves_previously_unobserved_background_modes(self):
        wide,narrow=context_fixture()
        condition=lambda n:n.singular_values[0]/n.singular_values[n.rank-1]
        self.assertGreater(condition(narrow.nuisance),1e8)
        self.assertLess(condition(wide.base.nuisance),100.)
        self.assertEqual(wide.base.nuisance.rank,len(narrow.delays))
        # The delayed-source model is unchanged: resolution comes from real
        # surrounding observations, not deleting singular vectors by fiat.
        np.testing.assert_array_equal(wide.base.delays,narrow.delays)

    def test_expanded_operator_reproduces_core_without_rewhitening_error(self):
        wide,narrow=context_fixture();f=wide.base.field;truth=np.array([.1,.6,-.3,.04])
        for model in (wide.base,narrow):
            raw,_=model.response(truth)
            signal=model.reduced(raw[:,:,None]*model.amplitude_basis[:,None,:])@np.array([.3,1.,.4,.8,.2])
            beta=(np.arange(len(narrow.delays))+1j*np.arange(len(narrow.delays))[::-1])*1e-3
            value=original_values(model.field,(signal+model.design@beta).reshape(model.field.y.shape))
            if model is wide.base:full=value
            else:np.testing.assert_allclose(full[:,6:11],value,atol=3e-15,rtol=1e-13)

    def test_two_echo_recovery_with_surrounding_background_constraints(self):
        model,_=context_fixture();b=model.base
        truth=np.array([-.2,.4,.25,-.08,.5,-.9,-.5,.13])
        strengths=np.array([[.2,1.4,.1,.8,.3],[.5,.2,1.1,.4,.7]])
        signal=np.zeros(b.field.y.size,complex)
        for p,a in zip(truth.reshape(2,4),strengths):
            raw,_=b.response(p);signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@a
        rng=np.random.default_rng(851)
        beta=(rng.normal(size=len(b.delays))+1j*rng.normal(size=len(b.delays)))*.02
        b.field.y=(signal+b.design@beta).reshape(b.field.y.shape)
        fitted,_=build_model(b.field,b.training,b.knots,None,'local',b.delays)
        fit=fitted.fit(truth+np.sin(np.arange(8))*.01,(-1.,1.))
        np.testing.assert_allclose(fit['parameters'],truth,atol=3e-7,rtol=0)
        self.assertLess(fit['objective'],1e-18)
        self.assertGreater(fitted.prediction(np.asarray(fit['parameters']),~b.training)['heldout_explained_fraction'],1-1e-12)


if __name__=='__main__':unittest.main()
