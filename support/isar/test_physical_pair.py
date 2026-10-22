from copy import deepcopy
import unittest

import numpy as np

from amplitude_arc import AmplitudeArc
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_pair import SharedPhysicalPair
from test_common_physical_arc import fixture as single_fixture


def fixture(antenna_offsets=None):
    single,_=single_fixture();bands=single.independent.bands;g=single.geometry
    g.local_transmitters+=np.array([[0.,0.,0.],[200.,-100.,70.],[-300.,250.,-30.],[20.,5.,20.],[260.,180.,-70.]])
    primary=np.r_[np.zeros(g.count),np.linspace(-.7,.8,5)]
    if antenna_offsets is not None:primary[:18]=antenna_offsets
    secondary=np.array([[.65,-1.1+.2*ch,-.55+.05*ch,.16] for ch in range(5)])
    rng=np.random.default_rng(541)
    for b,p in zip(bands,secondary):
        other=AmplitudeArc(b.field,b.training,b.knots)
        raw,_=b.response(primary);echo,_=other.response(p)
        signal=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@np.array([.7,1.2,.4,.9,1.,.6])
        signal+=b.reduced(echo[:,:,None]*b.amplitude_basis[:,None,:])@np.array([.2,.6,.8,.3,.5,.9])
        static=(rng.normal(size=b.design.shape[1])+1j*rng.normal(size=b.design.shape[1]))*.2
        b.field.y=(signal+b.design@static).reshape(b.field.y.shape)
    bands=[PhysicalAmplitudeArc(b.field,b.training,b.knots,g,b.channel,b.midpoint) for b in bands]
    return SharedPhysicalPair(bands),np.r_[primary,secondary.ravel()]


class PhysicalPairTests(unittest.TestCase):
    def test_known_truth_prediction_and_full_shared_residual_derivative(self):
        model,truth=fixture();r,j=model.evaluate(truth)
        self.assertLess(r@r,1e-20)
        prediction=model.prediction(truth,[~b.training for b in model.bands])
        self.assertTrue(all(p['heldout_explained_fraction']>1-1e-12 for p in prediction))
        x=truth+np.sin(np.arange(len(truth))+.3)*.0002
        residual,jac=model.evaluate(x)
        direction=np.cos(np.arange(len(truth)))*.01
        # Smaller probes become dominated by cancellation in ECEF distances;
        # verify two step sizes in the finite-difference convergence plateau.
        for step in (1e-3,5e-4):
            finite=(model.evaluate(x+step*direction)[0]-model.evaluate(x-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),2e-5)
        # Shared physical priors occur once, never once per receiving band.
        expected=sum(pair.energy*np.dot(pair.evaluate(x[index])['residual'],pair.evaluate(x[index])['residual'])
                     for pair,index in zip(model.pairs,model.indices))
        expected+=x[:model.geometry.count]@x[:model.geometry.count]
        self.assertAlmostEqual(residual@residual,expected/model.energy,places=14)

    def test_heldout_data_cannot_affect_either_component_or_information(self):
        model,truth=fixture();before=model.evaluate(truth)[0].copy()
        information=model.information(truth)
        self.assertEqual(information['phase_rank'],5)
        self.assertEqual(information['secondary_rank'],20)
        for b in model.bands:b.field.y[~b.training]*=np.exp(.9j)
        bands=[PhysicalAmplitudeArc(deepcopy(b.field),b.training,b.knots,b.geometry,b.channel,b.midpoint) for b in model.bands]
        other=SharedPhysicalPair(bands)
        np.testing.assert_array_equal(other.evaluate(truth)[0],before)
        np.testing.assert_allclose(other.information(truth)['matrix'],information['matrix'],rtol=1e-12,atol=1e-12)
        prediction=other.prediction(truth,[~b.training for b in bands])
        self.assertTrue(all(p['heldout_error']>1 for p in prediction))

    def test_recovery_with_fixed_antennas_and_both_unknown_components(self):
        model,truth=fixture(np.sin(np.arange(18))*.01);seed=truth.copy()
        seed[18:]+=np.sin(np.arange(len(truth)-18)+.4)*.0002
        fitted,audit=model.fit(seed,max_evaluations=100)
        np.testing.assert_array_equal(fitted[:18],truth[:18])
        self.assertLess(abs(audit['objective']-truth[:18]@truth[:18]/model.energy),1e-17)
        np.testing.assert_allclose(fitted,truth,atol=2e-6,rtol=0)

    def test_data_cannot_invent_a_geographic_rotation_reference(self):
        model,truth=fixture();g=model.geometry
        p,v,a,tx,rx=g.state(truth[:g.count]);_,jac=model.evaluate(truth)
        for axis in np.eye(3):
            direction=np.zeros(model.count)
            displacement=np.r_[np.cross(axis,tx-rx).ravel(),np.zeros(3)]
            direction[:18]=np.linalg.lstsq(g.antennas.factor,displacement,rcond=1e-12)[0]
            direction[18:27]=np.r_[np.cross(axis,p-rx),np.cross(axis,v),np.cross(axis,a)]/g.nuisance_sigma[:9]
            # Rotating every antenna and target state around the receiver
            # leaves each bistatic path invariant. Only geographical priors
            # can supply that orientation information.
            change=jac[:-g.count]@direction
            self.assertLess(np.linalg.norm(change),1e-9)


if __name__=='__main__':unittest.main()
