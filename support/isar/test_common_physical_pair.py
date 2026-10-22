import unittest

import numpy as np

from common_physical_pair import CommonPhysicalPair,antenna_information
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_pair import SharedPhysicalPair
from test_physical_pair import fixture as independent_fixture


def fixture(primary_state=None):
    independent,old=independent_fixture();model=CommonPhysicalPair(independent)
    truth=np.r_[old[:28],.73,old[33:]]
    if primary_state is not None:truth[18:29]=primary_state
    mapped,_=model.mapping(truth)
    rng=np.random.default_rng(1904)
    for pair,index in zip(independent.pairs,independent.indices):
        b=pair.base;raw,_=b.response(mapped[:33]);second,_=pair.secondary.response(mapped[index[-4:]])
        basis=b.amplitude_basis
        signal=b.reduced(raw[:,:,None]*basis[:,None,:])@np.array([.7,1.2,.4,.9,1.,.6])
        signal+=b.reduced(second[:,:,None]*basis[:,None,:])@np.array([.2,.6,.8,.3,.5,.9])
        static=(rng.normal(size=b.design.shape[1])+1j*rng.normal(size=b.design.shape[1]))*.2
        b.field.y=(signal+b.design@static).reshape(b.field.y.shape)
    bands=[PhysicalAmplitudeArc(b.field,b.training,b.knots,b.geometry,b.channel,b.midpoint) for b in independent.bands]
    return CommonPhysicalPair(SharedPhysicalPair(bands)),truth


class CommonPhysicalPairTests(unittest.TestCase):
    def test_physical_phase_operator_and_exact_nested_likelihood(self):
        model,truth=fixture();mapped,transform=model.mapping(truth)
        physical=model.phase_map.physical_parameters(truth[:29]);g=model.geometry
        p,v,a,tx,rx=g.state(truth[:28])
        for b in model.bands:
            dt=b.field.data['row_time']-b.midpoint
            offset=(p+dt[:,None]*v+.5*dt[:,None]**2*a-b.reference_local)@g.rotation
            expected=b.field.raw_trajectory(b.field.data['centre_ecef_m']+offset,(v+dt[:,None]*a)@g.rotation,
                transmitter=g.receiver+tx[b.channel]@g.rotation,receiver=g.receiver+rx@g.rotation,
                position_offset=offset)*np.exp(1j*physical[-1])
            np.testing.assert_allclose(b.response(mapped[:33])[0],expected,atol=1e-10,rtol=1e-10)
        residual,jac=model.evaluate(truth);independent,ij=model.independent.evaluate(mapped)
        np.testing.assert_array_equal(residual,independent)
        np.testing.assert_array_equal(jac,ij@transform)
        self.assertLess(residual@residual,1e-20)
        np.testing.assert_array_equal(mapped[33:],truth[29:])

    def test_full_residual_derivative_and_known_truth_recovery(self):
        model,truth=fixture();x=truth+np.sin(np.arange(len(truth)))*.0001
        residual,jac=model.evaluate(x);direction=np.cos(np.arange(len(x)))*.01
        for step in (1e-3,5e-4):
            finite=(model.evaluate(x+step*direction)[0]-model.evaluate(x-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),2e-5)
        seed=truth.copy();seed[18:]+=np.sin(np.arange(len(truth)-18))*.0001
        fitted,audit=model.fit(seed,free_antennas=False,max_evaluations=100)
        self.assertLess(audit['objective'],1e-17)
        np.testing.assert_allclose(fitted,truth,atol=3e-6,rtol=0)
        predictions=model.prediction(fitted,[~b.training for b in model.bands])
        self.assertTrue(all(p['heldout_explained_fraction']>1-1e-12 for p in predictions))

    def test_phase_reference_does_not_change_information_and_restriction_adds_no_fake_gauge(self):
        model,truth=fixture();mapped,_=model.mapping(truth)
        constrained=antenna_information(model,truth,[28],np.arange(29,49))
        free=antenna_information(model.independent,mapped,np.arange(28,33),np.arange(33,53))
        a,b=np.asarray(constrained['matrix']),np.asarray(free['matrix'])
        self.assertGreater(np.linalg.eigvalsh(a-b).min(),-1e-8*max(np.linalg.norm(a),1))
        alternate=CommonPhysicalPair(model.independent,reference_band=0)
        other=np.r_[alternate.phase_map.from_physical(model.phase_map.physical_parameters(truth[:29])),truth[29:]]
        np.testing.assert_allclose(alternate.mapping(other)[0],mapped,atol=1e-10,rtol=0)
        changed=np.asarray(antenna_information(alternate,other,[28],np.arange(29,49))['matrix'])
        self.assertLess(np.linalg.norm(changed-a)/max(np.linalg.norm(a),1),3e-6)
        g=model.geometry;p,v,acc,tx,rx=g.state(truth[:28]);_,jac=model.evaluate(truth)
        for axis in np.eye(3):
            direction=np.zeros(model.count)
            direction[:18]=np.linalg.lstsq(g.antennas.factor,np.r_[np.cross(axis,tx-rx).ravel(),np.zeros(3)],rcond=1e-12)[0]
            direction[18:27]=np.r_[np.cross(axis,p-rx),np.cross(axis,v),np.cross(axis,acc)]/g.nuisance_sigma[:9]
            self.assertLess(np.linalg.norm(jac[:-28]@direction),1e-9)


if __name__=='__main__':unittest.main()
