from copy import deepcopy
import unittest

import numpy as np

from common_physical_arc import CommonPhysicalFit
from physical_amplitude_arc import PhysicalAmplitudeArc,PhysicalAmplitudeFit
from test_physical_amplitude_arc import problem


def fixture():
    original,independent_truth = problem()
    g = original.geometry;bands=[]
    for ch,rf in enumerate([177.5,184.5,191.625,219.5,226.5]):
        f=deepcopy(original.field)
        factor=rf*1e6/f.metadata['rf_hz']
        f.metadata['rf_hz']=rf*1e6;f.k*=factor;f.data['centre_omega']*=factor
        bands.append(PhysicalAmplitudeArc(f,original.training,original.knots,g,ch,original.midpoint))
    truth=np.r_[independent_truth[:g.count],.73]
    common=CommonPhysicalFit(PhysicalAmplitudeFit(bands))
    mapped,_=common.mapping(truth)
    rng=np.random.default_rng(441)
    for b in bands:
        raw,_=b.response(mapped)
        design=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])
        static=(rng.normal(size=b.design.shape[1])+1j*rng.normal(size=b.design.shape[1]))*.05
        b.field.y=(design@np.array([.5,.7,.1,1.4,1.,.2])+b.design@static).reshape(b.field.y.shape)
    bands=[PhysicalAmplitudeArc(b.field,b.training,b.knots,g,b.channel,b.midpoint) for b in bands]
    return CommonPhysicalFit(PhysicalAmplitudeFit(bands)),truth


class CommonPhysicalArcTests(unittest.TestCase):
    def test_common_phase_restores_original_physical_carrier_not_midpoint_gauge(self):
        model,truth=fixture();mapped,_=model.mapping(truth)
        g=model.geometry;p,v,a,tx,rx=g.state(truth[:-1])
        for b in model.independent.bands:
            dt=b.field.data['row_time']-b.midpoint
            local=p+dt[:,None]*v+.5*dt[:,None]**2*a
            offset=(local-b.reference_local)@g.rotation
            position=b.field.data['centre_ecef_m']+offset
            velocity=(v+dt[:,None]*a)@g.rotation
            expected=b.field.raw_trajectory(position,velocity,transmitter=g.receiver+tx[b.channel]@g.rotation,
                receiver=g.receiver+rx@g.rotation,position_offset=offset)*np.exp(1j*truth[-1])
            np.testing.assert_allclose(b.response(mapped)[0],expected,rtol=1e-11,atol=1e-11)

    def test_physical_phase_chain_rule_and_full_profile_gradient(self):
        model,truth=fixture()
        self.assertAlmostEqual(model.objective(truth)[0],truth[:-1]@truth[:-1]/model.independent.energy,places=12)
        p=truth+np.sin(np.arange(len(truth)))*.00002
        direction=np.cos(np.arange(len(p)))*.01;step=1e-5
        _,gradient=model.objective(p)
        finite=(model.objective(p+step*direction)[0]-model.objective(p-step*direction)[0])/(2*step)
        self.assertAlmostEqual(finite,gradient@direction,delta=2e-5+abs(finite)*5e-5)

    def test_phase_branches_preserve_prediction_and_are_not_test_data_driven(self):
        model,truth=fixture();mapped,_=model.mapping(truth)
        seeds=model.phase_branches(mapped,adjacent=False)
        self.assertGreaterEqual(len(seeds),2)
        for seed in seeds:
            self.assertTrue(np.isfinite(seed['parameters']).all())
        prediction=model.prediction(truth,[~b.training for b in model.independent.bands])
        self.assertTrue(all(r['test_complex_prediction_fraction']>1-1e-12 for r in prediction))
        model.independent.bands[4].field.y[~model.independent.bands[4].training]*=np.exp(1.2j)
        again=model.phase_branches(mapped,adjacent=False)
        for a,b in zip(seeds,again):np.testing.assert_array_equal(a['parameters'],b['parameters'])
        prediction=model.prediction(truth,[~b.training for b in model.independent.bands])
        self.assertLess(prediction[4]['test_complex_prediction_fraction'],.5)

    def test_fixed_antenna_phase_initialization_only_moves_declared_point_coordinates(self):
        model,truth=fixture();mapped,_=model.mapping(truth);n=model.geometry.count
        allowed=np.arange(18,21)
        seeds=model.phase_branches(mapped,adjacent=False,free_coordinates=allowed)
        fixed=np.setdiff1d(np.arange(n),allowed)
        for seed in seeds:np.testing.assert_array_equal(seed['parameters'][fixed],mapped[fixed])


if __name__=='__main__':unittest.main()
