import unittest

import numpy as np

from antenna_motion import Antennas
from flash_geometry import Flash
from physical_amplitude_arc import PhysicalAmplitudeArc,PhysicalAmplitudeFit
from test_rigid_motion import fixture


def problem():
    f=fixture().windowed(2.,4.);midpoint=3.
    g=object.__new__(Flash)
    g.receiver=f.data['receiver_ecef_m'];g.rotation=np.eye(3)
    g.local_transmitters=np.tile(f.data['transmitter_ecef_m']-g.receiver,(5,1))
    g.antennas=Antennas(g.rotation,[0]*5,receiver_sigma=(0.,0.,7.5))
    g.free_antennas=True;g.nantenna=18
    g.nuisance_sigma=np.array([150.,150.,20.,.5,.5,.1,.1,.1,.05,1.])
    g.count=28;g.lower=np.full(g.count,-3.);g.upper=-g.lower
    g.ais_coefficients=np.polynomial.polynomial.polyfit(f.data['row_time']-midpoint,f.data['centre_ecef_m']-g.receiver,3)
    train=np.arange(len(f.time))%2==0
    knots=np.linspace(f.data['row_time'][0],f.data['row_time'][-1],6)
    f.y[:]=1
    b=PhysicalAmplitudeArc(f,train,knots,g,4,midpoint)
    truth=np.r_[np.sin(np.arange(g.count))*.01,[0.,0.,0.,0.,.7]]
    raw,_=b.response(truth)
    design=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])
    strength=np.array([.5,.7,.1,1.4,1.,.2])
    rng=np.random.default_rng(56)
    static=(rng.normal(size=b.design.shape[1])+1j*rng.normal(size=b.design.shape[1]))*.1
    f.y=(design@strength+b.design@static).reshape(f.y.shape)
    b=PhysicalAmplitudeArc(f,train,knots,g,4,midpoint)
    return b,truth


class PhysicalAmplitudeArcTests(unittest.TestCase):
    def test_response_derivatives_include_phase_gauge_and_antenna_baseline(self):
        b,truth=problem();value,jac=b.response(truth)
        direction=np.cos(np.arange(len(truth)))*.1
        step=2e-4
        finite=(b.response(truth+step*direction)[0]-b.response(truth-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),3e-5)

    def test_profiled_positive_strength_and_shared_physical_gradient(self):
        b,truth=problem()
        self.assertLess(b.evaluate(truth)[0],1e-18)
        joint=PhysicalAmplitudeFit([b]);p=truth+np.sin(np.arange(len(truth)))*.002
        value,gradient=joint.objective(p)
        direction=np.cos(np.arange(len(p)))*.1;step=2e-4
        finite=(joint.objective(p+step*direction)[0]-joint.objective(p-step*direction)[0])/(2*step)
        self.assertAlmostEqual(finite,gradient@direction,delta=2e-6+abs(finite)*3e-5)
        prediction=joint.prediction(truth,[~b.training])[0]
        self.assertGreater(prediction['test_complex_prediction_fraction'],1-1e-12)


if __name__=='__main__':unittest.main()
