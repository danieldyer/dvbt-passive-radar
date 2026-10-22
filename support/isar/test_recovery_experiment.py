import unittest

import numpy as np

from motion import enu_basis
from recovery_experiment import Experiment, SCALE, thin_field
from test_rigid_motion import fixture


def problem(points=1):
    fields=[thin_field(fixture(c),4) for c in range(5)]
    rotation=enu_basis(*np.radians([-37.87,144.93])).T
    rx=fields[0].data['receiver_ecef_m'];mid=8.
    d=fields[0].data;index=np.argmin(abs(d['row_time']-mid))
    v=d['centre_velocity_ecef_mps'][index]@rotation.T
    p=(d['centre_ecef_m'][index]-rx)@rotation.T+(mid-d['row_time'][index])*v
    tx=np.array([f.data['transmitter_ecef_m'] for f in fields])
    offsets=np.array([[0.,0.,0.],[17.,-21.,4.]])[:points]
    return Experiment(fields,rotation,rx,tx,[p,v,np.zeros(3)],mid,offsets=offsets)


class RecoveryExperimentTests(unittest.TestCase):
    def test_cartesian_forward_and_symbol_quadrature_agree_with_estimator(self):
        ex=problem(2);p=np.array([.02,-.03,.04,-.02,.01,.005])
        for c in range(5):
            alpha=np.array([.8+.3j,-.2+.7j])
            expected=ex.independent_signal(p,c,alpha)
            actual=ex.templates(p,c)@alpha
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-7)

    def test_every_motion_coordinate_and_profiled_residual_jacobian(self):
        ex=problem(2);p=np.array([.02,-.03,.04,-.02,.01,.005])
        rng=np.random.default_rng(81)
        y=[ex.independent_signal(p,c,[.8+.3j,-.2+.7j])+(rng.normal(size=f.y.size)+1j*rng.normal(size=f.y.size))*.2
           for c,f in enumerate(ex.fields)]
        at=p+np.array([.003,-.002,.001,-.001,.001,.0005]);r,j,_,_=ex.profile(at,y)
        for i in range(6):
            delta=np.eye(6)[i]*2e-5
            finite=(ex.profile(at+delta,y,derivative=False)[0]-ex.profile(at-delta,y,derivative=False)[0])/4e-5
            self.assertLess(np.linalg.norm(finite-j[:,i])/np.linalg.norm(finite),3e-4)

    def test_full_motion_recovery_from_a_nontrivial_seed_and_frozen_unused_prediction(self):
        ex=problem();truth=np.array([.02,-.03,.04,-.02,.01,.005])
        y=[ex.independent_signal(truth,c,[.8+.3j]) for c in range(5)]
        fitted,pred=ex.fit(y,truth+np.array([.03,-.04,.06,-.04,.03,-.02]),90)
        self.assertLess(np.linalg.norm((fitted['parameters']-truth)*SCALE),.02)
        self.assertLess(sum(fitted['test_rss'])/sum(np.vdot(z,z).real for z in y),1e-10)
        self.assertLess(max(np.linalg.norm(z-w)/np.linalg.norm(z) for z,w in zip(y,pred)),1e-5)

    def test_excluded_future_data_cannot_change_training_fit_or_band_coefficients(self):
        base=problem()
        masks=[f.time<8. for f in base.fields]
        ex=Experiment(base.fields,base.rotation,base.receiver,base.transmitters,
                      base.state,base.midpoint,training=masks)
        truth=np.array([.02,-.03,.04,-.02,.01,.005])
        y=[ex.independent_signal(truth,c,[.8+.3j]) for c in range(5)]
        changed=[v.copy() for v in y]
        for values,mask,field in zip(changed,masks,ex.fields):
            values[np.repeat(~mask,len(field.offsets))]+=1e6*(1+2j)
        original=ex.profile(truth,y)
        perturbed=ex.profile(truth,changed)
        for first,second in zip(original[:2],perturbed[:2]):
            np.testing.assert_array_equal(first,second)
        for first,second in zip(original[2],perturbed[2]):
            np.testing.assert_array_equal(first,second)

    def test_zero_acceleration_is_a_physical_reference_and_four_coordinate_fit(self):
        ex=problem();self.assertTrue(np.all(ex.state[2]==0))
        truth=np.array([.02,-.03,.04,-.02,0.,0.])
        y=[ex.independent_signal(truth,c,[.8+.3j]) for c in range(5)]
        seed=truth+np.array([.03,-.04,.06,-.04,0.,0.])
        fit,_=ex.fit(y,seed,80,free=[0,1,2,3])
        self.assertLess(np.linalg.norm((fit['parameters']-truth)*SCALE),.02)
        np.testing.assert_array_equal(np.asarray(fit['parameters'])[4:],[0.,0.])
        ex.state[2][0]=.001
        with self.assertRaisesRegex(ValueError,'zero-acceleration reference'):
            ex.acquire(y,np.zeros(6),constant_velocity=True)


if __name__=='__main__':unittest.main()
