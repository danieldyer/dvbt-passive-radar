import unittest

import numpy as np

from joint_physical_arcs import JointPhysicalArcs
from test_common_physical_arc import fixture


class JointPhysicalArcsTests(unittest.TestCase):
    def test_independent_band_phases_use_the_same_likelihood_and_shared_prior(self):
        first,p=fixture();second,common_q=fixture()
        independent=second.independent;q=second.mapping(common_q)[0]
        model=JointPhysicalArcs([first,independent])
        parameters=np.r_[p[:18],p[18:],q[18:]]
        value,gradient=model.objective(parameters)
        expected=(first.objective(p)[0]*first.energy+independent.objective(q)[0]*independent.energy-p[:18]@p[:18])/model.energy
        self.assertAlmostEqual(value,expected,places=14)
        self.assertEqual(model.count,44)
        self.assertEqual(np.count_nonzero(model.prior),38)
        parameters+=np.sin(np.arange(len(parameters)))*.0001
        residual,jacobian=model.evaluate(parameters)
        direction=np.cos(np.arange(len(parameters)))*.01;step=2e-5
        finite=(model.evaluate(parameters+step*direction)[0]-model.evaluate(parameters-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(finite-jacobian@direction)/np.linalg.norm(finite),3e-5)
        # Extra phase freedom may remove information, never add it.
        constrained=np.asarray(JointPhysicalArcs([second]).information(common_q)['matrix'])
        free=np.asarray(JointPhysicalArcs([independent]).information(q)['matrix'])
        self.assertGreater(np.linalg.eigvalsh(constrained-free).min(),-1e-7*max(np.linalg.norm(constrained),1.))

    def test_band_phase_controls_are_constant_and_cannot_change_phase_rate(self):
        common,p=fixture();model=common.independent;mapped=common.mapping(p)[0]
        changed=mapped.copy();changed[model.geometry.count+2]+=.41
        for ch,band in enumerate(model.bands):
            initial,_=band.response(mapped);after,_=band.response(changed)
            np.testing.assert_allclose(after,initial*np.exp(.41j if ch==2 else 0j),rtol=1e-12,atol=1e-12)

    def test_shared_antenna_prior_is_counted_once_and_full_gradient_matches(self):
        first,p=fixture();second,q=fixture()
        model=JointPhysicalArcs([first,second]);parameters=np.r_[p[:18],p[18:],q[18:]]
        value,gradient=model.objective(parameters)
        expected=(first.objective(p)[0]*first.independent.energy+second.objective(q)[0]*second.independent.energy-p[:18]@p[:18])/model.energy
        self.assertAlmostEqual(value,expected,places=14)
        parameters+=np.sin(np.arange(len(parameters)))*.0001
        value,gradient=model.objective(parameters)
        direction=np.cos(np.arange(len(parameters)))*.01;step=1e-5
        finite=(model.objective(parameters+step*direction)[0]-model.objective(parameters-step*direction)[0])/(2*step)
        self.assertAlmostEqual(finite,gradient@direction,delta=1e-5+abs(finite)*5e-5)
        residual,jacobian=model.evaluate(parameters)
        numeric=(model.evaluate(parameters+step*direction)[0]-model.evaluate(parameters-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(numeric-jacobian@direction)/np.linalg.norm(numeric),3e-5)
        for index in model.indices:np.testing.assert_array_equal(parameters[index][:18],parameters[:18])


if __name__=='__main__':unittest.main()
