import unittest

import numpy as np

from bounded_quadratic import bounded_step
from carrier_cycles import CycleSurrogate,joint_mapping
from test_carrier_cycles import models


class BoundedQuadraticTests(unittest.TestCase):
    def test_known_active_bound_solution(self):
        step,audit=bounded_step(np.eye(2),np.array([[1.,1.]]),np.array([1.]),np.zeros(2),
            np.array([0.,-np.inf]),np.array([.1,np.inf]),np.array([0.,1.]))
        np.testing.assert_allclose(step,[.1,.9],atol=1e-12)
        self.assertTrue(audit['success']);self.assertAlmostEqual(audit['cost'],.82,places=12)
        self.assertLess(audit['equality_error'],1e-12);self.assertLess(audit['bound_error'],1e-12)

    def test_inactive_bounds_match_exact_equality_solution_with_correlated_root(self):
        root=np.array([[2.,3.,.2],[0.,.3,.1],[0.,0.,1.2]])
        equal=np.array([[1.,-.2,.4]]);centre=np.array([.3,-.2,.6]);target=np.array([.17])
        normal=root.T@root;lift=np.linalg.solve(normal,equal.T)
        expected=centre+lift@np.linalg.solve(equal@lift,target-equal@centre)
        feasible=equal.T@np.linalg.solve(equal@equal.T,target)
        step,audit=bounded_step(root,equal,target,centre,np.full(3,-10.),np.full(3,10.),feasible)
        np.testing.assert_allclose(step,expected,atol=1e-10);self.assertTrue(audit['success'])

    def test_joint_cycles_remain_feasible_with_tight_geometry_bounds(self):
        independent,common,truth=models(distinct=False);origin=joint_mapping(common,independent,truth)[0]
        independent.lower[:18]=-1e-6;independent.upper[:18]=1e-6
        common.lower[:18]=-1e-6;common.upper[:18]=1e-6
        surrogate=CycleSurrogate(independent,common,origin)
        result=surrogate.project(np.array([1,-1,0,0,1,-1,0,0]),bounded_initializer=True)
        audit=result['bounded_linear_initializer']
        self.assertTrue(audit['success']);self.assertLess(audit['equality_error'],1e-8)
        self.assertLess(audit['bound_error'],1e-8)
        self.assertTrue(np.all(abs(np.array(result['parameters'])[:18])<=1e-6+1e-12))

    def test_whitened_nonlinear_projection_recovers_known_truth(self):
        independent,common,truth=models(distinct=False);origin=joint_mapping(common,independent,truth)[0]
        surrogate=CycleSurrogate(independent,common,origin)
        result=surrogate.project_whitened(np.zeros(8,dtype=int))
        self.assertTrue(result['projection_success'])
        self.assertLess(result['projection_equality_error'],1e-10)
        self.assertLess(result['projection_bound_error'],1e-10)
        self.assertLess(common.objective(np.array(result['parameters']))[0],1e-17)

    def test_equivalent_nonlinear_coordinate_formulations_and_feasibility(self):
        independent,common,truth=models(distinct=False);origin=joint_mapping(common,independent,truth)[0]
        surrogate=CycleSurrogate(independent,common,origin)
        # A nonzero integer branch exercises the nonlinear closure and bounds.
        integer=np.array([1,0,0,0,1,0,0,0])
        result=surrogate.project_whitened(integer)
        reference=surrogate.project(integer,bounded_initializer=True)
        # These are numerical initializers, not certified physical fits. Check
        # independent formulations and exact feasibility; preserve any solver
        # iteration-limit status rather than redefining it as convergence.
        np.testing.assert_allclose(result['projected_surrogate_cost'],reference['projected_surrogate_cost'],rtol=2e-7)
        self.assertLess(result['projection_equality_error'],1e-7)
        self.assertLess(result['projection_bound_error'],1e-7)
        p=np.asarray(result['parameters'])
        self.assertTrue(np.all(p>=common.lower)&np.all(p<=common.upper))


if __name__=='__main__':unittest.main()
