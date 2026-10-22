import unittest

import numpy as np

from carrier_cycles import CycleSurrogate,cycle_constraints,joint_mapping
from integer_least_squares import nearest_integers
from joint_physical_pairs import JointPhysicalPairs
from test_common_physical_pair import fixture


def models(distinct=True):
    a,p=fixture();b,q=fixture(np.r_[np.sin(np.arange(10))*.03,-.42] if distinct else None)
    common=JointPhysicalPairs([a,b]);independent=JointPhysicalPairs([a.independent,b.independent])
    truth=np.r_[p[:18],p[18:],q[18:]]
    return independent,common,truth


class CarrierCycleTests(unittest.TestCase):
    def test_joint_map_integer_phase_invariance_and_constraint_derivative(self):
        independent,common,truth=models();mapped,transform=joint_mapping(common,independent,truth)
        np.testing.assert_allclose(common.evaluate(truth)[0],independent.evaluate(mapped)[0],atol=1e-13)
        np.testing.assert_allclose(common.evaluate(truth)[1],independent.evaluate(mapped)[1]@transform,atol=1e-13)
        value,derivative,indices=cycle_constraints(common,independent,mapped)
        np.testing.assert_allclose(value,0,atol=3e-12)
        integers=np.array([3,-7,11,-9,21,4,-6,8]);mapped[indices]+=2*np.pi*integers
        np.testing.assert_allclose(cycle_constraints(common,independent,mapped)[0],integers,atol=3e-12)
        np.testing.assert_allclose(common.evaluate(truth)[0],independent.evaluate(mapped)[0],atol=3e-12)
        direction=np.cos(np.arange(independent.count))*.01;step=1e-3
        finite=(cycle_constraints(common,independent,mapped+step*direction)[0]-
                cycle_constraints(common,independent,mapped-step*direction)[0])/(2*step)
        np.testing.assert_allclose(finite,derivative@direction,rtol=1e-6,atol=2e-8)

    def test_schur_metric_matches_constrained_full_quadratic(self):
        independent,common,truth=models();origin=joint_mapping(common,independent,truth)[0]
        origin+=np.sin(np.arange(independent.count))*.0001
        surrogate=CycleSurrogate(independent,common,origin)
        residual,jacobian=independent.evaluate(origin)
        for integer in (np.zeros(8),np.array([1,-1,0,2,-2,0,1,0])):
            step=surrogate.displacement(integer)
            np.testing.assert_allclose(surrogate.value+surrogate.derivative@step,integer,atol=2e-8)
            remaining=residual+jacobian@step;full=float(remaining@remaining)*independent.energy
            delta=integer-surrogate.float_cycles
            expected=surrogate.minimum_cost+delta@np.linalg.solve(surrogate.metric,delta)
            np.testing.assert_allclose(full,expected,rtol=2e-11,atol=1e-10)
            gradient=jacobian.T@remaining
            null=gradient-surrogate.derivative.T@np.linalg.lstsq(surrogate.derivative.T,gradient,rcond=None)[0]
            self.assertLess(np.linalg.norm(null),1e-8)

    def test_cycle_recovery_from_many_turns_and_bounded_projection(self):
        independent,common,truth=models(distinct=False);origin=joint_mapping(common,independent,truth)[0]
        _,_,indices=cycle_constraints(common,independent,origin)
        turns=np.array([17,-23,13,-9,31,-18,14,-26]);origin[indices]+=2*np.pi*turns
        surrogate=CycleSurrogate(independent,common,origin)
        result=nearest_integers(surrogate.float_cycles,surrogate.metric,4)
        self.assertEqual(result['candidates'][0]['integer'],turns.tolist())
        projected=surrogate.project(turns)
        p=np.asarray(projected['parameters'])
        self.assertTrue(np.all(p>=common.lower)&np.all(p<=common.upper))
        self.assertLess(common.objective(p)[0],1e-17)
        np.testing.assert_allclose(p,truth,atol=3e-6)


if __name__=='__main__':unittest.main()
