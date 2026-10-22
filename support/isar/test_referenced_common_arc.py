import unittest

import numpy as np

from referenced_common_arc import ReferencedCommonArc
from test_common_physical_arc import fixture
from joint_physical_arcs import JointPhysicalArcs


class ReferencedCommonArcTests(unittest.TestCase):
    def test_exact_bijection_and_unchanged_complex_prediction(self):
        common,truth=fixture();referenced=ReferencedCommonArc(common.independent)
        p=truth+np.sin(np.arange(len(truth)))*.003
        chart=referenced.from_physical(p)
        np.testing.assert_allclose(referenced.physical_parameters(chart),p,atol=1e-12,rtol=1e-12)
        first=common.mapping(p)[0];second=referenced.mapping(chart)[0]
        np.testing.assert_allclose(first,second,atol=1e-12,rtol=1e-12)
        self.assertAlmostEqual(common.objective(p)[0],referenced.objective(chart)[0],places=12)
        _,transform=referenced.mapping(chart)
        np.testing.assert_allclose(transform[referenced.geometry.count+3,:-1],0.,atol=1e-12)

    def test_coordinate_jacobian_and_profile_gradient(self):
        common,truth=fixture();referenced=ReferencedCommonArc(common.independent)
        p=referenced.from_physical(truth)+np.sin(np.arange(len(truth)))*.0002
        direction=np.cos(np.arange(len(p)))*.01;step=2e-5
        value,gradient=referenced.objective(p)
        finite=(referenced.objective(p+step*direction)[0]-referenced.objective(p-step*direction)[0])/(2*step)
        self.assertAlmostEqual(finite,gradient@direction,delta=2e-5+abs(finite)*5e-5)

    def test_antenna_information_is_invariant_to_common_phase_coordinate(self):
        common,truth=fixture();referenced=ReferencedCommonArc(common.independent)
        physical=JointPhysicalArcs([common]);conditioned=JointPhysicalArcs([referenced])
        first=np.array(physical.information(truth)['matrix'])
        second=np.array(conditioned.information(referenced.from_physical(truth))['matrix'])
        self.assertLess(np.linalg.norm(first-second)/np.linalg.norm(first),1e-8)


if __name__=='__main__':unittest.main()
