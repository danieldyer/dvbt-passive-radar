import unittest

import numpy as np
from scipy.optimize import least_squares

from antenna_motion import Antennas
from geometry_profile import GeometryProfile
from shared_origin_geometry import OriginRoad, SharedOriginGeometry
from shared_target_geometry import SharedGeometry, complex_gain
from test_shared_target_geometry import road_fixture
import test_physical_profile as physical_tests


class SharedOriginGeometryTests(unittest.TestCase):
    def test_origin_zero_preserves_legacy_cost_and_physical_gradients(self):
        original,truth=road_fixture(Antennas(np.eye(3),[0]))
        road=OriginRoad(original);antenna=np.arange(road.antennas.count)*.001
        a=original.evaluate(truth*.7,antenna)
        b=road.evaluate(truth*.7,antenna,np.zeros(1))
        self.assertAlmostEqual(a['value'],b['value'],places=12)
        np.testing.assert_allclose(a['gradient'],b['gradient'][:-1],atol=1e-12)
        np.testing.assert_allclose(a['gauss_newton'],b['gauss_newton'][:-1,:-1],atol=1e-12)

    def test_road_derivatives_include_shared_envelope_origin(self):
        original,truth=road_fixture(Antennas(np.eye(3),[0]));road=OriginRoad(original)
        p=np.r_[truth*.7,np.arange(road.antennas.count)*.001,1.7]
        def evaluate(x):return road.evaluate(x[:road.nmotion],x[road.nmotion:-1],x[-1:])
        result=evaluate(p)
        for i in range(len(p)):
            step=np.eye(len(p))[i]*2e-5
            numeric=(evaluate(p+step)['value']-evaluate(p-step)['value'])/4e-5
            self.assertAlmostEqual(numeric,result['gradient'][i],delta=3e-5+abs(numeric)*3e-5)

    def test_joint_origin_gradient_counts_ship_and_vehicles(self):
        antennas=Antennas(np.eye(3),[0]);band,truth=physical_tests.PhysicalProfileTests().fixture()
        ship=GeometryProfile([band],truth*.3,antennas);road,_=road_fixture(antennas)
        legacy=SharedGeometry(ship,[road]);joint=SharedOriginGeometry(legacy)
        p=joint.initial.copy();p[joint.range_slice]=0
        self.assertAlmostEqual(legacy.objective(p)[0],joint.objective(p)[0],places=12)
        p[joint.range_slice]=1.1;result=joint.evaluate(p)
        independent=(ship.evaluate(p[:joint.ship_count])['value']*ship.energy+
            joint.roads[0].evaluate(p[joint.road_slices[0]],p[joint.antenna_slice],p[joint.range_slice])['value'])/joint.energy
        self.assertAlmostEqual(result['value'],independent,places=12)
        for i in [0,joint.antenna_slice.start,joint.antenna_slice.stop-1,joint.range_slice.start,len(p)-1]:
            step=np.eye(len(p))[i]*2e-5
            numeric=(joint.objective(p+step)[0]-joint.objective(p-step)[0])/4e-5
            self.assertAlmostEqual(numeric,result['gradient'][i],delta=3e-6+abs(numeric)*3e-5)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['gauss_newton']).min(),-1e-9)

    def test_known_nonzero_origin_recovery_with_independent_complex_gain(self):
        source,truth=road_fixture(Antennas(np.eye(3),[0]),noise=0)
        road=OriginRoad(source);antenna=np.zeros(road.antennas.count);origin=np.array([2.37])
        train=road.training[0];u,_=road.template(truth,antenna,origin,0,train);train.y=(.7-.4j)*u
        def residual(value,jacobian=False):
            u,du=road.template(truth,antenna,value,0,train)
            r,j,_,_=complex_gain(train.y,u,du[:,-1:])
            return np.r_[j.real,j.imag] if jacobian else np.r_[r.real,r.imag]
        fitted=least_squares(residual,np.zeros(1),jac=lambda x:residual(x,True),gtol=1e-12,xtol=1e-12,ftol=1e-12)
        self.assertTrue(fitted.success);np.testing.assert_allclose(fitted.x,origin,atol=1e-7)


if __name__=='__main__':unittest.main()
