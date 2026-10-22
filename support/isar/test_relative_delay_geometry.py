import unittest
import numpy as np

from relative_delay_geometry import relative_path,relative_delay_bound,relative_rate_bound,correlated_pair_bounds
from antenna_motion import Antennas
from flash_geometry import observables


class RelativeDelayGeometryTests(unittest.TestCase):
    def test_correlated_pair_rate_and_acceleration_bounds_enclose_exact_kinematics(self):
        rng=np.random.default_rng(73491)
        centre=np.array([100.,-500.,20.]);rotation=np.eye(3)
        tx=np.array([[30000.,5000.,700.],[30000.,5000.,700.],[30200.,4800.,760.]])
        factor=Antennas(rotation,['a','a','b']).factor[:-3,:-3]
        bound=correlated_pair_bounds(centre,tx,rotation,factor,scene_radius_m=500.,
            maximum_speed_mps=8.,maximum_acceleration_mps2=1.)
        for _ in range(200):
            shifted=tx+(factor@rng.uniform(-3,3,9)).reshape(3,3)
            p=centre+rng.normal(size=3)*50
            v=rng.normal(size=3);v*=rng.uniform(0,8)/np.linalg.norm(v)
            a=rng.normal(size=3);a*=rng.uniform(0,1)/np.linalg.norm(a)
            for receiver in (np.zeros(3),np.array([400.,-200.,30.])):
                measured=observables(p,v,a,shifted,receiver)
                for j,column in enumerate(('maximum_rate_difference_mps','maximum_acceleration_difference_mps2'),1):
                    difference=abs(measured[:,None,j]-measured[None,:,j])
                    self.assertTrue(np.all(difference<=bound[column]+1e-12))
        # Common survey displacement cancels from the baseline uncertainty.
        self.assertLess(bound['maximum_transmitter_baseline_m'][0,1],100.)

    def test_identical_transmitters_with_shared_offsets_have_zero_phase_difference(self):
        tx=np.tile([30000.,5000.,700.],(3,1))
        factor=np.tile(np.eye(3)*100,(3,1))
        bound=correlated_pair_bounds([0.,0.,0.],tx,np.eye(3),factor,
            scene_radius_m=500.,maximum_speed_mps=8.,maximum_acceleration_mps2=1.)
        np.testing.assert_array_equal(bound['maximum_rate_difference_mps'],0.)
        np.testing.assert_array_equal(bound['maximum_acceleration_difference_mps2'],0.)

    def test_rate_bound_encloses_exact_geometry_with_uncertain_transmitters(self):
        rng=np.random.default_rng(8401);centre=np.array([0.,0.,0.]);sites=np.array([[30000.,5000.,700.],[30200.,4800.,760.]])
        bound=relative_rate_bound(centre,sites,scene_radius_m=500.,maximum_point_speed_mps=8.,phase_centre_radius_m=300.)
        receiver=np.array([10000.,3000.,20.])
        for _ in range(100):
            p=rng.normal(size=3);p*=500*rng.random()/np.linalg.norm(p)
            delta=rng.normal(size=(2,3));delta*=300*rng.random((2,1))/np.linalg.norm(delta,axis=1)[:,None]
            tx=sites+delta;v=rng.normal(size=3);v*=8*rng.random()/np.linalg.norm(v)
            directions=(p-tx)/np.linalg.norm(p-tx,axis=1)[:,None]
            rate=(directions+(p-receiver)/np.linalg.norm(p-receiver))@v
            self.assertLessEqual(abs(rate[0]-rate[1]),bound['maximum_rate_difference_mps'])
    def test_receiver_and_common_per_band_delay_cancel_in_the_double_difference(self):
        p=np.array([5000.,8000.,20.]);q=p+np.array([80.,-160.,12.])
        tx=np.array([[30000.,20000.,500.],[30300.,20150.,650.]])
        differences=[]
        for rx in (np.zeros(3),np.array([100.,-200.,50.])):
            d=relative_path(p,q,tx,rx);differences.append(d[1]-d[0])
        self.assertAlmostEqual(*differences,places=10)
        original=np.linalg.norm(p-tx,axis=1)-np.linalg.norm(q-tx,axis=1)
        np.testing.assert_allclose(np.diff(original),np.diff(d),atol=1e-10)

    def test_random_bounded_points_and_antenna_offsets_respect_hessian_bound(self):
        rng=np.random.default_rng(923);centre=np.array([5000.,8000.,20.])
        sites=np.array([[30000.,20000.,500.],[30300.,20150.,650.],[29800.,20050.,550.]])
        bound=relative_delay_bound(centre,sites,scene_radius_m=500.,point_separation_m=500.,phase_centre_radius_m=100.)
        for _ in range(200):
            offset=rng.normal(size=3);offset*=rng.uniform(0,200)/np.linalg.norm(offset);p=centre+offset
            displacement=rng.normal(size=3);displacement*=rng.uniform(0,300)/np.linalg.norm(displacement);q=p+displacement
            antenna=rng.normal(size=sites.shape);antenna*=rng.uniform(0,100,(3,1))/np.linalg.norm(antenna,axis=1)[:,None]
            d=relative_path(p,q,sites+antenna,np.zeros(3))
            self.assertLessEqual(np.ptp(d),bound['maximum_range_difference_m']+1e-9)

    def test_coincident_transmitters_give_exactly_zero_difference(self):
        tx=np.tile([30000.,20000.,600.],(5,1))
        bound=relative_delay_bound([5000.,8000.,20.],tx,scene_radius_m=500.,point_separation_m=500.)
        self.assertEqual(bound['maximum_range_difference_m'],0.)


if __name__=='__main__':unittest.main()
