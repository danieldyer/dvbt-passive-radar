import unittest

import numpy as np

from compare_ais_phase import path_geometry
from nearby_echo_control import known_increment,project_reference_change
from cluster_phase import SplinePath


class KnownIncrementGeometryTests(unittest.TestCase):
    def test_reference_change_preserves_anchored_cubic_path_and_derivative(self):
        time=np.linspace(162.,168.,3001);curve=SplinePath(162.,168.,1.)
        u=time-165.;difference=17.+.07*u+.001*u**2-.0003*u**3
        coefficient,error=project_reference_change(curve,time,difference)
        self.assertLess(error,2e-14)
        np.testing.assert_allclose(curve(time)@coefficient,difference-17.,atol=2e-14)
        np.testing.assert_allclose(curve(time,1)@coefficient,.07+.002*u-.0009*u**2,atol=3e-14)

    def test_colocated_bands_have_zero_increment_without_changing_observer_doppler(self):
        time=np.linspace(-3,3,257);position=np.c_[2000-1.1*time,-11000+4.8*time,np.full(len(time),12.)]
        velocity=np.tile([-1.1,4.8,0.],(len(time),1));tx=np.array([31000.,-2000.,740.]);rx=np.array([0.,0.,40.])
        path,rate=known_increment(position,velocity,tx,tx,rx,time,0.,1.234)
        np.testing.assert_array_equal(path,0.);np.testing.assert_array_equal(rate,-1.234)

    def test_stationary_receiver_changes_only_interband_constant_not_path_change(self):
        time=np.linspace(-3,3,257);position=np.c_[2000-1.1*time,-11000+4.8*time,np.full(len(time),12.)]
        velocity=np.tile([-1.1,4.8,0.],(len(time),1));tx=np.array([31000.,-2000.,740.])
        reference=np.array([31200.,-1700.,760.]);receiver=np.array([0.,0.,40.])
        a,da=known_increment(position,velocity,tx,reference,receiver,time,0.,.731)
        b,db=known_increment(position,velocity,tx,reference,receiver+[30.,-20.,15.],time,0.,.731)
        np.testing.assert_allclose(a,b,atol=2e-11)
        np.testing.assert_allclose(da,db,atol=2e-15)

    def test_reference_site_change_preserves_physical_carrier_and_symbol_doppler(self):
        time=np.linspace(-3,3,257);position=np.c_[2000-1.1*time,-11000+4.8*time,np.full(len(time),12.)]
        velocity=np.tile([-1.1,4.8,0.],(len(time),1));rx=np.array([0.,0.,40.])
        tx=np.array([31000.,-2000.,740.]);old=np.array([31100.,-1900.,700.]);new=np.array([31200.,-1700.,760.])
        extra=.03*time**2;extra_rate=.06*time;observer=.731
        a,da=known_increment(position,velocity,tx,old,rx,time,0.,observer)
        b,db=known_increment(position,velocity,tx,new,rx,time,0.,observer)
        qa,va=path_geometry(position,velocity,old,rx);qa=qa-np.interp(0.,time,qa)+extra
        qb,vb=path_geometry(position,velocity,new,rx);qb=qb-np.interp(0.,time,qb)+extra
        np.testing.assert_allclose(a+qa,b+qb,atol=1e-11)
        for k in (3.7,4.7):
            np.testing.assert_allclose(np.exp(-1j*k*(a+qa)),np.exp(-1j*k*(b+qb)),atol=5e-11)
            original_omega=-k*.001024*observer
            np.testing.assert_allclose(original_omega-k*.001024*(da+va+extra_rate),
                original_omega-k*.001024*(db+vb+extra_rate),atol=1e-16)


if __name__=='__main__':unittest.main()
