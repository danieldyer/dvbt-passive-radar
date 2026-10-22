import unittest
from types import SimpleNamespace
import numpy as np
from spline_motion import knots,basis
from joint_ship_autofocus import physical_constraints,kkt_residual


class PhysicalConstraints(unittest.TestCase):
    def test_derivative_control_bounds_hold_between_all_knots(self):
        interval=[0.,30.];spacing=3.;cross=15.;npth=len(knots(interval,spacing))-4;nc=len(knots(interval,cross))-4
        ny=len(knots(interval,10.))-4;base=.03*np.sin(np.arange(npth))
        source=dict(interval_s=interval,path_spacing_seconds=spacing,heading_spacing_seconds=10.,parameters=np.r_[base,np.zeros(ny)].tolist())
        model=SimpleNamespace(count=npth+nc-2,offset_count=0,phase_count=0,path_count=npth-1,cross_count=nc-1,
            spacing=spacing,interval=interval,windows=[SimpleNamespace(result=dict(source_model=source),cross_spacing=cross)])
        a,lo,hi=physical_constraints(model);rng=np.random.default_rng(76);p=rng.normal(size=model.count)*.01
        self.assertTrue(np.all(a@p>=lo));self.assertTrue(np.all(a@p<=hi))
        t=np.linspace(*interval,10001)
        for order,limits in [(1,(3.,3.)),(2,(.15,.1))]:
            radial=basis(t,interval,spacing,derivative=order)@np.r_[p[:npth-1],0.]+basis(t,interval,spacing,derivative=order)@base
            transverse=basis(t,interval,cross,derivative=order)@np.r_[p[npth-1:],0.]
            self.assertLess(abs(radial).max(),limits[0]+1e-12);self.assertLess(abs(transverse).max(),limits[1]+1e-12)

    def test_active_physical_constraint_multiplier_signs(self):
        result=kkt_residual(np.zeros(2),np.array([-1.,2.]),[(-10.,10.)]*2,np.array([[1.,0.],[0.,-1.]]),np.full(2,-np.inf),np.zeros(2))
        self.assertLess(result['maximum_kkt_residual'],1e-12);self.assertEqual(result['maximum_constraint_violation'],0.)

if __name__=='__main__':unittest.main()
