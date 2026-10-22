import unittest
import numpy as np
from report_cv_geometry import path_derivatives,path_change,equivalent_initial_state


class CvGeometryTests(unittest.TestCase):
    def test_alternative_has_exact_range_rate_curvature_with_distinct_cv_state(self):
        p=np.array([-5300.,-6800.,-40.]);v=np.array([-2.,4.4,.003]);tx=np.array([31000.,-2700.,650.])
        g,h=path_derivatives(p,tx)
        for distance in [-500.,-100.,100.,500.]:
            q,w=equivalent_initial_state(p,v,tx,distance);gg,hh=path_derivatives(q,tx)
            self.assertLess(abs(path_change(p,q,tx)),1e-10)
            self.assertAlmostEqual(g@v,gg@w,delta=2e-14)
            self.assertAlmostEqual(v@h@v,w@hh@w,delta=2e-18)
            self.assertGreater(np.linalg.norm(q-p),abs(distance)-1e-8)
            # The CV constraint is not mistaken for a linear range history.
            self.assertGreater(abs(path_change(p+10*v,q+10*w,tx)),1e-9)

    def test_range_and_rate_differentials_at_one_epoch_have_only_two_rows(self):
        p=np.array([-5300.,-6800.,-40.]);v=np.array([-2.,4.4,.003]);tx=np.array([31000.,-2700.,650.])
        g,h=path_derivatives(p,tx)
        jac=np.vstack([np.r_[g[:2],0.,0.],np.r_[(h@v)[:2],g[:2]]])
        self.assertEqual(np.linalg.matrix_rank(jac),2)
        for c in range(4):
            step=np.eye(4)[c]*1e-3
            def observe(a):
                point=p+np.r_[a[:2],0.];velocity=v+np.r_[a[2:],0.]
                return np.array([path_change(p,point,tx),path_derivatives(point,tx)[0]@velocity])
            np.testing.assert_allclose((observe(step)-observe(-step))/.002,jac[:,c],rtol=1e-6,atol=1e-10)


if __name__=='__main__':unittest.main()
