import unittest

import numpy as np

from cluster_phase import PolynomialPath,SplinePath
from flexible_echo_cluster import FlexibleEchoCluster
from smooth_echo_cluster import SmoothEchoCluster,jerk_factor
from test_flexible_echo_cluster import cubic_scene


class SmoothEchoClusterTests(unittest.TestCase):
    def test_integrated_cubic_jerk_and_quadratic_nullspace(self):
        polynomial=PolynomialPath(2.,10.,3)
        factor=jerk_factor(polynomial)
        p=np.array([.4,-.2,.3])
        self.assertAlmostEqual(np.linalg.norm(factor@p)**2,(6*p[2]/polynomial.scale**3)**2*8,places=14)
        spline=SplinePath(2.,10.,1.);grid=np.linspace(2.,10.,301)
        q=np.linalg.lstsq(spline(grid),polynomial(grid)@p,rcond=None)[0]
        self.assertAlmostEqual(np.linalg.norm(jerk_factor(spline)@q)**2,np.linalg.norm(factor@p)**2,places=12)
        quadratic=np.linalg.lstsq(spline(grid),polynomial(grid)@np.r_[p[:2],0.],rcond=None)[0]
        self.assertLess(np.linalg.norm(jerk_factor(spline)@quadratic),1e-11)

    def test_missing_endpoints_do_not_create_unpenalized_spline_excursions(self):
        curve=SplinePath(0.,4.,.5);time=np.linspace(1.,3.,51)
        truth=.3*(time-2)+.07*(time-2)**2
        a=curve(time);factor=jerk_factor(curve)
        q=np.linalg.solve(a.T@a+factor.T@factor,a.T@truth)
        check=np.linspace(0.,4.,101)
        np.testing.assert_allclose(curve(check)@q,.3*(check-2)+.07*(check-2)**2,atol=2e-10,rtol=0)

    def test_shared_prior_counted_once_and_exact_residual_jacobian(self):
        original,p=cubic_scene()
        model=SmoothEchoCluster(original.bands,original.origins,1,original.curve,jerk_std=.03)
        r,j=model.evaluate(p);data,_=original.evaluate(p)
        np.testing.assert_array_equal(r[:len(data)],data)
        self.assertAlmostEqual(r@r*model.energy,data@data*model.energy+model.prior_cost(p),places=13)
        factor=jerk_factor(original.curve)/.03
        q=p[original.local_path_indices[0][1]]
        self.assertAlmostEqual(model.prior_cost(p),np.linalg.norm(factor@q)**2/2,places=13)
        again,jagain=model.evaluate(p)
        np.testing.assert_array_equal(again,r);np.testing.assert_array_equal(jagain,j)
        direction=np.cos(np.arange(model.count))*.01;step=1e-4
        finite=(model.evaluate(p+step*direction)[0]-model.evaluate(p-step*direction)[0])/(2*step)
        np.testing.assert_allclose(finite,j@direction,atol=1e-8,rtol=2e-5)


if __name__=='__main__':unittest.main()
