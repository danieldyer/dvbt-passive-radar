import unittest
import numpy as np
from scipy.interpolate import BSpline
from continuous_ship_fit import MotionCoordinates
from test_continuous_ship_fit import fixture
from kinematic_prior import KinematicPrior,derivative_quadrature
from spline_motion import knots


class KinematicPriorTests(unittest.TestCase):
    def test_exact_continuous_integrals_and_knot_spacing_invariance(self):
        for spacing in [2.,3.,5.]:
            grid=knots([0.,19.],spacing);t=np.linspace(0,19,151);b=BSpline.design_matrix(t,grid,3).toarray()
            values=2+.4*t-.003*t*t+.0002*t**3
            coef=np.linalg.lstsq(b,values,rcond=None)[0]
            _,w,d=derivative_quadrature(grid,3,3)
            self.assertAlmostEqual(float(w@((d@coef)**2)),19*(6*.0002)**2,places=13)
            # A quadratic trajectory has zero jerk; constant yaw rate has
            # zero angular acceleration, regardless of its nonzero value.
            coef=np.linalg.lstsq(b,2+.4*t-.003*t*t,rcond=None)[0]
            self.assertLess(abs(d@coef).max(),2e-13)
            _,w,d=derivative_quadrature(grid,3,2)
            coef=np.linalg.lstsq(b,.3+.01*t+.0004*t*t,rcond=None)[0]
            self.assertAlmostEqual(float(w@((d@coef)**2)),19*(2*.0004)**2,places=14)

    def test_absolute_motion_gradient_hessian_and_physical_scaling(self):
        b,m=fixture();model=b.model
        p=KinematicPrior(model,acceleration_change_mps2=.002,angular_rate_change_degps=.01,timescale_s=30.)
        q=np.linspace(-.02,.03,model.n);value,g=p.evaluate(q)
        for j in range(model.n):
            # The prior is exactly quadratic: a larger central step has no
            # truncation error and avoids cancellation against large terms.
            step=np.eye(model.n)[j]*2e-3
            numeric=(p.evaluate(q+step)[0]-p.evaluate(q-step)[0])/4e-3
            self.assertLess(abs(numeric-g[j]),1e-6*max(1,abs(g[j])))
            numeric_hessian=(p.evaluate(q+step)[1]-p.evaluate(q-step)[1])/4e-3
            self.assertLess(np.linalg.norm(numeric_hessian-p.gram[:,j])/max(1,np.linalg.norm(p.gram[:,j])),2e-10)
        stronger=KinematicPrior(model,acceleration_change_mps2=.004,angular_rate_change_degps=.02,timescale_s=30.)
        self.assertAlmostEqual(stronger.evaluate(q)[0]/value,.25,places=12)
        report=p.diagnostics(q);self.assertAlmostEqual(sum(d['negative_log_prior'] for d in report['components'])/value,1.,places=13)
        self.assertAlmostEqual(report['components'][0]['density'],.002**2/30)

    def test_jerk_prior_leaves_constant_acceleration_in_the_null_space(self):
        b,m=fixture();model=b.model;t=np.linspace(*m.interval,201);basis=BSpline.design_matrix(t,model.pk,3).toarray()
        base=np.column_stack([4.+.8*t+.003*t*t,-5.+.5*t-.002*t*t,np.zeros(len(t))])
        model.pc=np.linalg.lstsq(basis,base,rcond=None)[0]
        prior=KinematicPrior(model,acceleration_change_mps2=.001)
        self.assertLess(prior.evaluate(np.zeros(model.n))[0],1e-16)
        # Adding any constant/velocity/acceleration correction must likewise
        # remain free; the prior prefers smooth motion, not zero motion.
        coefficients=np.linalg.lstsq(basis,2.+.3*t+.001*t*t,rcond=None)[0]
        q=np.zeros(model.n);q[:model.np]=coefficients/model.scale[:model.np]
        self.assertLess(prior.evaluate(q)[0],1e-16)

    def test_no_implicit_penalty_and_invalid_process_scales(self):
        b,m=fixture();prior=KinematicPrior(b.model);q=np.ones(b.model.n)
        value,g=prior.evaluate(q);self.assertEqual(value,0.);np.testing.assert_array_equal(g,0.)
        with self.assertRaises(ValueError):KinematicPrior(b.model,acceleration_change_mps2=0)
        with self.assertRaises(ValueError):KinematicPrior(b.model,angular_rate_change_degps=-1)

if __name__=='__main__':unittest.main()
