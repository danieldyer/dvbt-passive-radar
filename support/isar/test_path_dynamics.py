import unittest
import numpy as np
from scipy.interpolate import BSpline
from spline_motion import knots
from path_dynamics import acceleration_matrix,project_initial,constraint,certificate


class PathDynamicsTests(unittest.TestCase):
    def test_exact_bounds_hold_between_knots_and_do_not_move_phase_anchor(self):
        grid=knots([3.2,601.8],3.);n=len(grid)-4;coefficient=14500+np.linspace(0,500,n)+.5*np.sin(np.arange(n)*.9)
        original=BSpline(grid,coefficient,3);projected,report=project_initial(grid,coefficient,.15,162.);curve=BSpline(grid,projected,3)
        t=np.linspace(3.2,601.8,60001);self.assertLessEqual(max(abs(curve(t,2))),.15+1e-8)
        self.assertAlmostEqual(float(curve(162.)),float(original(162.)),places=8)
        m=acceleration_matrix(grid);np.testing.assert_allclose(m@coefficient,original.derivative(2).c[:len(m)],atol=1e-11,rtol=0)
        bound=constraint(grid,projected,n+4,.15);parameters=np.zeros(n+4)
        self.assertLess(certificate(parameters,np.zeros(n+4),bound)['maximum_constraint_violation'],1e-8)

    def test_kkt_certificate_requires_correct_active_constraint_sign(self):
        from scipy.optimize import LinearConstraint
        bound=LinearConstraint(np.eye(2),[-1.,-1.],[1.,1.]);at=np.array([1.,0.])
        self.assertLess(certificate(at,np.array([-2.,0.]),bound)['maximum_kkt_residual'],1e-12)
        self.assertGreater(certificate(at,np.array([2.,0.]),bound)['maximum_kkt_residual'],1.)

if __name__=='__main__':unittest.main()
