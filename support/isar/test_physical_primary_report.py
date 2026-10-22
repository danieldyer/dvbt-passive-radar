import unittest

import numpy as np

from report_physical_primary_association import physical_path_change
from joint_path_information import expected_jacobian
from test_physical_mixed_nuisance import problem


class PhysicalPrimaryReportTests(unittest.TestCase):
    def test_physical_information_uses_mean_derivatives_and_preprojection_norms(self):
        model,p=problem(jerk_std=.3)
        matrix,norm=expected_jacobian(model,p)
        np.testing.assert_allclose(matrix,model.mean_derivative(p),atol=1e-12)
        self.assertTrue(np.all(np.linalg.norm(matrix,axis=0)<=norm+1e-10))
        transform=np.eye(model.count)[:,np.r_[18:27,28:model.count]]
        reduced,scale=expected_jacobian(model,p,transform)
        np.testing.assert_allclose(reduced,matrix@transform,atol=1e-12)
        np.testing.assert_allclose(scale,norm[np.r_[18:27,28:model.count]],atol=1e-12)

    def test_reported_path_increments_use_both_legs_and_preserve_motion_derivatives(self):
        model,p=problem();g=model.geometry;midpoint=model.bands[0].midpoint
        time=midpoint+np.array([-.4,0.,.3]);q=p[:g.count].copy()
        q[18:27]+=.04*np.cos(np.arange(9))
        value=physical_path_change(g,q,time,midpoint)
        point,velocity,acceleration,tx,rx=g.state(q)
        def exact_path(x):return np.linalg.norm(x-tx,axis=1)+np.linalg.norm(x-rx)-np.linalg.norm(tx-rx,axis=1)
        truth=np.array([exact_path(point+(t-midpoint)*velocity+.5*(t-midpoint)**2*acceleration)-exact_path(point) for t in time])
        np.testing.assert_allclose(value,truth,atol=2e-9)
        np.testing.assert_array_equal(value[1],0.)
        derivative=[]
        for index in range(18,27):
            z=np.asarray(q,complex);z[index]+=1e-20j
            derivative.append(physical_path_change(g,z,time,midpoint).imag/1e-20)
        direction=.01*np.sin(np.arange(9));perturb=np.zeros_like(q);perturb[18:27]=direction
        step=1e-3
        finite=(physical_path_change(g,q+step*perturb,time,midpoint)-physical_path_change(g,q-step*perturb,time,midpoint))/(2*step)
        np.testing.assert_allclose(finite,np.einsum('itc,i->tc',derivative,direction),atol=2e-7,rtol=2e-4)


if __name__=='__main__':unittest.main()
