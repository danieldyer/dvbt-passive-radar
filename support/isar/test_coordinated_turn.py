import unittest
import numpy as np
from numpy.polynomial.legendre import leggauss
from coordinated_turn import trajectory


class CoordinatedTurnTests(unittest.TestCase):
    def test_closed_form_matches_independent_velocity_quadrature_and_midpoint_state(self):
        time=np.linspace(-15,15,91);p=np.array([20.,-30.,4.]);v=np.array([-1.8,4.4,.003])
        for a in [np.zeros(3),np.array([-.018,-.013,0.]),np.array([.002,.001,0.])]:
            value,velocity,rate=trajectory(time,p,v,a)
            gamma=np.dot(v[:2],a[:2])/np.dot(v[:2],v[:2]);u,w=leggauss(32);s=time[:,None]*(u+1)/2
            cosine,sine=np.cos(rate*s),np.sin(rate*s)
            rotated=np.stack([cosine*v[0]-sine*v[1],sine*v[0]+cosine*v[1]],axis=-1)*(1+gamma*s[:,:,None])
            integral=np.einsum('ntj,t,n->nj',rotated,w,time/2)
            np.testing.assert_allclose(value[:,:2],p[:2]+integral,rtol=2e-12,atol=2e-11)
            at,vd,_=trajectory(np.array([0.]),p,v,a)
            np.testing.assert_array_equal(at[0],p);np.testing.assert_array_equal(vd[0],v)
            h=1e-4
            plus=trajectory(time+h,p,v,a);minus=trajectory(time-h,p,v,a)
            np.testing.assert_allclose((plus[0]-minus[0])/(2*h),velocity,atol=3e-9)
            numerical=(trajectory(np.array([h]),p,v,a)[1][0]-trajectory(np.array([-h]),p,v,a)[1][0])/(2*h)
            np.testing.assert_allclose(numerical,a,atol=3e-10)

    def test_complex_step_derivatives_are_valid_at_the_straight_line_limit(self):
        time=np.linspace(-15,15,41);state=np.array([[1.,2.,3.],[-1.8,4.4,.003],[0.,0.,0.]])
        for i,j in [(0,0),(1,0),(1,1),(2,0),(2,1)]:
            imaginary=state.astype(complex);imaginary[i,j]+=1e-20j
            analytic=trajectory(time,*imaginary)[0].imag/1e-20
            plus,minus=state.copy(),state.copy();plus[i,j]+=1e-6;minus[i,j]-=1e-6
            finite=(trajectory(time,*plus)[0]-trajectory(time,*minus)[0])/2e-6
            np.testing.assert_allclose(analytic,finite,atol=2e-7,rtol=2e-7)


if __name__=='__main__':unittest.main()
