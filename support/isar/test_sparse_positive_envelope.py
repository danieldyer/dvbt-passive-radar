import unittest
import numpy as np
from scipy.sparse import csr_matrix
from audit_ship_scattering_envelopes import linear_basis,statistics
from positive_envelope_scene import solve
from sparse_positive_envelope import Normal,direct_positive_quadratic


class SparseEnvelopeTests(unittest.TestCase):
    def test_small_spd_fallback_preserves_quadratic_and_boundaries(self):
        rng=np.random.default_rng(981);q,_=np.linalg.qr(rng.normal(size=(24,24)))
        gram=(q*np.geomspace(5e-5,2.,24))@q.T
        truth=rng.uniform(0,1,24);truth[::3]=0.;gradient=np.zeros(24);gradient[::3]=rng.uniform(.02,.1,8)
        target=gram@truth-gradient
        actual=direct_positive_quadratic(lambda x:gram@x,target)
        np.testing.assert_allclose(actual,truth,atol=2e-11,rtol=0)
        residual=gram@actual-target
        self.assertLess(max(abs(np.where(actual>1e-10,residual,np.minimum(residual,0)))),1e-13)
        self.assertIsNone(direct_positive_quadratic(lambda x:np.zeros_like(x),np.zeros(5)))
        design=np.abs(rng.normal(size=(8,12)));gram=design.T@design;target=gram@np.ones(12)
        value=direct_positive_quadratic(lambda x:gram@x,target)
        np.testing.assert_allclose(design@value,design@np.ones(12),atol=1e-12,rtol=0)
        g=gram@value-target
        self.assertLess(max(abs(np.where(value>1e-10,g,np.minimum(g,0)))),1e-11)
        with self.assertRaises(ValueError):direct_positive_quadratic(lambda x:-x,np.ones(5))

    def test_exact_normal_matrix_solution_and_phase_gradient(self):
        rng=np.random.default_rng(240);t=np.linspace(0,12,321);b,_=linear_basis(t,[0,12],3.);p=3;k=b.shape[1]
        h=rng.normal(size=(len(t),5,p))+1j*rng.normal(size=(len(t),5,p));root=1+rng.uniform(size=len(t));mask=np.arange(len(t))%4!=3
        phase=np.array([.3,-.2,.7]);a=rng.uniform(.2,1,size=(k,p));a[1,1]=0
        y=np.sum(h*(b@a*np.exp(1j*phase))[:,None,:],axis=2)+root[:,None]*(.4+.1j)
        normal=Normal(h,y,root,mask,csr_matrix(b));gram,rhs,energy=statistics(h,y,root,mask,b)
        # Dense reference uses point-major ordering, sparse uses knot-major.
        order=np.arange(k*p).reshape(p,k).T.ravel();gram=gram[order][:,order];rhs=rhs[order]
        v=rng.normal(size=k*p)+1j*rng.normal(size=k*p)
        np.testing.assert_allclose(normal.multiply(v),gram@v,atol=1e-10,rtol=1e-13);np.testing.assert_allclose(normal.rhs,rhs,atol=1e-11,rtol=1e-13)
        at=phase+np.array([.03,-.02,.04]);r=normal.solve(at);reference=solve(gram,rhs,energy,np.tile(np.arange(p),k),at)
        self.assertTrue(r['numerical_convergence'],r);self.assertLess(abs(r['cost']-reference['cost'])/energy,1e-12)
        np.testing.assert_allclose(r['coefficient'],reference['coefficient'],atol=2e-7,rtol=2e-7)
        for j in range(p):
            d=np.eye(p)[j]*1e-5;finite=(normal.solve(at+d)['cost']-normal.solve(at-d)['cost'])/2e-5
            self.assertLess(abs(finite-r['phase_gradient'][j])/energy,1e-7)

    def test_rank_deficiency_does_not_require_statistical_loading(self):
        rng=np.random.default_rng(241);t=np.linspace(0,10,201);b,_=linear_basis(t,[0,10],2.)
        h=rng.normal(size=(len(t),3,1))+1j*rng.normal(size=(len(t),3,1));h=np.repeat(h,2,axis=2);y=1.5*h[:,:,0]
        normal=Normal(h,y,np.ones(len(t)),np.ones(len(t),bool),csr_matrix(b));r=normal.solve(np.zeros(2))
        self.assertTrue(r['numerical_convergence']);self.assertLess(abs(r['cost'])/normal.energy,1e-12)
        scale=np.sqrt(normal.diagonal);power=max(abs(normal.rhs.real/scale))
        x=normal.design_nnls(np.ones(len(scale),complex),scale,power)
        coefficient=x*power/scale
        self.assertLess(abs(normal.energy-2*np.vdot(coefficient,normal.rhs).real+np.vdot(coefficient,normal.multiply(coefficient)).real)/normal.energy,1e-12)

if __name__=='__main__':unittest.main()
