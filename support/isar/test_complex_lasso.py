import unittest
import numpy as np
from complex_lasso import ComplexLasso,LowRankComplexLasso


class ComplexLassoTests(unittest.TestCase):
    def test_modal_solver_retains_physical_sparse_prior_and_null_space(self):
        rng=np.random.default_rng(619);v=np.linalg.qr(rng.normal(size=(9,5))+1j*rng.normal(size=(9,5)))[0]
        a=rng.normal(size=(70,5))+1j*rng.normal(size=(70,5));x=np.zeros(9,complex);x[[1,6]]=[1+.4j,-.3+.6j]
        y=a@v.conj().T@x+.01*(rng.normal(size=70)+1j*rng.normal(size=70))
        gram=a.conj().T@a;rhs=a.conj().T@y;energy=float(np.vdot(y,y).real)
        modal=LowRankComplexLasso(v,gram,rhs,energy);dense=ComplexLasso(v@gram@v.conj().T,v@rhs,energy)
        strength=.1*modal.linear_scale
        first,r=modal.solve(strength);second,s=dense.solve(strength)
        self.assertTrue(r['numerical_convergence']);self.assertTrue(s['numerical_convergence'])
        np.testing.assert_allclose(v.conj().T@first,v.conj().T@second,atol=2e-6,rtol=0)
        self.assertLess(abs(r['primal_objective']-s['primal_objective']),1e-6)
        trial=rng.normal(size=9)+1j*rng.normal(size=9)
        np.testing.assert_allclose(modal.ridge_step(trial,.13),dense.ridge_step(trial,.13),atol=1e-12,rtol=0)
    def test_diagonal_complex_soft_threshold(self):
        gram=np.diag([1.,2.,3.,4.]).astype(complex);rhs=np.array([1+2j,.1j,2-1j,-.05]);strength=.3
        expected=np.maximum(0.,1-strength/abs(rhs))*rhs/gram.diagonal()
        model=ComplexLasso(gram,rhs,20.);actual,report=model.solve(strength)
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0)
        self.assertTrue(report['numerical_convergence']);self.assertGreaterEqual(report['primal_dual_gap'],-1e-10)

    def test_correlated_exact_sparse_kkt_solution(self):
        rng=np.random.default_rng(818);a=rng.normal(size=(80,9))+1j*rng.normal(size=(80,9))
        a[:,1]=.98*a[:,0]+.15*a[:,1];gram=a.conj().T@a
        truth=np.zeros(9,complex);truth[[0,4,7]]=[.7+.8j,-.4+.2j,.3-.5j];strength=.15
        subgradient=rng.uniform(0,.5,9)*np.exp(1j*rng.uniform(-np.pi,np.pi,9))
        active=abs(truth)>0;subgradient[active]=truth[active]/abs(truth[active])
        y=a@truth+a@np.linalg.solve(gram,strength*subgradient)
        model=ComplexLasso(gram,a.conj().T@y,float(np.vdot(y,y).real));actual,report=model.solve(strength)
        np.testing.assert_allclose(actual,truth,atol=3e-5,rtol=0)
        self.assertTrue(report['numerical_convergence']);self.assertLess(report['maximum_scaled_kkt'],1e-6)
        self.assertLess(report['gap_relative_to_fitted_energy'],1e-5)
        gauge=np.exp(.6j);second=ComplexLasso(gram,model.rhs*gauge,model.energy)
        rotated,_=second.solve(strength);np.testing.assert_allclose(rotated,actual*gauge,atol=1e-9,rtol=0)


if __name__=='__main__':unittest.main()
