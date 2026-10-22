import unittest
import numpy as np
from scipy.linalg import block_diag,cholesky,solve_triangular
from scipy.optimize import nnls
from sparse_positive_envelope import block_positive_quadratic,polish_positive_quadratic,Normal
from scipy.sparse import csr_matrix


class BlockPositiveEnvelopeTests(unittest.TestCase):
    def test_nonstationary_warm_face_does_not_cycle_on_a_zero_step(self):
        gram=np.array([[1.,.9],[.9,1.]])
        rhs=np.array([1.,.5]);initial=np.array([.1,0.])
        # Both gradients are initially negative. Adding coordinate two before
        # optimizing coordinate one produces [2.895,-2.105] and a zero step.
        # The constrained optimum is instead [1,0].
        actual,iterations=polish_positive_quadratic(lambda x:gram@x,rhs,initial)
        np.testing.assert_allclose(actual,[1.,0.],atol=1e-12,rtol=0)
        self.assertLess(iterations,5)

    def test_global_stationary_schur_complement_matches_dense_nnls(self):
        rng=np.random.default_rng(20261018);blocks=[np.arange(0,5),np.arange(5,11),np.arange(11,18)]
        positive=[]
        for index in blocks:
            a=rng.normal(size=(2*len(index),len(index)));positive.append(a.T@a+np.eye(len(index)))
        raw=block_diag(*positive);mean=(rng.normal(size=(3,18))+1j*rng.normal(size=(3,18)))*.13
        gram=raw-np.real(mean.conj().T@mean);linear=rng.normal(size=18)
        lower=cholesky(gram,lower=True);expected=nnls(lower.T,solve_triangular(lower,linear,lower=True))[0]
        actual,sweeps=block_positive_quadratic(lambda i:raw[np.ix_(i,i)],mean,linear,blocks,np.zeros(18),lambda x:gram@x)
        np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=1e-8)
        self.assertLess(sweeps,50)

    def test_many_aspect_supports_retain_global_projection_and_kkt(self):
        rng=np.random.default_rng(492);windows=16;points=4;knots=9;rows=windows*25;ranges=7
        time=np.arange(rows);window=time//25;basis=np.zeros((rows,windows*knots))
        coordinate=(time%25)*(knots-1)/24;left=np.minimum(coordinate.astype(int),knots-2);fraction=coordinate-left
        basis[time,window*knots+left]=1-fraction;basis[time,window*knots+left+1]=fraction
        h=rng.normal(size=(rows,ranges,points))+1j*rng.normal(size=(rows,ranges,points))
        root=rng.uniform(.6,1.2,rows);selected=np.arange(rows)%4!=3;phase=rng.normal(size=windows*points)
        amplitude=rng.uniform(.2,1.,(windows*knots,points));gain=(basis@amplitude)*np.exp(1j*phase.reshape(windows,points)[window])
        y=np.sum(h*gain[:,None,:],axis=2)+root[:,None]*(.3+.4j)
        model=Normal(h,y,root,selected,csr_matrix(basis),np.zeros(rows,int))
        groups=np.concatenate([np.tile(np.arange(w*points,(w+1)*points),knots) for w in range(windows)])
        result=model.solve(phase,phase_groups=groups)
        self.assertEqual(len(model.temporal_blocks),windows)
        self.assertTrue(result['numerical_convergence'],result)
        np.testing.assert_allclose(result['amplitude'].reshape(-1,points),amplitude,atol=1e-7,rtol=1e-7)
        self.assertLess(abs(result['cost'])/model.energy,1e-12)


if __name__=='__main__':unittest.main()
