import unittest
import numpy as np
from polynomial_phase_acquisition import Band,magnitude_envelope
from rigid_patch import profile_static


class TestPolynomialAcquisition(unittest.TestCase):
    def test_training_brightness_is_invariant_to_motion_phase(self):
        rng=np.random.default_rng(829);t=np.linspace(-15,15,301);root=np.ones(len(t));mask=np.arange(len(t))%4!=3;groups=np.zeros(len(t),int)
        h=rng.normal(size=(len(t),5,2))+1j*rng.normal(size=(len(t),5,2));y=np.sum(h,axis=-1)+rng.normal(size=(len(t),5))
        first=magnitude_envelope(h,y,root,mask,groups,t)
        second=magnitude_envelope(h*np.exp(1j*np.sin(t)*5)[:,None,None],y,root,mask,groups,t)
        np.testing.assert_allclose(first,second,atol=7e-16,rtol=5e-16)
        altered=y.copy();altered[~mask]+=1e8
        np.testing.assert_array_equal(first,magnitude_envelope(h,altered,root,mask,groups,t))

    def test_stationary_projection_trial_gram_gradient_and_batch(self):
        rng=np.random.default_rng(8282);t=np.linspace(-15,15,301);root=rng.uniform(.7,1.3,len(t));selected=np.arange(len(t))%4!=3;groups=(t>0).astype(int)
        h=rng.normal(size=(len(t),5,3))+1j*rng.normal(size=(len(t),5,3));u=t/15;basis=np.c_[u,u*u,u*u*u]
        truth=np.array([-.8,.2,.9]);coefficient=np.array([.3+.7j,-.4+.1j,.9-.2j]);k=4.1
        y=np.einsum('trp,p->tr',h,coefficient)*np.exp(-1j*k*(basis@truth))[:,None]+root[:,None]*(.7+.3j)
        band=Band(h,y,root,selected,groups,t,k,0.,15.);at=np.array([-.6,.1,.8]);score,gradient=band.evaluate(at,True)
        for j in range(3):
            step=np.eye(3)[j]*1e-5;numeric=(band.evaluate(at+step)-band.evaluate(at-step))/2e-5
            self.assertLess(abs(numeric-gradient[j])/band.energy,1e-8)
        a=profile_static(h*np.exp(-1j*k*(basis@at))[:,None,None],root,selected,groups)[selected].reshape(-1,3)
        b=profile_static(y,root,selected,groups)[selected].ravel();fit=np.linalg.lstsq(a,b,rcond=None)[0]
        self.assertAlmostEqual(score,float(np.vdot(a@fit,a@fit).real),places=10)
        trials=np.array([at,truth,np.zeros(3)]);np.testing.assert_allclose(band.many(trials,batch=2),[band.evaluate(q) for q in trials],atol=1e-10,rtol=1e-14)
        self.assertLess(abs(band.evaluate(truth)-band.energy)/band.energy,1e-13)
        zero=np.concatenate([h,np.zeros_like(h[:,:,:1])],axis=-1)
        other=Band(zero,y,root,selected,groups,t,k,0.,15.)
        self.assertAlmostEqual(other.evaluate(at),score,places=10)
        single=Band(h,y,root,selected,groups,t,k,0.,15.,'best-point')
        score,gradient=single.evaluate(at,True)
        self.assertAlmostEqual(score,float(max(abs(a.conj().T@b)**2/np.sum(abs(a)**2,axis=0))),places=10)
        for j in range(3):
            step=np.eye(3)[j]*1e-5;finite=(single.evaluate(at+step)-single.evaluate(at-step))/2e-5
            self.assertLess(abs(finite-gradient[j])/single.energy,1e-8)
        np.testing.assert_allclose(single.many(trials,batch=2),[single.evaluate(q) for q in trials],atol=1e-10,rtol=1e-14)


if __name__=='__main__':unittest.main()
