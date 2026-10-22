import unittest
import numpy as np
from projected_aperture import Moments


class NativeProjectionTests(unittest.TestCase):
    def test_original_reference_projection_and_cross_band_phase(self):
        rng=np.random.default_rng(20261018);t=np.arange(2001)*.01;weight=.5+rng.uniform(size=len(t));weight[40:43]=0
        stationary=rng.normal(size=(2,7))+1j*rng.normal(size=(2,7));moving=np.exp(1j*(.4*t+.05*t*t))
        target=(.3+.5j)*moving[None,:,None]*np.array([1.,.03])[:,None,None]*np.linspace(.1,1,7)[None,None,:]
        original=stationary[:,None,:]+target
        for rf in [177.5e6,184.5e6,191.625e6,219.5e6,226.5e6]:
            k=2*np.pi*rf/299792458.;reference=14000+3*t+.01*t*t;saved=original*np.exp(1j*k*reference)[None,:,None]
            before=saved.copy();p=Moments(saved,reference,weight,k,chunk=57);actual=p[:,np.arange(len(t))]
            expected=target-np.einsum('mtr,t->mr',target,weight)[:,None,:]/weight.sum()
            np.testing.assert_allclose(actual*np.exp(-1j*k*reference)[None,:,None],expected,atol=4e-15,rtol=0)
            np.testing.assert_array_equal(saved,before)
            unphased=actual*np.exp(-1j*k*reference)[None,:,None]
            self.assertLess(np.max(abs(np.einsum('mtr,t->mr',unphased,weight))),1e-11)
            # A pure moving target through P has a real nonnegative matched
            # response: projection changes information, never carrier phase.
            matched=np.einsum('mtr,mtr,t->',target.conj(),unphased,weight)
            self.assertGreater(matched.real,0);self.assertLess(abs(matched.imag)/matched.real,1e-14)

if __name__=='__main__':unittest.main()
