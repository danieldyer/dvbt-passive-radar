import unittest
import numpy as np
from spatial_aperture_information import information


class SpatialApertureInformationTests(unittest.TestCase):
    def test_matches_independent_real_nuisance_projection(self):
        rng=np.random.default_rng(935)
        values=[rng.normal(size=(40,3))+1j*rng.normal(size=(40,3)) for _ in range(5)]
        derivatives=[rng.normal(size=(*h.shape,2))+1j*rng.normal(size=(*h.shape,2)) for h in values]
        h=np.concatenate([v.ravel() for v in values]);d=np.concatenate([v.reshape(-1,2) for v in derivatives])
        real_d=np.r_[d.real,d.imag];n=len(h)
        for coherent in [True,False]:
            nuisance=np.zeros((2*n,6 if coherent else 10));offset=0
            for c,v in enumerate(values):
                at=np.arange(offset,offset+v.size);offset+=v.size
                nuisance[at,c]=v.real.ravel();nuisance[n+at,c]=v.imag.ravel()
                p=5 if coherent else 5+c
                nuisance[at,p]=-v.imag.ravel();nuisance[n+at,p]=v.real.ravel()
            q=np.linalg.qr(nuisance,mode='reduced')[0];projected=real_d-q@(q.T@real_d)
            expected=2*projected.T@projected/np.vdot(h,h).real
            np.testing.assert_allclose(information(values,derivatives,coherent=coherent),expected,atol=2e-14,rtol=1e-13)

    def test_common_phase_gauge_and_independent_band_phase_information(self):
        values=[np.ones((21,1),complex) for _ in range(5)]
        derivative=[np.stack([1j*k*h,np.zeros_like(h)],axis=-1) for k,h in zip([3.,3.2,3.4,4.1,4.3],values)]
        independent=information(values,derivative,coherent=False)
        coherent=information(values,derivative)
        self.assertLess(np.max(abs(independent)),1e-13);self.assertGreater(coherent[0,0],.1)
        changed=[d+1j*h[:,:,None]*np.array([2.3,-.7]) for h,d in zip(values,derivative)]
        np.testing.assert_allclose(information(values,changed),coherent,atol=1e-14,rtol=1e-13)


if __name__=='__main__':unittest.main()
