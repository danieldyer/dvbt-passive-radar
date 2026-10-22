import unittest
import numpy as np
from audit_ship_scattering_envelopes import statistics,linear_basis
from rigid_patch import profile_static


class EnvelopeStatisticsTests(unittest.TestCase):
    def test_sufficient_statistics_match_explicit_projected_complex_dictionary(self):
        rng=np.random.default_rng(718);t=np.linspace(0,10,101)
        h=rng.normal(size=(len(t),4,3))+1j*rng.normal(size=(len(t),4,3));y=rng.normal(size=(len(t),4))+1j*rng.normal(size=(len(t),4))
        root=np.exp(.2*rng.normal(size=len(t)));selected=np.arange(len(t))%3!=1;b,_=linear_basis(t,[0,10],2.)
        a=np.einsum('trp,tk->trpk',h,b).reshape(len(t),4,-1)
        a=profile_static(a,root,selected)[selected].reshape(-1,a.shape[-1]);data=profile_static(y,root,selected)[selected].ravel()
        gram,rhs,energy=statistics(h,y,root,selected,b)
        np.testing.assert_allclose(gram,a.conj().T@a,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(rhs,a.conj().T@data,rtol=1e-12,atol=1e-12);self.assertAlmostEqual(energy,np.vdot(data,data).real,places=11)
        y[~selected]*=1e6
        g2,b2,e2=statistics(h,y,root,selected,b)
        np.testing.assert_array_equal(g2,gram);np.testing.assert_array_equal(b2,rhs);self.assertEqual(e2,energy)

    def test_positive_basis_does_not_extrapolate_or_modify_phase(self):
        t=np.linspace(0,10,1001);b,_=linear_basis(t,[0,10],2.);np.testing.assert_allclose(b.sum(axis=1),1.,atol=1e-15)
        self.assertGreaterEqual(b.min(),0.);self.assertLessEqual(b.max(),1.)
        with self.assertRaises(ValueError):linear_basis(np.array([-1.,5.]),[0,10],2.)


if __name__=='__main__':unittest.main()
