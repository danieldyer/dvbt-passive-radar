import unittest
import numpy as np
from coherent_amplitude_glrt import statistic,fixed_spectrum_statistic


class CoherentAmplitudeTests(unittest.TestCase):
    def test_fixed_spectrum_matches_a_single_concatenated_complex_template(self):
        rng=np.random.default_rng(913);gain=rng.normal(size=5)+1j*rng.normal(size=5)
        h=rng.normal(size=(5,17,11))+1j*rng.normal(size=(5,17,11))
        y=rng.normal(size=(5,17))+1j*rng.normal(size=(5,17))
        score=np.einsum('cnp,cn->cp',h.conj(),y);information=np.sum(abs(h)**2,axis=1)
        template=(h*gain[:,None,None]).reshape(-1,11);data=y.ravel()
        expected_z=template.conj().T@data;expected_i=np.sum(abs(template)**2,axis=0)
        power,amplitude=fixed_spectrum_statistic(score,information,gain)
        np.testing.assert_allclose(power,abs(expected_z)**2/expected_i,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(amplitude,expected_z/expected_i,rtol=1e-13,atol=1e-13)
        self.assertGreater(np.linalg.norm(power-fixed_spectrum_statistic(score,information,gain*np.exp(1j*np.arange(5)))[0]),.1)
        np.testing.assert_allclose(fixed_spectrum_statistic(score,information,gain*3*np.exp(.7j))[0],power,rtol=1e-13,atol=1e-13)

    def test_global_optimum_matches_exhaustive_phase_search(self):
        rng=np.random.default_rng(821);z=rng.normal(size=(5,31))+1j*rng.normal(size=(5,31));score,phi=statistic(z)
        reference=np.zeros(31)
        for theta in np.linspace(-np.pi,np.pi,100001):reference=np.maximum(reference,np.sum(np.maximum(np.real(z*np.exp(-1j*theta)),0.)**2,axis=0))
        np.testing.assert_allclose(score,reference,rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(score,np.sum(np.maximum(np.real(z*np.exp(-1j*phi)),0.)**2,axis=0),atol=1e-12)
        self.assertTrue(np.all(score<=np.sum(abs(z)**2,axis=0)+1e-12));self.assertTrue(np.all(score>=np.max(abs(z)**2,axis=0)-1e-12))

    def test_common_phase_unequal_brightness_and_opposed_bands(self):
        z=np.array([1.,.1,2.,.7,0.])[:,None]*np.exp(.7j);score,phi=statistic(z)
        np.testing.assert_allclose(score,np.sum(abs(z)**2,axis=0),atol=1e-12);np.testing.assert_allclose(phi,.7,atol=1e-12)
        score,_=statistic(np.array([1.,-1.]));self.assertAlmostEqual(float(score),1.,places=12)

    def test_common_phase_and_units_are_gauges_not_fitted_band_phases(self):
        rng=np.random.default_rng(713);z=rng.normal(size=(5,21))+1j*rng.normal(size=(5,21));score,_=statistic(z)
        changed,_=statistic(z*3*np.exp(.81j));np.testing.assert_allclose(changed,9*score,rtol=1e-12)
        np.testing.assert_allclose(statistic(z[::-1])[0],score,rtol=1e-12)
        self.assertGreater(np.linalg.norm(statistic(z*np.exp(1j*np.arange(5))[:,None])[0]-score),1.)
if __name__=='__main__':unittest.main()
