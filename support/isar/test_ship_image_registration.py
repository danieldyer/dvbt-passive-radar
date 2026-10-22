import unittest
from types import SimpleNamespace

import numpy as np

from omega_k_recording import project_stationary
from render_omega_k_ship import common_carrier,combine,resample_envelope


class RegistrationTests(unittest.TestCase):
    def test_five_band_carriers_recover_the_same_physical_point_phase(self):
        # Independent full matched-image point model. The full image phase is
        # q_c dot (p-p0); the saved envelope has q_c dot p removed.
        carriers=np.array([[3.1,.2],[3.3,.21],[3.5,.22],[3.7,.23],[3.9,.24]])
        p=np.array([37.,-19.]);qref=carriers[2]
        gains=np.exp(1j*np.arange(5)*.6)*np.arange(1,6)/3
        info=np.arange(5)+10.
        numerator=[]
        for q,g,n in zip(carriers,gains,info):
            envelope=g*n*np.exp(-1j*(q@p))
            numerator.append(common_carrier(envelope,q,qref,p))
        numerator=np.asarray(numerator)
        expected=gains*info*np.exp(-1j*(qref@p))
        np.testing.assert_allclose(numerator,expected,atol=1e-12)
        z=combine(numerator,info,gains)
        self.assertAlmostEqual(abs(z)**2,float(np.sum(abs(gains)**2*info)),places=10)

    def test_combination_has_unit_noise_variance_and_obeys_energy_bound(self):
        rng=np.random.default_rng(37);information=np.arange(1,6.)
        gain=rng.normal(size=5)+1j*rng.normal(size=5)
        noise=(rng.normal(size=(5,100000))+1j*rng.normal(size=(5,100000)))/np.sqrt(2)
        data=noise*np.sqrt(information[:,None]);z=combine(data,information,gain)
        self.assertLess(abs(np.mean(abs(z)**2)-1),.015)
        self.assertTrue(np.all(abs(z)**2<=np.sum(abs(data)**2/information[:,None],axis=0)*(1+1e-12)))

    def test_stationary_projection_is_orthogonal_and_keeps_epoch_sets_separate(self):
        rng=np.random.default_rng(4);root=np.linspace(1,2,51)
        f=SimpleNamespace(time=np.arange(51),root_information=root)
        y=rng.normal(size=(51,5))+1j*rng.normal(size=(51,5));use=np.arange(51)%3==0
        p=project_stationary(f,y,use)
        np.testing.assert_allclose(project_stationary(f,p,use),p,atol=1e-14)
        np.testing.assert_allclose(root@p,0,atol=1e-14)
        self.assertEqual(np.max(abs(p[~use])),0)
        self.assertAlmostEqual(np.vdot(y,p).real,np.vdot(p,p).real,places=11)

    def test_complex_envelope_interpolation_preserves_grid_coordinates(self):
        x=np.arange(-5.,6);y=np.arange(-7.,8);xx,yy=np.meshgrid(x,y)
        z=(1+.1j)*np.exp(.02j*xx+.03j*yy)
        np.testing.assert_allclose(resample_envelope(z,x,y,xx,yy),z,atol=1e-14)


if __name__=='__main__':unittest.main()
