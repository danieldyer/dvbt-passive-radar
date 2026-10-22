from types import SimpleNamespace
import unittest

import numpy as np

from block_pulse import BlockPulse
from field import Field
from fixed_range_field import fixed_reference
from midpoint_model import representative
from rigid_patch import RigidPatch
from template_information import norm_polynomials, reference,bin_moments
from test_fixed_range_field import aperture_for
from test_rigid_motion import fixture
from waveform_kernels import KernelField


def band_fixture(native=False):
    original=fixture();epoch=8.;aperture,_=aperture_for(original,epoch)
    f,_=fixed_reference(original,aperture,epoch,np.arange(-8,9))
    if not native:f=representative(f)
    kernels=BlockPulse(np.repeat(f.pulse.samples[None,:],len(f.time),axis=0))
    f=KernelField(f,kernels,np.repeat(np.arange(len(f.time)),f.metadata['block_rows']))
    original_time=original.data['row_time']
    times=f.data['row_time']
    centre=np.stack([np.interp(times,original_time,original.data['centre_ecef_m'][:,i]) for i in range(3)],axis=1)
    velocity=np.stack([np.interp(times,original_time,original.data['centre_velocity_ecef_mps'][:,i]) for i in range(3)],axis=1)
    frame=original.data['body_to_ecef'][0]
    kin=RigidPatch(times,epoch,centre,velocity,frame,f.data['transmitter_ecef_m'],f.data['receiver_ecef_m'])
    return SimpleNamespace(field=f,kinematics=kin,tx=f.data['transmitter_ecef_m'],rx=f.data['receiver_ecef_m'],origin=0.,training=np.ones(len(f.time),bool))


class TemplateInformationTests(unittest.TestCase):
    def test_norm_polynomial_matches_direct_whitening(self):
        b=band_fixture();f=b.field;coefficient=norm_polynomials(f,-10,20)
        for x in [-9.7,-3.45,.3,8.99]:
            i=int(np.floor(x));z=2*(x-i)-1
            poly=np.polynomial.chebyshev.chebval(z,coefficient[:,i+10].T)
            hp=f.pulse(f.offsets[None,:]-np.full((len(f.time),1),x))
            expected=np.sum(abs(hp@f.whiten.T)**2,axis=1)
            np.testing.assert_allclose(poly,expected,rtol=2e-14,atol=2e-14)

    def test_projected_information_bounds_and_brightness(self):
        b=band_fixture();parameters=np.array([.02,.001,0.,.05,0.]);points=np.array([[0.,0.,0.],[17.,-12.,3.]])
        values=reference(b,parameters,points)
        self.assertTrue(np.all(values[:,1]>0))
        self.assertTrue(np.all(values[:,1]<=values[:,0]))
        np.testing.assert_allclose(reference(b,parameters,points,amplitude=np.full(len(b.field.time),2.)),4*values,rtol=3e-15)

    def test_original_weighted_bin_carrier_and_brightness(self):
        rng=np.random.default_rng(937);rows=100+np.arange(128).reshape(-1,16)*.001088+rng.normal(size=(8,16))*1e-7
        w=rng.uniform(.5,2.,rows.shape);time=np.sum(w*rows,axis=1)/w.sum(axis=1)
        f=SimpleNamespace(metadata={'block_rows':16},data={'row_time':rows.ravel(),'row_information':w.ravel()},time=time)
        amplitude=rng.uniform(.5,1.5,rows.size);moments=bin_moments(f,amplitude);dt=rows-time[:,None]
        for frequency in [-20.,0.,17.]:
            expected=np.sum(w*amplitude.reshape(-1,16)*np.exp(1j*frequency*dt),axis=1)/w.sum(axis=1)
            actual=np.polynomial.polynomial.polyval(1j*frequency,moments.T,tensor=False)
            np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)

    def test_projected_information_calibrates_gaussian_noise(self):
        from stationary import Stationary
        from fixed_range_field import stationary_profile_design
        b=band_fixture();f=b.field;nu=Stationary(stationary_profile_design(f),np.ones(f.y.size,bool))
        h=(f.root_information[:,None]*np.exp(2j*np.pi*.01*f.time[:,None])*np.ones((1,len(f.offsets)))).ravel()
        residual=nu.remove(h);information=np.vdot(residual,residual).real;raw=np.vdot(h,h).real
        rng=np.random.default_rng(284);noise=(rng.normal(size=(len(h),1024))+1j*rng.normal(size=(len(h),1024)))/np.sqrt(2)
        power=abs(residual.conj()@noise)**2
        self.assertLess(abs(power.mean()/information-1),.1)
        self.assertLess(information/raw,.1)
        self.assertLess(power.mean()/raw,.11)


if __name__=='__main__':unittest.main()
