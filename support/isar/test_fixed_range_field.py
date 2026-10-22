from copy import deepcopy
import json
import unittest

import numpy as np

from field import Field,average,FS
from fixed_range_field import fixed_reference,stationary_profile_design
from motion import C,excess_path
from shared_echo_cluster import ReferencedAmplitudeArc,SharedEchoCluster
from stationary import Stationary
from symbol_moments import moments
from test_rigid_motion import fixture


def aperture_for(f,epoch):
    d=f.data;point=np.array([np.interp(epoch,d['row_time'],d['centre_ecef_m'][:,j]) for j in range(3)])
    path=float(excess_path(point[None],d['transmitter_ecef_m'],d['receiver_ecef_m'])[0][0])
    first=int(np.floor(path*FS/C))-128
    return dict(time_s=d['row_time'],information=d['row_information'],valid=np.ones(len(d['row_time']),bool),
        moments=np.zeros((2,len(d['row_time']),256),complex),metadata_json=json.dumps(dict(first_range_sample=first))),path


def recovery_fixture():
    original=fixture();epoch=8.;aperture,_=aperture_for(original,epoch)
    f,_=fixed_reference(original,aperture,epoch,np.arange(-8,9),omega_reference=-.01)
    f.y[:]=1.
    training=np.arange(len(f.time))%2==0;knots=np.linspace(f.data['row_time'][0],f.data['row_time'][-1],6)
    design=stationary_profile_design(f);rows=f.data['row_time'];first,last=rows[[0,-1]]
    # No AIS trajectory enters. The polynomial describes physical range
    # change directly, with the fixed C0/C1 Doppler reference restored.
    known=np.zeros(len(rows));rate=np.full(len(rows),-.01/(f.k*.001024))
    make=lambda field:ReferencedAmplitudeArc(field,training,knots,first,last,known,rate,stationary_design=design)
    band=make(f);truth=np.array([.3,.7,3.7,.13])
    raw,_=band.response(truth)
    moving=band.reduced(raw[:,:,None]*band.amplitude_basis[:,None,:])@np.array([.8,.6,1.,.4,.7,1.2])
    rng=np.random.default_rng(8493)
    static=(rng.normal(size=17)+1j*rng.normal(size=17))*100
    f.y=(moving+design@static).reshape(f.y.shape)
    return make(f),truth


class FixedRangeFieldTests(unittest.TestCase):
    def test_explicit_observer_keeps_range_and_carrier_geometrically_consistent(self):
        old=fixture();epoch=8.;aperture,_=aperture_for(old,epoch)
        rows=old.data['row_time'];observer=np.array([np.interp(epoch,rows,old.data['centre_ecef_m'][:,j]) for j in range(3)])+np.array([7.,-13.,2.])
        f,_=fixed_reference(old,aperture,epoch,np.arange(-8,9),observer_ecef_m=observer)
        np.testing.assert_array_equal(f.data['centre_ecef_m'],np.broadcast_to(observer,f.data['centre_ecef_m'].shape))
        expected=excess_path(observer[None],old.data['transmitter_ecef_m'],old.data['receiver_ecef_m'])[0][0]
        np.testing.assert_allclose(f.data['centre_path_m'],expected,atol=1e-12,rtol=0)
        at=np.broadcast_to(observer,(len(rows),3));delta,omega,_=f.path_kinematics(at,np.zeros_like(at))
        np.testing.assert_allclose(delta,0,atol=1e-12);np.testing.assert_allclose(omega,0,atol=1e-12)

    def test_empirical_path_and_constant_carrier_offset_have_an_explicit_fixed_reference(self):
        old=fixture();epoch=float(old.data['row_time'][len(old.data['row_time'])//2])
        aperture,_=aperture_for(old,epoch)
        aperture['moments'][0]=np.exp(.01j*np.arange(256))+.2*np.exp(-.05j*np.arange(256))
        aperture['moments'][1]=(.1+.02j)*aperture['moments'][0]
        geometric,gchange=fixed_reference(old,aperture,epoch,np.arange(-8,9),omega_reference=-.01)
        data=dict(old.data)
        for key in ('centre_ecef_m','centre_velocity_ecef_mps'):data.pop(key)
        offset=1234.5
        data['metadata_json']=json.dumps({**old.metadata,'phase_reference_offset_m':offset})
        data['value']=data['value']*np.exp(-1j*old.k*offset)
        empirical,echange=fixed_reference(Field(data,old.pulse.samples),aperture,epoch,np.arange(-8,9),omega_reference=-.01)
        np.testing.assert_allclose(empirical.data['value'],geometric.data['value'],atol=1e-12,rtol=1e-12)
        self.assertAlmostEqual(echange['constant_phase_change_rad']-gchange['constant_phase_change_rad'],old.k*offset,places=8)
        self.assertEqual(empirical.metadata['phase_reference_offset_m'],0.)
        self.assertNotIn('centre_ecef_m',empirical.data)
        self.assertEqual(stationary_profile_design(empirical).shape[1],17)

    def test_stationary_original_profiles_remain_constant_after_fixed_reference_change(self):
        old=fixture();aperture,_=aperture_for(old,8.)
        shape=np.exp(.2j*np.arange(256))+np.exp(-.37j*np.arange(256))*.3
        aperture['moments'][0]=shape
        f,change=fixed_reference(old,aperture,8.,np.arange(-8,9),omega_reference=-.02)
        good=f.data['information']>0
        np.testing.assert_allclose(f.data['value'][good],np.tile(f.data['value'][good][0],(good.sum(),1)),atol=2e-14,rtol=2e-14)
        np.testing.assert_array_equal(f.data['row_information'],old.data['row_information'])
        np.testing.assert_array_equal(f.data['information'],old.data['information'])
        self.assertTrue(np.all(f.data['centre_path_m']==change['reference_path_m']))
        self.assertAlmostEqual(change['constant_phase_change_rad'],
            -old.k*change['envelope_origin_change_samples']*C/FS,places=13)

    def test_independent_fourier_waveform_preserves_carrier_phase_and_both_symbol_moments(self):
        frequency=np.fft.fftfreq(256)
        spectrum=(abs(frequency)<.38)*(1+.3*np.sin(9*frequency));spectrum/=spectrum.sum()
        for channel in (0,4):
            old=fixture(channel);epoch=8.;aperture,reference=aperture_for(old,epoch)
            first=json.loads(aperture['metadata_json'])['first_range_sample']
            dt=old.data['row_time']-epoch;delta=.3*C/FS+.7*dt+.004*dt**2
            velocity=.7+.008*dt;omega=-old.k*velocity*.001024
            m0,m1,_,_=moments(omega);path=reference+delta
            for start in range(0,len(dt),32):
                use=slice(start,start+32)
                lag=first+np.arange(256)[None,:]-path[use,None]*FS/C
                pulse=np.exp(2j*np.pi*lag[:,:,None]*frequency)@spectrum
                carrier=np.exp(-1j*old.k*path[use,None])
                aperture['moments'][0,use]=pulse*carrier*m0[use,None]
                aperture['moments'][1,use]=pulse*carrier*m1[use,None]
            f,_=fixed_reference(old,aperture,epoch,np.arange(-8,9),omega_reference=-.014)
            expected=average(f.path_response(delta,omega),f.data['row_information'],f.metadata['block_rows'])[0]
            difference=np.linalg.norm(f.data['value']-expected)/np.linalg.norm(expected)
            self.assertLess(difference,2e-5)
            phase=abs(np.angle(np.vdot(expected,f.data['value'])))
            self.assertLess(phase,1e-5)

    def test_stationary_design_is_orthogonal_and_information_loss_matches_schur_complement(self):
        b,_=recovery_fixture();f=b.field;selected=b.selected;design=b.design
        weight=f.root_information[b.training]**2
        gram=design[selected].conj().T@design[selected]
        np.testing.assert_allclose(gram,np.eye(17)*weight.sum(),atol=1e-6,rtol=2e-15)
        self.assertLess(b.nuisance.singular_values.max()/b.nuisance.singular_values.min(),1+1e-12)
        shape=np.exp(.3j*np.arange(17))
        for frequency in (0.,.1,.5):
            tone=np.exp(2j*np.pi*frequency*f.time)
            h=(f.root_information[:,None]*tone[:,None]*shape).ravel()[selected]
            observed=np.vdot(b.nuisance.remove(h),b.nuisance.remove(h)).real/np.vdot(h,h).real
            expected=1-abs(np.sum(weight*tone[b.training])/weight.sum())**2
            self.assertAlmostEqual(observed,expected,places=13)

    def test_moving_phase_recovery_with_one_hundred_times_stronger_arbitrary_stationary_profile(self):
        b,truth=recovery_fixture();model=SharedEchoCluster([b],[.3],origin_radius=1.)
        fit=model.fit(truth+np.array([.002,-.002,.003,-.002]))
        np.testing.assert_allclose(fit['parameters'],truth,atol=3e-7,rtol=0)
        self.assertLess(fit['cost'],1e-12)
        self.assertGreater(model.prediction(np.array(fit['parameters']),[~b.training])[0]['heldout_explained_fraction'],1-1e-12)

    def test_stationary_and_moving_parameters_never_use_unused_observations(self):
        b,truth=recovery_fixture();before=SharedEchoCluster([b],[.3]);r,j=before.evaluate(truth)
        f=deepcopy(b.field);f.y[~b.training]*=np.exp(.9j)
        other=ReferencedAmplitudeArc(f,b.training,b.knots,b.epoch-b.scale,b.epoch+b.scale,
            b.known_path,b.known_rate,stationary_design=stationary_profile_design(f))
        changed=SharedEchoCluster([other],[.3]);rr,jj=changed.evaluate(truth)
        np.testing.assert_array_equal(rr,r);np.testing.assert_array_equal(jj,j)
        self.assertGreater(changed.prediction(truth,[~b.training])[0]['heldout_error'],1.)


if __name__=='__main__':unittest.main()
