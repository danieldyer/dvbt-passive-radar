import json
import unittest

import numpy as np

from field import Field,average,interpolate,FS
from fixed_range_field import fixed_reference,stationary_profile_design
from motion import C
from stationary import Stationary
from symbol_moments import moments


class SeparateCarrierReferenceTests(unittest.TestCase):
    def fixture(self):
        n=512;time=np.arange(n)*.001088;epoch=time.mean();rate=27.;rf=226.5e6;k=2*np.pi*rf/C
        path=1600+rate*(time-epoch);omega=np.full(n,-k*rate*.001024)
        frequency=np.fft.fftfreq(256);spectrum=(abs(frequency)<.38)*(1+.3*np.sin(9*frequency));spectrum/=spectrum.sum()
        pulse=np.exp(2j*np.pi*np.arange(-64,65)[:,None]*frequency)@spectrum
        weight=np.ones(n);mean,information=average(time,weight,16)
        data=dict(row_time=time,row_information=weight,time=mean,information=information,
            centre_path_m=path,centre_omega=omega,value=np.zeros((len(mean),17),complex),offset_samples=np.arange(-8,9),
            metadata_json=json.dumps(dict(rf_hz=rf,block_rows=16)))
        field=Field(data,pulse);first=int(np.floor(1600*FS/C))-128
        aperture=dict(time_s=time,information=weight,valid=np.ones(n,bool),
            moments=np.zeros((2,n,256),complex),metadata_json=json.dumps(dict(first_range_sample=first)))
        m0,m1,_,_=moments(omega)
        for start in range(0,n,16):
            rows=slice(start,start+16);lag=first+np.arange(256)[None,:]-path[rows,None]*FS/C
            value=np.exp(2j*np.pi*lag[:,:,None]*frequency)@spectrum
            value*=np.exp(-1j*k*path[rows,None])
            aperture['moments'][0,rows]=value*m0[rows,None]
            aperture['moments'][1,rows]=value*m1[rows,None]
        return field,aperture,epoch,rate

    def test_preserve_carrier_avoids_fast_vehicle_averaging_loss_and_matches_fourier_truth(self):
        old,aperture,epoch,rate=self.fixture();omega=old.data['centre_omega'][0]
        fixed,_=fixed_reference(old,aperture,epoch,np.arange(-8,9),omega_reference=omega)
        kept,_=fixed_reference(old,aperture,epoch,np.arange(-8,9),omega_reference=omega,preserve_carrier=True)
        delta=old.data['centre_path_m']-kept.data['centre_path_m']
        expected=average(kept.path_response(delta,old.data['centre_omega']),old.data['row_information'],16)[0]
        self.assertLess(np.linalg.norm(kept.data['value']-expected)/np.linalg.norm(expected),2e-5)
        self.assertLess(abs(np.angle(np.vdot(expected,kept.data['value']))),1e-5)
        fixed_delta=old.data['centre_path_m']-fixed.data['centre_path_m']
        fixed_expected=average(fixed.path_response(fixed_delta,old.data['centre_omega']),old.data['row_information'],16)[0]
        self.assertLess(np.linalg.norm(fixed.data['value']-fixed_expected)/np.linalg.norm(fixed_expected),2e-5)
        ratio=np.linalg.norm(fixed.y)**2/np.linalg.norm(kept.y)**2
        predicted=abs(np.mean(np.exp(-1j*old.k*rate*np.arange(16)*.001088)))**2
        # The scalar Dirichlet loss neglects envelope migration within each
        # 16-row mean. The full Fourier waveform above tests that migration;
        # this comparison only checks the small-envelope-change approximation.
        self.assertLess(abs(10*np.log10(ratio/predicted)),.01)
        self.assertLess(10*np.log10(ratio),-1.)

    def test_known_carrier_static_design_is_orthogonal_and_does_not_invent_information(self):
        old,aperture,epoch,_=self.fixture()
        kept,_=fixed_reference(old,aperture,epoch,np.arange(-8,9),omega_reference=-.01,preserve_carrier=True)
        design=stationary_profile_design(kept);selected=np.repeat(np.arange(len(kept.time))%2==0,17)
        nuisance=Stationary(design,selected)
        self.assertLess(nuisance.singular_values.max()/nuisance.singular_values.min(),1+1e-12)
        stationary=kept.stationary(1610.).ravel()[selected]
        self.assertLess(np.linalg.norm(nuisance.remove(stationary))/np.linalg.norm(stationary),1e-12)
        crop=kept.windowed(kept.time[3],kept.time[10])
        np.testing.assert_array_equal(crop.data['carrier_reference_path_m'],kept.data['carrier_reference_path_m'][48:160])


if __name__=='__main__':unittest.main()
