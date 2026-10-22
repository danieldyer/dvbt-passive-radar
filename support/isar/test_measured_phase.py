import unittest
import numpy as np

from measured_phase import local_average,anchored_history,interpolate_connected,range_rate_map,path_increments


class MeasuredPhaseTests(unittest.TestCase):
    def test_quadratic_acquisition_restores_physical_phase_at_the_weighted_epoch(self):
        rng=np.random.default_rng(193);t=np.arange(0.,5.,.004);w=rng.uniform(.2,2,len(t))
        centres=np.arange(.3,4.71,.05);origin=2.2;rate=.4;acceleration=-.3
        path=rate*(t-origin)+.5*acceleration*(t-origin)**2
        for rf in (177.5e6,226.5e6):
            k=2*np.pi*rf/299792458.;signal=np.exp(-1j*k*path+1.7j)
            epoch,value,_=local_average(t,signal,w,centres,.5,k,rate,
                acceleration_mps2=acceleration,rate_epoch_s=origin)
            expected=np.exp(-1j*k*(rate*(epoch-origin)+.5*acceleration*(epoch-origin)**2)+1.7j)
            np.testing.assert_allclose(value,expected,rtol=0,atol=3e-15)

    def test_unequal_carriers_and_saved_deramps_recover_same_physical_path(self):
        centres=np.arange(.3,2.71,.05);rng=np.random.default_rng(744)
        for ch,rf in enumerate([177.5e6,184.5e6,191.625e6,219.5e6,226.5e6]):
            t=np.arange(0.,3.,.004)+ch*.00003;k=2*np.pi*rf/299792458.
            old=14000+(-.7+.02*ch)*t;actual=14050-.3*t
            rate=.4-.02*ch;w=rng.uniform(.3,2.,len(t));w[::7]=0
            signal=np.exp(-1j*k*(actual-old)+1j*ch*.91)
            epoch,z,var=local_average(t,signal,w,centres,.5,k,rate)
            nominal=13900+(-.6+.01*ch)*epoch;old_at=14000+(-.7+.02*ch)*epoch
            h=anchored_history(epoch,z,var,k,old_at,nominal,20)
            np.testing.assert_allclose(h['measured_path_m'],-.3*(epoch-epoch[20]),atol=3e-12)
            np.testing.assert_allclose(abs(z),1,atol=1e-13)
            # The supplied acquisition rate is restored, not subtracted from
            # the physical measurement or fitted separately in each band.
            h2=anchored_history(epoch,z*np.exp(1.7j),var,k,old_at,nominal,20)
            np.testing.assert_allclose(h2['measured_path_m'],h['measured_path_m'],atol=1e-14)

    def test_fade_does_not_connect_unknown_integer_cycles(self):
        t=np.arange(11.)*.1;z=np.exp(.3j*t);z[4]=0;v=np.full(11,.01)
        h=anchored_history(t,z,v,4.,t,t,7)
        np.testing.assert_array_equal(h['connected'],np.arange(11)>4)
        self.assertTrue(np.isnan(h['measured_path_m'][:5]).all())
        out=interpolate_connected(t,h['measured_path_m'],np.arange(21)*.05)
        self.assertTrue(np.isnan(out[:10]).all())

    def test_conditional_noise_variance_of_weighted_complex_measurement(self):
        rng=np.random.default_rng(45);t=np.arange(101)*.01;w=np.linspace(.5,2.,101)
        centre=np.array([.5]);use=abs(t-.5)<=.25
        noise=(rng.normal(size=(3000,len(t)))+1j*rng.normal(size=(3000,len(t))))/np.sqrt(2*8192*w)
        estimate=noise[:,use]@w[use]/w[use].sum()
        _,_,var=local_average(t,noise[0],w,centre,.5,4.,0.)
        self.assertLess(abs(np.mean(abs(estimate)**2)/var[0]-1),.05)

    def test_periodogram_sign_and_scale(self):
        t=np.arange(200)*.01;k=4.;rates=np.array([-.4,0.,.4]);w=np.ones(len(t))
        y=np.exp(-1j*k*.4*t)[:,None]
        power=range_rate_map(t,y,w,k,rates)[:,0]
        self.assertEqual(power.argmax(),2)
        self.assertAlmostEqual(power[2],8192*len(t),places=8)

    def test_local_increments_ignore_constant_phase_and_do_not_bridge_a_fade(self):
        t=np.arange(81)*.05;old=14000-.7*t;actual=14050-.3*t+.01*t**2
        for rf in (177.5e6,226.5e6):
            k=2*np.pi*rf/299792458.;z=np.exp(-1j*k*(actual-old)+2.4j)
            z[40]=0;var=np.full(len(t),.001)
            _,delta,good=path_increments(t,z,var,k,old,5)
            truth=actual[5:]-actual[:-5]
            np.testing.assert_allclose(delta[good],truth[good],atol=4e-12)
            self.assertFalse(good[35]);self.assertFalse(good[40])
            self.assertTrue(good[34]);self.assertTrue(good[41])


if __name__=='__main__':unittest.main()
