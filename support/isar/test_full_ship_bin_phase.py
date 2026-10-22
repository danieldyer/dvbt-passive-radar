import unittest
import numpy as np
from seed_full_ship_phase import bin_carrier
from motion import C


class FullFieldBinTests(unittest.TestCase):
    def test_actual_epochs_preserve_phase_across_all_five_carriers(self):
        rng=np.random.default_rng(881);native=np.arange(16)*.001088
        weights=np.exp(.3*rng.normal(size=16));time=np.average(native,weights=weights)
        offsets=np.arange(-8,9);frequency=np.fft.fftfreq(256);spectrum=(abs(frequency)<.405)*(1+.2*np.cos(7*frequency));spectrum/=sum(spectrum)
        phases=[]
        for rf in [177.5e6,184.5e6,191.625e6,219.5e6,226.5e6]:
            for velocity in [-6.,-.1,5.]:
                k=2*np.pi*rf/C;delta0=.37;lag=offsets[None,:]-delta0-velocity*(native-time)[:,None]*8e6/C
                pulse=np.exp(2j*np.pi*lag[:,:,None]*frequency)@spectrum
                exact=np.sum(weights[:,None]*pulse*np.exp(-1j*k*velocity*(native-time))[:,None],axis=0)/sum(weights)
                factor=bin_carrier(np.array([time]),np.array([velocity]),native,weights,k)[0]
                central=np.exp(2j*np.pi*(offsets[:,None]-delta0)*frequency)@spectrum;approx=central*factor
                inner=np.vdot(exact,approx);loss=-10*np.log10(abs(inner)**2/(np.vdot(exact,exact).real*np.vdot(approx,approx).real))
                self.assertLess(loss,.0001);self.assertLess(abs(np.degrees(np.angle(inner))),.01)
                phases.append(abs(np.degrees(np.angle(inner))))
        self.assertLess(max(phases),.01)
        np.testing.assert_array_equal(bin_carrier(np.array([time]),np.array([5.]),native,np.zeros(16),1.),[1.+0j])
if __name__=='__main__':unittest.main()
