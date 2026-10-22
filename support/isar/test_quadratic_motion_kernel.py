import unittest
import numpy as np

from field import FS
from motion import C
from quadratic_motion_kernel import QuadraticMotionKernel,wavenumber_focus,fractional_angle
from recovery_experiment import Experiment
from test_recovery_experiment import problem


class QuadraticKernelTests(unittest.TestCase):
    def test_physical_range_and_carrier_bookkeeping_in_wavenumber_focus(self):
        n=127;spacing=3.;x=np.arange(n)*spacing;t=np.linspace(-1,1,61);rf=200e6
        delta=17.3*t+8.7*t*t
        frequency=np.fft.fftfreq(n,d=spacing);spectrum=np.exp(-((frequency-.01)/.025)**2)
        signal=np.fft.ifft(spectrum[None,:]*np.exp(-2j*np.pi*frequency[None,:]*delta[:,None]),axis=1)*np.exp(-2j*np.pi*rf/C*delta[:,None])
        focused=wavenumber_focus(signal,spacing,delta,rf,pad_factor=1)
        np.testing.assert_allclose(focused,np.broadcast_to(np.fft.ifft(spectrum),signal.shape),atol=2e-14)
        np.testing.assert_allclose(np.sum(abs(signal)**2,axis=1),np.sum(abs(focused)**2,axis=1),atol=1e-14)

    def test_all_three_profiled_path_derivatives_and_frft_sign(self):
        ex=problem();f=ex.fields[2];kernel=QuadraticMotionKernel(f,ex.training[2],ex.designs[2])
        p=kernel.nominal+np.array([6.,.12,.004]);y=kernel.response(p)*(.2+.1j)
        at=p+np.array([2.,.01,.001]);r,j,_,_=kernel.evaluate(at,y)
        for i,step in enumerate([.001,1e-5,1e-6]):
            delta=np.eye(3)[i]*step
            finite=(kernel.evaluate(at+delta,y,False)[0]-kernel.evaluate(at-delta,y,False)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-j[:,i])/np.linalg.norm(finite),3e-4)
        angle=fractional_angle(.02,200e6,10.)
        self.assertAlmostEqual(1/np.tan(angle),200e6/C*.02*100,places=12)

    def test_chirp_search_acquires_and_refines_migrating_quadratic_echo(self):
        ex=problem();f=ex.fields[2];kernel=QuadraticMotionKernel(f,ex.training[2],ex.designs[2])
        truth=kernel.nominal+np.array([9.,.17,.008]);y=kernel.response(truth)*(.2+.1j)
        seeds=kernel.search(y,np.arange(-20.,21.,10.),rate_halfwidth=.35,acceleration_halfwidth=.02,keep=3)
        bounds=([-40,kernel.nominal[1]-.4,kernel.nominal[2]-.04],[40,kernel.nominal[1]+.4,kernel.nominal[2]+.04])
        fits=[kernel.refine(y,s['parameters'],bounds=bounds) for s in seeds]
        best=min(fits,key=lambda r:r['training_rss'])
        np.testing.assert_allclose(best['parameters'],truth,atol=2e-5)
        self.assertLess(best['test_rss']/np.sum(abs(y)**2),1e-14)

    def test_static_clutter_is_profiled_and_unused_phase_cannot_change_the_fit(self):
        ex=problem();f=ex.fields[2];k=QuadraticMotionKernel(f,ex.training[2],ex.designs[2])
        truth=k.nominal+np.array([6.,.17,.008]);signal=k.response(truth)*(.2+.1j)
        beta=np.arange(ex.designs[2].shape[1])*(.05+.02j)
        y=signal+ex.designs[2]@beta
        bounds=([-40,k.nominal[1]-.4,k.nominal[2]-.04],[40,k.nominal[1]+.4,k.nominal[2]+.04])
        seed=truth+np.array([1.,.002,.0001]);a=k.refine(y,seed,bounds=bounds)
        changed=y.copy();changed[~k.selected]+=signal[~k.selected]*(np.exp(.37j)-1)
        b=k.refine(changed,seed,bounds=bounds)
        np.testing.assert_allclose(a['parameters'],truth,atol=1e-7)
        np.testing.assert_array_equal(a['parameters'],b['parameters'])
        np.testing.assert_array_equal(a['prediction'],b['prediction'])
        self.assertGreater(b['test_rss'],a['test_rss']+1.)


if __name__=='__main__':unittest.main()
