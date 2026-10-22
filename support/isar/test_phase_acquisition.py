import unittest

import numpy as np

from phase_acquisition import projected_periodogram


class PhaseAcquisitionTests(unittest.TestCase):
    def test_static_profile_cancels_and_tone_information_is_preserved(self):
        rng=np.random.default_rng(644);t=np.arange(300)*.011;w=rng.uniform(.3,2,len(t))
        k=4.1;rates=np.array([-.4,0.,.2]);amplitude=np.array([.7+.4j,-.2+.3j])
        y=np.exp(-1j*k*rates[0]*(t-t.mean()))[:,None]*amplitude
        score,retained=projected_periodogram(t,y,w,k,rates,1.08)
        shifted,_=projected_periodogram(t,y+np.array([1e6-2e6j,-3e6+1e6j]),w,k,rates,1.08)
        np.testing.assert_allclose(shifted,score,rtol=2e-8,atol=1e-10)
        np.testing.assert_allclose(score[0],8192*w.sum()/1.08*retained[0]*abs(amplitude)**2,rtol=2e-14)
        np.testing.assert_array_equal(score[1],0.)
        self.assertEqual(np.argmax(score[:,0]),0)

    def test_conditional_noise_score_has_unit_mean_without_rescaling(self):
        rng=np.random.default_rng(835);t=np.arange(200)*.01;w=np.linspace(.2,3,len(t));scale=1.08
        noise=(rng.normal(size=(len(t),2500))+1j*rng.normal(size=(len(t),2500)))*np.sqrt(scale/(2*8192*w[:,None]))
        score,_=projected_periodogram(t,noise,w,4.,np.array([-.05,.2,.8]),scale)
        np.testing.assert_allclose(score.mean(axis=1),1.,atol=.05)

    def test_accelerating_echo_acquires_with_correct_phase_sign(self):
        t=np.arange(300)*.013;w=np.ones(len(t));k=4.3;rates=np.linspace(-1,1,41)
        dt=t-t.mean();a=.35;r=-.4
        y=np.exp(-1j*k*(r*dt+.5*a*dt**2))[:,None]+(50+30j)
        score,retained=projected_periodogram(t,y,w,k,rates,1.08,acceleration=a)
        index=int(np.argmin(abs(rates-r)))
        self.assertEqual(score[:,0].argmax(),index)
        self.assertAlmostEqual(score[index,0]/(8192*w.sum()/1.08*retained[index]),1.,places=13)


if __name__=='__main__':unittest.main()
