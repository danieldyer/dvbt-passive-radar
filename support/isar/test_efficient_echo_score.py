import unittest

import numpy as np

from efficient_echo_score import scores


class EfficientEchoScoreTests(unittest.TestCase):
    def test_score_matches_independent_joint_real_gaussian_least_squares(self):
        rng=np.random.default_rng(647);n=83
        nuisance=rng.normal(size=(2*n,7));q=np.linalg.qr(nuisance)[0]
        residual=rng.normal(size=n)+1j*rng.normal(size=n)
        candidates=rng.normal(size=(n,5))+1j*rng.normal(size=(n,5))
        result=scores(candidates,residual,q);data=np.r_[residual.real,residual.imag]
        baseline=np.linalg.norm(data-nuisance@np.linalg.lstsq(nuisance,data,rcond=None)[0])**2
        for j,h in enumerate(candidates.T):
            design=np.c_[nuisance,np.r_[h.real,h.imag],np.r_[-h.imag,h.real]]
            error=np.linalg.norm(data-design@np.linalg.lstsq(design,data,rcond=None)[0])**2
            self.assertAlmostEqual(result['score'][j],baseline-error,places=11)

    def test_two_targets_at_the_same_range_are_separated_by_phase_rate(self):
        time=np.arange(401)*.01;rates=np.array([-.9,-.6,-.3,.0,.3,.6,.9])
        primary=np.exp(-2j*np.pi*.3*time);second=np.exp(2j*np.pi*.6*time)
        q=np.linalg.qr(np.c_[np.r_[primary.real,primary.imag],np.r_[-primary.imag,primary.real]])[0]
        result=scores(np.exp(-2j*np.pi*time[:,None]*rates),2*primary+.4*second,q)
        self.assertFalse(result['valid'][4])
        self.assertEqual(int(np.argmax(result['score'])),1)
        # No spatial/range separation appears anywhere in this likelihood.
        self.assertGreater(result['retained_information'][1],.9)

    def test_unobservable_echo_is_flagged_and_known_noise_is_not_renormalized(self):
        rng=np.random.default_rng(419);n=101;h=np.exp(1j*np.arange(n)*.11)
        q=np.linalg.qr(np.c_[np.r_[h.real,h.imag],np.r_[-h.imag,h.real]])[0]
        answer=scores(h[:,None],h,q)
        self.assertFalse(answer['valid'][0]);self.assertEqual(answer['score'][0],0.)
        q=np.empty((2*n,0));values=[]
        for _ in range(600):
            noise=(rng.normal(size=n)+1j*rng.normal(size=n))/np.sqrt(2)
            values.append(scores(h[:,None],noise,q)['score'][0])
        self.assertLess(abs(np.mean(values)-1),.12)


if __name__=='__main__':unittest.main()
