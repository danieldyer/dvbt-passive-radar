import unittest

import numpy as np

from matched_phase import ProfiledPhase, refine_profiled


class ProfiledPhaseTests(unittest.TestCase):
    def test_background_projection_gradient_and_known_phase_recovery(self):
        rng = np.random.default_rng(712)
        time = np.arange(241)*.025
        ranges = np.arange(5)
        template = (1+.2*np.sin(time[:, None]+ranges))*np.exp(1j*(time[:, None]*(.8+.2*ranges)+ranges))
        background = np.stack([np.broadcast_to(np.exp(.2j*k*ranges), template.shape)*
            np.exp(.3j*k*time[:, None]) for k in range(3)], axis=-1).reshape(-1, 3)
        selected = np.broadcast_to((np.arange(len(time))%2 == 0)[:, None], template.shape).copy()
        information = np.sum(abs(template)**2, axis=1)
        shape = ProfiledPhase(time, template, template, background, selected, .7, information)
        coefficients = rng.normal(0, .2, shape.curve.basis.shape[1])
        phase = shape.curve.phase(coefficients)
        y = template*np.exp(1j*phase[:, None])+(background@np.array([170+60j, -400+100j, 90j])).reshape(template.shape)
        likelihood = ProfiledPhase(time, y, template, background, selected, .7, information)
        probe = .4*coefficients
        value, gradient = likelihood.objective(probe)
        self.assertGreater(value, .001)
        for i in range(len(probe)):
            delta = np.eye(len(probe))[i]*1e-5
            numeric = (likelihood.objective(probe+delta)[0]-likelihood.objective(probe-delta)[0])/2e-5
            self.assertLess(abs(numeric-gradient[i]), 1e-8)
        fitted, detail = refine_profiled(time, y, template, background, selected, np.zeros(len(time)),
            knot_seconds=.7, gauge_information=information)
        self.assertTrue(detail['converged'])
        self.assertLess(np.max(abs(fitted-phase)), 5e-6)
        self.assertLess(likelihood.objective(coefficients)[0], 1e-22)
        # The synthetic phase also predicts ranges not used to fit it.
        selected[:, 1::2] = False
        fitted, detail = refine_profiled(time, y, template, background, selected, np.zeros(len(time)),
            knot_seconds=.7, gauge_information=information)
        self.assertTrue(detail['converged'])
        self.assertLess(np.max(abs(fitted-phase)), 1e-5)

    def test_small_phase_on_mismatched_template_has_strict_stationarity(self):
        rng = np.random.default_rng(719)
        time = np.arange(801)*.025
        r = np.arange(5)
        template = (1+.3*np.cos(time[:,None]+r))*np.exp(1j*(time[:,None]*(.3+.1*r)+r))
        background = np.stack([np.broadcast_to(np.exp(.1j*k*r),template.shape)*np.exp(.7j*k*time[:,None]) for k in range(3)],axis=-1).reshape(-1,3)
        selected = np.broadcast_to((np.arange(len(time))%2==0)[:,None],template.shape).copy()
        information = np.sum(abs(template)**2,axis=1)
        shape = ProfiledPhase(time,template,template,background,selected,1.,information)
        phase = shape.curve.phase(rng.normal(0,.02,shape.curve.basis.shape[1]))
        data = template*np.exp(1j*phase[:,None])+.2*(rng.normal(size=template.shape)+1j*rng.normal(size=template.shape))
        fitted, detail = refine_profiled(time,data,template,background,selected,np.zeros(len(time)),knot_seconds=1.,gauge_information=information)
        self.assertTrue(detail['converged'])
        self.assertLess(detail['maximum_gradient'],1e-9)
        self.assertLess(abs(np.sum(shape.curve.gauge_weight*fitted)),1e-10)
        self.assertLess(abs(np.sum(shape.curve.gauge_weight*shape.curve.affine[:,1]*fitted)),1e-10)
        if detail['stationarity_polish']:
            self.assertEqual(detail['stationarity_polish']['tangent_dimensions'],shape.curve.basis.shape[1]-2)
            self.assertGreater(min(detail['stationarity_polish']['curvature_eigenvalues']),0.)


if __name__ == '__main__':
    unittest.main()
