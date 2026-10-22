import unittest

import numpy as np

from channel_alignment import AlignmentBand
from fit_physical_profile import solve
from range_phase import RangePhase
import test_coherent_points as point_fixture


class RangePhaseTests(unittest.TestCase):
    def test_gradient_likelihood_and_known_origin_phase_recovery(self):
        original, truth = point_fixture.CoherentPointTests().fixture()
        points = original.coordinates(truth)
        exact = original.evaluate(truth)
        origins = np.array([7.5, -4.5, 2.5])
        bands = []
        for c, (b, out) in enumerate(zip(original.bands, exact['bands'])):
            shifted = AlignmentBand(b.field, b.state, points, b.selected_time)
            target, _ = shifted.design(origins[c])
            b.field.y += ((target-out['target'])@out['coefficient']).reshape(b.field.y.shape)
            bands.append(AlignmentBand(b.field, b.state, points, b.selected_time))
        seed = np.r_[exact['point_phase_radians'], np.asarray(exact['band_phase_radians'])[original.other]]
        model = RangePhase(bands, np.zeros(3), [seed, seed+np.array([.4, -.2, .2, -.4])], reference=2)
        x = origins/model.scale+[.15, -.1, .2]
        measured = model.evaluate(x)
        raw = sum(np.linalg.norm(b.clean-b.nuisance.remove(d[b.selected])@a)**2
            for b, d, a in zip(bands, measured['targets'], measured['coefficients']))/model.energy
        self.assertAlmostEqual(raw, measured['value'], places=12)
        for j in range(3):
            step = np.eye(3)[j]*1e-4
            numeric = (model.objective(x+step)[0]-model.objective(x-step)[0])/2e-4
            self.assertAlmostEqual(numeric, measured['gradient'][j], delta=2e-7)
        repeated = model.evaluate(x)
        self.assertAlmostEqual(repeated['value'], measured['value'], places=13)
        np.testing.assert_allclose(repeated['gradient'], measured['gradient'], atol=1e-11)
        fit, result = solve(model, np.empty((0, 3)), np.empty(0), np.empty(0),
            np.full(3, -7.5), np.full(3, 7.5), polish_stationarity=False)
        recovered = model.evaluate(fit)
        self.assertTrue(result['converged'])
        self.assertLess(recovered['value'], 1e-12)
        np.testing.assert_allclose(fit*model.scale, origins, atol=2e-4)
        np.testing.assert_allclose(np.exp(1j*recovered['band_phase_radians']),
            np.exp(1j*np.asarray(exact['band_phase_radians'])), atol=2e-6)
        joint = RangePhase(bands, np.zeros(3), [seed, seed+.2], reference=2, joint=True)
        joint.initialize_phases()
        trial = np.r_[x, seed+np.array([.12, -.1, .05, -.08])]
        value = joint.evaluate(trial)
        for j in range(len(trial)):
            h = 1e-4 if j < 3 else 1e-6
            step = np.eye(len(trial))[j]*h
            finite = (joint.objective(trial+step)[0]-joint.objective(trial-step)[0])/(2*h)
            self.assertAlmostEqual(finite, value['gradient'][j], delta=2e-7)
        fitted, report = solve(joint, np.empty((0, 7)), np.empty(0), np.empty(0),
            np.r_[np.full(3, -7.5), np.full(4, -np.inf)],
            np.r_[np.full(3, 7.5), np.full(4, np.inf)], polish_stationarity=False)
        self.assertTrue(report['converged'])
        self.assertLess(joint.evaluate(fitted)['value'], 1e-12)
        np.testing.assert_allclose(fitted[:3]*joint.scale, origins, atol=2e-4)
        np.testing.assert_allclose(np.exp(1j*joint.last['band_phase_radians']),
            np.exp(1j*np.asarray(exact['band_phase_radians'])), atol=2e-6)


if __name__ == '__main__':
    unittest.main()
