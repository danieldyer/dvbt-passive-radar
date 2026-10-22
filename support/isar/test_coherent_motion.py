import unittest

import numpy as np
from scipy.optimize import minimize

from antenna_motion import Antennas
from coherent_motion import CoherentMotion
from full_motion import Band
from motion import enu_basis
from spline_motion import SplineMotion
from test_attitude_motion import independent_echo, setup
from test_rigid_motion import fixture


class CoherentMotionTests(unittest.TestCase):
    @staticmethod
    def geometry_fixture():
        points = np.array([[-105., -12., 8.], [15., 14., 35.], [115., -6., 45.], [65., 16., 17.]])
        amplitudes = 100*np.array([.8+.3j, -.5+.4j, .4-.7j, -.3-.2j])
        gains = np.array([.8*np.exp(.7j), 1., .7*np.exp(-1.2j)])
        antennas = Antennas(enu_basis(*np.radians([-37.87, 144.93])).T, [0, 0, 2])
        truth = np.zeros(antennas.count)
        truth[[1, -3]] = [.3, -.2]
        biases = np.array([12., -4., -7.5])
        reference = fixture(3)
        tx, rx = reference.data['transmitter_ecef_m'], reference.data['receiver_ecef_m']
        bands = []
        for c, channel in enumerate((0, 3, 4)):
            field = fixture(channel)
            model, prior = setup(field)
            curve = SplineMotion(model)
            model.pop('rock_spacing_seconds')
            model['parameters'] = curve.parameters[:sum(curve.counts[:2])]
            curve = SplineMotion(model)
            state = curve.apply(field.data['row_time'], prior, tx, rx)
            dtx, drx = antennas.offsets(truth, c)
            field.y = gains[c]*independent_echo(field, state, points, amplitudes, range_bias_m=biases[c],
                transmitter=field.data['transmitter_ecef_m']+dtx, receiver=rx+drx)
            bands.append(Band(field, prior, tx, rx, points, model, np.arange(len(field.time))%2 == 0,
                              range_bias_m=biases[c]))
        joint = CoherentMotion(bands, np.ones(3), curve.parameters, profile_channel_gains=True,
            reference_channel=1, antennas=antennas, fit_range_origins=True)
        parameters = joint.initial.copy()
        parameters[joint.antenna_slice] = truth
        return joint, parameters, gains

    def test_joint_location_and_envelope_gradient_against_independent_echo(self):
        joint, truth, gains = self.geometry_fixture()
        probe = truth.copy()
        probe[:joint.nmotion] *= .7
        probe[joint.antenna_slice] *= .8
        probe[joint.range_slice] += [1., -2., .5]
        _, gradient = joint.objective(probe)
        for index in range(len(probe)):
            delta = np.eye(len(probe))[index]*.0003
            numeric = (joint.objective(probe+delta)[0]-joint.objective(probe-delta)[0])/.0006
            self.assertLess(abs(numeric-gradient[index]), 4e-6, index)
        self.assertLess(joint.objective(truth)[0], 2e-8)
        np.testing.assert_allclose(joint.project(truth)[-1], gains, atol=1e-4)
        information = joint.information(truth)
        np.testing.assert_allclose(information, information.T, atol=1e-5)
        self.assertGreater(np.linalg.eigvalsh(information).min(), -information.max()*1e-12)
        without_point_prior = joint.information(truth, profile_point_coordinates=True)
        self.assertGreater(np.linalg.eigvalsh(information-without_point_prior).min(), -information.max()*1e-10)
        self.assertLess(np.trace(without_point_prior), np.trace(information))
        all_checks = joint.validation(truth)
        for check in all_checks:
            self.assertLess(check['prediction_phase_rms_degrees'], .001)
        opposite_half = [(~b.selected_time)&(b.field.time > np.mean(b.model['interval_s'])) for b in joint.bands]
        limited = joint.validation(truth, test_masks=opposite_half)
        for whole, part in zip(all_checks, limited):
            self.assertLessEqual(part['prediction_error'], whole['prediction_error'])
            self.assertLess(len(part['prediction_phase_by_one_second_block']), len(whole['prediction_phase_by_one_second_block']))
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            joint.validation(truth, test_masks=[b.selected_time for b in joint.bands])

    def test_identifiable_antenna_offsets_recover_with_unknown_gains_and_delays(self):
        joint, truth, _ = self.geometry_fixture()
        # A deliberately identifiable conditional experiment: two physical
        # antenna directions, known motion/support; all gains and delays unknown.
        # Do not claim that all antenna coordinates are observable from one ship.
        active = np.r_[joint.nmotion+1, joint.nmotion+joint.nantenna-3,
                       np.arange(joint.nmotion+joint.nantenna, len(truth))]
        initial = truth.copy()
        initial[active[:2]] = 0
        initial[joint.range_slice] += [3., -2., 4.]
        information = joint.information(truth)[np.ix_(active, active)]/joint.energy
        eigen, vectors = np.linalg.eigh(information)
        self.assertGreater(eigen.min(), eigen.max()*1e-8)
        transform = (vectors/np.sqrt(eigen))@vectors.T
        def evaluate(x):
            p = initial.copy()
            p[active] += transform@x
            value, gradient = joint.objective(p)
            return value, transform.T@gradient[active]
        fit = minimize(evaluate, np.zeros(len(active)), jac=True, method='BFGS',
                       options={'gtol': 1e-8, 'maxiter': 80})
        recovered = initial.copy()
        recovered[active] += transform@fit.x
        self.assertLess(np.max(abs(fit.jac)), 1e-7)
        displacement = joint.antennas.factor@(recovered[joint.antenna_slice]-truth[joint.antenna_slice])
        self.assertLess(abs(displacement).max(), .03)
        self.assertLess(abs(recovered[joint.range_slice]-truth[joint.range_slice]).max(), .003)
        for check in joint.validation(recovered):
            self.assertLess(check['prediction_phase_rms_degrees'], .003)

    def test_shared_amplitude_profile_gradient_and_information(self):
        points = np.array([[-105., -12., 8.], [15., 14., 35.], [115., -6., 45.]])
        amplitudes = np.array([.8+.3j, -.5+.4j, .4-.7j])
        gains = np.array([.8*np.exp(.7j), 1., .7*np.exp(-1.2j)])
        biases = [12., 0., -7.5]
        bands = []
        reference = fixture(3)
        tx, rx = reference.data['transmitter_ecef_m'], reference.data['receiver_ecef_m']
        for channel, gain, bias in zip((0, 3, 4), gains, biases):
            field = fixture(channel)
            model, prior = setup(field)
            curve = SplineMotion(model)
            state = curve.apply(field.data['row_time'], prior, tx, rx)
            field.y = gain*independent_echo(field, state, points, amplitudes, range_bias_m=bias)
            bands.append(Band(field, prior, tx, rx, points, model, np.arange(len(field.time))%2 == 0,
                              range_bias_m=bias))
        joint = CoherentMotion(bands, gains, curve.parameters)
        p = joint.initial*.5
        _, gradient = joint.objective(p)
        for i in range(len(p)):
            delta = np.eye(len(p))[i]*.0002
            numeric = (joint.objective(p+delta)[0]-joint.objective(p-delta)[0])/.0004
            self.assertLess(abs(numeric-gradient[i]), 2e-6)
        self.assertLess(joint.objective(joint.initial)[0], 2e-8)
        information = joint.information(joint.initial)
        np.testing.assert_allclose(information, information.T, atol=1e-7)
        self.assertGreater(np.linalg.eigvalsh(information).min(), -1e-7)
        for validation in joint.validation(joint.initial):
            self.assertLess(validation['prediction_phase_rms_degrees'], .001)
        profiled = CoherentMotion(bands, np.ones(3), curve.parameters,
                                  profile_channel_gains=True, reference_channel=1)
        _, gradient = profiled.objective(p)
        # Selected directions span path, yaw, roll and pitch. The derivative must
        # include neither an omitted gain fit nor a derivative through its optimum.
        for i in np.linspace(0, len(p)-1, 8).astype(int):
            delta = np.eye(len(p))[i]*.0003
            numeric = (profiled.objective(p+delta)[0]-profiled.objective(p-delta)[0])/.0006
            self.assertLess(abs(numeric-gradient[i]), 3e-6)
        np.testing.assert_allclose(profiled.project(profiled.initial)[-1], gains, atol=1e-4)
        reduced = profiled.information(profiled.initial)
        same_gains = profiled.project(profiled.initial)[-1]
        fixed = CoherentMotion(bands, same_gains, curve.parameters)
        # Compare at identical nuisance values, not truth versus the tiny
        # interpolation-biased estimates obtained from the independent generator.
        self.assertGreater(np.linalg.eigvalsh(fixed.information(fixed.initial)-reduced).min(), -1e-5)


if __name__ == '__main__':
    unittest.main()
