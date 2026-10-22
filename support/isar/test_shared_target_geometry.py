import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.append(str(Path(__file__).resolve().parents[1]/'calibration'))
from antenna_motion import Antennas
from complex_likelihood import projected_profile
from geometry_profile import GeometryProfile
from range_fit import Field, Pulse
from road_geometry import RoadCurve
from road_model import RoadProblem, RoadTrajectory
from shared_target_geometry import RoadTarget, SharedGeometry, complex_gain, path_partials
import test_physical_profile as physical_tests


def road_fixture(antennas, *, noise=.01):
    seed = {'epoch': 0., 'scale': 3., 'duration_s': 6., 'rf_hz': 177.5e6}
    road = RoadCurve({'enu_m': [[300., -1000., 0.], [600., -950., 2.],
                              [900., -800., 4.], [1200., -700., 5.]],
                      'directions': [-1, 1], 'posted_speed_mps': 20.})
    sites = np.array([[31000., -3000., 700.]])
    model = RoadTrajectory(seed, road, [450., 30., .2, 0.], [0], np.zeros((1, 3)))
    time = np.linspace(-2.99, 2.99, 100)
    truth = np.sin(np.linspace(0, 2*np.pi, model.count))*.02
    position, jacobian, _ = model.positions(truth, time, sites)
    length, _ = path_partials(position, jacobian, sites[0], np.zeros(3))
    seed['path'] = np.polynomial.polynomial.polyfit(time/3, length, 5).tolist()
    pulse = Pulse(np.sinc(np.arange(-64, 65)*.82)*np.exp(-np.arange(-64, 65)**2/300))
    data = {'time': time, 'offset_samples': np.arange(-2, 3),
            'value': np.zeros((len(time), 5), complex), 'variance': np.full((len(time), 5), .1)}
    fields = [Field(data, pulse, seed, selected=(np.arange(len(time))%2 == parity)) for parity in (0, 1)]
    problem = object.__new__(RoadProblem)
    problem.model, problem.sites, problem.range_origins = model, sites, np.zeros(1)
    problem.motion = np.array([40., 3., 3.])
    target = RoadTarget(problem, fields[:1], fields[1:], truth*.3, sites, antennas)
    rng = np.random.default_rng(7163)
    for field in fields:
        u, _ = target.template(truth, np.zeros(antennas.count), 0, field)
        field.y = (.7-.4j)*u+noise*(rng.normal(size=len(u))+1j*rng.normal(size=len(u)))
    target.energy = float(np.vdot(fields[0].y, fields[0].y).real)
    return target, truth


class SharedTargetGeometryTests(unittest.TestCase):
    def test_path_derivatives_include_both_legs_and_the_direct_reference(self):
        rng = np.random.default_rng(1923)
        position = rng.normal(size=(11, 3))*1000+[1400., -3200., 50.]
        basis = rng.normal(size=(11, 3, 4))
        tx, rx = np.array([32000., -3000., 720.]), np.array([2., -1., 4.])
        _, derivative = path_partials(position, basis, tx, rx)
        for j in range(10):
            delta = np.eye(10)[j]*.01
            plus = path_partials(position+basis@delta[:4], basis, tx+delta[4:7], rx+delta[7:])[0]
            minus = path_partials(position-basis@delta[:4], basis, tx-delta[4:7], rx-delta[7:])[0]
            np.testing.assert_allclose((plus-minus)/.02, derivative[:, j], atol=1e-9, rtol=2e-7)

    def test_road_profile_derivatives_and_legacy_observation_equivalence(self):
        target, truth = road_fixture(Antennas(np.eye(3), [0]))
        p = np.r_[truth*.4, [.002, -.001, .003, -.002, .003, -.001]]
        result = target.evaluate(p[:target.nmotion], p[target.nmotion:])
        for j in range(len(p)):
            step = 2e-5
            delta = np.eye(len(p))[j]*step
            values = [target.evaluate(x[:target.nmotion], x[target.nmotion:])['value'] for x in (p+delta, p-delta)]
            numeric = (values[0]-values[1])/(2*step)
            self.assertAlmostEqual(numeric, result['gradient'][j], delta=2e-5+abs(numeric)*3e-5)
        field = target.training[0]
        u, du = target.template(truth, np.zeros(target.antennas.count), 0, field)
        path, partials = target.problem.model.paths(truth, field.time, np.zeros(len(field.time), int), target.problem.sites)
        original, jacobian = field.template(path, partials)
        np.testing.assert_allclose(u, original, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(du[:, :target.nmotion], jacobian, rtol=1e-12, atol=1e-12)
        old_r, old_j, old_a = projected_profile(field.y, u, du)
        new_r, new_j, new_a, mean = complex_gain(field.y, u, du)
        np.testing.assert_allclose(old_r, np.sqrt(2)*new_r)
        np.testing.assert_allclose(old_j, np.sqrt(2)*new_j)
        self.assertAlmostEqual(old_a, new_a)
        self.assertLess(np.linalg.norm(u.conj()@mean), 1e-10)

    def test_joint_likelihood_counts_one_antenna_prior_without_target_reweighting(self):
        antennas = Antennas(np.eye(3), [0])
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        ship = GeometryProfile([band], truth*.3, antennas)
        road, _ = road_fixture(antennas)
        joint = SharedGeometry(ship, [road])
        p = joint.initial.copy()
        p[joint.antenna_slice] = [.002, -.001, .003, -.002, .003, -.001]
        result = joint.evaluate(p)
        a = ship.evaluate(p[:joint.ship_count])['value']*ship.energy
        b = road.evaluate(p[joint.road_slices[0]], p[joint.antenna_slice])['value']
        self.assertAlmostEqual(result['value']*joint.energy, a+b, places=9)
        indices = [0, ship.nmotion-1, *range(joint.antenna_slice.start, joint.antenna_slice.stop), len(p)-1]
        for j in indices:
            step = 2e-5
            delta = np.eye(len(p))[j]*step
            numeric = (joint.objective(p+delta)[0]-joint.objective(p-delta)[0])/(2*step)
            self.assertAlmostEqual(numeric, result['gradient'][j], delta=3e-6+abs(numeric)*3e-5)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['gauss_newton']).min(), -1e-9)

    def test_nonzero_tx_rx_offsets_recover_from_known_spatially_diverse_motion(self):
        rng = np.random.default_rng(7122)
        position = rng.uniform([-12., -22., -3.], [25., 18., 15.], size=(120, 3))
        tx, rx = np.array([40., -25., 20.]), np.array([-15., 20., 8.])
        truth = np.array([.03, -.02, .01, -.015, .025, -.02])
        basis = np.zeros((len(position), 3, 0))
        path = path_partials(position, basis, tx+truth[:3], rx+truth[3:])[0]
        y = (.8+.3j)*np.exp(-4j*path)
        def residual(p, jacobian=False):
            length, derivative = path_partials(position, basis, tx+p[:3], rx+p[3:])
            u = np.exp(-4j*length)
            error, j, _, _ = complex_gain(y, u, -4j*u[:, None]*derivative)
            return np.r_[j.real, j.imag] if jacobian else np.r_[error.real, error.imag]
        fitted = least_squares(residual, np.zeros(6), jac=lambda p: residual(p, True),
                               xtol=1e-12, ftol=1e-12, gtol=1e-12)
        np.testing.assert_allclose(fitted.x, truth, atol=1e-9)

    def test_noisy_geometry_estimator_matches_conditional_six_coordinate_crlb(self):
        # Isolate the physical geometry/profiled-amplitude operator. The
        # spatial trajectory is known here; this is not a certificate for the
        # recorded ship, unknown road association, or the full autofocus fit.
        rng = np.random.default_rng(71631)
        position = rng.uniform([-12., -22., -3.], [25., 18., 15.], size=(120, 3))
        tx, rx = np.array([40., -25., 20.]), np.array([-15., 20., 8.])
        truth = np.array([.03, -.02, .01, -.015, .025, -.02])
        basis = np.zeros((len(position), 3, 0))
        path, derivative = path_partials(position, basis, tx+truth[:3], rx+truth[3:])
        u, amplitude, sigma = np.exp(-4j*path), .8+.3j, .02
        mean_derivative = -4j*amplitude*u[:, None]*derivative
        mean_derivative -= u[:, None]*(u.conj()@mean_derivative/np.vdot(u, u).real)
        fisher = 2*np.real(mean_derivative.conj().T@mean_derivative)/sigma**2
        factor = np.linalg.cholesky(fisher)
        errors = []
        for _ in range(256):
            y = amplitude*u+sigma/np.sqrt(2)*(rng.normal(size=len(u))+1j*rng.normal(size=len(u)))
            def residual(p, jacobian=False):
                length, derivative = path_partials(position, basis, tx+p[:3], rx+p[3:])
                v = np.exp(-4j*length)
                error, j, _, _ = complex_gain(y, v, -4j*v[:, None]*derivative)
                return np.r_[j.real, j.imag] if jacobian else np.r_[error.real, error.imag]
            fit = least_squares(residual, np.zeros(6), jac=lambda p: residual(p, True),
                                xtol=1e-11, ftol=1e-11, gtol=1e-11)
            self.assertTrue(fit.success)
            errors.append((fit.x-truth)@factor)
        errors = np.asarray(errors)
        self.assertLess(abs(np.mean(errors**2)-1.), .14)
        self.assertLess(np.max(abs(errors.mean(axis=0))), .25)


if __name__ == '__main__':
    unittest.main()
