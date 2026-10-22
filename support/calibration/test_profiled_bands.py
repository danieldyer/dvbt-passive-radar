import unittest
import numpy as np

from profiled_bands import common_phase, profile_bands, ProfiledBandProblem, MissingReferenceBand
from coherent_fit import CoherentProblem
import test_scatterer_validation as rigid_fixture
from geometry import central_jacobian
from road_model import RoadProblem
from antenna_offsets import AntennaProblem


class PhaseProjectionTest(unittest.TestCase):
    def test_zero_reference_band_keeps_the_valid_reduced_likelihood(self):
        fixture = rigid_fixture.RigidModelTest()
        fixture.setUp()
        pr = fixture.problem
        channels = [{'seed': field.seed, 'value': field.data['value'][:, 4],
                     'operator': field.q, 'time': field.time, 'k': field.k} for field in pr.fields]
        problem = RoadProblem(channels, {'t': np.zeros(1)}, pr.sites, pr.fields, pr.model)
        joint = CoherentProblem([problem])
        reduced = ProfiledBandProblem(joint)
        state = np.r_[fixture.state[-9:], fixture.state[:15], np.zeros(4)]
        for channel, field in enumerate(pr.fields):
            u, _ = reduced.channel_model(state, 0, channel)
            field.y = u*(0 if channel == 1 else 1)
        residual, amplitude = reduced.target(state, 0)
        np.testing.assert_allclose(residual, 0, atol=1e-12)
        self.assertEqual(amplitude[1], 0)
        with self.assertRaises(MissingReferenceBand):
            reduced.expanded_state(state)

    def test_exact_global_phase_compared_with_dense_search(self):
        rng = np.random.default_rng(1256)
        angles = np.linspace(0, 2*np.pi, 50000, endpoint=False)
        rotations = np.exp(-1j*angles[:, None])
        for _ in range(20):
            z = rng.normal(size=5)+1j*rng.normal(size=5)
            phase, amplitude = common_phase(z)
            dense = np.max(np.sum(np.maximum(np.real(rotations*z), 0)**2, axis=1))
            exact = np.sum(amplitude**2)
            self.assertGreaterEqual(exact+1e-12, dense)
            self.assertLess(exact-dense, 1e-7)
            np.testing.assert_allclose(amplitude, np.maximum(np.real(z*np.exp(-1j*phase)), 0), atol=1e-15)

    def test_noiseless_positive_band_response_is_preserved(self):
        rng = np.random.default_rng(918)
        templates = [rng.normal(size=30)+1j*rng.normal(size=30) for _ in range(5)]
        expected = np.array([.1, 2, .4, 1.7, .8])*np.exp(2.7j)
        data = [u*a for u, a in zip(templates, expected)]
        residual, actual = profile_bands(data, templates)
        np.testing.assert_allclose(residual, 0, atol=2e-14)
        np.testing.assert_allclose(actual, expected, atol=2e-15)

    def test_profiled_derivative_includes_phase_and_positive_gains(self):
        rng = np.random.default_rng(541)
        templates = [rng.normal(size=41)+1j*rng.normal(size=41) for _ in range(5)]
        derivatives = [rng.normal(size=(41, 7))+1j*rng.normal(size=(41, 7)) for _ in range(5)]
        data = [u*a+rng.normal(size=41) for u, a in zip(templates, [1+.3j, .4+.2j, -.7j, -2, .2j])]
        _, j, a = profile_bands(data, templates, derivatives)
        self.assertTrue(np.any(abs(a) == 0))
        h = 1e-5
        for column in range(7):
            plus = [u+h*du[:, column] for u, du in zip(templates, derivatives)]
            minus = [u-h*du[:, column] for u, du in zip(templates, derivatives)]
            numeric = (profile_bands(data, plus)[0]-profile_bands(data, minus)[0])/(2*h)
            np.testing.assert_allclose(j[:, column], numeric, rtol=2e-7, atol=3e-9)

    def test_same_physical_likelihood_after_gain_elimination(self):
        fixture = rigid_fixture.RigidModelTest()
        fixture.setUp()
        p = fixture.problem
        channels = [{'seed': field.seed, 'value': field.data['value'][:, 4],
                     'operator': field.q, 'time': field.time, 'k': field.k} for field in p.fields]
        problem = RoadProblem(channels, {'t': np.zeros(1)}, p.sites, p.fields, p.model)
        for ch, field in enumerate(p.fields):
            u, _ = problem.channel_model(fixture.state, ch, True)
            field.y = (.4+.3j)*u*(1+.2*ch)+.01*field.y
        joint = CoherentProblem([problem])
        trial = joint.initial([fixture.state[:p.model.count]], fixture.state[-9:])
        reduced = ProfiledBandProblem(joint)
        state, missing = reduced.expanded_state(trial[:reduced.count])
        r = reduced.residual(trial[:reduced.count])
        np.testing.assert_allclose(r, joint.residual(state), atol=3e-14)
        original = joint.residual(trial)
        self.assertLessEqual(r@r, original@original+1e-10)
        actual = reduced.residual(trial[:reduced.count], True, include_motion=False)
        numeric = central_jacobian(lambda x: reduced.residual(x, include_motion=False),
                                   trial[:reduced.count], 1e-4)
        self.assertLess(np.linalg.norm(actual-numeric)/np.linalg.norm(actual), 3e-7)

        antenna = AntennaProblem(joint)
        moved = antenna.initial(trial)
        # Moving a common site correction into its individual channel centres
        # must preserve every physical path, carrier phase and range template.
        np.testing.assert_allclose(antenna.residual(moved, include_motion=False, include_site_prior=False),
            reduced.residual(trial[:reduced.count], include_motion=False, include_site_prior=False), atol=2e-12)
        actual = antenna.residual(moved, True, include_motion=False)
        numeric = central_jacobian(lambda x:antenna.residual(x, include_motion=False), moved, 1e-4)
        self.assertLess(np.linalg.norm(actual-numeric)/np.linalg.norm(actual), 5e-7)
        np.testing.assert_array_equal(problem.model.channel_offsets, np.zeros((5, 3)))

    def test_correlated_tower_prior_does_not_count_site_survey_five_times(self):
        fixture = rigid_fixture.RigidModelTest()
        fixture.setUp()
        pr = fixture.problem
        pr.model.count = 15
        pr.channel_data = lambda ch:pr.fields[ch].y
        joint = CoherentProblem([pr])
        antenna = AntennaProblem(joint, (2., 3., 4.))
        covariance = antenna.prior_covariance_m2
        assignment = pr.model.channel_sites
        for a in range(5):
            for b in range(5):
                expected = np.eye(3)*(100 if assignment[a] == assignment[b] else 0)
                if a == b:
                    expected += np.diag([4., 9., 16.])
                np.testing.assert_array_equal(covariance[3*a:3*a+3, 3*b:3*b+3], expected)
        np.testing.assert_allclose(antenna.whiten@(covariance/100)@antenna.whiten.T, np.eye(15), atol=3e-15)

    def test_antenna_holdout_restores_licensed_prior_without_mutating_fitted_model(self):
        fixture = rigid_fixture.RigidModelTest()
        fixture.setUp()
        pr = fixture.problem
        pr.model.count = 15
        pr.channel_data = lambda ch:pr.fields[ch].y
        nominal = pr.model.channel_offsets.copy()
        fitted = nominal+np.arange(15).reshape(5, 3)
        pr.model.channel_offsets = fitted.copy()
        joint = CoherentProblem([pr])
        antenna = AntennaProblem(joint, nominal_offsets=nominal)
        np.testing.assert_array_equal(antenna.original_offsets, nominal)
        np.testing.assert_array_equal(antenna.joint.problems[0].model.channel_offsets, nominal)
        np.testing.assert_array_equal(joint.problems[0].model.channel_offsets, fitted)
        initial = antenna.initial(np.zeros(joint.count))
        np.testing.assert_array_equal(initial[antenna.offset:], np.zeros(15))

    def test_shared_range_origin_preserves_phase_gauge_and_derivatives(self):
        fixture = rigid_fixture.RigidModelTest()
        fixture.setUp()
        pr = fixture.problem
        channels = [{'seed': f.seed, 'value': f.data['value'][:, 4],
                     'operator': f.q, 'time': f.time, 'k': f.k} for f in pr.fields]
        problem = RoadProblem(channels, {'t': np.zeros(1)}, pr.sites, pr.fields, pr.model)
        biases = np.array([2., 0., -3., -16., -25.])
        for ch, field in enumerate(pr.fields):
            u, _ = problem.channel_model(fixture.state, ch, True, range_bias=biases[ch])
            field.y = (.4+.3j)*(1+.2*ch)*u
        joint = CoherentProblem([problem], range_bias=True)
        original = joint.initial([fixture.state[:pr.model.count]], fixture.state[-9:])
        original[joint.bias_offset:] = biases[[0, 2, 3, 4]]
        original[joint.beta_offset:joint.magnitude_offset] = 0
        reduced = ProfiledBandProblem(joint)
        p = reduced.initial(original)
        state, missing = reduced.expanded_state(p)
        self.assertEqual(len(state), joint.count)
        np.testing.assert_array_equal(state[joint.bias_offset:], biases[[0, 2, 3, 4]])
        np.testing.assert_allclose(reduced.residual(p, include_motion=False, include_site_prior=False), 0, atol=3e-14)
        np.testing.assert_allclose(reduced.residual(p), joint.residual(state), atol=3e-14)
        actual = reduced.residual(p, True, include_motion=False)
        numeric = central_jacobian(lambda x:reduced.residual(x, include_motion=False), p, 1e-4)
        self.assertLess(np.linalg.norm(actual-numeric)/np.linalg.norm(actual), 5e-7)
        self.assertEqual(len(reduced.bounds()[0]), reduced.count)
        # A per-band constant envelope delay has its carrier phase absorbed
        # into beta; it must not silently change the RF path-phase convention.
        for ch, field in enumerate(pr.fields):
            path = problem.model.paths(fixture.state, field.time,
                np.full(len(field.time), ch), pr.sites, True)[0]
            shifted = field.template(path+biases[ch])[0]
            offset = field.template(path, range_bias=biases[ch])[0]
            np.testing.assert_allclose(shifted*np.exp(1j*field.k*biases[ch]), offset, atol=2e-12)

    def test_short_phase_windows_keep_the_original_deramp_and_rf_origin(self):
        from time_band_phase import block_field
        fixture=rigid_fixture.RigidModelTest();fixture.setUp()
        field=fixture.problem.fields[4]
        path=fixture.problem.model.paths(fixture.state,field.time,np.full(len(field.time),4),
                                        fixture.problem.sites,True)[0]
        lag=field.offsets[None,:]-(path-field.seed_path)[:,None]*8e6/299792458.
        phase=np.exp(-1j*field.k*(path-(field.seed_path-field.seed['path'][0])))
        amplitude=.3+.7j
        field.data={**field.data,'value':amplitude*phase[:,None]*field.pulse(lag)}
        original=field.q.copy()
        sliced=block_field(field,44.,47.)
        keep=(field.time>=44)&(field.time<47)
        u,_=sliced.template(path[keep])
        np.testing.assert_allclose(sliced.y,amplitude*u,atol=3e-14)
        self.assertEqual(sliced.seed,field.seed)
        np.testing.assert_array_equal(sliced.seed_path,field.seed_path[keep])
        np.testing.assert_array_equal(field.q,original)
        self.assertTrue(np.all((sliced.time>=44)&(sliced.time<47)))


if __name__ == '__main__':
    unittest.main()
