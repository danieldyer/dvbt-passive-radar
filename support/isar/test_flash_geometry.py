import unittest
import numpy as np
from types import SimpleNamespace
from scipy.interpolate import CubicHermiteSpline

from flash_geometry import Flash, observables
from flash_common_phase import CommonPhase
from flash_siting_validation import phase_origin
from motion import ecef, enu_basis


class FlashGeometryTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([2000., -11000., 12.])
        self.v = np.array([-1.1, 4.8, .01])
        self.a = np.array([.01, -.02, .002])
        self.tx = np.array([[31000., -2000., 740.], [31200., -1780., 700.], [31100., -1900., 730.]])
        self.rx = np.array([10., 14., 40.])

    def test_rates_and_acceleration_against_geometric_finite_differences(self):
        def at(t):
            return observables(self.p+t*self.v+.5*t*t*self.a, self.v+t*self.a, self.a, self.tx, self.rx)
        value = at(0);dt = .01
        np.testing.assert_allclose((at(dt)[:, 0]-at(-dt)[:, 0])/(2*dt), value[:, 1], atol=2e-8)
        np.testing.assert_allclose((at(dt)[:, 1]-at(-dt)[:, 1])/(2*dt), value[:, 2], atol=1e-10)

    def test_common_translation_is_an_exact_single_flash_gauge(self):
        base = observables(self.p, self.v, self.a, self.tx, self.rx)
        for shift in ([30., -20., 10.], [-700., 500., -90.]):
            shift = np.array(shift)
            np.testing.assert_allclose(observables(self.p+shift, self.v, self.a, self.tx+shift, self.rx+shift), base, atol=1e-10)

    def test_unknown_constant_band_phase_absorbs_single_epoch_siting_phase(self):
        k = np.array([3.7, 4.0, 4.7]);gain = np.exp(1j*np.array([.2, -.5, 1.]))
        original = observables(self.p, self.v, self.a, self.tx, self.rx)[:, 0]
        changed = observables(self.p, self.v, self.a, self.tx+[200., -300., 10.], self.rx)[:, 0]
        data = gain*np.exp(-1j*k*original)
        profiled_gain = gain*np.exp(1j*k*(changed-original))
        np.testing.assert_allclose(profiled_gain*np.exp(-1j*k*changed), data, atol=5e-12)

    def test_joint_receiver_transmitter_and_motion_derivatives(self):
        lat, lon = np.radians([-37.85, 144.98])
        rx = ecef(lat, lon, 45.);rotation = enu_basis(lat, lon).T
        times = np.array([0., 60., 120.])
        values = np.column_stack([lat-.001+times*4/6371000, np.full(3, lon-.001)])
        slopes = np.tile([4/6371000, 0], (3, 1))
        track = SimpleNamespace(epoch=0., reference_height_ellipsoid_m=5., curve=CubicHermiteSpline(times, values, slopes))
        transmitters = rx+np.array([[31000+c*80, -2000+c*40, 740.] for c in range(5)])@rotation
        options = [{'transmitter_ecef_m': t, 'site_id': str(c%3)} for c,t in enumerate(transmitters)]
        problem = Flash(np.zeros((5, 3)), np.tile(np.diag([1., .01, .01]), (5, 1, 1)),
                        track, 60., rotation, rx, options)
        p = np.linspace(-.4, .5, problem.count)
        direction = np.linspace(.3, -.6, problem.count)
        derivative = np.imag(problem.residual(p.astype(complex)+1e-20j*direction))/1e-20
        difference = (problem.residual(p+1e-4*direction)-problem.residual(p-1e-4*direction))/2e-4
        np.testing.assert_allclose(derivative, difference, atol=1e-7, rtol=2e-5)
        self.assertLess(problem.ais_representation_error_m, 1e-6)
        fixed_receiver = Flash(np.zeros((5, 3)), np.tile(np.diag([1., .01, .01]), (5, 1, 1)),
            track, 60., rotation, rx, options, receiver_sigma=(0.,0.,7.5),
            receiver_up_bounds_m=(-10.,10.))
        constrained = p.copy();constrained[15:18]=[2.8,-2.7,1.]
        np.testing.assert_array_equal(fixed_receiver.state(constrained)[-1],[0.,0.,7.5])
        low,high=fixed_receiver.bounds()
        self.assertEqual(low[17],-10./7.5);self.assertEqual(high[17],10./7.5)
        exact=fixed_receiver.prediction(constrained)
        fixed_receiver.measured=exact
        fit=fixed_receiver.fit()
        self.assertTrue(-10 <= fixed_receiver.state(fit.x)[-1][2] <= 10)
        np.testing.assert_array_equal(fixed_receiver.state(fit.x)[-1][:2],[0.,0.])

    def test_common_phase_cycles_and_conditional_covariance(self):
        k = np.linspace(3.7, 4.7, 5)
        phase = np.linspace(-.4, .6, 5)
        measured = np.column_stack([np.arange(5.), np.zeros(5), np.zeros(5)])
        def prediction(p):
            return np.column_stack([np.arange(5.)+p[0]*np.arange(1, 6), np.full(5, p[1]), np.zeros(5)])
        base = SimpleNamespace(columns=np.array([1]), measured=measured, whiten=[np.array([[10.]])]*5,
                               prediction=prediction, count=2,
                               bounds=lambda:(np.full(2,-3.),np.full(2,3.)),
                               residual=lambda p:np.r_[np.full(5, 10*p[1]), p])
        cycles = np.array([1, -2, 3, 0, 5])
        problem = CommonPhase(base, phase-2*np.pi*cycles, np.full(5, .04), np.full(5, .01),
                              np.arange(5.), k, cycles)
        p = np.array([.05, .02, .3])
        phase_error = p[-1]-k*(prediction(p[:-1])[:, 0]-np.arange(5.))-phase
        covariance = np.array([[.01, .01], [.01, .04]])
        expected = sum(np.array([p[1], error])@np.linalg.solve(covariance, [p[1], error]) for error in phase_error)
        result = problem.residual(p)
        self.assertAlmostEqual(float(result@result-p[:2]@p[:2]), expected, places=11)
        changed = CommonPhase(base, phase, np.full(5, .04), np.full(5, .01), np.arange(5.), k, np.zeros(5))
        np.testing.assert_allclose(changed.residual(p), result, atol=1e-13)
        # A reversible data deramp changes the observed phase and its reference
        # together. A change of physical antenna prior must do neither.
        shift = np.array([.1, -.3, .6, .9, -.5])
        reframed = CommonPhase(base, phase+k*shift, np.full(5, .04), np.full(5, .01),
                              np.arange(5.)+shift, k, np.zeros(5))
        np.testing.assert_allclose(reframed.residual(p), result, atol=1e-13)
        coherent = CommonPhase(base, np.full(5, .7), np.full(5, .04), np.full(5, .01),
                               np.arange(5.), k, np.zeros(5))
        fit = coherent.fit(np.array([.01, .01, 0.]))
        np.testing.assert_allclose(fit.x, [0., 0., .7], atol=1e-8)
        base.bounds=lambda:(np.array([-.02,-3.]),np.array([.02,3.]))
        bounded = CommonPhase(base,.7-k*.2*np.arange(1,6),np.full(5,.04),np.full(5,.01),
                              np.arange(5.),k,np.zeros(5))
        limited=bounded.fit(np.array([0.,0.,.7]))
        self.assertAlmostEqual(limited.x[0],.02,places=7)

    def test_constant_phase_origin_never_uses_prediction_samples(self):
        rng = np.random.default_rng(19)
        h = rng.standard_normal((20, 5))+1j*rng.standard_normal((20, 5))
        amplitude = np.linspace(.5, 1.2, 20)
        training = np.arange(20)%2 == 0
        y = h*amplitude[:, None]*np.exp(.7j)
        original, _ = phase_origin(h, y, training)
        y[~training] *= np.exp(.9j)
        changed, _ = phase_origin(h, y, training)
        np.testing.assert_allclose(original[training], changed[training], atol=1e-13)
        np.testing.assert_allclose(changed[~training], original[~training]*np.exp(.9j), atol=1e-12)

    def test_full_phase_range_rate_curvature_covariance_matches_joint_likelihood(self):
        rng = np.random.default_rng(31)
        for columns in ([0,1], [0,1,2]):
            dimension = len(columns)
            transform = rng.normal(size=(5,dimension+1,dimension+1))
            covariance = transform@transform.transpose(0,2,1)+.3*np.eye(dimension+1)
            obs_cov = covariance[:,:dimension,:dimension]
            whiten = np.linalg.inv(np.linalg.cholesky(obs_cov))
            measured = rng.normal(size=(5,3))
            direction = rng.normal(size=(5,3))
            def prediction(p):
                return measured+p[0]*direction
            def residual(p):
                return np.r_[np.einsum('nij,nj->ni',whiten,(prediction(p)-measured)[:,columns]).ravel(),p]
            base = SimpleNamespace(columns=np.array(columns), measured=measured, whiten=whiten,
                prediction=prediction, residual=residual)
            nominal = measured[:,0]-.1
            phase = np.linspace(-.2,.3,5)
            k = np.linspace(3.7,4.7,5)
            problem = CommonPhase(base,phase,covariance[:,-1,-1],covariance[:,-1,:dimension],nominal,k,np.zeros(5))
            p = np.array([.07,.4])
            error = np.column_stack([(prediction(p[:-1])-measured)[:,columns],
                p[-1]-k*(prediction(p[:-1])[:,0]-nominal)-phase])
            expected = sum(e@np.linalg.solve(c,e) for e,c in zip(error,covariance))
            result = problem.residual(p)
            self.assertAlmostEqual(float(result@result-p[0]**2),expected,places=12)


if __name__ == '__main__':
    unittest.main()
