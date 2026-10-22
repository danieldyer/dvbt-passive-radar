import unittest

import numpy as np

from coherent_arc import CoherentArc
from field import Field
from fit_road_arc_raw import initialise_motion, solve
from flash_geometry import observables
from road_arc import observation, seed_path
from road_arc_geometry import RoadGeometry, road_state
from road_arc_raw import RawRoadGeometry, template, trajectory
from road_geometry import RoadCurve


class RawRoadTests(unittest.TestCase):
    def test_exact_position_velocity_jacobians_on_curved_road(self):
        x = np.linspace(0, 1000., 21)
        road = RoadCurve({'enu_m': np.column_stack([x, 30*np.sin(x/180), 3*np.sin(x/190)]).tolist()})
        state = np.array([432.7, 11.2, -.8, 3.2, 2.4]); t = np.linspace(-1., 1., 31)
        _, _, dp, dv = trajectory(road, state, t)
        for j in range(5):
            d = np.eye(5)[j]*1e-4
            plus, minus = trajectory(road, state+d, t), trajectory(road, state-d, t)
            np.testing.assert_allclose(dp[:, :, j], (plus[0]-minus[0])/2e-4, atol=1e-8)
            np.testing.assert_allclose(dv[:, :, j], (plus[1]-minus[1])/2e-4, atol=1e-8)

    def test_delay_phase_symbol_derivatives_include_both_antenna_legs(self):
        road = RoadCurve({'enu_m': [[-1000., 500., -10.], [0., 500., -10.], [1000., 500., -10.]]})
        state = np.array([1050., 10., -.2, 3., 2.]); time = 50.+np.arange(160)*.001088
        tx, rx = np.array([30000., -3500., 750.]), np.array([.3, -.4, 1.])
        position, velocity, _, _ = trajectory(road, state, time-50.)
        path = np.linalg.norm(position-tx, axis=1)+np.linalg.norm(position-rx, axis=1)-np.linalg.norm(tx-rx)
        coefficient = np.polynomial.polynomial.polyfit(time-50., path, 4)
        seed = {'epoch': 50., 'scale': 1., 'path': coefficient.tolist(), 'rf_hz': 191.625e6}
        pulse = np.sinc(np.arange(-64, 65)*.79)
        raw = {'row_time': time, 'row_information': np.ones(len(time))/8192,
            'row_value': np.ones((len(time), 5), complex), 'offset_samples': np.arange(-2, 3)}
        f = Field(observation(raw, seed, pulse), pulse)
        txmap, rxmap = np.eye(3), np.diag([-.2, .4, .7]); scale = np.array([1., 1., 1., 1., 1.])
        value, jacobian = template(f, road, state, 50., tx, rx, scale, txmap, rxmap, range_origin=3., origin_derivative=True)
        for j in range(9):
            h = 2e-5; direction = np.eye(9)[j]*h
            def evaluate(sign):
                d = sign*direction
                return template(f, road, state+d[:5], 50., tx+txmap@d[5:8], rx+rxmap@d[5:8],
                    scale, txmap, rxmap, range_origin=3.+30*d[-1], origin_derivative=True)[0]
            numeric = (evaluate(1)-evaluate(-1))/(2*h)
            np.testing.assert_allclose(jacobian[:, j], numeric, atol=3e-6, rtol=3e-5)

    def test_training_only_initialisation_escapes_coarse_doppler_sidelobe(self):
        description = {'enu_m': [[-1000., 500., -10.], [0., 500., -10.], [1000., 500., -10.]],
                       'directions': [-1, 1], 'posted_speed_mps': 20.}
        road = RoadCurve(description); truth = np.array([1050., 10., 0., 0., 2.])
        time = 49.+np.arange(1840)*.001088
        tx = np.tile([30000., -3500., 750.], (5, 1)); rx = np.zeros(3)
        target = observables(*road_state(road, truth), tx, rx)
        arc = {'label': 'Synthetic', 'first_s': 49., 'last_s': 51.,
               'initial_road_state': (truth+np.array([0., 1.7, 0., 0., 0.])).tolist(),
               'channels': [{'absolute_observables': row.tolist(),
                             'absolute_observable_covariance': np.eye(3).tolist()} for row in target]}
        g = RoadGeometry([arc], [description], tx, [0]*5, np.eye(3), free_antennas=False)
        position, _, _, _ = trajectory(road, truth, time-50.)
        path = np.linalg.norm(position-tx[0], axis=1)+np.linalg.norm(position, axis=1)-np.linalg.norm(tx[0])
        pulse = np.sinc(np.arange(-64, 65)*.79)
        bands = []
        for rf in [177.5, 184.5, 191.625, 219.5, 226.5]:
            seed = {'epoch': 50., 'scale': 1., 'rf_hz': rf*1e6,
                    'path': np.polynomial.polynomial.polyfit(time-50., path, 4).tolist()}
            raw = {'row_time': time, 'row_information': np.ones(len(time))/8192,
                   'row_value': np.ones((len(time), 5), complex), 'offset_samples': np.arange(-2, 3)}
            field = Field(observation(raw, seed, pulse), pulse)
            h, _ = template(field, road, truth, 50., tx[0], rx, g.scale, np.empty((3, 0)), np.empty((3, 0)))
            field.y = (2*np.exp(.7j)*h).reshape(field.y.shape)
            selected = ((field.time-49.)//.125).astype(int)%2 == 0
            bands.append(CoherentArc(field, selected))
        model = RawRoadGeometry(g, [bands])
        coarse = solve(model)
        initial, starts = initialise_motion(model, model.initial)
        fit = solve(model, initial)
        self.assertGreater(coarse.cost, fit.cost+10.)
        np.testing.assert_allclose(fit.x*g.scale, truth, atol=2e-5)
        self.assertLess(fit.cost, 1e-12)
        # Test rows cannot change either the seeds or their training ranking.
        for b in bands:
            b.field.y[~b.training] *= np.exp(1.3j)
        altered = RawRoadGeometry(g, [[CoherentArc(b.field, b.training) for b in bands]])
        other, _ = initialise_motion(altered, altered.initial)
        np.testing.assert_allclose(other, initial, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
