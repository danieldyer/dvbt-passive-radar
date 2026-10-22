import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from field import FS, OFFSETS, average
from sea_paths import geometry, point
from test_rigid_motion import fixture


def setup():
    field = fixture()
    d = field.data
    state = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
        'body_to_ecef': d['body_to_ecef'],
        'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
    # Arbitrary fixed plane below every physical antenna and point in the fixture.
    normal = np.array([0., 0., 1.])
    level = min(d['transmitter_ecef_m'][2], d['receiver_ecef_m'][2], d['centre_ecef_m'][:, 2].min())-20.
    return field, state, np.array([0., 0., level]), normal


class SeaPathTests(unittest.TestCase):
    def test_common_reference_baseline_and_doppler_from_independent_time_difference(self):
        time = np.linspace(0., 30., 31)
        tx, rx = np.array([32000., 2000., 760.]), np.array([100., -16000., 50.])
        p0, v = np.array([0., 0., 24.]), np.array([2., 4., .03])
        p = p0+time[:, None]*v
        result = geometry(p, np.broadcast_to(v, p.shape), tx, rx, np.zeros(3), np.array([0., 0., 1.]))
        self.assertEqual(result['physical_baseline_m'], np.linalg.norm(tx-rx))
        self.assertTrue(np.all(result['excess_path_m'][1:] > result['excess_path_m'][0]))
        dt = .001
        def independent(t):
            position = p0+t[:, None]*v
            paths = []
            for tx_sign, rx_sign in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                a, b = tx*[1., 1., tx_sign], rx*[1., 1., rx_sign]
                paths.append(np.sqrt(np.sum((position-a)**2, axis=1))+
                    np.sqrt(np.sum((position-b)**2, axis=1))-np.sqrt(np.sum((tx-rx)**2)))
            return np.array(paths)
        np.testing.assert_allclose(result['excess_path_m'], independent(time), atol=1e-10)
        np.testing.assert_allclose(result['path_rate_mps'], (independent(time+dt)-independent(time-dt))/(2*dt), atol=1e-8)
        # Subtracting image-antenna baselines separately would make these wrong.
        self.assertGreater(abs(np.linalg.norm(tx*[1, 1, -1]-rx)-np.linalg.norm(tx-rx)), 1.)

    def test_direct_limit_and_four_ray_echo_against_independent_fourier_quadrature(self):
        field, state, origin, normal = setup()
        location, bias = np.array([40., -12., 30.]), 3.2
        direct, _ = point(field, state, location, origin, normal, tx_reflection=0., rx_reflection=0., range_bias_m=bias)
        np.testing.assert_allclose(direct, field.point(location, motion=state, range_bias_m=bias), atol=1e-8)
        gt, gr = -.91+.03j, -.87-.02j
        actual, _ = point(field, state, location, origin, normal, tx_reflection=gt, rx_reflection=gr, range_bias_m=bias)
        d = field.data
        position = state['position_ecef_m']+state['body_to_ecef']@location
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@location
        tx, rx = d['transmitter_ecef_m'].copy(), d['receiver_ecef_m'].copy()
        ti, ri = tx.copy(), rx.copy()
        ti[2], ri[2] = 2*origin[2]-tx[2], 2*origin[2]-rx[2]
        frequency = np.fft.fftfreq(256)
        power = (abs(frequency) < .38)*(1+.3*np.sin(9*frequency)); power /= power.sum()
        u, weights = leggauss(32); u, weights = u/2, weights/2
        raw = np.zeros((len(position), len(OFFSETS)), complex)
        direct_product = np.linalg.norm(position-tx, axis=1)*np.linalg.norm(position-rx, axis=1)
        for a, b, gain in ((tx, rx, 1.), (ti, rx, gt), (tx, ri, gr), (ti, ri, gt*gr)):
            ra, rb = np.sqrt(np.sum((position-a)**2, axis=1)), np.sqrt(np.sum((position-b)**2, axis=1))
            delta = ra+rb-np.sqrt(np.sum((tx-rx)**2))-d['centre_path_m']
            rate = np.sum((position-a)*velocity, axis=1)/ra+np.sum((position-b)*velocity, axis=1)/rb
            lag = OFFSETS[None, :]-(delta[:, None]+bias)*FS/299792458.
            pulse = (np.exp(2j*np.pi*lag.reshape(-1, 1)*frequency)@power).reshape(lag.shape)
            tone = np.exp(-1j*field.k*.001024*rate[:, None]*u)
            moments = tone@weights-1j*d['centre_omega']*(tone@(weights*u))
            raw += gain*(direct_product/(ra*rb))[:, None]*pulse*np.exp(-1j*field.k*delta[:, None])*moments[:, None]
        expected = field.apply(average(raw, d['row_information'], field.metadata['block_rows'])[0])
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), 3e-4)

    def test_specular_plane_zero_height_limit_and_normal_translation(self):
        tx, rx = np.array([20., 100., 70.]), np.array([500., -10., 40.])
        p = np.array([[10., 20., 0.], [12., 23., 0.]])
        v = np.array([[2., 3., 0.], [2., 3., 0.]])
        normal, origin = np.array([0., 0., 1.]), np.zeros(3)
        paths = geometry(p, v, tx, rx, origin, normal)
        for key in ('excess_path_m', 'path_rate_mps', 'relative_spreading'):
            np.testing.assert_allclose(paths[key], np.broadcast_to(paths[key][0], paths[key].shape), atol=1e-12)
        # For unit negative reflection the coherent 1,-1,-1,+1 sum vanishes.
        k = 4.2
        response = np.exp(-1j*k*paths['excess_path_m'])*paths['relative_spreading']
        np.testing.assert_allclose(np.array([1., -1., -1., 1.])@response, 0., atol=1e-12)
        offset = np.array([-4e6, 3e6, -3.8e6])
        shifted = geometry(p+offset, v, tx+offset, rx+offset, origin+offset, normal)
        for key in paths:
            np.testing.assert_allclose(shifted[key], paths[key], atol=1e-9)


if __name__ == '__main__':
    unittest.main()
