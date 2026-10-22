import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from field import FS, OFFSETS, average
from motion import ecef, enu_basis
from spherical_sea import reflected_leg, horizontal_reflection, geometry, components
from test_rigid_motion import fixture


class SphericalSeaTests(unittest.TestCase):
    def test_four_observed_rays_against_fourier_and_symbol_quadrature(self):
        field = fixture()
        d = field.data
        state = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
            'body_to_ecef': d['body_to_ecef'],
            'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
        lat, lon = np.radians([-37.87, 144.93])
        radius = 6371000.
        centre = ecef(lat, lon, 0.)-radius*enu_basis(lat, lon)[:, 2]
        point = np.array([60., -12., 25.])
        bias = 3.2
        actual, paths = components(field, state, point, centre, radius, relative_permittivity=80.,
            conductivity_s_per_m=4., range_bias_m=bias)
        # Geometry and endpoint Doppler are independently checked below. Here
        # synthesize each resulting ray through Fourier pulse and quadrature
        # symbol moments, without Field.path_response or its moment formula.
        frequency = np.fft.fftfreq(256)
        power = (abs(frequency) < .38)*(1+.3*np.sin(9*frequency)); power /= power.sum()
        u, weight = leggauss(32); u, weight = u/2, weight/2
        dielectric = 80.-1j*4./(2*np.pi*field.metadata['rf_hz']*8.8541878128e-12)
        gains = []
        for ray in paths['reflected_legs']:
            sine = ray['sin_grazing']
            root = np.sqrt(dielectric-(1-sine*sine))
            gains.append((sine-root)/(sine+root))
        gt, gr = gains
        for j, gain in enumerate([1., gt, gr, gt*gr]):
            delta = paths['excess_path_m'][j]-d['centre_path_m']
            lag = OFFSETS[None, :]-(delta[:, None]+bias)*FS/299792458.
            pulse = (np.exp(2j*np.pi*lag.reshape(-1, 1)*frequency)@power).reshape(lag.shape)
            tone = np.exp(-1j*field.k*.001024*paths['path_rate_mps'][j, :, None]*u)
            moments = tone@weight-1j*d['centre_omega']*(tone@(weight*u))
            raw = pulse*np.exp(-1j*field.k*delta[:, None])*(gain*paths['relative_spreading'][j]*moments)[:, None]
            expected = field.apply(average(raw, d['row_information'], field.metadata['block_rows'])[0])
            self.assertLess(np.linalg.norm(actual[j]-expected)/np.linalg.norm(expected), 3e-4)
        np.testing.assert_allclose(actual[0], field.point(point, motion=state, range_bias_m=bias), atol=2e-7)

    def test_specular_lengths_against_independent_sag_coordinates_and_reflection_law(self):
        radius = 6371000.
        centre = np.array([0., 0., -radius])
        distances = np.array([1000., 7000., 15000., 35000.])
        height = np.array([3., 10., 35., 60.])
        position = np.c_[distances, np.zeros(4), np.sqrt((radius+height)**2-distances**2)-radius]
        antenna = np.array([0., 0., 760.])
        velocity = np.broadcast_to([2., 4., .03], position.shape)
        result = reflected_leg(position, velocity, antenna, centre, radius)
        for i, p in enumerate(position):
            def surface(x):
                root = np.sqrt(radius**2-x*x)
                return np.array([x, 0., -x*x/(radius+root)]), np.array([1., 0., -x/root])
            def derivative(x):
                s, tangent = surface(x)
                return sum(np.dot((s-end)/np.linalg.norm(s-end), tangent) for end in (antenna, p))
            x = brentq(derivative, 0., p[0], xtol=1e-9)
            s = surface(x)[0]
            expected = np.linalg.norm(s-antenna)+np.linalg.norm(s-p)
            self.assertAlmostEqual(result['length_m'][i], expected, delta=2e-8)
            np.testing.assert_allclose(result['specular_ecef_m'][i], s, atol=2e-6)
        normal = (result['specular_ecef_m']-centre)/radius
        incoming = antenna-result['specular_ecef_m']; incoming /= np.linalg.norm(incoming, axis=1)[:, None]
        outgoing = position-result['specular_ecef_m']; outgoing /= np.linalg.norm(outgoing, axis=1)[:, None]
        tangent_sum = incoming+outgoing-2*result['sin_grazing'][:, None]*normal
        self.assertLess(abs(tangent_sum).max(), 1e-9)
        self.assertTrue(np.all((result['divergence'] > 0)&(result['divergence'] < 1)))

    def test_doppler_and_common_baseline_under_spherical_reflection(self):
        radius = 6371000.; centre = np.array([0., 0., -radius])
        time = np.linspace(0., 20., 11)
        velocity = np.broadcast_to([2., 4., .01], (11, 3))
        position = np.array([14000., 3000., 12.])+time[:, None]*velocity
        tx, rx = np.array([0., 0., 760.]), np.array([5000., -2000., 45.])
        result = geometry(position, velocity, tx, rx, centre, radius)
        dt = .01
        plus = geometry(position+dt*velocity, velocity, tx, rx, centre, radius)['excess_path_m']
        minus = geometry(position-dt*velocity, velocity, tx, rx, centre, radius)['excess_path_m']
        np.testing.assert_allclose(result['path_rate_mps'], (plus-minus)/(2*dt), atol=5e-7)
        self.assertEqual(result['physical_baseline_m'], np.linalg.norm(tx-rx))
        self.assertTrue(np.all(result['excess_path_m'][1:] > result['excess_path_m'][0]))

    def test_fresnel_limits_and_passivity_with_declared_time_convention(self):
        sine = np.array([1e-7, .001, .01, .1, 1.])
        value = horizontal_reflection(sine, 200e6, relative_permittivity=80., conductivity_s_per_m=4.)
        self.assertTrue(np.all(abs(value) <= 1))
        self.assertTrue(np.all(value.imag >= 0))
        self.assertLess(abs(value[0]+1), 2e-8)
        normal = horizontal_reflection(np.array([1.]), 200e6, relative_permittivity=4., conductivity_s_per_m=0.)
        np.testing.assert_allclose(normal, -1/3, atol=1e-15)
        with self.assertRaises(ValueError):
            reflected_leg(np.array([[100000., 0., np.sqrt(6371002.**2-100000.**2)-6371000.]]),
                np.zeros((1, 3)), np.array([0., 0., 2.]),
                np.array([0., 0., -6371000.]), 6371000.)


if __name__ == '__main__':
    unittest.main()
