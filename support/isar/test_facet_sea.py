import unittest

import numpy as np

from facet_sea import response
from motion import C
from specular_facet import hh_current, response as direct_response, unit
from spherical_sea import geometry, horizontal_reflection


class SeaFacetTests(unittest.TestCase):
    def setUp(self):
        self.position = np.array([[0., 0., 25.]])
        self.tx, self.rx = np.array([37000., 3800., 700.]), np.array([5300., 6800., 31.])
        bisector = unit(self.tx-self.position[0])+unit(self.rx-self.position[0])
        angle = np.arctan2(bisector[0], bisector[1])-np.pi/2
        self.along = np.array([np.sin(angle), np.cos(angle), 0.])
        self.up = np.array([0., 0., 1.])
        self.sphere = np.array([0., 0., -6371000.])
        self.radius = 6371000.
        self.frequency = np.array([177.5e6, 226.5e6])

    def evaluate(self, length=286., height=30., **kwargs):
        return response(self.position, self.tx, self.rx, self.along, self.up,
                        length, height, self.frequency, self.sphere, self.radius, **kwargs)

    def test_direct_component_is_existing_exact_spherical_aperture(self):
        found = self.evaluate()[0]
        expected = direct_response(self.position, self.tx, self.rx, self.along, self.up,
                                   286., 30., self.frequency)['relative_field']
        np.testing.assert_allclose(found, expected, atol=5e-15)

    def test_point_limit_matches_independent_four_ray_model_and_current(self):
        paths = geometry(self.position, np.zeros((1, 3)), self.tx, self.rx, self.sphere, self.radius)
        normal = np.cross(self.along, self.up)
        centre_current = hh_current(unit(self.position-self.tx), unit(self.rx-self.position), normal, self.up)
        expected = []
        for index, choice in enumerate(((False, False), (True, False), (False, True), (True, True))):
            factor = np.ones((1, len(self.frequency)), complex)
            directions = []
            for leg, (antenna, reflected) in enumerate(zip((self.tx, self.rx), choice)):
                ray = paths['reflected_legs'][leg]
                endpoint = ray['specular_ecef_m'] if reflected else antenna
                directions.append(unit(self.position-endpoint))
                if reflected:
                    factor *= np.array([horizontal_reflection(ray['sin_grazing'], f,
                        relative_permittivity=80., conductivity_s_per_m=4.) for f in self.frequency]).T
            factor *= (hh_current(directions[0], -directions[1], normal, self.up)/centre_current)[:, None]
            factor *= paths['relative_spreading'][index, :, None]
            delay = paths['excess_path_m'][index]-paths['excess_path_m'][0]
            expected.append(factor*np.exp(-2j*np.pi*self.frequency/C*delay[:, None]))
        np.testing.assert_allclose(self.evaluate(.001, .001, order=(4, 4)), expected, atol=3e-8)

    def test_quadrature_and_coordinate_invariance(self):
        first = self.evaluate()
        np.testing.assert_allclose(first, self.evaluate(order=(96, 24)), atol=1e-8)
        rotation = np.linalg.qr(np.array([[1., 2., 3.], [4., 1., 6.], [7., 8., 2.]]))[0]
        shift = np.array([4e6, -3e6, 5e6])
        rotated = response(self.position@rotation+shift, self.tx@rotation+shift, self.rx@rotation+shift,
            self.along@rotation, self.up@rotation, 286., 30., self.frequency,
            self.sphere@rotation+shift, self.radius, up=self.up@rotation)
        np.testing.assert_allclose(first, rotated, atol=2e-8)


if __name__ == '__main__':
    unittest.main()
