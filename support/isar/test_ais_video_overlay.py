import unittest

import numpy as np

from ais_video_overlay import following_bounds, pixels
from geographic_movie import frame_transform, project_power


class AisOverlayTests(unittest.TestCase):
    def test_fixed_camera_cardinal_axes_and_pixel_centres(self):
        xy = pixels([[100., 250.], [110., 250.], [100., 260.]],
                    [50., 150., 200., 300.], (200, 100))
        np.testing.assert_allclose(xy, [[99.5, 49.5], [119.5, 49.5], [99.5, 39.5]])

    def test_moving_camera_translates_but_does_not_rotate_ais(self):
        track = np.array([[200., 300.], [240., 330.]])
        a = pixels(track, following_bounds([200., 300.], [800, 800]), (800, 800))
        b = pixels(track, following_bounds([220., 270.], [800, 800]), (800, 800))
        np.testing.assert_allclose(b-a, np.tile([-20., -30.], (2, 1)))
        np.testing.assert_allclose(b[1]-b[0], [40., -30.])

    def test_overlay_and_rotating_body_image_use_identical_world_coordinates(self):
        along = port = np.arange(-50., 51.)
        x, y = np.meshgrid(along, port)
        bounds, size = [-10., 10., -10., 10.], (20, 20)
        east, north = np.meshgrid(np.arange(-9.5, 10), np.arange(9.5, -10, -1))
        expected_pixels = pixels(np.c_[east.ravel(), north.ravel()], bounds, size)
        np.testing.assert_allclose(expected_pixels, np.c_[np.tile(np.arange(20), 20), np.repeat(np.arange(20), 20)])
        # A fixed geographic ramp is unchanged for every body attitude.
        for angle in (0., .7, -1.1):
            body = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
            matrix, shift = frame_transform(np.array([3., -2., 0.]), body, np.zeros(3), np.eye(3))
            power = 100+2*(matrix[0, 0]*x+matrix[0, 1]*y+shift[0])+3*(matrix[1, 0]*x+matrix[1, 1]*y+shift[1])
            actual = project_power(power, along, port, bounds, matrix, shift, size)
            np.testing.assert_allclose(actual, 100+2*east+3*north, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
