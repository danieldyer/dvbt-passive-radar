import unittest
import numpy as np

from geometry import ecef, enu_rotation
from rebase_receiver import rebase


class ReceiverGeometryTests(unittest.TestCase):
    def test_receiver_change_preserves_absolute_transmitter_positions(self):
        old = ecef(-37.810752, 144.987337, 48.45)
        old_rotation = enu_rotation(-37.810752, 144.987337)
        receiver = ecef(-37.81040454194318, 144.98744506389454, 43.)
        rotation = enu_rotation(-37.81040454194318, 144.98744506389454)
        positions = np.array([[31600., -3020., 640.], [31700., -2800., 610.], [31800., -2740., 608.]])
        offsets = np.array([[.02, -.002, 4.], [0, 0, 0], [.01, -.001, 3.], [.01, -.001, 3.], [0, 0, 0]])
        sites = [0, 0, 0, 0, 2]
        prior = {'receiver_ecef_m': old.tolist(), 'ecef_to_enu': old_rotation.tolist(),
                 'receiver': {}, 'enu_transmitters_m': positions.tolist()}
        main = {'hypothesis': 'main', 'channel_sites': sites, 'channels': [],
                'site_positions_enu_m': positions.tolist(), 'channel_offsets_enu_m': offsets.tolist(),
                'targets': [{'obsolete_fit': True}]}
        changed, centres = rebase(prior, main, receiver, rotation, {'antenna_agl_m': 10.})
        absolute = receiver+(np.array(centres['site_positions_enu_m'])[sites]+
                             np.array(centres['channel_offsets_enu_m']))@rotation
        np.testing.assert_allclose(absolute, old+(positions[sites]+offsets)@old_rotation, rtol=0, atol=2e-9)
        np.testing.assert_allclose(receiver+np.array(changed['enu_transmitters_m'])@rotation,
                                   old+positions@old_rotation, rtol=0, atol=2e-9)
        self.assertNotIn('targets', centres)
        np.testing.assert_array_equal(prior['receiver_ecef_m'], old)
        self.assertEqual(changed['receiver']['antenna_agl_m'], 10.)


if __name__ == '__main__':
    unittest.main()
