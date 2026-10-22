from copy import deepcopy
import unittest

import numpy as np

from common_physical_arc import CommonPhysicalFit
from physical_amplitude_arc import PhysicalAmplitudeArc, PhysicalAmplitudeFit
from ship_calibration_transfer import frozen_coordinates
from test_common_physical_arc import fixture


class CalibrationTransferTests(unittest.TestCase):
    def test_frozen_enu_displacements_preserve_coordinate_frame_and_input(self):
        rotation = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        assignment = [dict(transmitter_ecef_m=[100.+c,200.,300.]) for c in range(5)]
        original = deepcopy(assignment)
        receiver = np.array([3.,4.,5.])
        tx = np.arange(15.).reshape(5,3);rx = np.array([0.,0.,2.])
        updated, moved_rx = frozen_coordinates(assignment,receiver,rotation,
            dict(transmitter_offsets_enu_m=tx,receiver_offset_enu_m=rx))
        for before,after,offset in zip(assignment,updated,tx):
            np.testing.assert_array_equal(np.array(after['transmitter_ecef_m'])-before['transmitter_ecef_m'],offset@rotation)
        np.testing.assert_array_equal(moved_rx-receiver,rx@rotation)
        self.assertEqual(original,assignment)
        np.testing.assert_array_equal(receiver,[3.,4.,5.])

    def test_fixed_geometry_carrier_initialisation_cannot_move_antennas_or_motion(self):
        model,truth = fixture();g = deepcopy(model.geometry)
        p,v,a,tx,rx = g.state(truth[:-1])
        # Freeze the prior antennas; this test is of the search restrictions.
        g.free_antennas=False;g.nantenna=0;g.count=10
        g.lower=g.lower[18:];g.upper=g.upper[18:]
        bands = [PhysicalAmplitudeArc(b.field,b.training,b.knots,g,b.channel,b.midpoint)
                 for b in model.independent.bands]
        frozen = CommonPhysicalFit(PhysicalAmplitudeFit(bands))
        seed = np.r_[truth[18:-1],truth[-1]]
        mapped,_ = frozen.mapping(seed)
        before = g.state(seed[:-1])
        branches = frozen.phase_branches(mapped,adjacent=False)
        self.assertGreaterEqual(len(branches),2)
        for branch in branches:
            q = branch['parameters'][:-1]
            state = g.state(q)
            np.testing.assert_array_equal(q[3:],seed[3:-1])
            np.testing.assert_array_equal(state[3],before[3])
            np.testing.assert_array_equal(state[4],before[4])
        direction=np.sin(np.arange(len(seed)))*.01;step=1e-5
        value,gradient=frozen.objective(seed)
        finite=(frozen.objective(seed+step*direction)[0]-frozen.objective(seed-step*direction)[0])/(2*step)
        self.assertAlmostEqual(finite,gradient@direction,delta=2e-5+abs(finite)*5e-5)


if __name__ == '__main__':
    unittest.main()
