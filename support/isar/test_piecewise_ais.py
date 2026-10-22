import unittest

import numpy as np

from ais import AisTrack
from motion import ecef,enu_basis
from piecewise_ais import LocalAisReference
from flash_geometry import Flash


class PiecewiseAisTests(unittest.TestCase):
    def test_crossing_an_ais_knot_preserves_positions_velocities_and_derivatives(self):
        t=np.array([0.,8.,12.,20.]);nan=np.full(4,np.nan)
        track=AisTrack(t,np.radians([-37.81,-37.80995,-37.80991,-37.80990]),
            np.radians([144.98,144.98001,144.98002,144.9801]),nan,nan,nan,nan,nan,nan,None,5.)
        receiver=ecef(np.radians(-37.8104),np.radians(144.9874),40.)
        rotation=enu_basis(np.radians(-37.8104),np.radians(144.9874)).T
        reference=LocalAisReference(track,10.,rotation,receiver)
        self.assertEqual(len(reference.coefficients),3)
        for offset in np.linspace(-3.9,3.9,31):
            sample=track.sample([10.+offset],max_gap_s=10.)
            np.testing.assert_allclose(reference(offset),(sample['position_ecef_m'][0]-receiver)@rotation.T,atol=1e-5,rtol=0.)
            np.testing.assert_allclose(reference(offset,1),sample['velocity_ecef_mps'][0]@rotation.T,atol=2e-6,rtol=0.)
            np.testing.assert_allclose(reference(offset+1e-20j).imag/1e-20,reference(offset,1),atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(reference(offset+1e-20j,1).imag/1e-20,reference(offset,2),atol=1e-12,rtol=1e-12)
        offsets=np.array([[-3.9,-2.1,-2.],[0.,1.9,2.],[2.1,3.8,4.]])
        for derivative in range(4):
            expected=np.array([reference(t,derivative) for t in offsets.ravel()]).reshape(offsets.shape+(3,))
            np.testing.assert_allclose(reference.sample(offsets,derivative),expected,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(reference.sample(offsets+1e-20j).imag/1e-20,reference.sample(offsets,1),atol=1e-12,rtol=1e-12)
        with self.assertRaises(ValueError):reference(4.01)
        assignment=[dict(site_id=0,transmitter_ecef_m=(receiver+np.array([30000.+c*50,5000.,500.])@rotation).tolist()) for c in range(5)]
        model=Flash(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,10.,rotation,receiver,
                    assignment,free_antennas=False,ais_reference=reference)
        for offset in (-2.1,-1.9,1.9,2.1):
            q=np.zeros(model.count);q[-1]=offset
            direction=np.sin(np.arange(model.count))*.03;step=2e-4
            finite=(model.prediction(q+step*direction)-model.prediction(q-step*direction))/(2*step)
            analytic=model.prediction(q+1e-20j*direction).imag/1e-20
            np.testing.assert_allclose(finite,analytic,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
