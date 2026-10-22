import unittest

import numpy as np

from facet_raw import template
from field import average
from motion import enu_basis
from specular_facet import hh_current,unit
from test_rigid_motion import fixture


class FacetRawTests(unittest.TestCase):
    def setup_geometry(self):
        f=fixture().windowed(2.,3.)
        d=f.data
        up=enu_basis(*np.radians([-37.87,144.93]))[:,2]
        positions=d['centre_ecef_m']+20*up
        velocities=d['centre_velocity_ecef_mps']
        tx,rx=d['transmitter_ecef_m'],d['receiver_ecef_m']
        n=unit(tx-positions[len(positions)//2])+unit(rx-positions[len(positions)//2])
        a=unit(np.cross(up,n))
        along=np.broadcast_to(a,positions.shape)
        return f,positions,velocities,along,np.zeros_like(along),up,tx,rx

    def test_small_facet_matches_point_operator_including_amplitude(self):
        f,p,v,a,da,up,tx,rx=self.setup_geometry()
        found=template(f,p,v,a,da,up,tx,rx,.001,.001,order=(4,4))
        rt,rr=np.linalg.norm(p-tx,axis=1),np.linalg.norm(rx-p,axis=1)
        gain=hh_current(unit(p-tx),unit(rx-p),np.cross(a,up),up)
        middle=len(p)//2
        gain*=rt[middle]*rr[middle]/(rt*rr*abs(gain[middle]))
        raw=f.raw_trajectory(p,v)*gain[:,None]
        expected=f.apply(average(raw,f.data['row_information'],f.metadata['block_rows'])[0])
        self.assertLess(np.linalg.norm(found-expected)/np.linalg.norm(expected),2e-7)

    def test_quadrature_converges_with_range_envelope_and_row_binning(self):
        args=self.setup_geometry()
        coarse=template(*args,200.,20.,order=(24,6))
        dense=template(*args,200.,20.,order=(48,12))
        self.assertLess(np.linalg.norm(coarse-dense)/np.linalg.norm(dense),2e-6)


if __name__=='__main__':unittest.main()
