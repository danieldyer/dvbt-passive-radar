import unittest

import numpy as np

from motion import ecef,enu_basis
from sea_physical_arc import SeaPhysicalArc
from test_physical_amplitude_arc import problem


class SeaPhysicalArcTests(unittest.TestCase):
    def test_direct_limit_and_full_sea_response_jacobian(self):
        b,truth=problem();lat,lon=np.radians([-37.87,144.93]);radius=6371000.
        water=dict(centre_ecef_m=ecef(lat,lon,0.)-radius*enu_basis(lat,lon)[:,2],radius_m=radius)
        plain=SeaPhysicalArc(b.field,b.training,b.knots,b.geometry,b.channel,b.midpoint,water=water,family='direct')
        raw,jac=b.response(truth);actual,actual_jac=plain.response(truth)
        np.testing.assert_allclose(actual,raw,rtol=2e-8,atol=2e-8)
        np.testing.assert_allclose(actual_jac,jac,rtol=3e-8,atol=1e-7)
        for family in ('transmit','both'):
            model=SeaPhysicalArc(b.field,b.training,b.knots,b.geometry,b.channel,b.midpoint,water=water,family=family)
            raw,jac=model.response(truth)
            direction=np.cos(np.arange(len(truth)))*.1;step=2e-4
            finite=(model.response(truth+step*direction)[0]-model.response(truth-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),4e-5)


if __name__=='__main__':unittest.main()
