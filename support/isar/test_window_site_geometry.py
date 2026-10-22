import unittest
from types import SimpleNamespace

import numpy as np

from audit_window_site_geometry import physical_observables
from flash_geometry import observables


class WindowSiteGeometryTests(unittest.TestCase):
    def test_restore_absolute_range_and_known_interband_derivatives_against_physical_geometry(self):
        # Independent exact Euclidean trajectories, not a polynomial chosen to
        # reproduce the audit's coefficient extraction. A shared reference
        # band changes its range/rate/curvature; each antenna adds its own
        # known differential curve in the archived digital observation frame.
        p=np.array([2000.,-11000.,12.]);v=np.array([-1.1,4.8,.01]);a=np.array([.01,-.02,.002])
        tx=np.array([[31000+c*65.,-2000+c*43.,740.+7*c] for c in range(5)])
        rx=np.array([0.,0.,40.]);epoch=324.;time=np.linspace(-4,4,513)
        truth=observables(p,v,a,tx,rx);ranges=np.array([
            observables(p+t*v+.5*t*t*a,v+t*a,a,tx,rx)[:,0] for t in time])
        for c in range(5):
            centre=truth[c,0]-34.-4*c
            known=ranges[:,c]-ranges[:,2]-(truth[c,0]-truth[2,0])
            band=SimpleNamespace(epoch=epoch,known_path=known,
                field=SimpleNamespace(data=dict(row_time=epoch+time,centre_path_m=np.full(len(time),centre))))
            conditional=dict(range_origin_m_rate_residual_mps_curvature_residual_mps2=[
                [truth[c,0]-centre,truth[2,1],truth[2,2]]])
            actual,error=physical_observables(band,conditional)
            self.assertLess(error,1e-8)
            np.testing.assert_allclose(actual,truth[c],rtol=0,atol=2e-9)

    def test_unrepresented_known_path_is_not_silently_converted_to_a_precise_summary(self):
        time=np.linspace(-4,4,513)
        band=SimpleNamespace(epoch=324.,known_path=.01*np.sin(7*time),
            field=SimpleNamespace(data=dict(row_time=324+time,centre_path_m=np.full(len(time),14000.))))
        with self.assertRaises(ValueError):physical_observables(band,{})


if __name__=='__main__':unittest.main()
