import unittest

import numpy as np

from compare_ais_phase import path_geometry
from relative_scatterer_delay import RelativeScattererDelay
from test_flexible_echo_cluster import cubic_scene


class RelativeScattererDelayTests(unittest.TestCase):
    def test_within_band_delay_difference_cancels_direct_baseline_and_fixed_origin(self):
        receiver=np.array([0.,0.,40.]);transmitter=np.array([31000.,-2000.,740.])
        first=np.array([[2000.,-8000.,12.]]);second=first+[50.,90.,5.]
        physical=[]
        for origin in (-713.,0.,1413.):
            a,_=path_geometry(first,np.zeros_like(first),transmitter,receiver)
            b,_=path_geometry(second,np.zeros_like(second),transmitter,receiver)
            physical.append((b-origin)-(a-origin))
        np.testing.assert_array_equal(physical[0],physical[1]);np.testing.assert_array_equal(physical[0],physical[2])

    def test_affine_jacobian_and_known_scene_recovery_keep_band_phases_free(self):
        base,truth=cubic_scene();model=RelativeScattererDelay(base,1,2,[(0,1,2)])
        p=truth[model.kept];np.testing.assert_allclose(model.mapping(p),truth,atol=0.)
        probe=p+.0003*np.sin(np.arange(model.count));r,j=model.evaluate(probe)
        direction=.001*np.cos(np.arange(model.count));step=1e-4
        finite=(model.evaluate(probe+step*direction)[0]-model.evaluate(probe-step*direction)[0])/(2*step)
        np.testing.assert_allclose(finite,j@direction,rtol=2e-5,atol=2e-9)
        fitted,audit=model.fit(probe)
        np.testing.assert_allclose(model.mapping(fitted),truth,atol=3e-6)
        self.assertTrue(audit['mapped_envelope_guard']['valid'])
        phase=np.array([[6*c+3,6*c+5] for c in range(3)])
        self.assertTrue(np.all(np.isin(phase,model.kept)))
        prediction=base.prediction(model.mapping(fitted),[~b.training for b in base.bands])
        self.assertTrue(all(v['heldout_explained_fraction']>1-1e-12 for v in prediction))

    def test_outside_removed_envelope_guard_is_explicitly_invalid(self):
        base,truth=cubic_scene();model=RelativeScattererDelay(base,1,2,[(0,1,2)])
        p=truth.copy();p[model.origins[0,1]]+=1.2;p[model.origins[1,0]]+=1.2
        audit=model.guard_audit(p[model.kept])
        self.assertFalse(audit['valid']);self.assertIn(int(model.origins[1,1]),audit['outside_indices'])


if __name__=='__main__':unittest.main()
