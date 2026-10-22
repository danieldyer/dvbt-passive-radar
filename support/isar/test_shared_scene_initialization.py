import unittest

import numpy as np

from fit_shared_translation_scene import acquire_scene_phases,seed_offsets
from test_shared_translation_scene import shared_problem


class SharedSceneInitializationTests(unittest.TestCase):
    def test_phase_acquisition_is_training_only_and_does_not_change_shared_motion(self):
        scene,truth=shared_problem();p=truth.copy();first=acquire_scene_phases(scene,p)
        phases=np.r_[scene.phase_indices.ravel(),[v[1] for group in scene.nuisance_indices for v in group]]
        p[phases]+=.9*np.sin(phases)
        for b in scene.bands:b.field.y[~b.training]+=100*np.exp(.7j)
        second=acquire_scene_phases(scene,p)
        np.testing.assert_array_equal(first,second)
        remaining=np.setdiff1d(np.arange(scene.count),phases)
        np.testing.assert_array_equal(first[remaining],truth[remaining])

    def test_range_seed_has_the_requested_physical_linearized_offset(self):
        scene,truth=shared_problem();p=seed_offsets(scene,truth,20.,70.)
        g=scene.geometry;base=g.state(p[:28]);point=scene.point_summary(p)[1]
        offset=np.asarray(point['position_enu_m'])-base[0]
        vectors=(base[0]-base[3][2],base[0]-base[4])
        gradient=sum(v/np.linalg.norm(v) for v in vectors)
        self.assertAlmostEqual(gradient@offset,20.,places=10)
        difference=np.asarray(point['observables'])[2,0]-g.prediction(p[:28])[2,0]
        radius=np.linalg.norm(offset)
        # Hessian norm of each Euclidean distance is <= 1 / distance.
        bound=.5*radius**2*sum(1/(np.linalg.norm(v)-radius) for v in vectors)
        self.assertLess(abs(difference-20.),bound)
        self.assertLess(radius,scene.relative_radius_m)
        np.testing.assert_array_equal(p[:28],truth[:28])


if __name__=='__main__':unittest.main()
