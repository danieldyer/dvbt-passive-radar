from types import SimpleNamespace
import unittest

import numpy as np

from fit_physical_primary_association import acquire_constant_phases,geometry_proposal
from test_physical_mixed_nuisance import problem


class PhysicalPrimaryAssociationTests(unittest.TestCase):
    def test_acquisition_summary_keeps_complex_step_and_the_real_prior(self):
        truth=.1*np.cos(np.arange(9));scale=20.
        geometry=SimpleNamespace(count=28,lower=np.full(28,-3.),upper=np.full(28,3.))
        geometry.data_residual=lambda p:scale*(p[18:27]-truth)
        fitted,cost=geometry_proposal(geometry,np.zeros((5,3)))
        expected=truth*scale**2/(scale**2+.5)
        np.testing.assert_allclose(fitted[18:27],expected,atol=1e-12)
        np.testing.assert_array_equal(fitted[np.r_[np.arange(18),27]],0.)
        self.assertAlmostEqual(cost,np.sum(scale**2*(expected-truth)**2+expected**2/2),places=12)

    def test_cold_phase_acquisition_ignores_initial_phases_and_unused_observations(self):
        model,truth=problem();seed=truth.copy();phases=[]
        for c,(cluster,index) in enumerate(zip(model.clusters,model.indices)):
            phases.extend([model.geometry.count+c,*[index[s][1] for s in cluster.nuisance_slices]])
        acquired,details=acquire_constant_phases(model,seed)
        seed[phases]+=.7*np.cos(phases)
        for cluster in model.clusters:
            cluster.base.field.y[~cluster.base.training]+=100.*np.exp(.4j)
        repeated,again=acquire_constant_phases(model,seed)
        np.testing.assert_array_equal(acquired,repeated)
        for a,b in zip(details,again):
            np.testing.assert_array_equal(a['constant_complex_coefficients_real_imag'],b['constant_complex_coefficients_real_imag'])
        nonphase=np.setdiff1d(np.arange(model.count),phases)
        np.testing.assert_array_equal(acquired[nonphase],truth[nonphase])


if __name__=='__main__':unittest.main()
