import unittest

import numpy as np

from common_physical_pair import CommonPhysicalPair
from fit_physical_primary_coherence import common_free
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_echo_cluster import PhysicalEchoCluster
from physical_pair import fit_parameters
from test_physical_mixed_nuisance import problem


def coherent_problem(*,jerk_std=None):
    old,independent_truth=problem(jerk_std=jerk_std);common=CommonPhysicalPair(old)
    truth=np.r_[independent_truth[:28],.61,independent_truth[33:]]
    mapped,_=common.mapping(truth)
    for cluster,index in zip(old.clusters,old.indices):
        b=cluster.base;before,_=b.response(independent_truth[:33]);after,_=b.response(mapped[:33])
        brightness=b.amplitude_basis@np.array([.7,1.2,.4,.9,1.,.6])
        change=b.reduced(((after-before)*brightness[:,None])[:,:,None])[:,0]
        b.field.y+=change.reshape(b.field.y.shape)
    physical=[PhysicalAmplitudeArc(b.field,b.training,b.knots,b.geometry,b.channel,b.midpoint,
        stationary_design=b.design) for b in old.bands]
    model=PhysicalEchoCluster(physical,[c.empirical for c in old.clusters],old.origins,
        nuisance_arcs=[c.nuisance_arcs for c in old.clusters],nuisance_jerk_std=jerk_std)
    return CommonPhysicalPair(model),truth


class PhysicalPrimaryCoherenceTests(unittest.TestCase):
    def test_variable_degree_nuisances_and_priors_remain_exactly_nested(self):
        model,truth=coherent_problem(jerk_std=.3);mapped,transform=model.mapping(truth)
        residual,jacobian=model.evaluate(truth);old,derivative=model.independent.evaluate(mapped)
        np.testing.assert_array_equal(residual,old)
        np.testing.assert_array_equal(jacobian,derivative@transform)
        self.assertAlmostEqual(residual@residual*model.energy,model.independent.prior_cost(mapped),places=9)
        p=truth+.0001*np.cos(np.arange(model.count));_,j=model.evaluate(p)
        direction=.01*np.sin(np.arange(model.count));step=5e-4
        finite=(model.evaluate(p+step*direction)[0]-model.evaluate(p-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(finite-j@direction)/np.linalg.norm(finite),5e-5)

    def test_known_common_primary_with_flexible_neighbours_predicts_unused_samples(self):
        model,truth=coherent_problem();seed=truth.copy()
        free=np.r_[18,21,np.arange(28,model.count)]
        seed[free]+=.0004*np.cos(free)
        fitted,audit=fit_parameters(model,seed,free,max_evaluations=120)
        self.assertLess(audit['objective']*model.energy,1e-10)
        np.testing.assert_allclose(fitted[free],truth[free],atol=5e-5)
        self.assertTrue(all(p['heldout_explained_fraction']>1-1e-12
            for p in model.prediction(fitted,[~b.training for b in model.bands])))

    def test_estimated_coordinate_pullback_preserves_fixed_antenna_lag_and_nuisance(self):
        original=np.r_[np.arange(18,27),np.arange(28,60)]
        original=np.setdiff1d(original,[44,45])
        reduced=common_free(original)
        self.assertFalse(np.isin(reduced,np.r_[np.arange(18),27,40,41]).any())
        self.assertEqual(sum(reduced==28),1)
        np.testing.assert_array_equal(reduced,np.r_[np.arange(18,27),28,np.arange(29,40),np.arange(42,56)])


if __name__=='__main__':unittest.main()
