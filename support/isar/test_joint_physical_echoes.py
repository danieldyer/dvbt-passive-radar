from copy import deepcopy
import unittest

import numpy as np

from cluster_phase import PolynomialPath
from field import Field
from joint_physical_echoes import JointPhysicalEchoes
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_echo_cluster import PhysicalEchoCluster
from shared_echo_cluster import ReferencedAmplitudeArc
from test_physical_mixed_nuisance import problem
from test_physical_primary_coherence import coherent_problem


def shifted_model(original,offset,*,jerk_std=None):
    physical=[];empirical=[];neighbours=[]
    for b,cluster in zip(original.bands,original.clusters):
        data=deepcopy(b.field.data)
        for key in ('time','row_time'):data[key]=data[key]+offset
        f=Field(data,b.field.pulse.samples);f.y=b.field.y.copy()
        physical.append(PhysicalAmplitudeArc(f,b.training,b.knots+offset,b.geometry,b.channel,b.midpoint+offset,
            stationary_design=b.design))
        local=[]
        for old in (cluster.empirical,*cluster.nuisance_arcs):
            curve=old.base.curve;first,last=curve.epoch-curve.scale+offset,curve.epoch+curve.scale+offset
            local.append(ReferencedAmplitudeArc(f,b.training,b.knots+offset,first,last,old.known_path,old.known_rate,
                stationary_design=b.design,curve=PolynomialPath(first,last,curve.degree)))
        empirical.append(local[0]);neighbours.append(local[1:])
    return PhysicalEchoCluster(physical,empirical,original.origins,
        nuisance_arcs=neighbours,nuisance_jerk_std=jerk_std)


class JointPhysicalEchoesTests(unittest.TestCase):
    def test_data_once_geometry_priors_once_and_each_neighbour_jerk_once(self):
        first,a=problem(jerk_std=.3);second=shifted_model(first,10.,jerk_std=.3)
        model=JointPhysicalEchoes([first,second]);p=np.zeros(model.count)
        p[:18]=.03*np.cos(np.arange(18))
        for arc,index in zip(model.arcs,model.indices):p[index[18:]]=a[18:]
        r,j=model.evaluate(p)
        local=sum(arc.objective(p[index])[0]*arc.energy for arc,index in zip(model.arcs,model.indices))
        expected=local-p[:18]@p[:18]/2
        self.assertAlmostEqual(r@r*model.energy,expected,places=8)
        expected_prior=sum(arc.prior_cost(p[index]) for arc,index in zip(model.arcs,model.indices))-p[:18]@p[:18]/2
        self.assertAlmostEqual(model.prior_cost(p),expected_prior,places=11)
        self.assertEqual(len(r),sum(model.data_rows)+len(model.prior_factor))
        direction=.001*np.sin(np.arange(model.count))
        # Millimetric physical probes avoid subtracting sub-micrometric
        # propagation differences at ECEF-scale coordinates. A step sweep
        # verifies the analytic derivative before choosing these two scales.
        for step in (.1,.03):
            finite=(model.evaluate(p+step*direction)[0]-model.evaluate(p-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-j@direction)/np.linalg.norm(finite),1e-5)

    def test_overlapping_windows_cannot_supply_duplicate_calibration_information(self):
        first,_=problem();second=shifted_model(first,.1)
        with self.assertRaisesRegex(ValueError,'original observations'):
            JointPhysicalEchoes([first,second])

    def test_common_and_independent_phase_scenes_have_exact_local_data_and_information(self):
        common,a=coherent_problem(jerk_std=.3)
        second=shifted_model(common.independent,10.,jerk_std=.3)
        b=common.mapping(a)[0];model=JointPhysicalEchoes([common,second]);p=np.zeros(model.count)
        for index,value in zip(model.indices,(a,b)):p[index]=value
        r,_=model.evaluate(p)
        self.assertAlmostEqual(r@r*model.energy,model.prior_cost(p),places=9)
        mean,norm=model.expected_jacobian(p)
        self.assertEqual(mean.shape,(sum(model.data_rows),model.count))
        self.assertTrue(np.all(np.linalg.norm(mean,axis=0)<=norm+1e-8))
        for arc,index,rows in zip(model.arcs,model.indices,model.data_rows):
            derivative=arc.mean_derivative(p[index])
            self.assertEqual(len(derivative),rows)
        free=np.concatenate([index[np.r_[18,21,np.arange(28,arc.count)]] for arc,index in zip(model.arcs,model.indices)])
        seed=p.copy();seed[free]+=.0002*np.cos(free)
        # A no-jerk truth provides an exact recovery control; here the nonzero
        # jerk prior is deliberately retained and only data/prior ownership is
        # asserted. No prior-driven shift is described as recovered motion.
        residual,_=model.evaluate(seed)
        self.assertTrue(np.isfinite(residual).all())

    def test_disjoint_noiseless_scenes_recover_motion_and_predict_unused_rows(self):
        first,a=problem();second=shifted_model(first,10.)
        model=JointPhysicalEchoes([first,second]);truth=np.zeros(model.count)
        for index in model.indices:truth[index]=a
        free=np.concatenate([index[np.r_[18,21,np.arange(28,arc.count)]] for arc,index in zip(model.arcs,model.indices)])
        seed=truth.copy();seed[free]+=.0002*np.sin(free)
        fitted,audit=model.fit(seed,free,max_evaluations=100)
        self.assertLess(audit['objective']*model.energy,1e-10)
        np.testing.assert_allclose(fitted[free],truth[free],atol=5e-5)
        predictions=model.prediction(fitted,[[~b.training for b in a.bands] for a in model.arcs])
        self.assertTrue(all(v['heldout_explained_fraction']>1-1e-12 for scene in predictions for v in scene))


if __name__=='__main__':unittest.main()
