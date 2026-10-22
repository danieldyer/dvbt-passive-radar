from copy import deepcopy
import unittest

import numpy as np

from cluster_phase import PolynomialPath
from component_phase import physical_primary_measurements
from field import Field
from fixed_range_field import stationary_profile_design
from motion import excess_path
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_echo_cluster import PhysicalEchoCluster
from physical_pair import fit_parameters
from shared_echo_cluster import ReferencedAmplitudeArc
from test_physical_echo_cluster import fixture as original_fixture


def problem(*,jerk_std=None,missing_channel=None):
    old,_=original_fixture();geometry=old.geometry;fields=[];primary=[];empirical=[];arcs=[]
    for b in old.bands:
        data=deepcopy(b.field.data);time=data['row_time']
        point=np.array([np.interp(b.midpoint,time,data['centre_ecef_m'][:,j]) for j in range(3)])
        data['centre_ecef_m']=np.broadcast_to(point,data['centre_ecef_m'].shape).copy()
        data['centre_velocity_ecef_mps']=np.zeros_like(data['centre_velocity_ecef_mps'])
        length=excess_path(point[None,:],data['transmitter_ecef_m'],data['receiver_ecef_m'])[0][0]
        data['centre_path_m']=np.full(len(time),length)
        data['centre_omega']=np.full(len(time),np.interp(b.midpoint,time,data['centre_omega']))
        f=Field(data,b.field.pulse.samples);f.y[:]=np.exp(.3j*f.time)[:,None]
        fields.append(f)
    def construct():
        primary=[];empirical=[];arcs=[]
        for old_band,f in zip(old.bands,fields):
            design=stationary_profile_design(f);first,last=old_band.epoch-old_band.scale,old_band.epoch+old_band.scale
            p=PhysicalAmplitudeArc(f,old_band.training,old_band.knots,geometry,old_band.channel,old_band.midpoint,
                stationary_design=design)
            known=np.zeros(len(f.data['row_time']));rate=f.data['centre_omega']/(f.k*.001024)
            e=ReferencedAmplitudeArc(f,p.training,p.knots,first,last,known,rate,stationary_design=design)
            cubic=ReferencedAmplitudeArc(f,p.training,p.knots,first,last,known,rate,
                stationary_design=design,curve=PolynomialPath(first,last,3))
            primary.append(p);empirical.append(e);arcs.append((e,cubic,e))
        return PhysicalEchoCluster(primary,empirical,[-3.,1.2,4.],nuisance_arcs=arcs,nuisance_jerk_std=jerk_std)
    model=construct();truth=np.zeros(model.count)
    truth[geometry.count:model.primary_count]=np.linspace(-.7,.8,5)
    for c,(cluster,index) in enumerate(zip(model.clusters,model.indices)):
        values=([ -3.,.4+.1*c,-.4,.06 ],[1.2,-.8+.2*c,.3,-.09,.07],[4.,1.7-.2*c,-.8,.1])
        for block,value in zip(cluster.nuisance_slices,values):truth[index[block]]=value
    strength=np.array([[.7,1.2,.4,.9,1.,.6],[.3,.5,.2,.6,.4,.7],
        [.4,.3,.7,.5,.2,.6],[.6,.3,.5,.2,.8,.4]])
    for c,(cluster,index) in enumerate(zip(model.clusters,model.indices)):
        b=cluster.base;signal=np.zeros(b.field.y.size,complex)
        for s,((raw,_),amplitude) in enumerate(zip(cluster.response_blocks(truth[index]),strength)):
            amplitude=np.zeros_like(amplitude) if c==missing_channel and s==0 else amplitude
            signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@amplitude
        static=b.design@((np.arange(len(b.field.offsets))+1j*np.arange(len(b.field.offsets))[::-1])*20.)
        b.field.y=(signal+static).reshape(b.field.y.shape)
    return construct(),truth


class PhysicalMixedNuisanceTests(unittest.TestCase):
    def test_explicit_quadratic_arcs_preserve_the_original_operator(self):
        original,truth=original_fixture()
        empirical=[c.empirical for c in original.clusters]
        explicit=PhysicalEchoCluster(original.bands,empirical,original.origins,
            nuisance_arcs=[(e,)*3 for e in empirical])
        self.assertEqual(explicit.count,original.count)
        a,da=original.evaluate(truth);b,db=explicit.evaluate(truth)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(da,db)

    def test_mixed_operator_exact_gradient_and_once_counted_jerk_prior(self):
        model,truth=problem(jerk_std=.3);r,_=model.evaluate(truth)
        expected=0.
        for cluster,index in zip(model.clusters,model.indices):
            curve=cluster.nuisance_arcs[1].base.curve;coefficient=truth[index[cluster.nuisance_slices[1]]][-1]
            expected+=36*coefficient**2/(curve.scale**5*.3**2)
        self.assertAlmostEqual(model.prior_cost(truth),expected,places=11)
        self.assertAlmostEqual(r@r*model.energy,expected,places=10)
        probe=truth+.0001*np.sin(np.arange(model.count));_,j=model.evaluate(probe)
        direction=.01*np.cos(np.arange(model.count));step=5e-4
        finite=(model.evaluate(probe+step*direction)[0]-model.evaluate(probe-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(finite-j@direction)/np.linalg.norm(finite),5e-5)

    def test_known_physical_motion_and_flexible_neighbours_recover_on_unused_rows(self):
        model,truth=problem();n=model.geometry.count
        free=np.r_[18,21,np.arange(n,model.count)]
        seed=truth.copy();seed[free]+=.0005*np.sin(free)
        fitted,audit=fit_parameters(model,seed,free,max_evaluations=140)
        np.testing.assert_allclose(fitted[free],truth[free],atol=5e-5)
        self.assertLess(model.objective(fitted)[0]*model.energy,1e-10)
        prediction=model.prediction(fitted,[~b.training for b in model.bands])
        self.assertTrue(all(p['heldout_explained_fraction']>1-1e-12 for p in prediction))

    def test_absent_primary_and_unused_phase_changes_are_not_replaced_by_the_model(self):
        model,truth=problem(missing_channel=2)
        missing=physical_primary_measurements(model.clusters[2],truth[model.indices[2]])
        power=abs(missing['complex_coefficient'])**2*8192*missing['conditional_information']
        self.assertLess(np.max(power),1e-6)
        c=1;cluster=model.clusters[c];p=truth[model.indices[c]];b=cluster.base
        before=physical_primary_measurements(cluster,p)
        b.field.y[~b.training]+=before['frozen_component_prediction'][~b.training]*(np.exp(.37j)-1)
        after=physical_primary_measurements(cluster,p);valid=before['conditional_information']>0
        ratio=after['complex_coefficient'][valid]/before['complex_coefficient'][valid]
        np.testing.assert_allclose(ratio[b.training[valid]],1.,atol=1e-9)
        np.testing.assert_allclose(ratio[~b.training[valid]],np.exp(.37j),atol=1e-9)
        np.testing.assert_array_equal(before['frozen_component_prediction'],after['frozen_component_prediction'])
        np.testing.assert_array_equal(before['physical_path_change_m'],after['physical_path_change_m'])


if __name__=='__main__':unittest.main()
