import unittest

import numpy as np

from physical_amplitude_arc import PhysicalAmplitudeArc
from shared_translation_scene import SharedTranslationScene
from test_physical_mixed_nuisance import problem


def shared_problem(*,jerk_std=None,missing_second_channel=None):
    old,old_truth=problem();g=old.geometry
    def construct(points=2):
        bands=[PhysicalAmplitudeArc(b.field,b.training,b.knots,g,b.channel,b.midpoint,
            stationary_design=b.design) for b in old.bands]
        return SharedTranslationScene(bands,[c.nuisance_arcs for c in old.clusters],old.origins,
            points=points,origin_radius=2.,jerk_std=jerk_std)
    scene=construct();truth=np.zeros(scene.count)
    truth[scene.offset_indices[1]]=[.14,.6,.25]
    truth[scene.phase_indices[0]]=np.linspace(-.7,.8,5)
    truth[scene.phase_indices[1]]=np.linspace(1.1,-.2,5)
    for c,private in enumerate(scene.nuisance_indices):
        previous=old_truth[old.indices[c]]
        for target,source in zip(private,old.clusters[c].nuisance_slices):truth[target]=previous[source]
    strengths=np.array([[.7,1.2,.4,.9,1.,.6],[.2,.6,.9,.3,.7,.8],
        [.3,.5,.2,.6,.4,.7],[.4,.3,.7,.5,.2,.6],[.6,.3,.5,.2,.8,.4]])
    for c,(cluster,transform) in enumerate(zip(scene.clusters,scene.maps)):
        b=cluster.base;signal=np.zeros(b.field.y.size,complex)
        for s,((raw,_),amplitude) in enumerate(zip(cluster.response_blocks(transform@truth),strengths)):
            a=np.zeros_like(amplitude) if c==missing_second_channel and s==1 else amplitude
            signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@a
        static=b.design@((np.arange(len(b.field.offsets))+1j*np.arange(len(b.field.offsets))[::-1])*20.)
        b.field.y=(signal+static).reshape(b.field.y.shape)
    return construct(),truth


class SharedTranslationSceneTests(unittest.TestCase):
    def test_single_physical_point_preserves_existing_residual_and_derivative_exactly(self):
        old,p=problem(jerk_std=.3)
        scene=SharedTranslationScene(old.bands,[c.nuisance_arcs for c in old.clusters],old.origins,
            points=1,origin_radius=2.,jerk_std=.3)
        self.assertEqual(scene.count,old.count)
        a,da=old.evaluate(p);b,db=scene.evaluate(p)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(da,db)
        np.testing.assert_array_equal(old.lower,scene.lower);np.testing.assert_array_equal(old.upper,scene.upper)
        self.assertEqual(scene.prior_cost(p),old.prior_cost(p))

    def test_shared_physical_kinematics_point_height_and_compact_horizontal_support(self):
        scene,p=shared_problem();g=scene.geometry;point=scene.point_geometries[1]
        q=np.r_[p[:28],p[scene.offset_indices[1]]]
        base=g.state(q[:28]);second=point.state(q)
        expected=np.array([21*np.cos(.6),21*np.sin(.6),5.])
        np.testing.assert_allclose(second[0]-base[0],expected,atol=1e-12)
        for a,b in zip(second[1:],base[1:]):np.testing.assert_array_equal(a,b)
        moved=q.copy();moved[20]+=.2
        self.assertAlmostEqual(point.state(moved)[0][2],second[0][2],places=12)
        edge=q.copy();edge[28]=scene.upper[scene.offset_indices[1][0]]
        self.assertAlmostEqual(np.linalg.norm((point.state(edge)[0]-g.state(edge[:28])[0])[:2]),300.,places=10)
        b=scene.point_arcs[2][1];local=np.r_[q,p[scene.phase_indices[1]]]
        raw,_=b.response(local);dt=b.field.data['row_time']-b.midpoint
        position=base[0]+expected+dt[:,None]*base[1]+.5*dt[:,None]**2*base[2]
        offset=(position-b.reference_local)@g.rotation
        independent=b.field.raw_trajectory(b.field.data['centre_ecef_m']+offset,(base[1]+dt[:,None]*base[2])@g.rotation,
            transmitter=g.receiver+base[3][b.channel]@g.rotation,receiver=g.receiver+base[4]@g.rotation,position_offset=offset)
        length=point.prediction(q)[b.channel,0]
        independent*=np.exp(1j*(b.field.k*(length-b.reference_path)+p[scene.phase_indices[1,b.channel]]))
        np.testing.assert_allclose(raw,independent,rtol=2e-10,atol=1e-10)

    def test_shared_profiled_derivative_and_once_counted_physical_and_jerk_priors(self):
        scene,p=shared_problem(jerk_std=.3);r,_=scene.evaluate(p)
        self.assertAlmostEqual(r@r*scene.energy,scene.prior_cost(p),places=9)
        expected=.5*(.14**2+.25**2)
        for cluster,transform,private in zip(scene.clusters,scene.maps,scene.nuisance_indices):
            curve=cluster.arcs[3].base.curve;coefficient=p[private[1]][-1]
            expected+=36*coefficient**2/(curve.scale**5*.3**2)
        self.assertAlmostEqual(scene.prior_cost(p),expected,places=10)
        changed=p.copy();changed[scene.offset_indices[1][1]]+=1.7
        self.assertEqual(scene.prior_cost(p),scene.prior_cost(changed))
        probe=p+.0001*np.sin(np.arange(scene.count));_,j=scene.evaluate(probe)
        direction=.01*np.cos(np.arange(scene.count))
        for step in (.003,.001):
            finite=(scene.evaluate(probe+step*direction)[0]-scene.evaluate(probe-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-j@direction)/np.linalg.norm(finite),8e-5)
        mean,norm=scene.expected_jacobian(p)
        self.assertTrue(np.all(np.linalg.norm(mean,axis=0)<=norm+1e-8))

    def test_common_motion_recovery_and_original_unused_prediction_with_free_neighbours(self):
        scene,truth=shared_problem();free=np.r_[18,21,np.arange(scene.phase_start,scene.count)]
        seed=truth.copy();seed[free]+=.0005*np.sin(free)
        fitted,audit=scene.fit(seed,free,max_evaluations=120)
        np.testing.assert_allclose(fitted[free],truth[free],atol=5e-5)
        rows=sum(len(c.base.clean)*2 for c in scene.clusters)
        r,_=scene.evaluate(fitted);self.assertLess(r[:rows]@r[:rows]*scene.energy,1e-10)
        predictions=scene.prediction(fitted,[~b.training for b in scene.bands])
        self.assertTrue(all(v['heldout_explained_fraction']>1-1e-12 for v in predictions))

    def test_missing_second_point_and_unused_phase_perturbation_remain_measured(self):
        scene,p=shared_problem(missing_second_channel=2);before=scene.measurements(p)
        missing=before[2][1];power=abs(missing['complex_coefficient'])**2*8192*missing['conditional_information']
        self.assertLess(np.max(power),1e-5)
        b=scene.bands[1];b.field.y[~b.training]+=before[1][1]['frozen_component_prediction'][~b.training]*(np.exp(.37j)-1)
        after=scene.measurements(p)
        ratio=after[1][1]['complex_coefficient']/before[1][1]['complex_coefficient']
        np.testing.assert_allclose(ratio[b.training],1.,atol=2e-8)
        np.testing.assert_allclose(ratio[~b.training],np.exp(.37j),atol=2e-8)
        np.testing.assert_array_equal(before[1][1]['physical_path_change_m'],after[1][1]['physical_path_change_m'])


if __name__=='__main__':unittest.main()
