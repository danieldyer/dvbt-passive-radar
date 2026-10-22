from copy import deepcopy
import unittest

import numpy as np

from amplitude_arc import AmplitudeArc
from shared_echo_cluster import SharedEchoCluster,ReferencedAmplitudeArc
from test_background_context import context_fixture


def fixture():
    old,_=context_fixture();bands=[]
    for c in range(3):
        f=deepcopy(old.base.field);f.k*=1+.05*c
        bands.append(AmplitudeArc(f,old.base.training,old.base.knots,stationary_paths=[]))
    model=SharedEchoCluster(bands,[-3.,0.,4.],origin_radius=.75);p=np.zeros(model.count);n=3
    for c in range(3):
        for s in range(3):p[2*n*c+2*s:2*n*c+2*s+2]=[model.origins[s]+.03*np.sin(c+s),.4*(c-s)]
    p[-6:]=[.2,-.05,-.15,.04,.08,.03]
    strength=np.array([[.3,.8,.4,.9,.5],[.7,.3,.8,.5,1.],[.4,.2,.3,.6,.4]])
    for cluster,index in zip(model.clusters,model.indices):
        b=cluster.base;signal=np.zeros(b.field.y.size,complex)
        for (raw,_),amplitude in zip(cluster.response_blocks(p[index]),strength):
            signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@amplitude
        b.field.y=signal.reshape(b.field.y.shape)
    return SharedEchoCluster([AmplitudeArc(b.field,b.training,b.knots,stationary_paths=[]) for b in bands],model.origins,origin_radius=.75),p


class SharedEchoClusterTests(unittest.TestCase):
    def test_one_shared_point_does_not_force_neighbouring_points_to_share_motion(self):
        original,p=fixture()
        model=SharedEchoCluster(original.bands,original.origins,origin_radius=.75,shared_components=[1])
        q=np.zeros(model.count)
        for c,index in enumerate(model.indices):q[index]=p[original.indices[c]]
        # A band-specific neighbour has an independent path. The selected
        # component still has exactly one common pair of path coordinates.
        q[model.phase_indices[2,0]] += [.06,-.025]
        self.assertEqual(model.count,32)
        np.testing.assert_array_equal(model.phase_indices[:,1],np.tile(model.phase_indices[0,1],(3,1)))
        self.assertEqual(len(np.unique(model.phase_indices[:,0])),6)
        strengths=np.array([[.3,.8,.4,.9,.5],[.7,.3,.8,.5,1.],[.4,.2,.3,.6,.4]])
        for cluster,index in zip(model.clusters,model.indices):
            b=cluster.base;value=np.zeros(b.field.y.size,complex)
            for (raw,_),strength in zip(cluster.response_blocks(q[index]),strengths):
                value+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@strength
            b.field.y=value.reshape(b.field.y.shape)
        fitted=SharedEchoCluster([AmplitudeArc(b.field,b.training,b.knots,stationary_paths=[])
            for b in model.bands],model.origins,origin_radius=.75,shared_components=[1])
        fit=fitted.fit(q+.002*np.sin(np.arange(model.count)))
        np.testing.assert_allclose(fit['parameters'],q,atol=3e-7)
        self.assertLess(fit['cost'],1e-15)

    def test_independent_and_default_layouts_reproduce_the_same_local_likelihoods(self):
        common,p=fixture()
        for shared in ([],[0],[1,2]):
            model=SharedEchoCluster(common.bands,common.origins,origin_radius=.75,shared_components=shared)
            q=np.zeros(model.count)
            for c,index in enumerate(model.indices):q[index]=p[common.indices[c]]
            a,_=common.evaluate(p);b,j=model.evaluate(q)
            np.testing.assert_allclose(a,b,atol=1e-14)
            probe=q+.003*np.cos(np.arange(model.count));_,j=model.evaluate(probe)
            d=np.sin(np.arange(model.count))*.02;step=1e-4
            finite=(model.evaluate(probe+step*d)[0]-model.evaluate(probe-step*d)[0])/(2*step)
            np.testing.assert_allclose(finite,j@d,atol=1e-8,rtol=2e-5)

    def test_external_guide_coordinates_remain_fixed_during_nuisance_refit(self):
        model,p=fixture();fixed=np.arange(model.count-2,model.count);free=np.setdiff1d(np.arange(model.count),fixed)
        seed=p+.004*np.sin(np.arange(model.count));seed[fixed]=p[fixed]
        fit=model.fit(seed,free_indices=free)
        np.testing.assert_array_equal(np.asarray(fit['parameters'])[fixed],p[fixed])
        np.testing.assert_allclose(fit['parameters'],p,atol=3e-7)
        self.assertLess(fit['cost'],1e-15)

    def test_no_new_background_and_exact_three_echo_profiled_derivative(self):
        model,p=fixture()
        for b in model.bands:
            self.assertEqual(b.design.shape[1],0);self.assertEqual(b.nuisance.rank,0)
            np.testing.assert_array_equal(b.clean,b.field.y.ravel()[b.selected])
        r,j=model.evaluate(p);self.assertLess(r@r,1e-24)
        x=p+.002*np.sin(np.arange(model.count));_,j=model.evaluate(x);direction=np.cos(np.arange(model.count))*.03
        step=1e-4
        finite=(model.evaluate(x+step*direction)[0]-model.evaluate(x-step*direction)[0])/(2*step)
        np.testing.assert_allclose(finite,j@direction,atol=1e-8,rtol=2e-5)

    def test_known_three_scatterer_recovery_and_unused_prediction(self):
        model,p=fixture();fit=model.fit(p+.005*np.sin(np.arange(model.count)))
        np.testing.assert_allclose(fit['parameters'],p,atol=2e-7)
        prediction=model.prediction(np.array(fit['parameters']),[~b.training for b in model.bands])
        self.assertTrue(all(v['heldout_explained_fraction']>1-1e-12 for v in prediction))

    def test_reference_path_changes_envelope_phase_and_symbol_doppler_consistently(self):
        model,p=fixture();b=model.bands[0];rows=b.field.data['row_time'];epoch=b.epoch;scale=b.scale
        u=(rows-epoch)/scale;known=.11*u+.03*u**2;rate=(.11+.06*u)/scale
        referenced=ReferencedAmplitudeArc(b.field,b.training,b.knots,epoch-scale,epoch+scale,known,rate)
        q=np.array([.12,.3,.16,-.04]);expected=q.copy();expected[2:]+=[.11,.03]
        a,da=b.response(expected);h,dh=referenced.response(q)
        np.testing.assert_allclose(a,h,atol=2e-15,rtol=1e-12)
        np.testing.assert_allclose(da,dh,atol=2e-15,rtol=1e-12)


if __name__=='__main__':unittest.main()
