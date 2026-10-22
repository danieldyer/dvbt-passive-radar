import unittest

import numpy as np

from cluster_phase import PolynomialPath
from flexible_echo_cluster import FlexibleEchoCluster
from smooth_echo_cluster import SmoothEchoCluster,jerk_factor
from test_flexible_echo_cluster import cubic_scene


class MultiFlexiblePathTests(unittest.TestCase):
    def fixture(self):
        source,p=cubic_scene();curve=source.curve
        model=SmoothEchoCluster(source.bands,source.origins,(1,2),curve,
            shared_components=(1,),jerk_std=.03)
        q=np.zeros(model.count);n=source.components
        q[:2*n*len(source.bands)]=p[:2*n*len(source.bands)]
        for c,b in enumerate(model.bands):
            for s,index in enumerate(model.local_path_indices[c]):
                old=p[source.local_path_indices[c][s]]
                q[index]=old if len(old)==len(index) else np.r_[old,0.]
        return source,p,model,q

    def test_shared_and_independent_paths_reproduce_the_original_operator(self):
        source,p,model,q=self.fixture()
        for old,new,oi,ni in zip(source.clusters,model.clusters,source.indices,model.indices):
            for (a,_),(b,_) in zip(old.response_blocks(p[oi]),new.response_blocks(q[ni])):
                np.testing.assert_allclose(a,b,atol=1e-14)
        np.testing.assert_array_equal(model.local_path_indices[0][1],model.local_path_indices[1][1])
        self.assertFalse(np.array_equal(model.local_path_indices[0][2],model.local_path_indices[1][2]))

    def test_motion_priors_count_each_distinct_path_once_and_keep_exact_jacobian(self):
        _,_,model,q=self.fixture();q=q+.001*np.cos(np.arange(len(q)))
        r,j=model.evaluate(q);factor=jerk_factor(model.curve)/.03
        paths=[model.local_path_indices[0][1],*[p[2] for p in model.local_path_indices]]
        expected=sum(np.linalg.norm(factor@q[p])**2/2 for p in paths)
        self.assertAlmostEqual(model.prior_cost(q),expected,places=11)
        direction=.002*np.sin(np.arange(len(q)));step=1e-4
        numerical=(model.evaluate(q+step*direction)[0]-model.evaluate(q-step*direction)[0])/(2*step)
        np.testing.assert_allclose(numerical,j@direction,rtol=3e-5,atol=1e-8)

    def test_recover_common_point_with_independently_curved_neighbours(self):
        _,_,template,truth=self.fixture()
        model=FlexibleEchoCluster(template.bands,template.origins,(1,2),template.curve,shared_components=(1,))
        for c in range(len(model.bands)):
            truth[model.local_path_indices[c][2]]+=c*np.array([.03,-.02,.07])
        for c,(cluster,index) in enumerate(zip(model.clusters,model.indices)):
            b=cluster.base;signal=np.zeros(b.field.y.size,complex)
            for s,(raw,_) in enumerate(cluster.response_blocks(truth[index])):
                amplitude=.5+.2*np.sin(np.arange(len(b.knots))+c+s)
                signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@amplitude
            b.field.y=signal.reshape(b.field.y.shape)
            b.clean=b.nuisance.remove(signal[b.selected]);b.energy=float(np.vdot(b.clean,b.clean).real)
        model=FlexibleEchoCluster(model.bands,model.origins,(1,2),model.curve,shared_components=(1,))
        fit=model.fit(truth+.002*np.sin(np.arange(model.count)))
        np.testing.assert_allclose(fit['parameters'],truth,atol=5e-7)
        self.assertLess(fit['cost'],1e-15)
        prediction=model.prediction(np.asarray(fit['parameters']),[~b.training for b in model.bands])
        self.assertTrue(all(v['heldout_explained_fraction']>1-1e-12 for v in prediction))

    def test_quadratic_neighbour_information_uses_its_own_basis_in_a_mixed_model(self):
        from release_fixed_grid_phase_guide import path_information
        _,_,model,q=self.fixture();cluster=model.clusters[0]
        value=cluster.evaluate(q[model.indices[0]]);j=value['mean_derivative']
        covariance=np.linalg.solve(2*j.T@j,np.eye(j.shape[1]))
        info=path_information(model,q,component=0)
        self.assertEqual(info['target_rank'],2)
        block=cluster.slices[0];curve=cluster.arcs[0].base.curve
        g=np.zeros((len(info['time_s']),j.shape[1]));g[:,block.start+2:block.stop]=curve(info['time_s'])
        variance=np.einsum('ni,ij,nj->n',g,covariance,g)
        np.testing.assert_allclose(np.array(info['path_standard_deviation_m'])**2,variance,rtol=2e-7,atol=1e-16)

    def test_two_shared_paths_with_an_independent_neighbour_recover_and_predict(self):
        source,p=cubic_scene();model=FlexibleEchoCluster(source.bands,source.origins,(1,2),source.curve,shared_components=(1,2))
        truth=np.zeros(model.count);truth[:18]=p[:18]
        for c in range(len(model.bands)):
            truth[model.local_path_indices[c][0]]=[.1+.03*c,-.02]
        truth[model.local_path_indices[0][1]]=[.16,-.04,.12]
        truth[model.local_path_indices[0][2]]=[-.12,.03,-.08]
        for c,(cluster,index) in enumerate(zip(model.clusters,model.indices)):
            b=cluster.base;signal=np.zeros(b.field.y.size,complex)
            for s,(raw,_) in enumerate(cluster.response_blocks(truth[index])):
                a=.6+.2*np.sin(np.arange(len(b.knots))+c+s)
                signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@a
            b.field.y=signal.reshape(b.field.y.shape);b.clean=b.nuisance.remove(signal[b.selected])
        model=FlexibleEchoCluster(model.bands,model.origins,(1,2),model.curve,shared_components=(1,2))
        fitted=model.fit(truth+.0001*np.sin(np.arange(model.count)))
        np.testing.assert_allclose(fitted['parameters'],truth,atol=1e-6)
        self.assertLess(fitted['cost'],1e-15)
        prediction=model.prediction(np.array(fitted['parameters']),[~b.training for b in model.bands])
        self.assertTrue(all(v['heldout_explained_fraction']>1-1e-12 for v in prediction))
        smooth=SmoothEchoCluster(model.bands,model.origins,(1,2),model.curve,shared_components=(1,2),jerk_std=.03)
        factor=jerk_factor(model.curve)/.03
        expected=sum(np.linalg.norm(factor@truth[model.local_path_indices[0][s]])**2/2 for s in (1,2))
        self.assertAlmostEqual(smooth.prior_cost(truth),expected,places=10)


if __name__=='__main__':unittest.main()
