import unittest

import numpy as np

from cluster_phase import PolynomialPath
from flexible_echo_cluster import FlexibleEchoCluster
from shared_echo_cluster import ReferencedAmplitudeArc
from test_shared_echo_cluster import fixture


def cubic_scene():
        original,_=fixture();bands=[]
        for b in original.bands:
            rows=b.field.data['row_time']
            bands.append(ReferencedAmplitudeArc(b.field,b.training,b.knots,b.epoch-b.scale,b.epoch+b.scale,
                np.zeros(len(rows)),np.zeros(len(rows))))
        curve=PolynomialPath(bands[0].epoch-bands[0].scale,bands[0].epoch+bands[0].scale,3)
        model=FlexibleEchoCluster(bands,original.origins,1,curve)
        p=np.zeros(model.count)
        for c in range(3):
            for s,origin in enumerate(model.origins):
                p[6*c+2*s:6*c+2*s+2]=[origin,.3*(c-s)]
                p[model.local_path_indices[c][s]]=[.16,-.04,.12] if s==1 else [.1,-.02]
        for c,(cluster,index) in enumerate(zip(model.clusters,model.indices)):
            b=cluster.base;signal=np.zeros(b.field.y.size,complex)
            for s,(raw,_) in enumerate(cluster.response_blocks(p[index])):
                amplitude=.5+.25*np.sin(np.arange(len(b.knots))+c+s)
                signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@amplitude
            b.field.y=signal.reshape(b.field.y.shape)
        # Reconstruct the cached training data after generating the independent
        # known scene; fitting must not keep the fixture's previous observations.
        for b in bands:b.clean=b.nuisance.remove(b.field.y.ravel()[b.selected]);b.energy=float(np.vdot(b.clean,b.clean).real)
        model=FlexibleEchoCluster(bands,original.origins,1,curve)
        return model,p


class FlexibleEchoClusterTests(unittest.TestCase):
    def test_shared_cubic_path_recovery_and_exact_profiled_derivative(self):
        model,p=cubic_scene();bands=model.bands
        x=p+.001*np.sin(np.arange(model.count));r,j=model.evaluate(x)
        direction=np.cos(np.arange(model.count))*.01;step=1e-4
        finite=(model.evaluate(x+step*direction)[0]-model.evaluate(x-step*direction)[0])/(2*step)
        np.testing.assert_allclose(finite,j@direction,atol=1e-8,rtol=2e-5)
        fit=model.fit(x)
        np.testing.assert_allclose(fit['parameters'],p,atol=4e-7)
        self.assertLess(fit['cost'],1e-15)
        prediction=model.prediction(np.asarray(fit['parameters']),[~b.training for b in bands])
        self.assertTrue(all(v['heldout_explained_fraction']>1-1e-12 for v in prediction))

    def test_different_brightness_weights_bias_quadratic_coefficients_for_one_shared_path(self):
        truth,p=cubic_scene();curve=PolynomialPath(truth.curve.epoch-truth.curve.scale,
            truth.curve.epoch+truth.curve.scale,2)
        model=FlexibleEchoCluster(truth.bands,truth.origins,1,curve,shared=False)
        seed=np.zeros(model.count);n=model.components
        for c in range(len(truth.bands)):
            seed[2*n*c:2*n*(c+1)]=p[2*n*c:2*n*(c+1)]
            for s,index in enumerate(model.local_path_indices[c]):
                seed[index]=p[truth.local_path_indices[c][s]][:2]
        fit=model.fit(seed,200)
        coefficients=np.array([np.asarray(fit['parameters'])[paths[1]] for paths in model.local_path_indices])
        # There are no independent physical channel paths in the generated
        # scene. The low-order approximation and different temporal weights
        # alone create a spurious interband coefficient difference.
        self.assertGreater(np.linalg.norm(coefficients-coefficients[0]),1e-3)
        self.assertGreater(fit['cost'],1e-4)


if __name__=='__main__':unittest.main()
