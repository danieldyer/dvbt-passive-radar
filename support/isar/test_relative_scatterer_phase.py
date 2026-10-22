import unittest

import numpy as np

from relative_scatterer_phase import RelativeScattererPhase
from motion import C,excess_path
import test_flexible_echo_cluster as fixture


class RelativeScattererPhaseTests(unittest.TestCase):
    def test_referenced_carriers_from_exact_two_leg_geometry_obey_the_restriction(self):
        base,p=fixture.cubic_scene();model=RelativeScattererPhase(base,0,1,reference_band=1)
        receiver=np.array([0.,0.,40.]);points=np.array([[2000.,-11000.,12.],[2050.,-10960.,30.]])
        for c,b in enumerate(base.bands):
            transmitter=np.array([31000.+65*c,-2000.+43*c,740.+7*c])
            path=excess_path(points,transmitter,receiver)[0];observer=path[0]-18.4*c
            # Independent physical propagation and a common intrinsic phase
            # per scatterer, with arbitrary per-band constant system phases.
            p[model.origin_indices[c]]=(path-observer)*8e6/C
            p[model.phase_indices[c]]=[.7,-.9]+np.full(2,1.3*c)-b.field.k*(path-observer)
        mapped=model.mapping(p[model.kept])
        np.testing.assert_allclose(np.exp(1j*mapped[model.phase_indices]),np.exp(1j*p[model.phase_indices]),atol=3e-13)

    def test_mapping_preserves_physical_relative_phase_for_unequal_carriers(self):
        base,p=fixture.cubic_scene();model=RelativeScattererPhase(base,0,1,reference_band=1)
        q=model.seed(p,.7);mapped=model.mapping(q)
        r=mapped[model.origin_indices];phase=mapped[model.phase_indices]
        intrinsic=phase[:,1]-phase[:,0]+model.phase_per_sample*(r[:,1]-r[:,0])
        np.testing.assert_allclose(intrinsic,intrinsic[1],atol=3e-13)
        # Acquiring a phase-compatible range branch does not replace the
        # independent temporal paths or align their slopes/curvature.
        np.testing.assert_array_equal(mapped[2*base.components*len(base.bands):],p[2*base.components*len(base.bands):])
        np.testing.assert_allclose(np.exp(1j*mapped[model.phase_indices]),np.exp(1j*p[model.phase_indices]),atol=3e-13)

    def test_exact_profiled_jacobian_retains_envelope_carrier_chain_rule(self):
        base,p=fixture.cubic_scene();model=RelativeScattererPhase(base,0,1,reference_band=1)
        q=model.seed(p,.2);direction=.0001*np.cos(np.arange(model.count));h=1e-4
        r,j=model.evaluate(q)
        finite=(model.evaluate(q+h*direction)[0]-model.evaluate(q-h*direction)[0])/(2*h)
        np.testing.assert_allclose(finite,j@direction,rtol=3e-5,atol=2e-8)

    def test_noiseless_common_points_recover_and_predict_unused_complex_data(self):
        base,p=fixture.cubic_scene();model=RelativeScattererPhase(base,0,1,reference_band=1)
        q=model.seed(p,.1);truth=model.mapping(q)
        # Generate an explicit constant-intrinsic-phase multiband scene; the
        # existing independent fixture deliberately did not assume this.
        for c,(cluster,index) in enumerate(zip(base.clusters,base.indices)):
            b=cluster.base;signal=np.zeros(b.field.y.size,complex)
            for s,(raw,_) in enumerate(cluster.response_blocks(truth[index])):
                strength=.7+.15*np.sin(np.arange(len(b.knots))+c+s)
                signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@strength
            b.field.y=signal.reshape(b.field.y.shape);b.clean=b.nuisance.remove(signal[b.selected])
        base.last=None
        for cluster in base.clusters:cluster.last=None
        fitted,audit=model.fit(q+.00001*np.sin(np.arange(model.count)),max_evaluations=100)
        np.testing.assert_allclose(fitted,q,atol=2e-6)
        self.assertLess(audit['objective'],1e-15)
        predictions=model.prediction(fitted,[~b.training for b in base.bands])
        self.assertTrue(all(v['heldout_explained_fraction']>1-1e-12 for v in predictions))


if __name__=='__main__':unittest.main()
