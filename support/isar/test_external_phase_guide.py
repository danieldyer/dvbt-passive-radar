import unittest

import numpy as np

from component_phase import component_measurements
from fit_flexible_flash import diagnostics
from test_flexible_echo_cluster import cubic_scene


class ExternalPhaseGuideTests(unittest.TestCase):
    def test_path_is_frozen_but_unused_measured_phase_is_not(self):
        model,truth=cubic_scene()
        fixed=model.local_path_indices[0][model.selected]
        free=np.setdiff1d(np.arange(model.count),fixed)
        seed=truth+.003*np.sin(np.arange(model.count));seed[fixed]=truth[fixed]
        fit=model.fit(seed,free_indices=free)
        p=np.asarray(fit['parameters'])
        np.testing.assert_array_equal(p[fixed],truth[fixed])
        np.testing.assert_allclose(p,truth,atol=5e-7)
        self.assertLess(fit['cost'],1e-20)
        parts=[np.where(b.training,0,3) for b in model.bands]
        before=diagnostics(model,p,parts)
        cluster=model.clusters[2];band=cluster.base;value=cluster.evaluate(p[model.indices[2]])
        measured_before=component_measurements(cluster,p[model.indices[2]])
        k=len(band.knots);s=model.selected
        prediction=(value['design'][:,s*k:(s+1)*k]@value['amplitude'][s*k:(s+1)*k]).reshape(band.field.y.shape)
        band.field.y[~band.training]+=prediction[~band.training]*(np.exp(.31j)-1)
        after=diagnostics(model,p,parts)
        measured_after=component_measurements(cluster,p[model.indices[2]])
        # Real brightness varies inside a retained 16-row mean, so the phase
        # against a unit-brightness template need not be exactly zero even for
        # noiseless data. The measured change must still equal the injected
        # rotation, with no change to the fitted external path.
        change=np.radians(np.asarray(after[2]['phase_residual_deg'])-before[2]['phase_residual_deg'])
        np.testing.assert_allclose(np.angle(np.exp(1j*change[~band.training])),.31,atol=1e-13)
        ratio=measured_after[s]['complex_coefficient']/measured_before[s]['complex_coefficient']
        np.testing.assert_allclose(ratio[~band.training],np.exp(.31j),atol=1e-13)
        for a,b in zip(measured_before,measured_after):
            np.testing.assert_array_equal(a['frozen_component_prediction'],b['frozen_component_prediction'])
        np.testing.assert_array_equal(p[fixed],truth[fixed])
        for a,b in zip(before[:2],after[:2]):
            np.testing.assert_array_equal(a['phase_residual_deg'],b['phase_residual_deg'])


if __name__=='__main__':unittest.main()
