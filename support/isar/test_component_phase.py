from copy import deepcopy
import unittest

import numpy as np

from component_phase import component_measurements
from fixed_range_field import stationary_profile_design
from shared_echo_cluster import ReferencedAmplitudeArc, SharedEchoCluster
from test_fixed_range_field import recovery_fixture


def fixture():
    b, _ = recovery_fixture()
    make = lambda field: ReferencedAmplitudeArc(field, b.training, b.knots,
        b.epoch-b.scale, b.epoch+b.scale, b.known_path, b.known_rate,
        stationary_design=stationary_profile_design(field))
    model = SharedEchoCluster([b], [-4., .3, 4.5])
    p = np.array([-4., .3, .3, -.4, 4.5, .8, 3.7, .13, -2.2, .06, 1.9, -.05])
    cluster = model.clusters[0]
    signal = np.zeros(b.field.y.size, complex)
    for (raw, _), strength in zip(cluster.response_blocks(p[model.indices[0]]), (.7, 1., .9)):
        signal += b.reduced(raw[:, :, None]*b.amplitude_basis[:, None, :])@np.full(len(b.knots), strength)
    static = b.design@((np.arange(17)+1j*np.arange(17)[::-1])*100)
    f = deepcopy(b.field)
    f.y = (signal+static).reshape(f.y.shape)
    return SharedEchoCluster([make(f)], [-4., .3, 4.5]), p


class ComponentPhaseTests(unittest.TestCase):
    def test_separate_three_components_and_restore_physical_phase(self):
        model, p = fixture()
        cluster = model.clusters[0]
        measured = component_measurements(cluster, p[model.indices[0]])
        for record, strength in zip(measured, (.7, 1., .9)):
            restored = record['complex_coefficient']*np.exp(1j*cluster.base.field.k*record['physical_path_change_m'])
            valid = record['conditional_information']>0
            np.testing.assert_allclose(restored[valid], strength, atol=2e-9, rtol=0)
            np.testing.assert_array_equal(valid, cluster.base.field.data['information']>0)
            np.testing.assert_array_equal(restored[~valid], 0.)

    def test_unused_phase_is_measured_and_never_replaced_by_fitted_curve(self):
        model, p = fixture()
        cluster = model.clusters[0]
        local = p[model.indices[0]]
        original = component_measurements(cluster, local)
        b = cluster.base
        delta = .31
        b.field.y[~b.training] += original[1]['frozen_component_prediction'][~b.training]*(np.exp(1j*delta)-1)
        changed = component_measurements(cluster, local)
        valid = original[1]['conditional_information']>0
        ratio = np.divide(changed[1]['complex_coefficient'], original[1]['complex_coefficient'],
            out=np.zeros(len(valid), complex), where=valid)
        np.testing.assert_allclose(ratio[b.training&valid], 1., atol=2e-11)
        np.testing.assert_allclose(ratio[~b.training&valid], np.exp(1j*delta), atol=2e-11)
        for a, z in zip(original, changed):
            np.testing.assert_array_equal(a['frozen_component_prediction'], z['frozen_component_prediction'])


if __name__ == '__main__':
    unittest.main()
