import unittest

import numpy as np

from audit_scattered_information import information_cases
from scattered_origin_geometry import ScatteredOriginRoad
from scattered_road import ScatteredRoad
from shared_origin_geometry import OriginRoad
from test_scattered_road import fixture


class ScatteredInformationTests(unittest.TestCase):
    def test_independent_full_projection_and_unknown_offsets_cannot_add_information(self):
        rng = np.random.default_rng(1298)
        # Three antenna coordinates, two origins, independent ship/road motion.
        ship = rng.normal(size=(37, 9)); shared = np.arange(4, 9)
        roads = [(rng.normal(size=(n, 10)), 2, 5) for n in (31, 29)]
        cases = information_cases(ship, shared, np.arange(4), roads, 3, cutoff=1e-12)
        for case in cases:
            local_count = 5 if case['free_vehicle_reflectors'] else 2
            target = np.concatenate([ship[:, 4:7], *[j[:, 5:8] for j, _, _ in roads]])
            nuisance = np.zeros((len(target), 4+2*local_count+2))
            nuisance[:len(ship), :4] = ship[:, :4]
            nuisance[:, -2:] = np.concatenate([ship[:, -2:], *[j[:, -2:] for j, _, _ in roads]])
            start = len(ship)
            for i, (j, _, _) in enumerate(roads):
                nuisance[start:start+len(j), 4+i*local_count:4+(i+1)*local_count] = j[:, :local_count]
                start += len(j)
            residual = target-nuisance@np.linalg.lstsq(nuisance, target, rcond=None)[0]
            np.testing.assert_allclose(case['combined']['information'], residual.T@residual, atol=1e-12)
        difference = cases[0]['combined']['information']-cases[1]['combined']['information']
        self.assertGreater(np.linalg.eigvalsh(difference).min(), 0.)

    def test_physical_zero_offset_information_matches_original_when_offsets_known(self):
        road, truth = fixture(); original = OriginRoad(road)
        expanded = ScatteredOriginRoad(ScatteredRoad(road, [2., -1.], reference=1))
        antenna, origins = np.zeros(road.antennas.count), np.array([2., -1.])
        p = np.r_[truth, np.zeros(3)]
        old = original.evaluate(truth, antenna, origins, include_motion_prior=False)['expected_real_jacobian']
        new = expanded.evaluate(p, antenna, origins, include_motion_prior=False)['expected_real_jacobian']
        n, m, a = original.nmotion, expanded.nmotion, road.antennas.count
        np.testing.assert_allclose(new[:, np.r_[np.arange(n), np.arange(m, new.shape[1])]], old, atol=1e-12)
        ship = np.zeros((1, a+2))
        for cutoff in (1e-8, 1e-10, 1e-12, 1e-14):
            control = information_cases(ship, np.arange(a+2), np.array([], int),
                [(old, n, n)], a, cutoff=cutoff)
            tested = information_cases(ship, np.arange(a+2), np.array([], int),
                [(new, n, m)], a, cutoff=cutoff)
            before = control[0]['combined']['information']
            known, unknown = [c['combined']['information'] for c in tested]
            np.testing.assert_allclose(before, known, atol=1e-15, rtol=1e-8)
            self.assertGreaterEqual(np.linalg.eigvalsh(known-unknown).min(),
                -1e-12*np.linalg.norm(known))


if __name__ == '__main__':
    unittest.main()
