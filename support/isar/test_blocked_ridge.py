import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from blocked_ridge import RidgePath, tune
from fit_blocked_ridge import main as nested_main
from fit_cached_scene import main as scene_main
from linear_observation import LinearObservation
from scene_dictionary import fingerprint


class BlockedRidgeTests(unittest.TestCase):
    def test_complex_path_matches_augmented_least_squares(self):
        rng = np.random.default_rng(222)
        a = rng.normal(size=(83, 21))+1j*rng.normal(size=(83, 21))
        y = rng.normal(size=83)+1j*rng.normal(size=83)
        strengths = [1e-5, .1, 10., 1000.]
        actual = RidgePath(a, y).coefficients(strengths)
        for j, strength in enumerate(strengths):
            expected = np.linalg.lstsq(np.r_[a, np.sqrt(strength)*np.eye(21)], np.r_[y, np.zeros(21)], rcond=None)[0]
            np.testing.assert_allclose(actual[:, j], expected, rtol=1e-10, atol=1e-12)

    def test_inner_validation_selection_matches_direct_errors_and_qr(self):
        rng = np.random.default_rng(293)
        a = rng.normal(size=(150, 30))+1j*rng.normal(size=(150, 30))
        y = a@(rng.normal(size=30)+1j*rng.normal(size=30))+3*(rng.normal(size=150)+1j*rng.normal(size=150))
        train = LinearObservation(a[:100], y[:100], 0., 100)
        validation = LinearObservation(a[100:], y[100:], 0., 50)
        grid = np.logspace(-6, 2, 17)
        strength, result = tune(train, validation, grid)
        path = RidgePath(train.design, train.data)
        coefficients = path.coefficients(result['absolute_strengths'])
        scores = [validation.error(c) for c in coefficients.T]
        np.testing.assert_allclose(result['inner_prediction_errors'], scores, rtol=1e-12)
        self.assertEqual(strength, result['absolute_strengths'][np.argmin(scores)])
        compressed = [LinearObservation.compress(o.design, o.data) for o in (train, validation)]
        check, detail = tune(*compressed, grid)
        self.assertAlmostEqual(check/strength, 1., places=12)
        np.testing.assert_allclose(detail['inner_prediction_errors'], scores, rtol=1e-12)

    def test_nested_cli_never_uses_outer_test_to_fit_or_choose_penalty(self):
        rng = np.random.default_rng(3712)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root/'cache'
            cache.mkdir()
            times = np.arange(240)*.05
            points = np.c_[np.arange(12), np.zeros(12), np.full(12, 20.)]
            manifest = {'complete': True, 'source_motion': 'synthetic', 'points_body_m': points.tolist(),
                'interval_s': [0., 12.], 'channels': []}
            for c in range(5):
                design = rng.normal(size=(720, 12))+1j*rng.normal(size=(720, 12))
                background = rng.normal(size=(720, 4))+1j*rng.normal(size=(720, 4))
                coefficient = rng.normal(size=12)+1j*rng.normal(size=12)
                y = design@coefficient+background@np.ones(4)+.1*(rng.normal(size=720)+1j*rng.normal(size=720))
                destination = cache/f'channel-{c}-design.npy'
                np.save(destination, design)
                np.savez(cache/f'channel-{c}-observation.npz', time=times, data=y.reshape(-1, 3), background=background)
                manifest['channels'].append({'channel': c, 'shape': list(design.shape), 'design_sha256': fingerprint(destination)})
            (cache/'dictionary.json').write_text(json.dumps(manifest))
            def run(name):
                statistics, output = root/(name+'-stats'), root/name
                with patch.object(sys, 'argv', ['scene', '--dictionary', str(cache), '--output', str(statistics), '--variants', 'independent']), contextlib.redirect_stdout(io.StringIO()):
                    scene_main()
                with patch.object(sys, 'argv', ['nested', '--statistics', str(statistics), '--output', str(output)]), contextlib.redirect_stdout(io.StringIO()):
                    nested_main()
                return json.loads((output/'fit.json').read_text())
            original = run('original')
            self.assertTrue(original['complete'])
            self.assertGreater(original['heldout_explained_fraction'], .99)
            outer = np.floor(times).astype(int)%2 == 1
            for c in range(5):
                destination = cache/f'channel-{c}-observation.npz'
                with np.load(destination) as f:
                    data = dict(f)
                data['data'][outer] *= 2*np.exp(.6j)
                np.savez(destination, **data)
            changed = run('changed-outer-data')
            for a, b in zip(original['channels'], changed['channels']):
                self.assertEqual(a['absolute_penalty'], b['absolute_penalty'])
                np.testing.assert_array_equal(a['coefficient_real_imag'], b['coefficient_real_imag'])
                self.assertNotEqual(a['prediction_error'], b['prediction_error'])
                self.assertLess(a['regenerated_prediction_error_relative_check'], 1e-10)


if __name__ == '__main__':
    unittest.main()
