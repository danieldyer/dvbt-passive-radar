"""Training-only finite-clutter covariance test on frozen strong-road motion.

The complete original time-bin observations are retained. For each of two
previously fitted physical trajectories, compare no clutter, deterministic
unregularized clutter, and a finite Gaussian clutter prior. No trajectory,
antenna, observation split or data normalization is refitted in this control.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from full_stationary_road import FullRoad
from gaussian_stationary import GaussianStationary
from geographic_io import read_profiles
from scene_dictionary import fingerprint
from stationary_sufficient_road import coefficients
from strong_road_controls import state
from wide_road_templates import stationary, widen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('controls', 'geometry-fit', 'ship-fit', 'road-fit', 'templates', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    args = parser.parse_args()
    controls, geometry, saved = [json.loads(p.read_text()) for p in (args.controls, args.geometry_fit, args.road_fit)]
    if not controls['complete']:
        raise ValueError('Complete the matched strong-road controls first')
    for group in ('sources', 'implementation_sha256'):
        for path, digest in controls['identity'][group].items():
            if fingerprint(Path(path)) != digest:
                raise ValueError('Source control changed: '+path)
    for path in (args.geometry_fit, args.ship_fit, args.road_fit, args.templates):
        if controls['identity']['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('Use the identical geometry, road observations and PSFs')
    joint, _, _, labels, _ = load_experiment(args.ship_fit, args.road_fit, args.fields,
        args.geometry, geometry['identity']['labels'], geometry['identity']['nodes'])
    antenna = np.asarray(geometry['parameters'])[joint.antenna_slice]
    _, templates = read_profiles(args.templates)
    result = {'scope': __doc__, 'complete': False, 'targets': [], 'accepted_calibration': False,
        'production_changed': False, 'source_sha256': {str(p): fingerprint(p) for p in
            (args.controls, args.geometry_fit, args.ship_fit, args.road_fit, args.templates)},
        'implementation_sha256': {str(p): fingerprint(p) for p in
            [Path(__file__), Path(__file__).with_name('gaussian_stationary.py')]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Preserve the existing covariance experiment')
    def save():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(args.output)
    started = time.monotonic()
    for target in controls['targets']:
        label = target['label']
        source, _ = widen(joint.roads[labels.index(label)], templates)
        patches = [Path(saved['input_patches'])/f'{label}-{c}.npz' for c in range(5)]
        previous, binned = stationary(source, patches)
        full = FullRoad(previous, binned)
        cases = []
        for motion_name in ('full_moving_only', 'full_with_stationary'):
            motion = state(target['fits'][motion_name])
            channels = []
            for c, (train, test) in enumerate(zip(full.training, full.validation)):
                u, _ = full.raw_template(motion, antenna, c, train)
                v, _ = full.raw_template(motion, antenna, c, test)
                prior = GaussianStationary(train.original.y, u, train.stationary)
                hyper = prior.fit()
                prediction, uncertainty = prior.predict(test.stationary, v, hyper['eta'])
                a, b = coefficients(full, motion, antenna, c)
                a0 = np.vdot(u, train.original.y)/np.vdot(u, u).real
                estimates = [('moving_only', a0*v, a0),
                    ('unregularized_stationary', a*v+test.stationary@b, a),
                    ('finite_stationary', prediction, uncertainty['amplitude'])]
                errors = {}
                for name, predicted, amplitude in estimates:
                    residual = test.original.y-predicted
                    moving_residual = test.remove(test.original.y-amplitude*v)
                    errors[name] = {'prediction_error': float(np.vdot(residual, residual).real),
                        'moving_prediction_error_with_identical_test_nuisance': float(np.vdot(moving_residual, moving_residual).real),
                        'amplitude_real_imag': [float(amplitude.real), float(amplitude.imag)]}
                uncertainty = {k: val for k, val in uncertainty.items() if k != 'amplitude'}
                channels.append({'channel': c, 'training_variances': hyper, 'errors': errors,
                    'conditional_prediction_uncertainty': uncertainty,
                    'observed_over_conditional_error': errors['finite_stationary']['prediction_error']/uncertainty['expected_prediction_error']})
            totals = {name: {metric: sum(c['errors'][name][metric] for c in channels)
                for metric in ('prediction_error', 'moving_prediction_error_with_identical_test_nuisance')}
                for name in ('moving_only', 'unregularized_stationary', 'finite_stationary')}
            expected = sum(c['conditional_prediction_uncertainty']['expected_prediction_error'] for c in channels)
            case = {'frozen_motion': motion_name, 'channels': channels, 'totals': totals,
                'observed_over_conditional_error': totals['finite_stationary']['prediction_error']/expected,
                'all_variance_fits_stationary': all(c['training_variances']['stationary'] for c in channels)}
            cases.append(case)
            print(json.dumps({'label': label, 'motion': motion_name,
                'stationary': case['all_variance_fits_stationary'], 'totals': totals,
                'observed_over_conditional_error': case['observed_over_conditional_error'],
                'seconds': time.monotonic()-started}), flush=True)
        result['targets'].append({'label': label, 'cases': cases}); save()
    result.update(complete=True, seconds=time.monotonic()-started); save()


if __name__ == '__main__':
    main()
