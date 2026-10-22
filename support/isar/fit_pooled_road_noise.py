"""Matched recorded-data check of pooled vehicle noise-scale estimation.

Both full-bin likelihoods use the same measured observations, pulse support,
stationary dictionary, nominal geometry, common origins and motion starts.
Only the declared bin variances differ. All predictions are scored on both
identical test operators, with no refit of moving/static coefficients on test.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from full_road_likelihood import build
from geographic_io import read_profiles
from pooled_road_noise import replace
from scene_dictionary import fingerprint
from shared_origin_geometry import OriginRoad
from shared_origin_transfer import FrozenOriginRoad
from shared_target_geometry import RoadTarget
from validate_shared_geometry import fit_target


def prediction(fitted, scoring, motion, antenna, origins):
    results = []
    for c, train in enumerate(fitted.training):
        u = fitted.template(motion, antenna, origins, c, train)[0]
        amplitude = np.vdot(u, train.y)/np.vdot(u, u).real
        raw_u = fitted.template(motion, antenna, origins, c, train.original)[0]
        static = train.coefficients(train.original.y-amplitude*raw_u)
        row = {}
        for name, model in scoring.items():
            test = model.validation[c]
            np.testing.assert_array_equal(train.delays, test.delays)
            v = model.template(motion, antenna, origins, c, test.original)[0]
            moving = test.remove(amplitude*v)
            error = test.y-moving
            waveform_error = test.original.y-amplitude*v-test.stationary@static
            norm = np.vdot(moving, moving).real
            row[name] = {'moving_prediction_error': float(np.vdot(error, error).real),
                'moving_matched_energy': float(abs(np.vdot(moving, test.y))**2/norm) if norm else 0.,
                'full_waveform_prediction_error': float(np.vdot(waveform_error, waveform_error).real)}
        results.append(row)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'wide-psf', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--labels', nargs='+', default=['A', 'O', 'K', 'Q'])
    p.add_argument('--iterations', type=int, default=180)
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve existing experiments')
    source, saved = [json.loads(path.read_text()) for path in (args.source, args.road_fit)]
    if not source['complete'] or not source['fit']['converged'] or source['original_row_maximum_kkt_residual'] > 1e-7:
        raise ValueError('Require a stationary common-origin source')
    identity = source['identity']; inherited = identity['legacy_identity']
    for path in (args.ship_fit, args.road_fit):
        if fingerprint(path) != inherited['sources'][str(path)]:
            raise ValueError('Geometry source changed')
    joint, _, _, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], inherited['nodes'])
    if dropped or any(label not in labels for label in args.labels):
        raise ValueError('Require the same disjoint road hypotheses')
    parameters = np.asarray(source['parameters']); antenna = parameters[joint.antenna_slice]
    origins = np.asarray(source['range_origins_m']); patches = Path(saved['input_patches'])
    records = read_profiles(args.wide_psf)[1]
    paths = [args.source, args.ship_fit, args.road_fit, args.wide_psf, Path(__file__),
        *[Path(__file__).with_name(name) for name in ('full_road_likelihood.py', 'pooled_road_noise.py',
            'profiled_road_field.py', 'shared_origin_geometry.py', 'shared_origin_transfer.py',
            'validate_shared_geometry.py', 'fit_physical_profile.py')],
        *[patches/f'{label}-{c}.npz' for label in args.labels for c in range(5)]]
    result = {'complete': False, 'scope': __doc__, 'targets': [],
        'source_sha256': {str(path): fingerprint(path) for path in paths},
        'accepted_calibration': False, 'production_changed': False,
        'limitations': 'Both covariance estimates use this recording. The new estimate is independent of the fitted bin mean only under its stated temporal mean and separable Gaussian covariance. Test observation operators are fixed and identical within each comparison. Real scattering, road associations, covariance shape and within-bin mean variation remain conditional.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        index = labels.index(label); road = joint.roads[index]
        files = [patches/f'{label}-{c}.npz' for c in range(5)]
        baseline, baseline_exact, template_check = build(road, files, records)
        raw = RoadTarget(road.problem, [f.original for f in baseline.training],
            [f.original for f in baseline.validation], road.initial, road.transmitters, road.antennas)
        pooled, checks = replace(raw, files)
        revised, revised_exact, _ = build(pooled, files)
        alternatives = {'empirical_centre': baseline, 'pooled_innovations': revised}
        exact = dict(zip(alternatives, (OriginRoad(baseline_exact), OriginRoad(revised_exact))))
        models = {name: OriginRoad(model) for name, model in alternatives.items()}
        starts = [parameters[joint.road_slices[index]], road.initial]
        trials = []
        for name, model in alternatives.items():
            fit = fit_target(FrozenOriginRoad(model, origins), antenna, starts, args.iterations)
            selected = fit['starts'][fit['selected_start']]; motion = np.asarray(selected['parameters'])
            scores = prediction(models[name], models, motion, antenna, origins)
            losses = []
            for c, field in enumerate(model.validation):
                a = models[name].template(motion, antenna, origins, c, field)[0]
                b = exact[name].template(motion, antenna, origins, c, exact[name].validation[c])[0]
                ratio = abs(np.vdot(a, b))**2/(np.vdot(a, a).real*np.vdot(b, b).real)
                losses.append(float(-10*np.log10(min(1., ratio))))
            trials.append({'noise_model': name, 'fit': fit, 'prediction': scores,
                'maximum_original_row_moving_template_loss_db': max(losses)})
            print(json.dumps({'label': label, 'noise_model': name,
                'stationary': selected['fit']['converged'], 'kkt': selected['fit']['maximum_kkt_residual'],
                'seconds': time.monotonic()-started}), flush=True)
        comparisons = {}
        for name in models:
            before, after = [{key: sum(c[name][key] for c in trial['prediction']) for key in
                ('moving_prediction_error', 'moving_matched_energy', 'full_waveform_prediction_error')}
                for trial in trials]
            comparisons[name] = {key+'_change_percent': 100*(after[key]/before[key]-1) for key in before}
            comparisons[name]['moving_matched_energy_change_db'] = float(
                10*np.log10(after['moving_matched_energy']/before['moving_matched_energy']))
        result['targets'].append({'label': label, 'trials': trials, 'comparison': comparisons,
            'template_check': template_check, 'pooled_noise_checks': checks})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    result.update(complete=True, seconds=time.monotonic()-started,
        all_selected_fits_stationary=all(trial['fit']['starts'][trial['fit']['selected_start']]['fit']['converged']
            for target in result['targets'] for trial in target['trials']))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
