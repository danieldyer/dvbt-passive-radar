"""Full-bin calibration controls for strong road targets using measured PSF tails.

The nominal antenna coordinates, road hypotheses and archived observations stay
fixed. Three estimators compare the original moving-only Doppler gate, all
saved bins with a moving-only response, and all bins with stationary nuisance.
Every waveform prediction is evaluated on identical held-out observations.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from full_stationary_road import FullRoad, predict_at
from geographic_io import read_profiles
from integrated_stationary_road import IntegratedRoad
from road_prediction_uncertainty import prediction_noise_cost
from scene_dictionary import fingerprint
from shared_target_geometry import RoadTarget
from stationary_sufficient_road import coefficients
from validate_shared_geometry import fit_target
from wide_road_templates import stationary, widen


def state(fit):
    return np.asarray(fit['starts'][fit['selected_start']]['parameters'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('geometry-fit', 'ship-fit', 'road-fit', 'templates', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--labels', nargs='+', default=['A', 'K'])
    parser.add_argument('--iterations', type=int, default=180)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    geometry, saved = [json.loads(p.read_text()) for p in (args.geometry_fit, args.road_fit)]
    if not geometry['complete']:
        raise ValueError('Use a completed geometry control')
    for group in ('sources', 'implementation_sha256'):
        for path, digest in geometry['identity'][group].items():
            if fingerprint(Path(path)) != digest:
                raise ValueError('Geometry input changed: '+path)
    for path in (args.ship_fit, args.road_fit):
        if geometry['identity']['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('Use the same ship and road sources as the geometry control')
    joint, _, _, labels, _ = load_experiment(args.ship_fit, args.road_fit, args.fields,
        args.geometry, geometry['identity']['labels'], geometry['identity']['nodes'])
    if set(args.labels)-set(labels):
        raise ValueError('Requested targets are not part of the original geometry control')
    parameters = np.asarray(geometry['parameters'])
    antenna = parameters[joint.antenna_slice]
    header, templates = read_profiles(args.templates)
    sources = [args.geometry_fit, args.ship_fit, args.road_fit, args.templates,
        *[Path(saved['input_patches'])/f'{label}-{c}.npz' for label in args.labels for c in range(5)]]
    files = ('strong_road_controls.py', 'wide_road_templates.py', 'full_stationary_road.py',
        'stationary_sufficient_road.py', 'integrated_stationary_road.py', 'road_prediction_uncertainty.py')
    implementation = [*[Path(__file__).with_name(n) for n in files],
        Path('support/calibration/geographic_io.py'), Path('support/calibration/geographic_profile_export.cpp')]
    identity = {'sources': {str(p): fingerprint(p) for p in sources}, 'labels': args.labels,
        'iterations': args.iterations, 'stationary_cutoff': 1e-10,
        'implementation_sha256': {str(p): fingerprint(p) for p in implementation}}
    result = {'identity': identity, 'scope': __doc__, 'targets': [], 'complete': False,
        'accepted_calibration': False, 'production_changed': False,
        'template_scope': 'Wider PSF and its central range covariance are regenerated from the saved IQ. Original observation values and empirical time variances are retained; central template differences are explicitly checked and recorded.',
        'remaining_approximation': 'Moving response is evaluated at the bin mean; stationary response averages the exact original row times. Full-bin covariance remains the declared empirical model.'}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume changed strong-road controls')
    def save():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.output)
    started = time.monotonic()
    for label in args.labels:
        if any(t['label'] == label for t in result['targets']):
            continue
        i = labels.index(label)
        source, checks = widen(joint.roads[i], templates)
        patches = [Path(saved['input_patches'])/f'{label}-{c}.npz' for c in range(5)]
        previous, binned = stationary(source, patches)
        gated, full = IntegratedRoad(previous, binned), FullRoad(previous, binned)
        moving = RoadTarget(source.problem, [f.original for f in full.training],
            [f.original for f in full.validation], source.initial, source.transmitters, source.antennas)
        moving.energy = source.energy
        models = [('original_gate', source), ('full_moving_only', moving), ('full_with_stationary', full)]
        starts = [parameters[joint.road_slices[i]], source.initial]
        fits = []
        for name, model in models:
            fitted = fit_target(model, antenna, starts, args.iterations)
            fits.append(fitted)
            starts = [state(fitted), source.initial]
            print(json.dumps({'label': label, 'model': name,
                'stationary': fitted['starts'][fitted['selected_start']]['fit']['converged'],
                'seconds': time.monotonic()-started}), flush=True)
        channels = []
        for c in range(5):
            record = {'channel': c}
            for (name, model), fitted in zip(models, fits):
                motion = state(fitted)
                if model is full:
                    amplitude, beta = coefficients(full, motion, antenna, c)
                else:
                    amplitude = complex(*fitted['prediction'][c]['coefficient_real_imag'])
                    beta = np.zeros(len(full.training[c].delays), complex)
                record[name] = {space: predict_at(destination, motion, antenna, c, amplitude, beta)
                    for space, destination in [('original_gate', gated), ('full_bins', full)]}
            motion = state(fits[-1])
            u, _ = full.raw_template(motion, antenna, c, full.training[c])
            v, _ = full.raw_template(motion, antenna, c, full.validation[c])
            record['conditional_prediction_uncertainty'] = prediction_noise_cost(full.training[c], full.validation[c], u, v)
            channels.append(record)
        summary = {space: {name: {metric: sum(c[name][space][metric] for c in channels)
            for metric in ('prediction_error', 'moving_prediction_error_with_identical_test_nuisance')}
            for name, _ in models} for space in ('original_gate', 'full_bins')}
        item = {'label': label, 'template_checks': checks, 'channels': channels, 'summary': summary,
            'fits': {name: fit for (name, _), fit in zip(models, fits)},
            'selected_fits_stationary': [fit['starts'][fit['selected_start']]['fit']['converged'] for fit in fits]}
        result['targets'].append(item); save()
        print(json.dumps({'label': label, 'summary': summary, 'seconds': time.monotonic()-started}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started); save()


if __name__ == '__main__':
    main()
