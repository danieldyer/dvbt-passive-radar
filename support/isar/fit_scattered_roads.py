"""Test fixed per-band vehicle reflecting points before fitting antenna sites.

All observations, full-bin covariance, static nuisance, common origins and
nominal antennas remain fixed. A shared translating trajectory is retained.
SBS defines its reflecting-point gauge; other bands have bounded constant
offsets in a fixed road-aligned frame. Two declared extents are sensitivity
hypotheses, not measured vehicle sizes. No rotations or arbitrary phase curves
are fitted, and withheld blocks never select the numerical start.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from full_road_likelihood import build
from geographic_io import read_profiles
from scattered_road import ScatteredRoad
from scene_dictionary import fingerprint
from validate_shared_geometry import fit_target


def selected(fit):
    return fit['starts'][fit['selected_start']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'wide-psf', 'baseline', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--labels', nargs='+', default=['A', 'O', 'K', 'Q'])
    p.add_argument('--iterations', type=int, default=180)
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve existing scene-model experiments')
    source, saved, previous = [json.loads(path.read_text()) for path in
        (args.source, args.road_fit, args.baseline)]
    if (not source['complete'] or not source['fit']['converged']
            or source['original_row_maximum_kkt_residual'] > 1e-7
            or not previous['complete'] or not previous['all_selected_fits_stationary']):
        raise ValueError('Require stationary common-origin and matched point-model sources')
    for path, expected in previous['source_sha256'].items():
        if fingerprint(Path(path)) != expected:
            raise ValueError('Baseline scientific input changed: '+path)
    for path in (args.source, args.ship_fit, args.road_fit, args.wide_psf):
        if fingerprint(path) != previous['source_sha256'][str(path)]:
            raise ValueError('Use identical point-model source files')
    identity = source['identity']; inherited = identity['legacy_identity']
    joint, _, _, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], inherited['nodes'])
    if dropped or any(label not in labels for label in args.labels):
        raise ValueError('Use the same disjoint road hypotheses')
    parameters = np.asarray(source['parameters']); antenna = parameters[joint.antenna_slice]
    origins = np.asarray(source['range_origins_m']); patches = Path(saved['input_patches'])
    records = read_profiles(args.wide_psf)[1]
    sources = [args.source, args.ship_fit, args.road_fit, args.baseline, args.wide_psf, Path(__file__),
        *[Path(__file__).with_name(name) for name in ('scattered_road.py', 'full_road_likelihood.py',
            'shared_target_geometry.py', 'validate_shared_geometry.py', 'fit_physical_profile.py')]]
    result = {'complete': False, 'scope': __doc__, 'targets': [],
        'source_sha256': {str(path): fingerprint(path) for path in sources},
        'accepted_calibration': False, 'production_changed': False,
        'reference_channel': 1, 'offset_prior': 'No additional Gaussian offset prior; only the declared box extents. Original road/motion priors are unchanged.',
        'literature_context': {'url': 'https://www.iwaenc.org/proceedings/1997/nsip97/pdf/scan/ns970424.pdf',
            'scope': 'Gerry et al. model frequency/aspect-dependent localized and distributed scattering at high frequency. This motivates testing the shared-point assumption, but does not validate a one-point-per-band model for these VHF vehicles. Exact bistatic paths are used here rather than their monostatic far-field phase approximation.'}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        index = labels.index(label); road = joint.roads[index]
        fast, exact, checks = build(road, [patches/f'{label}-{c}.npz' for c in range(5)], records)
        record = next(t for t in previous['targets'] if t['label'] == label)
        point_fit = next(t['fit'] for t in record['trials'] if t['noise_model'] == 'empirical_centre')
        point_state = np.asarray(selected(point_fit)['parameters'])
        seed = ScatteredRoad(fast, origins)
        point = np.r_[point_state, np.zeros(seed.nmotion-fast.nmotion)]
        np.testing.assert_allclose(seed.evaluate(point, antenna)['value']/seed.energy,
            selected(point_fit)['fit']['objective'], rtol=2e-11, atol=2e-12)
        point_prediction = seed.predict(point, antenna)
        for a, b in zip(point_prediction, point_fit['prediction']):
            for key in ('prediction_error', 'matched_energy'):
                np.testing.assert_allclose(a[key], b[key], rtol=2e-10, atol=2e-10)
        trials = []; warm = point
        for name, limits in [('compact', (3., 2., 2.)), ('broad', (10., 3., 3.))]:
            model = ScatteredRoad(fast, origins, limits=limits)
            # The broad nested model includes the preceding compact solution.
            # Both retain the original common-point start, selected on training.
            fit = fit_target(model, antenna, [warm, point], args.iterations)
            best = selected(fit); state = np.asarray(best['parameters'])
            exact_model = ScatteredRoad(exact, origins, limits=limits)
            losses = []
            for c, field in enumerate(model.validation):
                a = model.template(state, antenna, c, field)[0]
                b = exact_model.template(state, antenna, c, exact_model.validation[c])[0]
                ratio = abs(np.vdot(a, b))**2/(np.vdot(a, a).real*np.vdot(b, b).real)
                losses.append(float(-10*np.log10(min(1., ratio))))
            before_error = sum(c['prediction_error'] for c in point_prediction)
            before_energy = sum(c['matched_energy'] for c in point_prediction)
            error = sum(c['prediction_error'] for c in fit['prediction'])
            energy = sum(c['matched_energy'] for c in fit['prediction'])
            offsets = model.offsets(state)
            trial = {'name': name, 'limits_along_across_up_m': list(limits), 'fit': fit,
                'offsets_along_across_up_m': offsets.tolist(), 'axes_enu': model.axes.tolist(),
                'offset_bounds_hit': int(np.sum(np.isclose(abs(offsets), np.asarray(limits), atol=1e-5, rtol=0))),
                'comparison': {'prediction_error_change_percent': 100*(error/before_error-1),
                    'matched_energy_change_db': float(10*np.log10(energy/before_energy))},
                'maximum_original_row_moving_template_loss_db': max(losses)}
            trials.append(trial); warm = state
            print(json.dumps({'label': label, 'model': name, 'stationary': best['fit']['converged'],
                'kkt': best['fit']['maximum_kkt_residual'], 'comparison': trial['comparison'],
                'seconds': time.monotonic()-started}), flush=True)
        result['targets'].append({'label': label, 'point_fit': point_fit,
            'point_prediction': point_prediction, 'template_checks': checks, 'trials': trials})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    result.update(complete=True, seconds=time.monotonic()-started,
        all_selected_fits_stationary=all(selected(trial['fit'])['fit']['converged']
            for target in result['targets'] for trial in target['trials']))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
