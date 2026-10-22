"""Matched full-bin geometry transfer allowing per-band vehicle reflectors.

Retain the completed point experiment's data, covariance, stationary nuisance,
common origins, road identities, physical priors and excluded vehicle. Expand
only the physical vehicle mean, verify the nested objective and recorded-data
derivatives, then fit nominal/free antenna controls and frozen-geometry transfer.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from full_road_likelihood import build
from geographic_io import read_profiles
from scattered_origin_geometry import ScatteredOriginGeometry, FrozenScatteredRoad
from scene_dictionary import fingerprint
from shared_origin_geometry import OriginRoad
from shared_origin_transfer import fit_control, subset
from validate_shared_geometry import fit_target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('point-experiment', 'ship-fit', 'road-fit', 'wide-psf', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--limits', type=float, nargs=3, default=(3., 2., 2.),
        metavar=('ALONG', 'ACROSS', 'UP'))
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--iterations', type=int, default=180)
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve existing geometry experiments')
    base_paths = [args.point_experiment/name for name in ('experiment.json', 'fixed.json', 'fitted.json')]
    base, point_fixed, point_free = [json.loads(path.read_text()) for path in base_paths]
    if (not base['complete'] or any(not f['fit']['converged'] or
            f['original_row_maximum_kkt_residual'] > 1e-7 for f in (point_fixed, point_free))):
        raise ValueError('Require completed stationary point-model controls')
    old_identity = base['identity']
    if any(f['identity'] != old_identity for f in (point_fixed, point_free)):
        raise ValueError('Point-model controls are not matched')
    source_path = Path(old_identity['source']); source = json.loads(source_path.read_text())
    if fingerprint(source_path) != old_identity['source_sha256']:
        raise ValueError('The point-model source changed')
    for group in ('implementation_sha256', 'additional_sources_sha256'):
        for path, expected in old_identity[group].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError('The point-model experiment changed: '+path)
    inherited = source['identity']['legacy_identity']
    for path in (args.ship_fit, args.road_fit):
        if fingerprint(path) != inherited['sources'][str(path)]:
            raise ValueError('Use the identical original geometry sources')
    if fingerprint(args.wide_psf) != old_identity['additional_sources_sha256'][str(args.wide_psf)]:
        raise ValueError('Use identical measured pulse support')
    legacy, ship, bounds, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, source['identity']['labels'], inherited['nodes'])
    calibration, excluded = old_identity['calibration_labels'], old_identity['excluded_label']
    if dropped or excluded in calibration:
        raise ValueError('Require disjoint calibration and excluded targets')
    saved = json.loads(args.road_fit.read_text()); patches = Path(saved['input_patches'])
    records = read_profiles(args.wide_psf)[1]; exact_roads = {}; checks = {}
    for label in [*calibration, excluded]:
        i = labels.index(label)
        fast, exact, check = build(legacy.roads[i], [patches/f'{label}-{c}.npz' for c in range(5)], records)
        legacy.roads[i] = fast; exact_roads[i] = exact; checks[label] = check
        if check != old_identity['full_bin_road_checks'][label]:
            raise ValueError('Vehicle observation construction changed: '+label)
    indices = [labels.index(label) for label in calibration]
    fast, exact, bounds = subset(legacy, ship, bounds, source['parameters'], indices)
    exact.roads = [OriginRoad(exact_roads[i]) for i in indices]
    fast.initial = exact.initial = np.asarray(point_fixed['parameters']).copy()
    base_value = exact.evaluate(exact.initial)
    np.testing.assert_allclose(base_value['value'], point_fixed['fit']['objective'], atol=1e-11, rtol=0)
    reference = 1
    expanded_fast = ScatteredOriginGeometry(fast, limits=args.limits, reference=reference)
    expanded = ScatteredOriginGeometry(exact, limits=args.limits, reference=reference)
    value = expanded.evaluate(expanded.initial)
    np.testing.assert_allclose(value['value'], base_value['value'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(value['gradient'][expanded.base_columns], base_value['gradient'], atol=1e-11, rtol=0)
    expanded_bounds = expanded.expand_bounds(bounds)
    source_files = [*base_paths, source_path, args.ship_fit, args.road_fit, args.wide_psf]
    modules = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ('scattered_origin_geometry.py', 'scattered_road.py', 'shared_origin_transfer.py',
         'shared_origin_geometry.py', 'full_road_likelihood.py', 'fit_physical_profile.py',
         'stationarity.py', 'validate_shared_geometry.py')]]
    identity = {**old_identity, 'scope': __doc__, 'source': str(base_paths[1]),
        'source_sha256': fingerprint(base_paths[1]), 'point_experiment_identity': old_identity,
        'road_observation_model': old_identity['road_observation_model']+
            ' Additional constant per-band vehicle reflecting points; SBS fixes the trajectory gauge.',
        'reflector_extent_along_across_up_m': list(args.limits), 'reference_channel': reference,
        'reflector_prior': 'Box sensitivity bounds only; original road/height uncertainty retained. No measured vehicle dimensions or additional Gaussian offset prior.',
        'implementation_sha256': {str(path): fingerprint(path) for path in modules},
        'additional_sources_sha256': {str(path): fingerprint(path) for path in source_files}}
    args.output.mkdir(parents=True)
    state = {'complete': False, 'identity': identity, 'accepted_calibration': False,
        'nested_value_difference': value['value']-base_value['value'],
        'nested_gradient_maximum_difference': float(np.max(abs(
            value['gradient'][expanded.base_columns]-base_value['gradient'])))}
    (args.output/'experiment.json').write_text(json.dumps(state, indent=2)+'\n')
    # Audit the newly introduced physical point coordinates and common origin
    # on the exact recorded-row likelihood before starting an expensive fit.
    q = expanded.initial.copy(); first = expanded.road_slices[0]
    q[first.stop-3:first.stop] = [.5, -.25, .1]
    observed = expanded.evaluate(q); derivative_checks = []
    for coordinate in (first.stop-3, first.stop-1, expanded.range_slice.start):
        delta = np.eye(len(q))[coordinate]*1e-3
        numerical = (expanded.objective(q+delta)[0]-expanded.objective(q-delta)[0])/.002
        analytic = observed['gradient'][coordinate]
        difference = abs(numerical-analytic)
        if difference > 3e-9+abs(numerical)*5e-4:
            raise ValueError('Recorded-data physical derivative audit failed')
        derivative_checks.append({'coordinate': coordinate, 'analytic': float(analytic),
            'numerical': numerical, 'absolute_error': difference})
        print(json.dumps({'stage': 'recorded_derivative', **derivative_checks[-1],
            'seconds': time.monotonic()-started}), flush=True)
    state['recorded_derivative_checks'] = derivative_checks
    (args.output/'experiment.json').write_text(json.dumps(state, indent=2)+'\n')
    fixed = fit_control(expanded_fast, expanded, expanded_bounds, expanded.initial,
        True, args.iterations, args.output/'fixed.json', identity, calibration)
    if not fixed['fit']['converged']:
        raise ValueError('Expanded fixed-antenna control failed stationarity')
    free = fit_control(expanded_fast, expanded, expanded_bounds, fixed['parameters'],
        False, args.iterations, args.output/'fitted.json', identity, calibration)
    if not free['fit']['converged']:
        raise ValueError('Expanded free-antenna control failed stationarity')
    offsets = []
    for case in (fixed, free):
        parameters = np.asarray(case['parameters'])
        offsets.append({label: road.original.offsets(parameters[section]).tolist()
            for label, road, section in zip(calibration, expanded.roads, expanded.road_slices)})
    road = exact_roads[labels.index(excluded)]; old = saved['labels'].index(excluded)
    extension = np.zeros(3*(len(road.training)-1))
    starts = [np.r_[saved['selected_roads'][old]['state'], extension], np.r_[road.initial, extension]]
    transfers = []
    for case in (fixed, free):
        target = FrozenScatteredRoad(road, case['range_origins_m'], limits=args.limits, reference=reference)
        parameters = np.asarray(case['parameters'])
        fit = fit_target(target, parameters[expanded.antenna_slice], starts, args.iterations)
        transfers.append(fit)
        print(json.dumps({'stage': 'excluded_target', 'fixed_antennas': case['fixed_antennas'],
            'stationary': fit['starts'][fit['selected_start']]['fit']['converged'],
            'seconds': time.monotonic()-started}), flush=True)
    errors = [sum(c['prediction_error'] for c in t['prediction']) for t in transfers]
    energies = [sum(c['matched_energy'] for c in t['prediction']) for t in transfers]
    stationary = [t['starts'][t['selected_start']]['fit']['converged'] for t in transfers]
    state.update(complete=bool(all(stationary)), transfer=transfers,
        fitted_reflector_offsets_along_across_up_m=offsets,
        summary={'prediction_error_change_percent': 100*(errors[1]/errors[0]-1),
            'matched_energy_change_db': float(10*np.log10(energies[1]/energies[0])),
            'selected_fits_stationary': stationary}, seconds=time.monotonic()-started)
    (args.output/'experiment.json').write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', **state['summary'], 'seconds': state['seconds']}), flush=True)


if __name__ == '__main__':
    main()
