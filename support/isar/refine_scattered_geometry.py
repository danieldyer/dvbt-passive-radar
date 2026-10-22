"""Resume a terminal near-stationary geometry fit without changing its model.

Rebuild the original likelihood and bounds, refresh optimizer scaling at the
saved point and use original-row gradients for any final curvature refinement.
The completed fixed-antenna control is reused byte-for-byte. No gate is relaxed.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from audit_scattered_information import load_model
from fit_shared_target_geometry import load_experiment
from full_road_likelihood import build
from geographic_io import read_profiles
from refine_full_capture_motion import constraints
from scattered_origin_geometry import FrozenScatteredRoad
from scene_dictionary import fingerprint
from shared_origin_transfer import fit_control
from spline_motion import SplineMotion
from validate_shared_geometry import fit_target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'ship-fit', 'road-fit', 'wide-psf', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve the earlier fit and use a fresh output directory')
    paths = [args.experiment/name for name in ('experiment.json', 'fixed.json', 'fitted.json')]
    state, fixed, previous = [json.loads(path.read_text()) for path in paths]
    identity = state['identity']
    if fixed['identity'] != identity or previous['identity'] != identity:
        raise ValueError('Controls are not from the same recorded experiment')
    if (not previous['complete'] or previous['fit']['converged']
            or previous['original_row_maximum_kkt_residual'] >= 1e-5):
        raise ValueError('Use a completed near-stationary failed fit')
    joint, _ = load_model(fixed, args)
    initial = np.asarray(previous['parameters'])
    np.testing.assert_allclose(joint.objective(initial)[0], previous['fit']['objective'], atol=1e-11, rtol=0)
    # Identical physical constraints to load_experiment, with the additional
    # reflecting-point boxes supplied by the same model that built the fit.
    _, matrix, lo, hi, lower, upper = constraints(joint.ship.bands[0].model, common_path=True)
    origins = np.asarray(json.loads(args.ship_fit.read_text())['initial_range_bias_m'])
    lower = np.r_[lower, np.full(joint.ship.antennas.count, -3.), origins-75.]
    upper = np.r_[upper, np.full(joint.ship.antennas.count, 3.), origins+75.]
    curve = SplineMotion(joint.ship.bands[0].model)
    yaw = ((joint.ship.active >= curve.sections[1].start)
           & (joint.ship.active < curve.sections[1].stop))
    lower[np.flatnonzero(yaw)] = upper[np.flatnonzero(yaw)] = 0.
    for road in joint.roads:
        a, b = road.original.problem.model.bounds()
        lower, upper = np.r_[lower, a], np.r_[upper, b]
    matrix = np.pad(matrix, ((0, 0), (0, len(initial)-matrix.shape[1])))
    bounds = matrix, lo, hi, lower, upper
    from stationarity import check
    kkt, violation = check(initial, joint.objective(initial)[1], *bounds)
    np.testing.assert_allclose(kkt, previous['original_row_maximum_kkt_residual'], atol=1e-10, rtol=0)
    if violation > 1e-7:
        raise ValueError('The saved point is infeasible under the reproduced constraints')
    refinement = {'scope': __doc__, 'source_sha256': {str(path): fingerprint(path) for path in paths},
        'implementation_sha256': {str(path): fingerprint(path) for path in [Path(__file__),
            Path(__file__).with_name('audit_scattered_information.py')]},
        'initial_original_row_kkt': kkt}
    args.output.mkdir(parents=True)
    shutil.copyfile(paths[1], args.output/'fixed.json')
    state.update(numerical_refinement=refinement, complete=False)
    (args.output/'experiment.json').write_text(json.dumps(state, indent=2)+'\n')
    fitted = fit_control(joint, joint, bounds, initial, False, 180,
        args.output/'fitted.json', identity, identity['calibration_labels'])
    fitted['numerical_refinement'] = refinement
    (args.output/'fitted.json').write_text(json.dumps(fitted, indent=2)+'\n')
    if not fitted['fit']['converged']:
        raise ValueError('Refinement did not meet the unchanged stationarity gate')
    offsets = [{label: road.original.offsets(np.asarray(case['parameters'])[section]).tolist()
        for label, road, section in zip(identity['calibration_labels'], joint.roads, joint.road_slices)}
        for case in (fixed, fitted)]
    antenna_slice = joint.antenna_slice
    # Release the joint model before rebuilding the excluded target's data.
    del joint
    excluded = identity['excluded_label']
    loaded = load_experiment(args.ship_fit, args.road_fit, args.fields, args.geometry, [excluded], 2)
    road = loaded[0].roads[0]; del loaded
    saved = json.loads(args.road_fit.read_text()); patches = Path(saved['input_patches'])
    _, road, audit = build(road, [patches/f'{excluded}-{c}.npz' for c in range(5)], read_profiles(args.wide_psf)[1])
    if audit != identity['full_bin_road_checks'][excluded]:
        raise ValueError('Excluded observation model changed')
    old = saved['labels'].index(excluded); extension = np.zeros(3*(len(road.training)-1))
    starts = [np.r_[saved['selected_roads'][old]['state'], extension], np.r_[road.initial, extension]]
    transfers = []
    for case in (fixed, fitted):
        target = FrozenScatteredRoad(road, case['range_origins_m'],
            limits=identity['reflector_extent_along_across_up_m'], reference=identity['reference_channel'])
        transfers.append(fit_target(target, np.asarray(case['parameters'])[antenna_slice], starts, 180))
        print(json.dumps({'stage': 'excluded_target', 'fixed_antennas': case['fixed_antennas']}), flush=True)
    errors = [sum(c['prediction_error'] for c in t['prediction']) for t in transfers]
    energies = [sum(c['matched_energy'] for c in t['prediction']) for t in transfers]
    stationary = [t['starts'][t['selected_start']]['fit']['converged'] for t in transfers]
    state.update(complete=all(stationary), transfer=transfers,
        fitted_reflector_offsets_along_across_up_m=offsets,
        summary={'prediction_error_change_percent': 100*(errors[1]/errors[0]-1),
            'matched_energy_change_db': float(10*np.log10(energies[1]/energies[0])),
            'selected_fits_stationary': stationary}, seconds=time.monotonic()-started)
    (args.output/'experiment.json').write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', **state['summary']}), flush=True)


if __name__ == '__main__':
    main()
