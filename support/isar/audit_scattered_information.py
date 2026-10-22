"""Conditional antenna information with uncertain vehicle reflecting points.

Keep the fitted mean fixed while comparing known versus unknown body offsets.
Eliminate independent complex gains and each target's motion, then the shared
range origins and the body offsets. The last projection is nested explicitly:
unknown offsets cannot create information through different numerical ranks.
Bounds and motion penalties are not measured information. Results are local and conditional on the stipulated
covariance, road associations and ship scene; they are not geographic error bars.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_shared_geometry_information import summary
from fit_shared_target_geometry import load_experiment
from full_road_likelihood import build
from geographic_io import read_profiles
from geometry_information import real_jacobian
from profile_information import from_jacobian
from scattered_origin_geometry import ScatteredOriginGeometry
from scene_dictionary import fingerprint
from shared_origin_geometry import OriginRoad
from shared_origin_information import local_root
from shared_origin_transfer import subset
from spline_motion import SplineMotion


def information_cases(ship_jacobian, shared_columns, ship_nuisance, roads,
                      antenna_count, *, cutoff):
    """Road blocks contain a mean Jacobian, base motion count and total count.

    Total count includes per-band body offsets, preceding shared antennas and
    origins. Fixed/free cases use the identical mean and amplitude estimates.
    """
    ship_root, ship_rank = local_root(ship_jacobian, shared_columns, ship_nuisance, cutoff)
    roots, ranks = [], []
    for jacobian, motion_count, total_count in roads:
        # Retain body columns for the final, explicitly nested projection.
        # Separate all-nuisance SVDs can retain different near-null directions
        # and spuriously make an extra unknown appear to add information.
        interest = np.r_[np.arange(total_count, jacobian.shape[1]),
                         np.arange(motion_count, total_count)]
        root, rank = local_root(jacobian, interest, np.arange(motion_count), cutoff)
        roots.append(root); ranks.append(rank)
    cases = [{'free_vehicle_reflectors': free, 'ship_motion_elimination': ship_rank,
              'road_motion_elimination': ranks} for free in (False, True)]
    shared_count = len(shared_columns)
    for name, selected in (('ship_only', [ship_root]), ('vehicles_only', roots),
                           ('combined', [ship_root, *roots])):
        body_count = sum(r.shape[1]-shared_count for r in selected)
        stacked = np.zeros((sum(len(r) for r in selected), shared_count+body_count))
        row, column = 0, shared_count
        for root in selected:
            count = root.shape[1]-shared_count
            stacked[row:row+len(root), :shared_count] = root[:, :shared_count]
            stacked[row:row+len(root), column:column+count] = root[:, shared_count:]
            row += len(root); column += count
        origins = np.arange(antenna_count, shared_count)
        retained = np.r_[np.arange(antenna_count), np.arange(shared_count, column)]
        projected, origin_rank = local_root(stacked, retained, origins, cutoff)
        fixed = projected[:, :antenna_count]
        known = fixed.T@fixed
        unknown, body_rank = from_jacobian(projected, np.arange(antenna_count),
            np.arange(antenna_count, projected.shape[1]), cutoff=cutoff)
        cases[0][name] = {'information': known, 'origin_elimination': origin_rank,
                          'body_offset_parameters': body_count}
        cases[1][name] = {'information': unknown, 'origin_elimination': origin_rank,
                          'body_offset_elimination': body_rank}
    return cases


def load_model(fit, args):
    identity = fit['identity']
    if not (fit['complete'] and fit['fit']['converged']
            and fit['original_row_maximum_kkt_residual'] <= 1e-7
            and fit['original_row_constraint_violation'] <= 1e-7):
        raise ValueError('Use a completed stationary fit')
    for group in ('implementation_sha256', 'additional_sources_sha256'):
        for path, expected in identity[group].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError('Fit input changed: '+path)
    for path in (args.ship_fit, args.road_fit, args.wide_psf):
        if fingerprint(path) != identity['additional_sources_sha256'].get(str(path)):
            raise ValueError('Use the identical recorded-data inputs: '+str(path))
    point_identity = identity['point_experiment_identity']
    source_path = Path(point_identity['source'])
    if fingerprint(source_path) != point_identity['source_sha256']:
        raise ValueError('Original common-origin calibration changed')
    source = json.loads(source_path.read_text())
    inherited = source['identity']['legacy_identity']
    legacy, ship, bounds, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, source['identity']['labels'], inherited['nodes'])
    if dropped:
        raise ValueError('Target set changed')
    records = read_profiles(args.wide_psf)[1]
    patches = Path(json.loads(args.road_fit.read_text())['input_patches'])
    exact_roads = []
    indices = [labels.index(label) for label in identity['calibration_labels']]
    for i in indices:
        fast, exact, check = build(legacy.roads[i],
            [patches/f'{labels[i]}-{c}.npz' for c in range(5)], records)
        if check != identity['full_bin_road_checks'][labels[i]]:
            raise ValueError('Recorded-row observation construction changed')
        legacy.roads[i] = fast; exact_roads.append(exact)
    _, exact, _ = subset(legacy, ship, bounds, source['parameters'], indices)
    exact.roads = [OriginRoad(road) for road in exact_roads]
    model = ScatteredOriginGeometry(exact,
        limits=identity['reflector_extent_along_across_up_m'],
        reference=identity['reference_channel'])
    parameters = np.asarray(fit['parameters'])
    value = model.evaluate(parameters)
    np.testing.assert_allclose(value['value'], fit['fit']['objective'], atol=1e-11, rtol=0)
    return model, parameters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'ship-fit', 'road-fit', 'wide-psf', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing information audit')
    fit = json.loads(args.fit.read_text())
    joint, parameters = load_model(fit, args)
    antenna_count = joint.ship.antennas.count
    shared = np.r_[np.arange(joint.ship_count)[joint.antenna_slice],
                   np.arange(joint.ship_count)[joint.range_slice]]
    curve = SplineMotion(joint.ship.bands[0].model)
    yaw = ((joint.ship.active >= curve.sections[1].start)
           & (joint.ship.active < curve.sections[1].stop))
    nuisance = np.setdiff1d(np.arange(joint.ship.nmotion), np.flatnonzero(yaw))
    roads = [(road.evaluate(parameters[section], parameters[joint.antenna_slice],
        parameters[joint.range_slice], include_motion_prior=False)['expected_real_jacobian'],
        road.original.base_count, road.nmotion) for road, section in zip(joint.roads, joint.road_slices)]
    sources = [args.fit, Path(__file__), *[Path(__file__).with_name(name) for name in
        ('shared_origin_information.py', 'geometry_information.py', 'profile_information.py')]]
    result = {'complete': False, 'scope': __doc__, 'fit': str(args.fit),
        'calibration_labels': fit['identity']['calibration_labels'], 'cases': [],
        'accepted_calibration': False,
        'source_sha256': {str(path): fingerprint(path) for path in sources}}
    for free_points in (False, True):
        for cutoff in (1e-8, 1e-10, 1e-12):
            jacobian, ranks = real_jacobian(joint.ship, parameters[:joint.ship_count],
                point_coordinates=free_points, cutoff=cutoff)
            cases = information_cases(jacobian, shared, nuisance, roads, antenna_count, cutoff=cutoff)
            for case in cases:
                for name in ('ship_only', 'vehicles_only', 'combined'):
                    info = case[name].pop('information')
                    case[name].update(summary(info, joint.ship.antennas.factor))
                    case[name]['information_before_prior'] = info.tolist()
                case.update(free_ship_reflector_positions=free_points,
                    ship_point_nuisance_ranks=ranks, cutoff=cutoff)
                result['cases'].append(case)
                print(json.dumps({'free_ship_points': free_points, 'cutoff': cutoff,
                    'free_vehicle_points': case['free_vehicle_reflectors'],
                    'weak_combined': case['combined']['directions_weaker_than_declared_prior']}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
