"""Expected location information in the combined target model.

Unknown complex reflectivities and each target's real motion are eliminated
before adding target information. No antenna prior, optimizer bound or road
motion penalty is counted as measured information. Results remain conditional
on road associations, reflector support and the empirical noise covariance.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fit_shared_target_geometry import load_experiment
from geometry_information import real_jacobian
from profile_information import from_jacobian
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def summary(information, factor):
    eigen = np.linalg.eigvalsh(information)
    if eigen.min() < -1e-9*max(1., eigen.max()):
        raise ValueError('Substantial negative expected information')
    covariance = np.linalg.solve(np.eye(len(information))+information, np.eye(len(information)))
    physical = factor@covariance@factor.T
    return {'information_eigenvalues_before_prior': eigen.tolist(),
        'directions_weaker_than_declared_prior': int(np.sum(eigen < 1.)),
        'conditional_sigma_enu_m_with_one_prior': np.sqrt(np.diag(physical)).reshape(-1, 3).tolist(),
        'prior_sigma_enu_m': np.sqrt(np.diag(factor@factor.T)).reshape(-1, 3).tolist(),
        'conditional_covariance_m2_with_one_prior': physical.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('joint', 'ship-fit', 'road-fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    args = p.parse_args()
    fitted = json.loads(args.joint.read_text())
    if not fitted['complete'] or fitted.get('kind') == 'recorded_data_derivative_audit':
        raise ValueError('Use a completed recorded-data fit')
    for path, digest in fitted['identity']['sources'].items():
        if fingerprint(Path(path)) != digest:
            raise ValueError('A source changed since the fitted result: '+path)
    joint, ship, _, labels, _ = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, fitted['identity']['labels'], fitted['identity']['nodes'])
    parameters = np.array(fitted['parameters'])
    antenna = np.arange(joint.ship_count)[joint.antenna_slice]
    nuisance = np.setdiff1d(np.arange(joint.ship_count), antenna)
    curve = SplineMotion(ship.bands[0].model)
    yaw = (ship.active >= curve.sections[1].start)&(ship.active < curve.sections[1].stop)
    nuisance = np.setdiff1d(nuisance, np.flatnonzero(yaw))
    road_jacobians = [road.evaluate(parameters[section], parameters[joint.antenna_slice],
        include_motion_prior=False)['expected_real_jacobian'] for road, section in zip(joint.roads, joint.road_slices)]
    report = {'scope': __doc__, 'source': str(args.joint), 'source_sha256': fingerprint(args.joint),
        'converged_fit': fitted['fit']['converged'], 'original_row_kkt': fitted['original_row_maximum_kkt_residual'],
        'cases': [], 'complete': False, 'implementation_sha256': {str(path): fingerprint(path) for path in
            [Path(__file__), Path(__file__).with_name('shared_target_geometry.py'),
             Path(__file__).with_name('geometry_information.py'), Path(__file__).with_name('profile_information.py')]}}
    for free_points in (False, True):
        for cutoff in (1e-8, 1e-10, 1e-12):
            ship_jacobian, point_ranks = real_jacobian(ship, parameters[:joint.ship_count],
                point_coordinates=free_points, cutoff=cutoff)
            ship_information, ship_ranks = from_jacobian(ship_jacobian, antenna, nuisance, cutoff=cutoff)
            road_information, road_ranks, individual = np.zeros_like(ship_information), [], []
            for label, road, jacobian in zip(labels, joint.roads, road_jacobians):
                information, ranks = from_jacobian(jacobian,
                    np.arange(road.nmotion, road.nmotion+road.antennas.count), np.arange(road.nmotion), cutoff=cutoff)
                road_information += information
                road_ranks.append({'label': label, **ranks})
                individual.append({'label': label, 'information_trace': float(np.trace(information))})
            factor = ship.antennas.factor
            record = {'ship_reflector_coordinates_eliminated': free_points, 'cutoff': cutoff,
                'ship_point_nuisance_ranks': point_ranks, 'ship_motion_nuisance': ship_ranks,
                'road_motion_nuisance': road_ranks, 'individual_road_information': individual,
                'ship_only': summary(ship_information, factor), 'vehicles_only': summary(road_information, factor),
                'combined': summary(ship_information+road_information, factor)}
            report['cases'].append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'free_ship_points': free_points, 'cutoff': cutoff,
                'weak_ship': record['ship_only']['directions_weaker_than_declared_prior'],
                'weak_vehicles': record['vehicles_only']['directions_weaker_than_declared_prior'],
                'weak_combined': record['combined']['directions_weaker_than_declared_prior']}), flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
