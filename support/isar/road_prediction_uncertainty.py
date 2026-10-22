"""Conditional prediction uncertainty of the full-bin vehicle response model.

Motion and geometry are held at the fitted values. The moving amplitude and
stationary complex coefficients are the same linear Gaussian estimators used
by the fit. Their propagated noise covariance is calculated without forming
a dense time-by-time matrix. This is not uncertainty from unknown motion,
wrong roads, empirical covariance error or unmodelled reflectivity changes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fit_shared_target_geometry import load_experiment
from full_stationary_road import FullRoad
from integrated_stationary_road import RecordedBins
from scene_dictionary import fingerprint
from stationary_road_control import StationaryRoad
from stationary_sufficient_road import coefficients


def prediction_noise_cost(train, test, u, v):
    """Trace of the exact frozen-motion coefficient-prediction covariance.

    With B = U S V^H, stationary transfer is D = Btest V/S. The moving
    amplitude uses a = P_B u / ||P_B u||^2, orthogonal to U. Consequently the
    two contributions to prediction covariance have zero cross covariance.
    Their traces are ||D||_F^2 and ||v - D U^H u||^2 / ||P_B u||^2.
    """
    transfer = test.stationary@train.inverse
    perpendicular = train.remove(u)
    information = float(np.vdot(perpendicular, perpendicular).real)
    if information <= 0:
        raise ValueError('Moving amplitude is not identifiable from stationary nuisance')
    innovation = v-transfer@(train.basis.conj().T@u)
    static_variance = float(np.linalg.norm(transfer)**2)
    moving_variance = float(np.vdot(innovation, innovation).real/information)
    return {'stationary_coefficient_prediction_variance': static_variance,
        'moving_coefficient_prediction_variance': moving_variance,
        'total_coefficient_prediction_variance': static_variance+moving_variance,
        'test_noise_energy_expectation': len(v),
        'total_test_error_expectation': len(v)+static_variance+moving_variance,
        'moving_amplitude_variance': 1/information,
        'moving_information_fraction_after_static_nuisance': information/float(np.vdot(u, u).real)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('full', 'geometry-fit', 'ship-fit', 'road-fit', 'control', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    args = parser.parse_args()
    fitted, geometry, saved, control = [json.loads(p.read_text()) for p in
        (args.full, args.geometry_fit, args.road_fit, args.control)]
    if not fitted['complete']:
        raise ValueError('Use a completed full-bin reference')
    for group in ('sources', 'implementation_sha256'):
        for path, digest in fitted['identity'][group].items():
            if fingerprint(Path(path)) != digest:
                raise ValueError('Full-bin fit input changed: '+path)
    for path in (args.geometry_fit, args.ship_fit, args.road_fit, args.control):
        if fitted['identity']['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('Different inputs from the full-bin fit')
    joint, _, _, labels, _ = load_experiment(args.ship_fit, args.road_fit, args.fields,
        args.geometry, geometry['identity']['labels'], geometry['identity']['nodes'])
    antenna = np.asarray(geometry['parameters'])[joint.antenna_slice]
    targets = []
    for record in fitted['targets']:
        label = record['label']
        previous = StationaryRoad(joint.roads[labels.index(label)], cutoff=control['identity']['cutoff'])
        binned = []
        for c, field in enumerate(previous.training):
            with np.load(Path(saved['input_patches'])/f'{label}-{c}.npz') as raw:
                bins = RecordedBins(field.original, raw['row_time'], raw['row_information'])
                binned.append(bins.stationary(field.original, field.delays, previous.origins[c]))
        model = FullRoad(previous, binned)
        fit = record['fit']
        motion = np.asarray(fit['starts'][fit['selected_start']]['parameters'])
        channels = []
        for c, (train, test) in enumerate(zip(model.training, model.validation)):
            u, _ = model.raw_template(motion, antenna, c, train)
            v, _ = model.raw_template(motion, antenna, c, test)
            noise = prediction_noise_cost(train, test, u, v)
            alpha, beta = coefficients(model, motion, antenna, c)
            error = test.original.y-alpha*v-test.stationary@beta
            training = train.original.y-alpha*u-train.stationary@beta
            observed = float(np.vdot(error, error).real)
            noise.update(channel=c, observed_test_error=observed,
                observed_over_frozen_model_expectation=observed/noise['total_test_error_expectation'],
                observed_training_error=float(np.vdot(training, training).real),
                frozen_motion_training_residual_degrees_of_freedom=len(u)-train.basis.shape[1]-1,
                stationary_rank=train.basis.shape[1],
                declared_stationary_columns=len(train.delays))
            channels.append(noise)
        expected = sum(c['total_test_error_expectation'] for c in channels)
        observed = sum(c['observed_test_error'] for c in channels)
        targets.append({'label': label, 'channels': channels, 'expected_test_error': expected,
            'observed_test_error': observed, 'observed_over_frozen_model_expectation': observed/expected})
    sources = [args.full, args.geometry_fit, args.ship_fit, args.road_fit, args.control]
    report = {'complete': True, 'scope': __doc__, 'targets': targets,
        'source_sha256': {str(p): fingerprint(p) for p in sources},
        'implementation_sha256': {str(Path(__file__)): fingerprint(Path(__file__))},
        'accepted_calibration': False, 'production_changed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({r['label']: r['observed_over_frozen_model_expectation'] for r in targets}))


if __name__ == '__main__':
    main()
