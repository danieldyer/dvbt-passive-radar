"""Predict untouched range/slow-time data with a physical ship model.

Unlike point-history reimaging diagnostics, both models below predict exactly
the same observations. Independent complex scatterer responses and stationary
nuisance amplitudes are fitted on training data, then held fixed for validation.
"""
import argparse
import json
from pathlib import Path
import time as clock

import numpy as np

from ais import read_csv, utc_seconds
from dense_scatterers import points
from field import Field
from imaging import physical_state, sites
from spline_motion import SplineMotion
from stationary import Stationary, make_design


def validate(field, target, nuisance, selected_time, interval, *, coefficients=None, test_time=None,
             stationary_mode='predict'):
    test_time = ~selected_time if test_time is None else np.asarray(test_time, dtype=bool)
    if test_time.shape != selected_time.shape or not test_time.any() or np.any(test_time&selected_time):
        raise ValueError('Validation epochs must be nonempty and disjoint from training epochs')
    selected = np.repeat(selected_time, field.y.shape[1])
    test = np.repeat(test_time, field.y.shape[1])
    y = field.y.ravel()
    clean = nuisance.remove(y[selected])
    design = nuisance.remove(target[selected])
    if coefficients is None:
        coefficients = np.linalg.lstsq(design, clean, rcond=None)[0]
    beta = nuisance.solve(y[selected]-target[selected]@coefficients)
    prediction = target@coefficients
    reference_beta = nuisance.solve(y[selected])
    reference = y-nuisance.design@beta
    test_design, model = target[test], prediction[test]
    stationary_error = float(np.sum(abs(y[test]-nuisance.design[test]@reference_beta)**2))
    retention, test_rank = 1., None
    if stationary_mode == 'project':
        # Profile only the declared stationary nuisance in the validation data.
        # Target motion, gains and complex coefficients remain frozen. Applying
        # the identical orthogonal projection to both data and target avoids
        # extrapolating unobservable stationary coefficients across a range crop.
        test_nuisance = Stationary(nuisance.design, test)
        test_design = test_nuisance.remove(target[test])
        model = test_design@coefficients
        reference[test] = test_nuisance.remove(y[test])
        retention = float(np.vdot(model, model).real/np.sum(abs(prediction[test])**2))
        prediction[test] = model
        stationary_error = float(np.sum(abs(reference[test])**2))
        test_rank = test_nuisance.rank
    elif stationary_mode != 'predict':
        raise ValueError('Stationary validation mode must be predict or project')
    product = np.vdot(model, reference[test])
    captured = float(abs(product)**2/np.vdot(model, model).real)
    separate = np.linalg.lstsq(test_design, reference[test], rcond=None)[0]
    upper = float(np.sum(abs(test_design@separate)**2))
    phase = []
    for first in np.arange(interval[0], interval[1], 1.):
        keep = test_time&(field.time >= first)&(field.time < first+1)
        if not keep.any():
            continue
        keep = np.repeat(keep, field.y.shape[1])
        power = float(np.sum(abs(prediction[keep])**2))
        cross = np.vdot(prediction[keep], reference[keep])
        phase.append({'time_s': first+.5, 'phase_radians': float(np.angle(cross)),
                      'prediction_energy': power, 'matched_amplitude': float(abs(cross)/power)})
    phase_rms = np.sqrt(np.average([p['phase_radians']**2 for p in phase],
                                  weights=[p['prediction_energy'] for p in phase]))
    error = float(np.sum(abs(reference[test]-model)**2))
    return {'prediction_error': error, 'stationary_only_error': stationary_error,
            'heldout_error_reduction_fraction': 1-error/stationary_error,
            'heldout_matched_energy': captured, 'independently_fitted_test_energy': upper,
            'heldout_prediction_energy': float(np.vdot(model, model).real),
            'heldout_cross_real_imag': [float(product.real), float(product.imag)],
            'conditional_efficiency_loss_db': float(-10*np.log10(captured/upper)),
            'prediction_phase_rms_degrees': float(np.degrees(phase_rms)),
            'prediction_phase_by_one_second_block': phase,
            'training_gram_condition': float(np.linalg.cond(design.conj().T@design)),
            'stationary_validation_mode': stationary_mode,
            'raw_heldout_energy': float(np.sum(abs(y[test])**2)),
            'extrapolated_stationary_energy': float(np.sum(abs(nuisance.design[test]@beta)**2)),
            'test_stationary_rank': test_rank,
            'test_target_information_retained_fraction': retention,
            'point_responses_real_imag': np.c_[coefficients.real, coefficients.imag].tolist(),
            'phase_scope': 'Phase of held-out data projected on the fitted multi-point prediction. Includes model mismatch, noise and glint, and depends on the assumed point support.'}


def summarize(checks):
    """One coherent score across channels, without per-channel phase refitting."""
    cross = sum(complex(*c['heldout_cross_real_imag']) for c in checks)
    predicted = sum(c['heldout_prediction_energy'] for c in checks)
    upper = sum(c['independently_fitted_test_energy'] for c in checks)
    captured = abs(cross)**2/predicted
    return {'prediction_error': sum(c['prediction_error'] for c in checks),
        'stationary_only_error': sum(c['stationary_only_error'] for c in checks),
        'matched_energy': captured, 'independent_dictionary_energy': upper,
        'conditional_coherent_loss_db': float(-10*np.log10(captured/upper)),
        'phase_rms_degrees_by_channel': [c['prediction_phase_rms_degrees'] for c in checks]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'images', 'ais', 'geometry-directory', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--points', type=int, default=8)
    p.add_argument('--holdout-seconds', type=float, default=2.)
    args = p.parse_args()
    motion = SplineMotion(json.loads(args.motion.read_text()))
    m = motion.description
    epoch = utc_seconds(m['nominal_start_utc'])
    track = read_csv(args.ais, reference_height_ellipsoid_m=m['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    result = {'scope': __doc__, 'motion': str(args.motion), 'fields': str(args.fields),
              'points': args.points, 'holdout_seconds': args.holdout_seconds, 'channels': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for channel in range(5):
        started = clock.perf_counter()
        with np.load(args.fields/f'channel-{channel}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(motion.interval[0]+.02, motion.interval[1]-.02)
        with np.load(args.images/f'channel-{channel}-image.npz') as image:
            locations = points(image, args.points, 30.)
        time = field.data['row_time']
        prior = physical_state(track, epoch+time, m['attitude_prior'])
        corrected = motion.apply(time, prior, tx[3], rx)
        selected = (np.arange(len(field.time))%2 == 0 if args.holdout_seconds == 0 else
                    np.floor((field.time-motion.interval[0])/args.holdout_seconds).astype(int)%2 == 0)
        stationary, _ = make_design(field)
        nuisance = Stationary(stationary, np.repeat(selected, field.y.shape[1]))
        stages = []
        for state in (prior, corrected):
            target = np.column_stack([field.point(point, motion=state).ravel() for point in locations])
            stages.append(validate(field, target, nuisance, selected, motion.interval))
        record = {'channel': channel, 'points_body_m': locations.tolist(), 'before': stages[0], 'after': stages[1],
                  'seconds': clock.perf_counter()-started,
                  'heldout_prediction_error_change_percent': 100*(stages[1]['prediction_error']/stages[0]['prediction_error']-1)}
        result['channels'].append(record)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: record[k] for k in ('channel', 'seconds', 'heldout_prediction_error_change_percent')}), flush=True)


if __name__ == '__main__':
    main()
