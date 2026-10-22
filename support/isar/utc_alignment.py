"""Bound capture-to-AIS UTC using file closure, then profile that nuisance.

The bound assumes the filesystem clock did not jump and the original archive
was not subsequently rewritten. Closure includes queue drainage, so the upper
bound is not an estimated start timestamp. No large offset is introduced to
force radar association.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from field import Field, OFFSETS
from imaging import attitude_basis
from stationary import Stationary, make_design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--field', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--nominal-start-utc', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.capture.parent/'summary.json').read_text())
    if any(summary[key] for key in ('overflows', 'missing_samples', 'timestamp_gaps', 'host_discarded_samples')):
        raise ValueError('Closure bound needs a continuous capture')
    epoch = utc_seconds(args.nominal_start_utc)
    duration = summary['received_samples']/summary['actual_rate_hz']
    upper = args.capture.stat().st_mtime-duration-epoch
    lower = args.capture.stat().st_birthtime-epoch
    if not 0 <= lower < upper < 60:
        raise ValueError('Unexpected wall-clock chronology; do not silently fit an unrestricted offset')
    description = json.loads(args.positions.read_text())
    with np.load(args.field) as data:
        field = Field(dict(data), data['pulse_samples']).windowed(*description['interval_s'])
    track = read_csv(args.ais, reference_height_ellipsoid_m=5, mmsi=563052200)
    original = track.sample(epoch+field.data['row_time'], max_gap_s=180)
    original['body_to_ecef'] = attitude_basis(track, original, description['attitude'])
    points = np.array(description['initial_points_body_m'])
    middle = len(field.data['row_time'])//2
    anchored = original['position_ecef_m'][middle]+points@original['body_to_ecef'][middle].T
    static, _ = make_design(field)
    selected = np.repeat(((field.time-description['interval_s'][0])//2)%2 == 0, len(OFFSETS))
    nuisance = Stationary(static, selected)
    y = field.y.ravel()
    clean = nuisance.remove(y[selected])
    trials = []
    for offset in np.linspace(lower, upper, 25):
        state = track.sample(epoch+offset+field.data['row_time'], max_gap_s=180)
        state['body_to_ecef'] = attitude_basis(track, state, description['attitude'])
        positions = (anchored-state['position_ecef_m'][middle])@state['body_to_ecef'][middle]
        design = np.column_stack([field.point(point, motion=state).ravel() for point in positions])
        projected = nuisance.remove(design[selected])
        amplitude = np.linalg.lstsq(projected, clean, rcond=None)[0]
        beta = nuisance.solve(y[selected]-design[selected]@amplitude)
        predicted = design[~selected]@amplitude
        validation = y[~selected]-static[~selected]@beta
        independent = np.linalg.lstsq(design[~selected], validation, rcond=None)[0]
        bound = float(np.sum(abs(design[~selected]@independent)**2))
        captured = float(abs(np.vdot(predicted, validation))**2/np.vdot(predicted, predicted).real)
        trials.append({'offset_s': float(offset), 'training_captured_energy': float(np.sum(abs(projected@amplitude)**2)),
                       'held_out_prediction_error': float(np.sum(abs(validation-predicted)**2)),
                       'held_out_matched_energy': captured, 'held_out_independent_upper_energy': bound,
                       'held_out_efficiency_loss_db': float(-10*np.log10(captured/bound))})
    chosen = int(np.argmax([trial['training_captured_energy'] for trial in trials]))
    result = {'nominal_start_utc': args.nominal_start_utc, 'recorded_sample_duration_s': duration,
              'offset_bounds_s': [lower, upper], 'selected_by_training_only': trials[chosen],
              'selected_at_bound': chosen in (0, len(trials)-1), 'trials': trials,
              'scope': 'Filesystem chronology bound, unchanged system-clock assumption. Exact AIS motion/heading-time shift; midpoint physical scatterer positions held fixed as a coordinate gauge. No calibrated UTC or antenna fit adopted.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'trials'}, indent=2))


if __name__ == '__main__':
    main()
