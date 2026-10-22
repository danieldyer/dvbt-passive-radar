"""Compare held-out coherent prediction over several fixed-duration apertures.

This estimates loss relative to independently refitting the test amplitudes in
the same point dictionary. It is a conditional model diagnostic, not a bound on
all recoverable ship information or a validation of absolute antenna positions.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ais import read_csv, utc_seconds
from field import Field
from imaging import physical_state, sites
from spline_motion import SplineMotion
from stationary import Stationary, make_design
from validate_motion import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'ais', 'geometry-directory', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--durations', type=float, nargs='+', default=[3., 5., 10., 15., 20., 30.])
    p.add_argument('--centre-step', type=float, default=2.)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    motion = SplineMotion(model)
    epoch = utc_seconds(model['nominal_start_utc'])
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    reference_tx, reference_rx = tx[3].copy(), rx.copy()
    if model.get('antenna_fit'):
        a = model['antenna_fit']; rotation = np.array(a['ecef_to_enu'])
        tx += np.array(a['transmitter_offsets_enu_m'])@rotation
        rx += np.array(a['receiver_offset_enu_m'])@rotation
    result = {'scope': __doc__, 'motion': str(args.motion), 'channels': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for channel in range(5):
        with np.load(args.fields/f'channel-{channel}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(motion.interval[0]+.02, motion.interval[1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        state = motion.apply(field.data['row_time'], prior, reference_tx, reference_rx)
        target = np.column_stack([field.trajectory(state['position_ecef_m']+state['body_to_ecef']@point,
            state['velocity_ecef_mps']+state['frame_velocity']@point, transmitter=tx[channel], receiver=rx).ravel()
            for point in np.array(model['points_body_m'][channel])])
        stationary, _ = make_design(field)
        rows = field.y.shape[1]
        block = model.get('holdout_seconds', 2.)
        train = (np.arange(len(field.time))%2 == 0 if block == 0 else
                 np.floor((field.time-motion.interval[0])/block).astype(int)%2 == 0)
        record = {'channel': channel, 'apertures': []}
        for duration in args.durations:
            centres = np.arange(motion.interval[0]+duration/2, motion.interval[1]-duration/2+.001, args.centre_step)
            scores = []
            for centre in centres:
                window = [centre-duration/2, centre+duration/2]
                keep = (field.time >= window[0])&(field.time < window[1])
                if np.sum(train[keep]) < 4 or np.sum(~train[keep]) < 4:
                    continue
                observed = SimpleNamespace(y=field.y[keep], time=field.time[keep])
                flat = np.repeat(keep, rows)
                nuisance = Stationary(stationary[flat], np.repeat(train[keep], rows))
                score = validate(observed, target[flat], nuisance, train[keep], window)
                scores.append({'centre_s': float(centre), **{k: score[k] for k in (
                    'conditional_efficiency_loss_db', 'heldout_error_reduction_fraction',
                    'prediction_phase_rms_degrees', 'heldout_matched_energy', 'training_gram_condition')}})
            record['apertures'].append({'duration_s': duration, 'windows': scores})
        result['channels'].append(record)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'channel': channel, 'apertures': [{'duration_s': a['duration_s'],
            'median_conditional_loss_db': float(np.median([w['conditional_efficiency_loss_db'] for w in a['windows']]))}
            for a in record['apertures']]}), flush=True)


if __name__ == '__main__':
    main()
