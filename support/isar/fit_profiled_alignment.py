"""Test channel path-rate residuals with independent scene profiling and folds."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from audit_residual_phase import score
from channel_alignment import AlignmentBand
from field import Field
from imaging import physical_state, sites
from profiled_alignment import ProfiledAlignment
from spline_motion import SplineMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'support', 'alignment', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--channels', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--range-folds', action='store_true')
    args = parser.parse_args()
    model, support, alignment = [json.loads(path.read_text()) for path in (args.motion, args.support, args.alignment)]
    points = np.array(support['points_body_m'][:args.count])
    if model.get('antenna_fit') or model.get('rock_spacing_seconds'):
        raise ValueError('Use the nominal-location, fixed-roll/pitch control')
    curve = SplineMotion(model)
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    result = {'source_motion': str(args.motion), 'source_support': str(args.support), 'count': args.count,
        'scope': 'Effective channel envelope origins and excess-path rates; physical ship motion and point coordinates fixed. Complex point responses and stationary nuisance are profiled at every candidate. Constant carrier phases are unidentifiable in this independent-band diagnostic. Odd interior one-second blocks and complementary whitened range components test prediction.',
        'complete': False, 'correction_adopted': False, 'channels': []}
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for c in args.channels:
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        state = curve.apply(field.data['row_time'], prior, tx[3], rx)
        block = np.floor(field.time-curve.interval[0]).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        band = AlignmentBand(field, state, points, selected)
        ranges = np.arange(field.y.shape[1])
        folds = [('all', ranges, ranges)]
        if args.range_folds:
            folds += [('even to odd', ranges[::2], ranges[1::2]), ('odd to even', ranges[1::2], ranges[::2])]
        channel = {'channel': c, 'trials': []}
        for label, train_range, test_range in folds:
            fit = ProfiledAlignment(band, float(np.mean(curve.interval)), train_range)
            original = fit.evaluate(np.array([model['range_bias_m'][c], 0.])/fit.scale)
            before = score(field, band.nuisance.design, (original['target']@original['coefficient']).reshape(field.y.shape),
                np.zeros(len(field.time)), selected, test_range)
            del original
            fitted, details = fit.fit(model['range_bias_m'][c])
            after = score(field, band.nuisance.design, (fitted['target']@fitted['coefficient']).reshape(field.y.shape),
                np.zeros(len(field.time)), selected, test_range)
            trial = {'fold': label, 'fit': details, 'before': before, 'after': after,
                'matched_gain_db': float(10*np.log10(after['matched_energy']/before['matched_energy'])),
                'prediction_error_change_percent': 100*(after['prediction_error']/before['prediction_error']-1)}
            channel['trials'].append(trial)
            result['pending_channel'] = channel
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'channel': c, 'fold': label, 'range_m': details['range_origin_m'],
                'rate_mps': details['path_rate_mps'], 'gradient': details['maximum_projected_gradient'],
                'converged': details['converged'], 'gain_db': trial['matched_gain_db'],
                'error_change_percent': trial['prediction_error_change_percent'],
                'seconds': time.perf_counter()-started}), flush=True)
        result['channels'].append(channel)
        del result['pending_channel']
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    result['complete'] = True
    result['seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
