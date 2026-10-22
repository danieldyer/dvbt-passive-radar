"""Screen every licensed site combination under fixed physical ship motion.

Independent complex band reflectivities make the likelihood factor by channel;
enumerating the Cartesian product therefore scores every combination exactly.
The mean envelope origin is aligned, but no target phase/time trajectory is
refitted. These are conditional geometry controls, not accepted site estimates.
"""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from audit_residual_phase import score
from channel_alignment import AlignmentBand
from field import Field
from imaging import physical_state, sites
from motion import excess_path
from scene_dictionary import fingerprint
from ship_site_signature import licensed_options
from spline_motion import SplineMotion
from stationary import Stationary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'support', 'alignment', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    args = parser.parse_args()
    model, support, alignment = [json.loads(p.read_text()) for p in (args.motion, args.support, args.alignment)]
    curve = SplineMotion(model)
    if model.get('antenna_fit') or model.get('rock_spacing_seconds'):
        raise ValueError('Use the fixed-roll/pitch nominal-site motion reference')
    points = np.array(support['points_body_m'][:args.count])
    tx, rx, _ = sites(args.geometry_directory)
    options = licensed_options(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'scope': __doc__, 'source_motion': str(args.motion), 'source_support': str(args.support),
        'source_sha256': {str(p): fingerprint(p) for p in (args.motion, args.support, args.alignment, args.ais)},
        'count': len(points), 'complete': False, 'channels': [], 'correction_adopted': False}
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        state = curve.apply(field.data['row_time'], prior, tx[3], rx)
        block = np.floor(field.time-curve.interval[0]).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        ranges = np.arange(field.y.shape[1])
        folds = [('all', ranges, ranges), ('even to odd', ranges[::2], ranges[1::2]),
            ('odd to even', ranges[1::2], ranges[::2])]
        centre_path = excess_path(state['position_ecef_m'], tx[c], rx)[0]
        channel = {'channel': c, 'options': []}
        for option in [o for o in options if o['channel'] == c]:
            alternate = np.array(option['transmitter_ecef_m'])
            delta = excess_path(state['position_ecef_m'], alternate, rx)[0]-centre_path
            origin = model['range_bias_m'][c]-float(np.average(delta, weights=field.data['row_information']))
            band = AlignmentBand(field, state, points, selected, transmitter=alternate)
            target = band.design(origin)[0]
            filename = f'channel-{c}-site-{option["site_id"]}.npy'
            np.save(args.output/filename, target)
            trials = []
            for label, train_range, test_range in folds:
                mask = np.zeros(field.y.shape, bool)
                mask[np.ix_(selected, train_range)] = True
                mask = mask.ravel()
                nuisance = Stationary(band.nuisance.design, mask)
                design, data = nuisance.remove(target[mask]), nuisance.remove(field.y.ravel()[mask])
                coefficient, _, rank, singular = np.linalg.lstsq(design, data, rcond=None)
                if rank != len(points):
                    raise ValueError('Site control changed numerical scene rank')
                residual = data-design@coefficient
                test = score(field, band.nuisance.design, (target@coefficient).reshape(field.y.shape),
                    np.zeros(len(field.time)), selected, test_range)
                trials.append({'fold': label, 'training_error': float(np.vdot(residual, residual).real),
                    'training_energy': float(np.vdot(data, data).real),
                    'condition': float(singular[0]/singular[-1]), 'test': test,
                    'coefficients_real_imag': np.c_[coefficient.real, coefficient.imag].tolist()})
            channel['options'].append({**option, 'range_origin_m': origin, 'trials': trials, 'design': filename})
            print(json.dumps({'channel': c, 'site': option['site_id'], 'kind': option['station_type'],
                'origin_m': origin, 'training_error': trials[0]['training_error'],
                'test_error': trials[0]['test']['prediction_error']}), flush=True)
        np.savez(args.output/f'channel-{c}-observation.npz', data=field.y, time=field.time,
            selected=selected, background=band.nuisance.design)
        result['channels'].append(channel)
        (args.output/'fit.json').write_text(json.dumps(result, indent=2)+'\n')
    result['combinations'] = []
    for label in ['all', 'even to odd', 'odd to even']:
        combinations = []
        for combination in product(*[c['options'] for c in result['channels']]):
            trials = [next(t for t in o['trials'] if t['fold'] == label) for o in combination]
            combinations.append({'sites': [o['site_id'] for o in combination],
                'training_error': sum(t['training_error'] for t in trials),
                'test_error': sum(t['test']['prediction_error'] for t in trials),
                'test_matched_energy': sum(t['test']['matched_energy'] for t in trials)})
        combinations.sort(key=lambda c: c['training_error'])
        result['combinations'].append({'fold': label, 'training_selected': combinations[0], 'all': combinations})
    result['complete'] = True
    (args.output/'fit.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
