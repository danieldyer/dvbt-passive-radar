"""Test sub-grid physical support before interpreting phase mismatch as calibration.

Hold motion, sites, envelope origins and point heights fixed. Refine each shared
horizontal coordinate within one metre of its declared two-metre grid point,
profiling independent complex band responses. Withheld interior one-second
blocks test both independent prediction and subsequent common-phase alignment.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from channel_alignment import AlignmentBand
from field import Field
from fit_physical_profile import solve
from imaging import physical_state, sites
from point_profile import PointProfile
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'support', 'alignment', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    model, support, alignment = [json.loads(p.read_text()) for p in (args.motion, args.support, args.alignment)]
    if model.get('antenna_fit') or model.get('rock_spacing_seconds') or np.any(model.get('channel_path_rate_bias_mps', 0.)):
        raise ValueError('Use fixed roll/pitch, nominal antenna sites and no channel rate correction')
    points = np.array(support['points_body_m'][:args.count])
    curve = SplineMotion(model)
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch, bands = utc_seconds(model['nominal_start_utc']), []
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as loaded:
            field = Field(dict(loaded), loaded['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        state = curve.apply(field.data['row_time'], prior, tx[3], rx)
        block = np.floor(field.time-curve.interval[0]).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        bands.append(AlignmentBand(field, state, points, selected))
    joint = PointProfile(bands, points, model['range_bias_m'])
    result = {'scope': __doc__, 'source_motion': str(args.motion), 'source_support': str(args.support),
        'source_sha256': {str(p): fingerprint(p) for p in (args.motion, args.support, args.alignment, args.ais)},
        'initial_points_body_m': points.tolist(), 'position_bound_m': 1., 'correction_adopted': False, 'complete': False,
        'limitation': 'Motion and point discovery used this aperture. These are conditional prediction controls, not independent absolute position or heading calibration.'}
    result['before'] = {key: joint.validation(joint.initial, coupled=coupled, reference=3)
        for key, coupled in [('independent', False), ('coherent', True)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    def checkpoint(parameters, trace):
        args.output.with_suffix('.checkpoint.json').write_text(json.dumps({'parameters': parameters.tolist(),
            'points_body_m': joint.coordinates(parameters).tolist(), 'trace': trace}, indent=2)+'\n')
        print(json.dumps(trace[-1]), flush=True)
    count = len(joint.initial)
    parameters, details = solve(joint, np.empty((0, count)), np.empty(0), np.empty(0),
        np.full(count, -1.), np.full(count, 1.), iterations=args.iterations, checkpoint=checkpoint)
    result.update(fit=details, parameters=parameters.tolist(), points_body_m=joint.coordinates(parameters).tolist(),
        bound_hits=int(np.sum(abs(parameters) >= 1.-1e-6)),
        after={key: joint.validation(parameters, coupled=coupled, reference=3)
            for key, coupled in [('independent', False), ('coherent', True)]}, complete=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'converged': details['converged'], 'gradient': details['maximum_kkt_residual'],
        'bound_hits': result['bound_hits'], 'prediction_error_change_percent': {
            key: 100*(result['after'][key]['prediction_error']/value['prediction_error']-1)
            for key, value in result['before'].items()}}), flush=True)


if __name__ == '__main__':
    main()
