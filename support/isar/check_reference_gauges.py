"""Verify body-frame and independent-band phase ambiguities on recorded geometry.

This relabels an unknown scattering model; it does not claim every relabelling
is compatible with an independently identified physical hull. The AIS antenna
trajectory stays the same physical trajectory under the body relabelling.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from field import Field
from imaging import physical_state, sites
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationary import Stationary, make_design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('support', 'alignment', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    support = json.loads(args.support.read_text())
    motion_path = Path(support['source_motion'])
    motion = json.loads(motion_path.read_text())
    alignment = json.loads(args.alignment.read_text())
    if not support['complete'] or not support['fit']['converged']:
        raise ValueError('Use a completed physical point reference')
    if motion.get('rock_spacing_seconds') or motion.get('antenna_fit'):
        raise ValueError('This reference keeps roll/pitch and antenna sites fixed')
    points = np.asarray(support['points_body_m'])
    curve = SplineMotion(motion)
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=motion['reference_height_ellipsoid_m'], mmsi=563052200)
    shift = np.array([80., 5., 0.])
    angle = np.radians(15.)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    new_points = (points-shift)@rotation
    antenna_in_new_body = -rotation.T@shift
    band_rotation = np.array([.7, -1.1, 1.4, 0., -2.2])
    result = {'scope': __doc__, 'source_support': str(args.support),
        'sources_sha256': {str(p): fingerprint(p) for p in (args.support, args.alignment, args.ais, motion_path, Path(__file__))},
        'body_origin_shift_m': shift.tolist(), 'body_yaw_relabelling_degrees': 15.,
        'ais_antenna_coordinates_in_new_body_m': antenna_in_new_body.tolist(),
        'band_phase_relabelling_radians': band_rotation.tolist(), 'channels': [], 'complete': False,
        'correction_adopted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as data:
            field = Field(dict(data), data['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        prior = physical_state(track, utc_seconds(motion['nominal_start_utc'])+field.data['row_time'], motion['attitude_prior'])
        state = curve.apply(field.data['row_time'], prior, tx[3], rx)
        relabelled = {**state,
            'position_ecef_m': state['position_ecef_m']+state['body_to_ecef']@shift,
            'velocity_ecef_mps': state['velocity_ecef_mps']+state['frame_velocity']@shift,
            'body_to_ecef': state['body_to_ecef']@rotation,
            'frame_velocity': state['frame_velocity']@rotation}
        antenna = relabelled['position_ecef_m']+relabelled['body_to_ecef']@antenna_in_new_body
        origin_error = float(np.max(np.linalg.norm(antenna-state['position_ecef_m'], axis=1)))
        old, changed = [], []
        position_error, velocity_error = 0., 0.
        for point, replacement in zip(points, new_points):
            for key, frame, label in [('position_ecef_m', 'body_to_ecef', 'position'),
                                      ('velocity_ecef_mps', 'frame_velocity', 'velocity')]:
                delta = relabelled[key]+relabelled[frame]@replacement-(state[key]+state[frame]@point)
                maximum = float(np.max(np.linalg.norm(delta, axis=1)))
                if label == 'position': position_error = max(position_error, maximum)
                else: velocity_error = max(velocity_error, maximum)
            old.append(field.point(point, motion=state, range_bias_m=motion['range_bias_m'][c]).ravel())
            changed.append(field.point(replacement, motion=relabelled, range_bias_m=motion['range_bias_m'][c]).ravel())
        old, changed = np.column_stack(old), np.column_stack(changed)
        dictionary_error = float(np.linalg.norm(changed-old)/np.linalg.norm(old))
        block = np.floor(field.time-curve.interval[0]).astype(int)
        selected = np.repeat((block%2 == 0)|(block == block.max()), field.y.shape[1])
        background, _ = make_design(field)
        nuisance = Stationary(background, selected)
        coefficient = np.linalg.lstsq(nuisance.remove(old[selected]), nuisance.remove(field.y.ravel()[selected]), rcond=None)[0]
        prediction = old@coefficient
        phase = np.exp(1j*band_rotation[c])
        rotated = (phase*old)@(coefficient/phase)
        phase_error = float(np.linalg.norm(rotated-prediction)/np.linalg.norm(prediction))
        if max(position_error, origin_error) > 5e-9 or velocity_error > 1e-12 or dictionary_error > 2e-8 or phase_error > 1e-12:
            raise ValueError('A declared reference relabelling changed the physical prediction')
        result['channels'].append({'channel': c, 'maximum_point_position_difference_m': position_error,
            'maximum_point_velocity_difference_mps': velocity_error,
            'maximum_ais_antenna_position_difference_m': origin_error,
            'body_relabelling_dictionary_relative_error': dictionary_error,
            'band_relabelling_prediction_relative_error': phase_error})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['channels'][-1]), flush=True)
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
