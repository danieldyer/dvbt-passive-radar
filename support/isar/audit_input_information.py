"""Measure conditional delay/motion information through the ship-input crop.

Use recorded epochs, weights, geometry and continuously refined support. No
phase or motion correction is fitted, and no scientific observation is changed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from compression_information import range_crop_information, temporal_information
from field import Field, FS
from imaging import physical_state, sites
from motion import C
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('support', 'alignment', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    args = parser.parse_args()
    support, alignment = [json.loads(p.read_text()) for p in (args.support, args.alignment)]
    model_path = Path(support['source_motion'])
    model = json.loads(model_path.read_text())
    curve = SplineMotion(model)
    points = np.asarray(support['points_body_m'][:args.count])
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    report = {'scope': __doc__, 'interval_s': model['interval_s'], 'channels': [], 'complete': False,
        'theory': 'Complex Gaussian, known separable range covariance; independent original rows; unknown complex point amplitude removed by Schur complement.',
        'parameters': ['constant physical path (m)', 'path rate (m/s)', 'path acceleration (m/s^2)'],
        'temporal_model': 'Recorded mean pulse and information; uniform within-symbol energy for C0/C1 covariance. This is a conditional model, not a measured real-noise covariance certificate.',
        'range_model': 'Independent exact flat occupied-spectrum Fourier model; not a universal bound for arbitrary recorded spectral weights.',
        'sources_sha256': {str(p): fingerprint(p) for p in (args.support, args.alignment, args.ais, model_path)},
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_input_information.py', 'compression_information.py', 'field.py')},
        'correction_adopted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        t = field.data['row_time']
        prior = physical_state(track, utc_seconds(model['nominal_start_utc'])+t, model['attitude_prior'])
        state = curve.apply(t, prior, tx[model['reference_channel']], rx)
        elapsed = t-np.mean(model['interval_s'])
        results = []
        for i, point in enumerate(points):
            position = state['position_ecef_m']+state['body_to_ecef']@point
            velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
            delta, omega, _ = field.path_kinematics(position, velocity)
            signal, path, doppler, _ = field.path_response(delta, omega, range_bias_m=model['range_bias_m'][c], derivatives=True)
            derivative = np.stack([path, elapsed[:, None]*path-field.k*.001024*doppler,
                .5*elapsed[:, None]**2*path-field.k*.001024*elapsed[:, None]*doppler], axis=-1)
            result = temporal_information(field, signal, derivative)
            offsets = (delta+model['range_bias_m'][c])*FS/C
            result.update(point_index=i, point_body_m=point.tolist(), range_offset_samples_minmax=[float(offsets.min()), float(offsets.max())])
            results.append(result)
        worst = max(-10*np.log10(min(p['relative_fisher_eigenvalues'])) for p in results)
        record = {'channel': c, 'points': results,
            'worst_motion_information_loss_db': float(worst),
            'worst_matched_snr_loss_db': float(max(-10*np.log10(p['matched_snr_fraction']) for p in results))}
        report['channels'].append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'points'}), flush=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    w = 2*np.pi*np.arange(-3408, 3409)/8192
    q = np.ones(len(w))
    report['range_crop'] = [{'radius_samples': radius, 'offset_samples': float(offset),
        **range_crop_information(w, q, np.arange(-radius, radius+1), offset)}
        for radius in (8, 12) for offset in np.arange(-9., 9.01, .25)]
    report['complete'] = True
    report['seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
