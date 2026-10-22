"""Measure fast-model discrepancy against original epochs over the whole capture."""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from field import Field
from imaging import physical_state, sites
from midpoint_model import discrepancy, representative
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'motion', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    model = json.loads(args.motion.read_text()); curve = SplineMotion(model)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    epoch = utc_seconds(model['nominal_start_utc'])
    points = np.array([[a, p, 20.] for a in (-180., 0., 180.) for p in (-60., 0., 60.)])
    output = {'scope': __doc__, 'points_body_m': points.tolist(), 'results': [], 'complete': False}
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples'])
        for start in (20., 142., 280., 420., 550.):
            exact = field.windowed(start, start+30.)
            fast = representative(exact)
            states = []
            for item in (exact, fast):
                t = item.data['row_time']
                prior = physical_state(track, epoch+t, model['attitude_prior'], max_gap_s=400.)
                states.append(curve.apply(t, prior, tx[3], rx))
            records = [discrepancy(exact.point(point, motion=states[0], range_bias_m=model['range_bias_m'][c]),
                fast.point(point, motion=states[1], range_bias_m=model['range_bias_m'][c])) for point in points]
            output['results'].append({'channel': c, 'interval_s': [start, start+30], 'points': records})
            print(json.dumps({'channel': c, 'start_s': start,
                'maximum_snr_loss_db': max(v['matched_snr_loss_db'] for v in records),
                'maximum_phase_degrees': max(abs(v['constant_phase_difference_degrees']) for v in records)}), flush=True)
    output['complete'] = True
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
