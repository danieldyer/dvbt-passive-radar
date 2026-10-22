"""Retain disjoint secondary range crops through the exact autofocus operator.

Range shift zero is the ship-bearing control; other shifts are secondary
observations, not asserted target-free noise. Original 16-row bin boundaries,
phase deramp, information and range whitening are identical to full-fields.
The explicitly profiled stationary response is saved separately from the data.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from extract_full_aperture import load
from field import Field, OFFSETS, observation
from imaging import sites
from scene_dictionary import fingerprint
from stationary import Stationary, make_design


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('aperture', 'fields', 'motion', 'ais', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--interval', type=float, nargs=2, default=[120., 210.])
    p.add_argument('--shifts', type=int, nargs='+', default=[-96, -64, -32, 0, 32, 64, 96])
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    tx, rx, _ = sites(Path(model['geometry_directory']))
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'scope': __doc__, 'interval_s': args.interval, 'range_shifts_samples': args.shifts,
        'sources_sha256': {str(v): fingerprint(v) for v in (args.motion, args.ais)},
        'secondary_guard': 'Nearest secondary sample is 24 profile samples (899 m excess path) from the ship-crop centre.',
        'channels': [], 'complete': False}
    for c in range(5):
        target = args.output/f'channel-{c}.npz'
        if target.exists():
            raise ValueError('Choose a fresh output directory; preserve the previous secondary observations')
        full = load(args.aperture/f'channel-{c}')
        with np.load(args.fields/f'channel-{c}.npz') as a:
            primary = Field(dict(a), a['pulse_samples']).windowed(*args.interval)
            pulse = a['pulse_samples']
        # Match the exact original boundaries of the existing observation.
        start = int(np.searchsorted(full['time_s'], primary.data['row_time'][0]))
        stop = start+len(primary.data['row_time'])
        if not np.array_equal(full['time_s'][start:stop], primary.data['row_time']):
            raise ValueError('Primary field and original profile epochs differ')
        aperture = {key: full[key][start:stop] for key in ('time_s', 'information', 'valid')}
        aperture.update(moments=full['moments'][:, start:stop], metadata_json=full['metadata_json'])
        data, residuals, nuisance_q, ranks = [], [], [], []
        for shift in args.shifts:
            d = observation(aperture, track, utc_seconds(model['nominal_start_utc']), tx[c], rx,
                model['attitude_prior'], offsets=OFFSETS+shift, max_ais_gap_s=400.)
            field = Field(d, pulse)
            np.testing.assert_array_equal(field.time, primary.time)
            if shift == 0:
                np.testing.assert_allclose(field.y, primary.y, rtol=0., atol=1e-11)
            design, _ = make_design(field)
            nuisance = Stationary(design, np.ones(field.y.size, bool))
            data.append(field.y)
            residuals.append(nuisance.remove(field.y.ravel()).reshape(field.y.shape))
            nuisance_q.append(nuisance.q.reshape(*field.y.shape, -1))
            ranks.append(nuisance.rank)
            print(json.dumps({'channel': c, 'shift': shift, 'stationary_rank': nuisance.rank,
                'raw_mean_power': float(np.mean(abs(data[-1])**2)),
                'residual_mean_power': float(np.mean(abs(residuals[-1])**2))}), flush=True)
        # Ranks may differ by one at a crop boundary. Store each Q explicitly.
        np.savez(target, time=primary.time, information=primary.data['information'],
            shifts=args.shifts, original=np.array(data), residual=np.array(residuals),
            **{f'nuisance_q_{j}': q for j, q in enumerate(nuisance_q)})
        identity['channels'].append({'channel': c, 'path': str(target), 'sha256': fingerprint(target),
            'primary_field_sha256': fingerprint(args.fields/f'channel-{c}.npz'),
            'time_s': primary.time[[0, -1]].tolist(), 'stationary_ranks': ranks})
        (args.output/'secondary.json').write_text(json.dumps(identity, indent=2)+'\n')
    identity['complete'] = True
    (args.output/'secondary.json').write_text(json.dumps(identity, indent=2)+'\n')


if __name__ == '__main__':
    main()
