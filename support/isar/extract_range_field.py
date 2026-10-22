"""Extract an offset autofocus crop with the original epochs and linear operator."""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from extract_full_aperture import load
from field import Field, OFFSETS, observation
from imaging import sites
from scene_dictionary import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('aperture', 'fields', 'motion', 'ais', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--interval', type=float, nargs=2, required=True)
    p.add_argument('--shift-samples', type=int, required=True)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    tx, rx, _ = sites(Path(model['geometry_directory']))
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'scope': __doc__, 'interval_s': args.interval, 'range_shift_samples': args.shift_samples,
        'sources_sha256': {str(p): fingerprint(p) for p in (args.motion, args.ais)},
        'channels': [], 'complete': False}
    for c in range(5):
        target = args.output/f'channel-{c}.npz'
        if target.exists():
            raise ValueError('Preserve existing observations; choose a fresh output directory')
        full = load(args.aperture/f'channel-{c}')
        with np.load(args.fields/f'channel-{c}.npz') as d:
            primary = Field(dict(d), d['pulse_samples']).windowed(*args.interval)
        start = int(np.searchsorted(full['time_s'], primary.data['row_time'][0]))
        stop = start+len(primary.data['row_time'])
        np.testing.assert_array_equal(full['time_s'][start:stop], primary.data['row_time'])
        aperture = {k: full[k][start:stop] for k in ('time_s', 'information', 'valid')}
        aperture.update(moments=full['moments'][:, start:stop], metadata_json=full['metadata_json'])
        data = observation(aperture, track, utc_seconds(model['nominal_start_utc']), tx[c], rx,
            model['attitude_prior'], offsets=OFFSETS+args.shift_samples, max_ais_gap_s=400.)
        np.testing.assert_array_equal(data['time'], primary.time)
        np.savez(target, **data, pulse_samples=primary.pulse.samples)
        manifest['channels'].append({'channel': c, 'path': str(target), 'sha256': fingerprint(target),
            'primary_field_sha256': fingerprint(args.fields/f'channel-{c}.npz')})
        (args.output/'fields.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps({'channel': c, 'bins': len(data['time'])}), flush=True)
    manifest['complete'] = True
    (args.output/'fields.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
