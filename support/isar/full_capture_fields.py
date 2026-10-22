"""Form the fixed linear autofocus observation once over the full recording."""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from extract_full_aperture import load
from field import observation
from imaging import sites


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('aperture', 'ais', 'geometry-directory', 'registration', 'pulses', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    model = json.loads(args.registration.read_text())
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    args.output.mkdir(parents=True, exist_ok=True)
    for c in range(5):
        path = args.output/f'channel-{c}.npz'
        if path.exists():
            continue
        aperture = load(args.aperture/f'channel-{c}')
        data = observation(aperture, track, utc_seconds(model['nominal_start_utc']), tx[c], rx,
            model['attitude_prior'], max_ais_gap_s=400.)
        with np.load(args.pulses/f'channel-{c}.npz') as f:
            data['pulse_samples'] = f['pulse_samples']
        np.savez(path, **data)
        print(json.dumps({'channel': c, 'bins': len(data['time']), 'original_rows': len(data['row_time'])}), flush=True)


if __name__ == '__main__':
    main()
