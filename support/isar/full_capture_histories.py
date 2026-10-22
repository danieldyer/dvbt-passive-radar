"""Exact original-row BP histories used to initialize full-capture radar motion.

AIS specifies only the coarse position/course search frame. These complex
histories retain measured radar phase; no phase correction is estimated here.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np

from ais import read_csv, utc_seconds
from extract_full_aperture import load
from imaging import geometry_from_state, physical_state, sites, write_input
from motion import C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('aperture', 'ais', 'geometry-directory', 'registration', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    p.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--chunk-rows', type=int, default=16384)
    p.add_argument('--max-ais-gap-s', type=float, default=400.)
    args = p.parse_args()
    if args.chunk_rows%16:
        raise ValueError('Chunks must preserve the 16-row history partition')
    registration = json.loads(args.registration.read_text())
    epoch = utc_seconds(registration['nominal_start_utc'])
    track = read_csv(args.ais, reference_height_ellipsoid_m=registration['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    x, y = np.meshgrid(np.arange(-240., 241., 12.), np.arange(-96., 97., 12.))
    pixels = np.c_[x.ravel(), y.ravel(), np.full(x.size, 20.), np.zeros(x.size)]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for c in args.channels:
        destination = args.output/f'channel-{c}'
        destination.mkdir(exist_ok=True)
        if (destination/'metadata.json').exists():
            raise ValueError('History extraction already complete; choose remaining channels')
        aperture = load(args.aperture/f'channel-{c}')
        meta = json.loads(aperture['metadata_json'])
        rf = meta['channel_configuration']['rf_hz']
        total = len(aperture['time_s'])
        blocks = (total+15)//16
        values = np.lib.format.open_memmap(destination/'values.npy', mode='w+', dtype='<c8', shape=(blocks, len(pixels)))
        times, information = np.empty(blocks), np.empty(blocks)
        source, result = destination/'working-input.bin', destination/'working-output.bin'
        for first in range(0, total, args.chunk_rows):
            stop = min(total, first+args.chunk_rows)
            t = aperture['time_s'][first:stop]
            weight = aperture['information'][first:stop]*aperture['valid'][first:stop]
            state = physical_state(track, epoch+t, registration['attitude_prior'], max_gap_s=args.max_ais_gap_s)
            geom, _ = geometry_from_state(state, tx[c], rx, rf, meta['first_range_sample'], weight)
            geom[:, 9, 0] += registration['range_bias_m'][c]*8e6/C
            write_input(source, geom, aperture['moments'][:, first:stop], pixels, rf)
            subprocess.run([str(args.executable), str(source), str(result), '--histories'], check=True, stdout=subprocess.DEVNULL)
            with result.open('rb') as stream:
                magic, count, block, count_pixels = struct.unpack('<8sIII', stream.read(20))
                if magic != b'ISARHI1\0' or block != 16 or count_pixels != len(pixels):
                    raise ValueError('Invalid GPU history layout')
                out = np.fromfile(stream, '<f4').reshape(count, len(pixels), 4)
            begin = np.arange(0, len(t), 16)
            w = np.add.reduceat(weight, begin)
            if not np.allclose(out[:, :, 2], w[:, None], rtol=3e-6, atol=1e-30):
                raise ValueError('A history leaves its retained range crop')
            section = slice(first//16, first//16+count)
            values[section] = np.divide(out[:, :, 0]+1j*out[:, :, 1], w[:, None],
                out=np.zeros((count, len(pixels)), complex), where=w[:, None] > 0)
            information[section] = w
            times[section] = np.divide(np.add.reduceat(t*weight, begin), w,
                out=np.add.reduceat(t, begin)/np.diff(np.r_[begin, len(t)]), where=w > 0)
            source.unlink(); result.unlink()
            print(json.dumps({'channel': c, 'last_s': float(t[-1]), 'seconds': time.perf_counter()-started}), flush=True)
        values.flush()
        np.savez(destination/'coordinates.npz', time=times, information=information, pixels=pixels)
        (destination/'metadata.json').write_text(json.dumps({'channel': c, 'rf_hz': rf,
            'interval_s': times[[0, -1]].tolist(), 'coarse_prior': str(args.ais),
            'max_ais_gap_s': args.max_ais_gap_s, 'registration': str(args.registration),
            'nominal_start_utc': registration['nominal_start_utc'], 'complete': True,
            'operator': __doc__}, indent=2)+'\n')


if __name__ == '__main__':
    main()
