"""Cache exact physical BP contributions for sliding-aperture ship videos.

Only the display support tool uses this cache. Prefix sums retain complex phase
and information; different coherent apertures do not require repeated imaging.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    args = parser.parse_args()
    description = json.loads((args.images/'image-description.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for channel in range(5):
        with np.load(args.images/f'channel-{channel}-image.npz') as image:
            t, along, port = image['time_s'], image['along_m'], image['port_m']
        with (args.images/f'channel-{channel}.bin').open('rb') as source:
            magic, rows, ranges, count, k = struct.unpack('<8sIIIf', source.read(24))
            if magic != b'ISARBP1\0' or rows != len(t) or count != len(along)*len(port):
                raise ValueError('Source image/operator dimensions disagree')
            geom = np.fromfile(source, '<f4', count=rows*40).reshape(rows, 10, 4)
        binary = args.output/f'channel-{channel}-histories.bin'
        subprocess.run([str(args.executable), str(args.images/f'channel-{channel}.bin'), str(binary), '--histories'], check=True)
        with binary.open('rb') as source:
            magic, blocks, block_rows, returned_pixels = struct.unpack('<8sIII', source.read(20))
        if magic != b'ISARHI1\0' or returned_pixels != count:
            raise ValueError('Invalid history output')
        data = np.memmap(binary, dtype='<f4', mode='r', offset=20, shape=(blocks, count, 4))
        weight = np.add.reduceat(geom[:, 9, 3].astype(float), np.arange(0, rows, block_rows))
        prefix = np.lib.format.open_memmap(args.output/f'channel-{channel}-prefix.npy', mode='w+',
            dtype='<c8', shape=(blocks+1, len(port), len(along)))
        prefix[0] = 0
        total = np.zeros(count, complex)
        for first in range(0, blocks, 32):
            section = data[first:first+32]
            if not np.allclose(section[:, :, 2], weight[first:first+len(section), None], rtol=2e-6):
                raise ValueError('A movie pixel leaves the valid range crop')
            cumulative = np.cumsum(section[:, :, 0].astype(float)+1j*section[:, :, 1], axis=0)+total
            prefix[first+1:first+len(section)+1] = cumulative.reshape(-1, len(port), len(along))
            total = cumulative[-1]
        prefix.flush()
        step = float(np.median(np.diff(t)))
        first = np.arange(0, rows, block_rows)
        last = np.minimum(first+block_rows-1, rows-1)
        np.savez(args.output/f'channel-{channel}-timing.npz', weight_prefix=np.r_[0, np.cumsum(weight)],
            first_time=t[first]-step/2, last_time=t[last]+step/2, rows_per_block=last-first+1,
            along_m=along, port_m=port, block_rows=block_rows, symbol_period_s=step)
        print(json.dumps({'channel': channel, 'blocks': blocks, 'pixels': count, 'block_rows': block_rows}), flush=True)
    (args.output/'description.json').write_text(json.dumps({**description, 'source_images': str(args.images),
        'cache': 'Double-precision prefix accumulation, stored complex float. Original symbol timing and information retained; video windows contain a fixed number of complete blocks.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
