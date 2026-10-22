"""Extract fixed-point GPU histories and test independent/shared phase models.

Points are selected from an already formed AIS-prior image, so temporal holdout
tests the subsequent phase fit, not the prior point-selection step. Images and
measurements are retained independently of the fit; no archive is modified.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from autofocus import fit
from imaging import write_input
from motion import C


def anchors(image, along, port, count=8):
    x, y = np.meshgrid(along, port)
    power = abs(image)**2
    allowed = (abs(x) <= 220)&(abs(y) <= 65)
    pixels = []
    for _ in range(count):
        if not allowed.any():
            break
        index = np.argmax(np.where(allowed, power, -1))
        row, column = np.unravel_index(index, power.shape)
        pixels.append([x[row, column], y[row, column], 20, 0])
        allowed &= (x-x[row, column])**2+(y-y[row, column])**2 >= 50**2
    return np.asarray(pixels)


def extract(directory, output, channel, executable, height, *, pixels=None):
    with np.load(directory/f'channel-{channel}-image.npz') as image:
        if pixels is None:
            pixels = anchors(image['complex_image'], image['along_m'], image['port_m'])
        t = image['time_s']
    pixels[:, 2] = height
    with (directory/f'channel-{channel}.bin').open('rb') as source:
        magic, rows, ranges, count, k = struct.unpack('<8sIIIf', source.read(24))
        if magic != b'ISARBP1\0' or rows != len(t):
            raise ValueError('Image/operator input mismatch')
        geom = np.fromfile(source, '<f4', count=rows*40).reshape(rows, 10, 4)
        source.seek(count*16, 1)
        moments = np.fromfile(source, '<c8').reshape(2, rows, ranges)
    input_path, result_path = output/f'channel-{channel}-anchors.bin', output/f'channel-{channel}-histories.bin'
    write_input(input_path, geom, moments, pixels, k*C/(2*np.pi))
    subprocess.run([str(executable), str(input_path), str(result_path), '--histories'], check=True)
    with result_path.open('rb') as source:
        magic, blocks, block_size, count = struct.unpack('<8sIII', source.read(20))
        values = np.fromfile(source, '<f4').reshape(blocks, count, 4)
    if magic != b'ISARHI1\0' or count != len(pixels):
        raise ValueError('Invalid history output')
    weights = geom[:, 9, 3].astype(float)
    start = np.arange(0, rows, block_size)
    w = np.add.reduceat(weights, start)
    time = np.add.reduceat(weights*t, start)/w
    if not np.allclose(values[:, :, 2], w[:, None], rtol=2e-6):
        raise ValueError('Point history loses crop support')
    z = (values[:, :, 0]+1j*values[:, :, 1])/values[:, :, 2]
    data = {'time': time, 'value': z, 'information': w, 'pixels': pixels, 'phase_scale': k}
    np.savez(output/f'channel-{channel}-histories.npz', **data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--degree', type=int, default=10)
    parser.add_argument('--histories', type=Path, help='Reuse already extracted histories')
    parser.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    args = parser.parse_args()
    description = json.loads((args.images/'image-description.json').read_text())
    interval = description['interval_s']
    args.output.mkdir(parents=True, exist_ok=True)
    groups, independent = [], []
    for channel in range(5):
        if args.histories:
            with np.load(args.histories/f'channel-{channel}-histories.npz') as loaded:
                data = dict(loaded)
        else:
            data = extract(args.images, args.output, channel, args.executable, description['height_offset_m'])
        time = data['time']
        train = ((time-interval[0])//2)%2 == 0
        group = {**data, 'channel': channel, 'phase_scale': 1., 'train': train}
        model = fit([group], interval, degree=args.degree)
        model['units'] = 'radians';model['channel'] = channel
        independent.append(model)
        (args.output/f'channel-{channel}-phase.json').write_text(json.dumps(model, indent=2))
        print(json.dumps({'independent': channel, **model}), flush=True)
        groups.append({**group, 'phase_scale': float(data['phase_scale'])})
    initial = np.mean([np.array(model['coefficients'])/g['phase_scale'] for model, g in zip(independent, groups)], axis=0)
    shared = fit(groups, interval, degree=args.degree, initial=initial)
    shared['units'] = 'metres of phase divided by wavenumber; negative of propagation-path error'
    shared['selection'] = 'Points selected on prior full-aperture image; holdout applies to phase fitting only'
    (args.output/'shared-path-phase.json').write_text(json.dumps(shared, indent=2))
    print(json.dumps({'shared': shared}), flush=True)


if __name__ == '__main__':
    main()
