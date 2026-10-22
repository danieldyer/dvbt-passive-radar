#!/usr/bin/env python3
"""Convert the independent 64 MS/s int16 fixture to the B200mini's 61.44 MS/s.

Validation tooling only. SciPy's symmetric rational FIR is independent of the
production Vulkan channel bank. A fixed 25-input/24-output phase relation and
overlapping context preserve the common I/Q time origin across every block.
"""
import argparse
import struct
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def blocks(samples, block=25*65536, cyclic=False):
    if block <= 0 or block % 25:
        raise ValueError('Block must be a positive multiple of 25 input samples')
    if cyclic and len(samples) % 25:
        raise ValueError('Cyclic input must close the rational resampling phase')
    halo = 100  # Greater than the FIR half-support, and a multiple of 25.
    for start in range(0, len(samples), block):
        count = min(block, len(samples)-start)
        before, after = max(0, start-halo), min(len(samples), start+count+halo)
        values = np.asarray(samples[before:after], dtype=np.float32)
        left, right = max(0, halo-start), max(0, start+count+halo-len(samples))
        if cyclic:
            values = np.concatenate([np.asarray(samples[np.arange(-left, 0) % len(samples)], dtype=np.float32), values,
                                     np.asarray(samples[np.arange(right) % len(samples)], dtype=np.float32)])
        else:
            values = np.pad(values, ((left, right), (0, 0)))
        output = resample_poly(values, 24, 25, axis=0, window=('kaiser', 8.6))
        yield output[96:96+(count*24+24)//25]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--cyclic', action='store_true', help='Wrap FIR context for a verified cyclic fixture')
    a = p.parse_args()
    with a.input.open('rb') as source:
        header = source.read(44)
    if (len(header) != 44 or header[:4] != b'RIFF' or header[8:12] != b'WAVE'
            or header[12:16] != b'fmt ' or header[36:40] != b'data'
            or struct.unpack_from('<HHI', header, 20) != (1, 2, 64000000)):
        raise ValueError('Expected the minimal stereo int16 64 MS/s simulator WAV')
    count = struct.unpack_from('<I', header, 40)[0]//4
    data = np.memmap(a.input, mode='r', dtype='<i2', offset=44, shape=(count, 2))
    length = (count*24+24)//25
    result_header = bytearray(header)
    struct.pack_into('<I', result_header, 4, 36+length*4)
    struct.pack_into('<II', result_header, 24, 61440000, 61440000*4)
    struct.pack_into('<I', result_header, 40, length*4)
    written = 0
    with a.output.open('xb') as target:
        target.write(result_header)
        for result in blocks(data, cyclic=a.cyclic):
            target.write(np.clip(np.rint(result), -32768, 32767).astype('<i2').tobytes())
            written += len(result)
    if written != length:
        raise RuntimeError('Rational fixture resampling changed the sample count')
    print(f'{written} I/Q samples at 61.44 MS/s; one common zero-phase rational conversion')


if __name__ == '__main__':
    main()
