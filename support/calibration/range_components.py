"""Inspect resolved range components of a candidate on a common physical grid."""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

from phase_history import index_archive, interpolate, C, FS, TAU
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'live'))
from read_archive import exact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--histories', type=Path, required=True)
    p.add_argument('--label', default='D')
    p.add_argument('--seed-channel', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    archive = index_archive(args.capture, args.histories/'archive-index.json')
    description = json.loads((args.histories/f'{args.label}.json').read_text())
    seed = description['channels'][args.seed_channel]
    start, end = description['start'], description['end']
    offsets = np.arange(-8, 8.001, .25)*C/FS
    values = np.zeros((5, 30, len(offsets)), dtype=complex)
    weight = np.zeros((5, 30)); times = np.zeros((5, 30))
    with args.capture.open('rb') as f:
        for position, ch, first, last in archive['records']:
            if last < start-.002 or first>end:
                continue
            f.seek(position)
            tag, length, size = struct.unpack('<4sIQ', exact(f, 16))
            meta = json.loads(exact(f, length)); count = meta['count']
            if tag != b'PROF' or meta['moments'] != 2 or meta['size'] != 1024:
                raise ValueError('Unsupported profile layout')
            data = np.frombuffer(exact(f, count*2048*8), '<c8').reshape(2,count,1024)
            rows = np.asarray(meta['rows'], dtype=float)
            t = rows[:,1]+4095.5/FS
            keep = (t>=start)&(t<end)&(rows[:,5]!=0)
            t = t[keep]
            if not len(t):
                continue
            path = np.polynomial.polynomial.polyval((t-seed['epoch'])/15, seed['path'])
            speed = np.polynomial.polynomial.polyval((t-seed['epoch'])/15,
                        np.polynomial.polynomial.polyder(seed['path']))/15
            rf = archive['header']['applied_channels'][ch]['rf_hz']
            doppler = -rf*speed/C
            phase = np.exp(1j*TAU*rf/C*(path-seed['path'][0]))
            bins = np.floor(t-start).astype(int)
            w = meta['information']
            weight[ch] += np.bincount(bins, weights=np.full(len(t),w), minlength=30)
            times[ch] += np.bincount(bins, weights=w*t, minlength=30)
            zero, one = data[0,keep], data[1,keep]
            for j, offset in enumerate(offsets):
                coordinate = (path+offset)*FS/C
                z = (interpolate(zero,coordinate)-1j*TAU*doppler*.001024*interpolate(one,coordinate))*phase
                values[ch,:,j] += (np.bincount(bins, weights=w*z.real, minlength=30)+
                                   1j*np.bincount(bins, weights=w*z.imag, minlength=30))
    values /= weight[:,:,None]; times /= weight
    np.savez_compressed(args.output, values=values, time=times, offset_m=offsets,
                         information=weight, reference_seed=np.array(seed['path']), epoch=seed['epoch'])
    for ch in range(5):
        power = np.mean(abs(values[ch])**2,axis=0)
        print(args.label, ch, 'mean-power peak offset m',offsets[np.argmax(power)],flush=True)


if __name__=='__main__':
    main()
