"""Extract narrow complex echo histories for phase-centre analysis.

Uses archived complex profiles, their actual epochs and information weights.
The seed is only a reversible deramp; no fitted phase is removed from the data.
"""
import argparse
import copy
import json
from pathlib import Path
import struct
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'live'))
from read_archive import read_header, exact
from candidates import C

FS = 8e6
TAU = 2 * np.pi


def index_archive(path, cache):
    stat = path.stat()
    identity = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['identity'] == identity:
            return saved
    entries = []
    with path.open('rb') as f:
        header = read_header(f)
        while True:
            pos = f.tell()
            tag, length, size = struct.unpack('<4sIQ', exact(f, 16))
            if not 0 < length <= 1024**2 or size > 64 * 1024**2:
                raise ValueError('Invalid record bounds')
            meta = json.loads(exact(f, length))
            if tag == b'PROF':
                rows = meta['rows']
                entries.append([pos, meta['channel'], rows[0][1], rows[-1][1]])
            f.seek(size, 1)
            if tag == b'DONE':
                if f.read(1):
                    raise ValueError('Unexpected data after DONE')
                break
    saved = {'identity': identity, 'header': header, 'records': entries}
    cache.write_text(json.dumps(saved))
    return saved


def fit_track_seed(tr, start, end):
    """One physical deramp from the selected source track, shared by all bands.

    Weak channels must not select unrelated peaks and then discard the intended
    echo through a narrow gate around that selection. Per-band geometry remains
    a parameter of the subsequent complex likelihood, not an extraction guess.
    """
    epoch = (start + end) / 2
    reference = tr[(tr[:, 0] >= start) & (tr[:, 0] <= end)]
    if len(reference)<4 or np.any(np.diff(reference[:,0])<=0):
        raise ValueError('A seed needs at least four ordered source-track detections')
    fit_points = reference
    t = (fit_points[:, 0] - epoch) / 15
    design = np.polynomial.polynomial.polyvander(t, 3)
    v0 = np.linalg.lstsq(design, fit_points[:, 2], rcond=None)[0]
    v = least_squares(lambda p: (design @ p - fit_points[:, 2]) / .3, v0, loss='soft_l1').x
    path = np.r_[0., v * 15 / np.arange(1, 5)]
    offsets = fit_points[:, 1] - np.polynomial.polynomial.polyval(t, path)
    path[0] = float(np.median(offsets))
    return {'epoch': epoch, 'scale': 15, 'duration_s':end-start, 'path': path.tolist(),
            'coarse_observations': reference.tolist(), 'coarse_points': len(reference)}


def channel_seeds(track, start, end, source_channel, frequencies):
    shared = fit_track_seed(track, start, end)
    result = []
    for channel, frequency in enumerate(frequencies):
        seed = copy.deepcopy(shared)
        seed['rf_hz'] = frequency
        seed['velocity_seed_from_other_channel'] = channel != source_channel
        if channel != source_channel:
            # A copied source measurement is not an independent observation.
            # Other bands contribute their actual complex samples to fitting.
            seed['coarse_observations'] = []
            seed['coarse_points'] = 0
        result.append(seed)
    return result


def interpolate(profiles, coordinates):
    """64-tap, symmetric windowed-sinc interpolation of the complex range axis."""
    centre = np.floor(coordinates).astype(int)
    indices = centre[:, None] + np.arange(-31, 33)
    delta = coordinates[:, None] - indices
    weights = np.sinc(delta) * np.i0(8 * np.sqrt(np.maximum(0, 1 - (delta / 32)**2))) / np.i0(8)
    weights /= weights.sum(axis=1)[:, None]
    if np.any(indices < -64) or np.any(indices > 928):
        raise ValueError('Calibration echo outside saved positive delay support')
    return np.sum(np.take_along_axis(profiles, indices % 1024, axis=1) * weights, axis=1)


def extract(path, archive, channel, seed, start, end, bin_seconds=.1):
    rf = archive['header']['applied_channels'][channel]['rf_hz']
    time, measured, information, range_power = [], [], [], []
    with path.open('rb') as f:
        for position, ch, first, last in archive['records']:
            if ch != channel or last < start - .002 or first > end:
                continue
            f.seek(position)
            tag, length, size = struct.unpack('<4sIQ', exact(f, 16))
            m = json.loads(exact(f, length))
            count = m['count']
            if tag != b'PROF' or m['moments'] != 2 or m['size'] != 1024:
                raise ValueError('Invalid matched-profile layout')
            data = np.frombuffer(exact(f, count * 2048 * 8), '<c8').reshape(2, count, 1024)
            rows = np.asarray(m['rows'], dtype=float)
            t = rows[:, 1] + 4095.5 / FS
            keep = (t >= start) & (t <= end) & (rows[:, 5] != 0)
            if not keep.any():
                continue
            t = t[keep]
            length_m = np.polynomial.polynomial.polyval((t-seed['epoch'])/seed['scale'], seed['path'])
            speed = np.polynomial.polynomial.polyval((t-seed['epoch'])/seed['scale'],
                        np.polynomial.polynomial.polyder(seed['path'])) / seed['scale']
            fd = -rf * speed / C
            coordinates = length_m * FS / C
            zero = interpolate(data[0, keep], coordinates)
            one = interpolate(data[1, keep], coordinates)
            signal = zero - 1j * TAU * fd * .001024 * one
            # Its inverse is part of the later forward model, including seed[0].
            phase = TAU * rf / C * (length_m - seed['path'][0])
            signal *= np.exp(1j * np.remainder(phase, TAU))
            time.extend(t);measured.extend(signal);information.extend(np.full(len(t), m['information']))
            range_power.extend(np.abs(zero)**2)
    t, z, w = map(np.asarray, (time, measured, information))
    if not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError('Missing/reordered calibration rows')
    bin_index = np.floor((t-start) / bin_seconds).astype(int)
    n = int(np.ceil((end-start) / bin_seconds))
    good = bin_index < n
    t, z, w, bin_index = t[good], z[good], w[good], bin_index[good]
    weight = np.bincount(bin_index, weights=w, minlength=n)
    norm = np.maximum(weight, 1e-30)
    average = (np.bincount(bin_index, weights=w*z.real, minlength=n) +
               1j*np.bincount(bin_index, weights=w*z.imag, minlength=n)) / norm
    times = np.bincount(bin_index, weights=w*t, minlength=n) / norm
    count = np.bincount(bin_index, minlength=n)
    scatter = np.bincount(bin_index, weights=w**2 * abs(z-average[bin_index])**2, minlength=n)
    variance = scatter / norm**2 * count / np.maximum(count-1, 1)
    # A diagnostic conditional thermal prediction, not a measured CRLB claim.
    thermal = 1 / (8192 * norm)
    return {'time': times, 'value': average, 'variance': variance, 'thermal_variance': thermal,
            'rows': count, 'information': weight, 'row_time': t, 'row_value': z,
            'row_information': w}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--tracks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--jobs', type=Path, help='JSON list of [label, channel, track index, start, end]')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = index_archive(args.capture, args.output / 'archive-index.json')
    tracks = json.loads(args.tracks.read_text())['tracks']
    jobs = (json.loads(args.jobs.read_text()) if args.jobs else
            [('A', 1, 1, 35., 65.), ('B', 2, 5, 455., 485.), ('C', 2, 6, 469., 499.)])
    for label, channel, index, first, last in jobs:
        tr = np.array(tracks[channel][index])
        frequencies = [c['rf_hz'] for c in archive['header']['applied_channels']]
        seeds = channel_seeds(tr, first, last, channel, frequencies)
        description = {'label': label, 'start': first, 'end': last, 'channels': seeds,
                       'reference_channel': channel, 'source_track': index,
                       'candidate_source': str(args.candidates), 'track_source': str(args.tracks),
                       'seed_policy': 'one selected source track shared across bands; no independent peak reselection'}
        for ch, seed in enumerate(seeds):
            data = extract(args.capture, archive, ch, seed, first, last)
            np.savez_compressed(args.output / f'{label}-{ch}.npz', **data)
            snr = 10*np.log10(np.maximum(abs(data['value'])**2/data['variance'], 1e-30))
            print(label, ch, 'source seed channel', channel, 'source detections', seeds[channel]['coarse_points'],
                  '0.1s bin diagnostic SNR median', np.median(snr), flush=True)
        (args.output / f'{label}.json').write_text(json.dumps(description, indent=2))


if __name__ == '__main__':
    main()
