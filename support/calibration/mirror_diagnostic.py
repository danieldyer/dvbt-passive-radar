"""Measure mirror-band covariance in archived IQ without correcting the data.

The diagnostic uses E[X_a(f) X_b(-f)], not the ordinary cross spectrum.
It can reveal conjugate leakage but is not an image-rejection calibration:
independent broadcast signals can also share deterministic pilot structure.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'live'))
from read_archive import exact, read_header


def channel_rate(header):
    return header['corrected_rate_hz']/header['decimation']


def spectra(samples, pairs, nfft=4096, group=64):
    """Non-overlapping rectangular FFTs, then sum adjacent frequency bins."""
    if nfft % group or samples.shape[1] < nfft:
        raise ValueError('Insufficient samples or incompatible frequency grouping')
    count = samples.shape[1] // nfft
    x = np.fft.fft(samples[:, :count*nfft].reshape(len(samples), count, nfft), axis=-1)
    mirror = (-np.arange(nfft)) % nfft
    power = np.mean(abs(x)**2, axis=1)
    cross = np.array([np.mean(x[a]*x[b][:, mirror], axis=0) for a, b in pairs])
    # Preserve negative-frequency pairing before shifting/grouping either side.
    pair_power = np.array([[power[a], power[b, mirror]] for a, b in pairs])
    def grouped(value):
        return np.fft.fftshift(value, axes=-1).reshape(*value.shape[:-1], -1, group).mean(axis=-1)
    return count, grouped(cross), grouped(pair_power)


def summarize(records, pairs, rf, rate, nfft, group):
    counts = np.array([r['fft_count'] for r in records])
    cross = np.array([r['cross_real'] for r in records])+1j*np.array([r['cross_imag'] for r in records])
    powers = np.array([r['pair_power'] for r in records])
    mean = np.average(cross, axis=0, weights=counts)
    power = np.average(powers, axis=0, weights=counts)
    rho = abs(mean)/np.sqrt(power[:, 0]*power[:, 1])
    # Record-level scatter is empirical: do not assume all OFDM samples or bins
    # are independent, or label this as a confidence bound on hardware leakage.
    scatter = np.sqrt(np.average(abs(cross-mean)**2, axis=0, weights=counts))
    window_results = []
    for window in sorted({r['window_start_s'] for r in records}):
        mask = np.array([r['window_start_s'] == window for r in records])
        c = np.average(cross[mask], axis=0, weights=counts[mask])
        p = np.average(powers[mask], axis=0, weights=counts[mask])
        window_results.append({'start_s': window, 'coherence_db': (20*np.log10(np.maximum(abs(c)/np.sqrt(p[:, 0]*p[:, 1]), 1e-30))).tolist()})
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, 1/rate)).reshape(-1, group).mean(axis=1)
    occupied = abs(frequencies) < 3.3e6
    return {'pairs': pairs, 'rf_hz': rf, 'sample_rate_hz': rate,
            'fft_size': nfft, 'frequency_group_bins': group, 'records': len(records),
            'analysed_seconds': float(counts.sum()*nfft/rate),
            'frequency_offset_hz': frequencies.tolist(),
            'coherence_db': (20*np.log10(np.maximum(rho, 1e-30))).tolist(),
            'record_scatter_coherence_db': (20*np.log10(scatter/np.sqrt(power[:, 0]*power[:, 1]))).tolist(),
            'occupied_coherence_percentiles_db': np.percentile(20*np.log10(np.maximum(rho[:, occupied], 1e-30)), [50, 95, 100], axis=1).T.tolist(),
            'windows': window_results,
            'scope': 'Descriptive nonconjugated covariance, not a hardware image-rejection bound. No correction applied.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--starts', nargs='+', type=float, default=[10., 50., 250., 550.])
    p.add_argument('--seconds', type=float, default=1.)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    nfft, group = 4096, 64
    records = []
    with args.capture.open('rb') as source:
        header = read_header(source)
        rate = channel_rate(header)
        rf = [ch['rf_hz'] for ch in header['applied_channels']]
        lo = header['applied_capture']['centre_hz']
        pairs = [[a, b] for a in range(len(rf)) for b in range(a+1, len(rf)) if abs(rf[a]+rf[b]-2*lo) < 1.]
        if not pairs:
            raise ValueError('No mirror-frequency pairs in the recording metadata')
        while True:
            tag, length, size = struct.unpack('<4sIQ', exact(source, 16))
            if length > 1024**2 or size > 64*1024**2:
                raise ValueError('Invalid record dimensions')
            meta = json.loads(exact(source, length))
            intervals = []
            if tag == b'CHIQ':
                for start in args.starts:
                    first = max(0, round(start*rate)-meta['first'])
                    last = min(meta['count'], round((start+args.seconds)*rate)-meta['first'])
                    if last-first >= nfft:
                        intervals.append((start, first, last))
            if intervals:
                if size != meta['channels']*meta['count']*8:
                    raise ValueError('Invalid channelised IQ dimensions')
                iq = np.frombuffer(exact(source, size), '<c8').reshape(meta['channels'], meta['count'])
                for start, first, last in intervals:
                    count, cross, power = spectra(iq[:, first:last], pairs, nfft, group)
                    records.append({'window_start_s': start, 'fft_count': count,
                                    'cross_real': cross.real.tolist(), 'cross_imag': cross.imag.tolist(),
                                    'pair_power': power.tolist()})
            else:
                source.seek(size, 1)
            if tag == b'DONE' or (tag == b'CHIQ' and meta['first']/rate > max(args.starts)+args.seconds):
                break
    if not records:
        raise ValueError('No IQ samples in requested windows')
    result = summarize(records, pairs, rf, rate, nfft, group)
    result['source'] = str(args.capture)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('pairs', 'analysed_seconds', 'occupied_coherence_percentiles_db')}, indent=2))


if __name__ == '__main__':
    main()
