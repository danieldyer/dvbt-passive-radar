"""Bound a time-averaged range-pulse approximation using recorded metadata.

This is a conditional point-template check, not an autofocus correction. The
replayed 30-symbol circular PSF is not the exact original finite-waveform
ambiguity function. Original archive weights are measured separately.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from field import Field, OFFSETS
from geographic_io import read_profiles
from range_fit import Pulse


def mismatch_loss(actual, predicted, weight, whiten):
    a = (actual@whiten.T)*np.sqrt(weight[:, None])
    b = (np.broadcast_to(predicted, actual.shape)@whiten.T)*np.sqrt(weight[:, None])
    return float(10*np.log10(np.vdot(a, a).real*np.vdot(b, b).real/abs(np.vdot(a, b))**2))


def archived_spectra(capture, archive, channel, interval):
    spectra, information = [], []
    with capture.open('rb') as source:
        for offset, c, first, last in archive['records']:
            if c != channel or last < interval[0] or first >= interval[1]:
                continue
            source.seek(offset)
            tag, size, _ = struct.unpack('<4sIQ', source.read(16))
            meta = json.loads(source.read(size))
            if tag != b'PROF' or meta['response_carriers'] != 6817 or meta['noise_carriers'] != 6817:
                raise ValueError('Original filter spectra are missing')
            source.seek(8*(meta['count']*meta['moments']*meta['size']+meta['stationary_carriers']), 1)
            h = np.frombuffer(source.read(6817*8), '<c8').astype(complex)
            noise = np.frombuffer(source.read(6817*4), '<f4').astype(float)
            weight = abs(h)**2/noise
            spectra.append(weight/weight.sum())
            information.append(weight.sum()/8192)
    return np.array(spectra), np.array(information)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture', 'index', 'templates', 'fields', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--interval', type=float, nargs=2, required=True)
    args = parser.parse_args()
    archive = json.loads(args.index.read_text())
    if Path(archive['identity'][0]).resolve() != args.capture.resolve():
        raise ValueError('Archive index belongs to a different capture')
    _, records = read_profiles(args.templates)
    report = {'interval_s': args.interval, 'scope': __doc__, 'channels': [],
        'inputs': {key: str(getattr(args, key)) for key in ('capture', 'index', 'templates', 'fields')}}
    offsets = np.arange(-6, 6.01, .5)
    # Pulse interpolation is linear; evaluate its exact implemented matrix.
    interpolation = np.array([Pulse(column)(OFFSETS[None, :]-offsets[:, None]) for column in np.eye(129)])
    for channel in range(5):
        with np.load(args.fields/f'channel-{channel}.npz') as f:
            field = Field(dict(f), f['pulse_samples'])
        selected = records[(records['channel'] == channel)&(records['time'] >= args.interval[0])&
            (records['time'] < args.interval[1])&(records['information'] > 0)]
        psf, information = selected['psf'].astype(complex), selected['information']
        mean = np.average(psf, axis=0, weights=information)
        actual = np.einsum('ni,ior->nor', psf, interpolation)
        old = np.einsum('i,ior->or', field.pulse.samples, interpolation)
        local = np.einsum('i,ior->or', mean, interpolation)
        old_loss = [mismatch_loss(actual[:, i], old[i], information, field.whiten) for i in range(len(offsets))]
        local_loss = [mismatch_loss(actual[:, i], local[i], information, field.whiten) for i in range(len(offsets))]
        spectra, gains = archived_spectra(args.capture, archive, channel, args.interval)
        smean = spectra.mean(axis=0)
        entry = {'channel': channel, 'replay_blocks': len(psf), 'original_records': len(spectra),
            'psf_rms_variation_fraction': float(np.linalg.norm(psf-mean)/np.linalg.norm(np.broadcast_to(mean, psf.shape))),
            'archived_spectrum_rms_variation_fraction': float(np.linalg.norm(spectra-smean)/np.linalg.norm(np.broadcast_to(smean, spectra.shape))),
            'archived_information_max_to_min_db': float(10*np.log10(gains.max()/gains.min())),
            'offset_samples': offsets.tolist(), 'conditional_point_loss_old_mean_db': old_loss,
            'conditional_point_loss_local_mean_db': local_loss,
            'worst_point_loss_old_mean_db': max(old_loss), 'worst_point_loss_local_mean_db': max(local_loss)}
        report['channels'].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if not isinstance(v, list)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
