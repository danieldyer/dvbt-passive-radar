"""Check the conditional field covariance on declared neighboring range crops.

High residual-Doppler samples are diagnostic secondary observations, not assumed
target-free truth. No covariance correction or scientific normalization is
applied. Independent time blocks expose variability and contamination.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from field import Field, OFFSETS, observation
from imaging import sites
from scene_dictionary import fingerprint


def spectral_covariance(field, *, length=128, low_hz=15., high_hz=25.):
    step = float(np.median(np.diff(field.time)))
    frequency = np.fft.fftfreq(length, step)
    selected = (abs(frequency) >= low_hz)&(abs(frequency) <= high_hz)
    window = np.hanning(length)
    window /= np.linalg.norm(window)
    spectra = []
    for start in range(0, len(field.time)-length+1, length):
        rows = slice(start, start+length)
        if not np.all(field.data['information'][rows] > 0):
            continue
        time = field.time[rows]-field.time[start]
        transform = np.exp(-2j*np.pi*frequency[:, None]*time)*window
        spectra.append(transform@field.y[rows])
    spectra = np.asarray(spectra)
    if len(spectra) < 6 or selected.sum() < field.y.shape[1]:
        raise ValueError('Insufficient independent time blocks or frequency samples')
    covariances = np.einsum('tfr,tfs->trs', spectra[:, selected], spectra[:, selected].conj())/selected.sum()
    covariance = covariances.mean(axis=0)
    diagonal = covariance.diagonal().real
    correlation = covariance/np.sqrt(diagonal[:, None]*diagonal)
    split = len(covariances)//2
    rng = np.random.default_rng(7721)
    boot = np.array([covariances[rng.integers(len(covariances), size=len(covariances))].mean(axis=0).diagonal().real
                     for _ in range(1000)])
    return {'time_blocks': len(spectra), 'samples_per_block': length, 'doppler_band_hz': [low_hz, high_hz],
        'frequency_hz': frequency.tolist(), 'mean_spectrum': np.mean(abs(spectra)**2, axis=(0, 2)).tolist(),
        'covariance_real': covariance.real.tolist(), 'covariance_imag': covariance.imag.tolist(),
        'covariance_eigenvalues': np.linalg.eigvalsh(covariance).tolist(),
        'mean_variance': float(diagonal.mean()), 'range_variance': diagonal.tolist(),
        'range_variance_block_bootstrap_95_interval': np.quantile(boot, [.025, .975], axis=0).T.tolist(),
        'maximum_range_correlation': float(np.max(abs(correlation-np.eye(len(correlation))))),
        'first_half_mean_variance': float(np.trace(covariances[:split].mean(axis=0)).real/len(diagonal)),
        'second_half_mean_variance': float(np.trace(covariances[split:].mean(axis=0)).real/len(diagonal)),
        'maximum_interval_error_s': float(abs(np.diff(field.time)-step).max()),
        'uncertainty_scope': 'Bootstrap resamples disjoint time blocks, retaining frequency/range dependence within each block. It is descriptive under temporal exchangeability, not proof that samples are free of target or interference.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aperture', 'fields', 'motion', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--range-shifts', type=int, nargs='+', default=[-12, 0, 16, 32])
    args = parser.parse_args()
    model = json.loads(args.motion.read_text())
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    report = {'scope': __doc__, 'interval_s': model['interval_s'], 'channels': [], 'complete': False,
        'expected': 'Identity range covariance and unit spectral power after field whitening, under its declared independent-row conditional Gaussian model.',
        'interpretation': 'Excess or correlated power can be noise-model error, structured interference, or unmodelled echoes. These controls do not select or apply a new covariance from the ship-bearing data.',
        'sources_sha256': {str(p): fingerprint(p) for p in (args.motion, args.ais)},
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_field_covariance.py', 'field.py')}, 'correction_adopted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for c in range(5):
        aperture_path = args.aperture/f'channel-{c}.npz'
        with np.load(aperture_path) as f:
            t = f['time_s']
            selected = (t >= model['interval_s'][0])&(t < model['interval_s'][1])
            aperture = {key: f[key][selected] for key in ('time_s', 'information', 'valid')}
            aperture['moments'] = f['moments'][:, selected]
            aperture['metadata_json'] = f['metadata_json']
        with np.load(args.fields/f'channel-{c}.npz') as f:
            pulse = f['pulse_samples']
        record = {'channel': c, 'aperture': str(aperture_path), 'crops': []}
        for shift in args.range_shifts:
            data = observation(aperture, track, utc_seconds(model['nominal_start_utc']), tx[c], rx,
                               model['attitude_prior'], offsets=OFFSETS+shift)
            field = Field(data, pulse)
            measured = spectral_covariance(field)
            record['crops'].append({'range_shift_samples': shift, **measured})
            print(json.dumps({'channel': c, 'shift': shift, 'mean_variance': measured['mean_variance'],
                'maximum_range_correlation': measured['maximum_range_correlation']}), flush=True)
        report['channels'].append(record)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
