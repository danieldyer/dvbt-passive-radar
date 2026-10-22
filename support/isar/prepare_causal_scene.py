"""Cache exact original-row causal templates and fixed-fold QR statistics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from causal_scattering import response, uniform_delays
from ais import utc_seconds
from full_motion import Band
from imaging import corrected_sites, motion_state, sites
from mixed_experiment import load
from scene_dictionary import fingerprint
from scene_statistics import compress
from spline_motion import SplineMotion
from stationary import Stationary


def unique_points(groups):
    points = []
    for p in np.concatenate(groups):
        if not points or min(np.linalg.norm(p-q) for q in points) > 1e-8:
            points.append(p)
    return np.asarray(points)


def read_records(directory, channel, fold):
    with np.load(Path(directory)/f'channel-{channel}.npz') as data:
        return {key: data[fold+'_'+key].item() if key in ('outside', 'observations') else data[fold+'_'+key]
                for key in ('design', 'data', 'outside', 'observations')}


def motion_override(experiment, path):
    """Change the physical model while preserving fields, support and folds."""
    model = json.loads(path.read_text())
    if not model['complete'] or not model['fit']['converged'] or model['original_row_kkt'] > 1e-7:
        raise ValueError('Use a completed original-row stationary physical motion fit')
    for name, digest in model.get('source_sha256', {}).items():
        if fingerprint(Path(name)) != digest:
            raise ValueError('Motion source changed: '+name)
    curve = SplineMotion(model)
    if not model['fixed_ne_frame'] or np.any(curve.yaw_coefficients):
        raise ValueError('Keep yaw exactly zero')
    for key in ('interval_s', 'nominal_start_utc', 'linear_reference', 'anchor_path',
                'anchor_transverse', 'path_reference_time_s', 'reference_channel',
                'path_spacing_seconds', 'heading_spacing_seconds', 'transverse_spacing_seconds'):
        if model.get(key) != experiment.model.get(key):
            raise ValueError('Physical motion override changed the reference frame or aperture: '+key)
    for original, new in zip(experiment.model['core_points_enu_m'], model['selected_points_body_m']):
        np.testing.assert_array_equal(original, new)
    np.testing.assert_array_equal(model['channel_gains_real_imag'], experiment.model['channel_gains_real_imag'])
    tx, rx, _ = sites(Path(model['geometry_directory']))
    actual_tx, actual_rx = corrected_sites(tx, rx, model.get('antenna_fit'))
    bands, offsets = [], []
    for c, original in enumerate(experiment.bands):
        prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+original.field.data['row_time'], model)
        bands.append(Band(original.field, prior, tx[model['reference_channel']], rx,
            original.points, model, original.selected_time, range_bias_m=model['range_bias_m'][c]))
        np.testing.assert_array_equal(bands[-1].selected, original.selected)
        np.testing.assert_allclose(bands[-1].clean, original.clean, rtol=0, atol=0)
        offsets.append(np.array([actual_tx[c]-original.field.data['transmitter_ecef_m'],
                                 actual_rx-original.field.data['receiver_ecef_m']]))
    return model, bands, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--delay-width-m', type=float, required=True)
    parser.add_argument('--nodes', type=int, help='Override the node count, retaining both convergence checks')
    parser.add_argument('--motion', type=Path, help='Stationary physical model; keep the source data, spatial support and validation folds')
    args = parser.parse_args()
    started = time.monotonic()
    experiment = load(args.source)
    model, bands = experiment.model, experiment.bands
    physical, offsets = model, [None]*len(bands)
    if args.motion:
        physical, bands, offsets = motion_override(experiment, args.motion)
    points = unique_points([np.asarray(p) for p in model['core_points_enu_m']])
    frequencies = np.array([b.field.metadata['rf_hz'] for b in bands])
    difference = np.linspace(-np.ptp(frequencies)-7e6, np.ptp(frequencies)+7e6, 2001)/299792458
    exact_kernel = (np.exp(-1j*np.pi*difference*args.delay_width_m)*
                    np.sinc(difference*args.delay_width_m))
    nodes = args.nodes or 8
    while True:
        delays, weights = uniform_delays(args.delay_width_m, nodes)
        error = float(abs(np.exp(-2j*np.pi*difference[:, None]*delays)@weights-exact_kernel).max())
        if error < 1e-8:
            break
        if args.nodes or nodes >= 64:
            raise ValueError('The delay quadrature fails the analytic cross-band covariance check')
        nodes += 4
    check_delays, check_weights = uniform_delays(args.delay_width_m, nodes+8)
    implementation = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ('causal_scattering.py', 'scene_statistics.py', 'field.py', 'full_motion.py', 'spline_motion.py',
         'imaging.py', 'mixed_experiment.py')]]
    identity = {'source': str(args.source), 'source_sha256': fingerprint(args.source),
        'delay_width_m': args.delay_width_m, 'nodes': len(delays),
        'training_sha256': model['training_sha256'],
        'fields_sha256': {str(Path(experiment.training['fields'])/f'channel-{c}.npz'):
                          experiment.training['channels'][c]['field_sha256'] for c in range(5)},
        'implementation_sha256': {str(p): fingerprint(p) for p in implementation}}
    if args.motion:
        identity['motion_override'] = {'source': str(args.motion), 'sha256': fingerprint(args.motion),
            'scope': 'Same original fields, point coordinates, stationary nuisance and split; only physical motion, envelope origins and declared antenna offsets change.'}
    result = {'identity': identity, 'scope': __doc__, 'complete': False,
        'motion_fitted': False, 'yaw_fitted': False, 'antenna_positions_fitted': False,
        'physical_model_source': str(args.motion or args.source),
        'range_origins_m': physical['range_bias_m'],
        'antenna_fit': physical.get('antenna_fit'),
        'points_enu_m': points.tolist(), 'delays_m': delays.tolist(), 'delay_weights': weights.tolist(),
        'power_delay_prior': 'Independent proper-complex Gaussian scattering with uniform power per internal delay over [0, width]; the zero-width limit is one common complex point response.',
        'column_order': 'delay-major, then source-union spatial point; unit integrated delay prior power per point',
        'analytic_covariance_maximum_absolute_error': error,
        'template_rows': 'All original symbol rows, followed by the exact saved averaging and whitening.',
        'folds': 'Outer training even half-second blocks; outer test odd. Inner training blocks 0 mod 4, inner test 2 mod 4. Each fold profiles its own stationary nuisance.',
        'source_baseline': model['summaries']['after'], 'source_baseline_channels': model['after'],
        'limitation': 'Motion, source support and envelope origins were previously estimated using this recording. Conditional test of frequency-response coupling; not independent target discovery or accepted calibration.',
        'channels': []}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'cache.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['identity'] != identity:
            raise ValueError('Existing cache uses different inputs or implementation')
        result = saved
    for c, band in enumerate(bands):
        destination = args.output/f'channel-{c}.npz'
        if c < len(result['channels']):
            if fingerprint(destination) != result['channels'][c]['sha256']:
                raise ValueError('An existing cache channel changed')
            continue
        target = response(band, physical['parameters'], points, delays, weights=weights,
                          antenna_offsets=offsets[c])
        block = np.floor((band.field.time-model['interval_s'][0])/.5).astype(int)
        masks = {'outer_train': block%2 == 0, 'outer_test': block%2 == 1,
                 'inner_train': block%4 == 0, 'inner_test': block%4 == 2}
        arrays, checks = {}, {}
        for name, mask in masks.items():
            selected = np.repeat(mask, band.field.y.shape[1])
            nuisance = Stationary(band.nuisance.design, selected)
            a = nuisance.remove(target[selected])
            y = nuisance.remove(band.field.y.ravel()[selected])
            record = compress(a, y)
            record['observations'] -= nuisance.rank
            rng = np.random.default_rng(16223+c)
            coefficient = (rng.normal(size=a.shape[1])+1j*rng.normal(size=a.shape[1]))/band.energy**.5
            direct = float(np.linalg.norm(y-a@coefficient)**2)
            compact = float(np.linalg.norm(record['data']-record['design']@coefficient)**2)+record['outside']
            discrepancy = abs(direct-compact)/max(direct, 1.)
            if discrepancy > 1e-11:
                raise ValueError('QR failed to preserve the original complex residual norm')
            checks[name] = {'relative_likelihood_discrepancy': discrepancy,
                'stationary_rank': nuisance.rank, 'observations': record['observations']}
            for key, value in record.items():
                arrays[name+'_'+key] = value
            if name == 'outer_test':
                expected = model['after'][c]['stationary_only_energy']
                np.testing.assert_allclose(np.vdot(y, y).real, expected, rtol=1e-12)
        probe_points = np.unique(np.linspace(0, len(points)-1, 3, dtype=int))
        probe_rows = np.unique(np.linspace(0, len(target)-1, 64, dtype=int))
        columns = np.concatenate([probe_points+j*len(points) for j in range(len(delays))])
        arrays['probe'] = target[np.ix_(probe_rows, columns)]
        arrays['probe_refined'] = response(band, physical['parameters'], points[probe_points],
            check_delays, weights=check_weights, antenna_offsets=offsets[c])[probe_rows]
        np.savez(destination, **arrays)
        result['channels'].append({'channel': c, 'file': str(destination), 'sha256': fingerprint(destination),
            'checks': checks, 'seconds': time.monotonic()-started})
        manifest.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'channel': c, 'points': len(points), 'columns': target.shape[1],
                          'seconds': time.monotonic()-started}), flush=True)
    probes, refined = [], []
    for c in range(5):
        with np.load(args.output/f'channel-{c}.npz') as data:
            probes.append(data['probe']); refined.append(data['probe_refined'])
    a, b = np.concatenate(probes), np.concatenate(refined)
    reference = b@b.conj().T
    relative = float(np.linalg.norm(a@a.conj().T-reference)/np.linalg.norm(reference))
    result['original_operator_covariance_probe_relative_error'] = relative
    result['quadrature_probe_scope'] = 'Cross-band covariance on three spatial points and 64 retained complex components per band, through the original-row waveform and whitening; compared with eight more quadrature nodes.'
    result.update(complete=bool(relative < 1e-7), seconds=time.monotonic()-started)
    manifest.write_text(json.dumps(result, indent=2)+'\n')
    if not result['complete']:
        raise ValueError(f'Original-operator delay quadrature error {relative:g} exceeds 1e-7')


if __name__ == '__main__':
    main()
