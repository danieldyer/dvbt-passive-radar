"""Finite-waveform information retention for coherent mixtures of ship points.

Known reconstructed symbols, response and diagonal occupied-carrier noise are
held fixed. Original sampled epochs follow the fitted physical trajectory.
This tests arbitrary complex amplitudes in the stated fixed spatial support,
not actual scene correctness, unknown clutter or motion-estimator efficiency.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import block_diag, solve_triangular

from block_pulse import BlockPulse
from finite_reference import (FS, NFFT, circular_profile, coefficient_planes,
    covariance, echo, probes)
from geographic_movie import state_at
from imaging import corrected_sites, sites
from motion import C, excess_path
from prepare_causal_scene import unique_points
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from symbol_moments import moments


def append_root(root, observation):
    """Streaming QR preserves the full raw design without normal equations."""
    return np.linalg.qr(np.concatenate((root, observation)), mode='r')


def retained_information(raw_root, compressed, relative_floor):
    """All complex scene directions resolvable at a declared relative power."""
    _, singular, right = np.linalg.svd(raw_root, full_matrices=False)
    keep = singular**2 > singular[0]**2*relative_floor
    white = right.conj().T[:, keep]/singular[keep]
    value = np.linalg.svd(compressed@white, compute_uv=False)**2
    value = np.sort(np.r_[value, np.zeros(int(keep.sum())-len(value))])
    if not keep.any() or value[0] < -1e-10 or value[-1] > 1+2e-6:
        raise ValueError('Linear compression increased the scene information')
    return {'relative_power_floor': relative_floor, 'resolved_modes': int(keep.sum()),
        'unresolved_modes': int((~keep).sum()), 'relative_information_eigenvalues': value.tolist(),
        'raw_relative_information_eigenvalues': (singular**2/singular[0]**2).tolist(),
        'worst_information_loss_db': float(-10*np.log10(max(value[0], np.finfo(float).tiny))),
        'isotropic_amplitude_average_retention': float(np.linalg.norm(compressed)**2/np.linalg.norm(raw_root)**2)}


def projected_statistics(template, exact, relative_floor=1e-24):
    """Best linear filters in the mean-template span, with explicit rank audit."""
    u, singular, _ = np.linalg.svd(template, full_matrices=False)
    keep = singular**2 > singular[0]**2*relative_floor
    return u[:, keep].conj().T@exact, {'resolved_model_modes': int(keep.sum()),
        'unresolved_model_modes': int((~keep).sum()), 'relative_power_floor': relative_floor,
        'relative_model_singular_values': (singular/singular[0]).tolist()}


def amplitude_risk(raw_root, exact, template):
    """Noise and bias of mean-template GLS in raw-data Fisher coordinates.

    Unlike a template-span bound, this uses the uncorrected mean-model inverse.
    The prior used only to report average bias is equal independent complex
    point powers, scaled to a declared total expected raw-observation SNR.
    """
    _, raw_singular, raw_right = np.linalg.svd(raw_root, full_matrices=False)
    u, singular, right = np.linalg.svd(template, full_matrices=False)
    if min(raw_singular[-1]/raw_singular[0], singular[-1]/singular[0]) <= 1e-12:
        return {'complete': False, 'reason': 'Full physical amplitude inverse is numerically unresolved; no silent rank reduction.'}
    p = template.shape[1]
    factor = raw_singular[:, None]*raw_right
    inverse = right.conj().T/singular
    mapping = inverse@(u.conj().T@exact)
    bias = factor@(mapping-np.eye(p))
    variance = float(np.linalg.norm(factor@inverse)**2/p)
    average_bias_per_snr = float(np.linalg.norm(bias)**2/(p*np.linalg.norm(raw_root)**2))
    unit_raw = raw_right.conj().T/raw_singular
    worst_bias_per_snr = float(np.linalg.norm(bias@unit_raw, 2)**2/p)
    return {'complete': True, 'physical_complex_amplitudes': p,
        'noise_risk_over_raw_crlb': variance,
        'average_squared_bias_over_raw_crlb_per_snr': average_bias_per_snr,
        'worst_squared_bias_over_raw_crlb_per_snr': worst_bias_per_snr,
        'mean_model_amplitude_mapping_real_imag': np.stack([mapping.real, mapping.imag], axis=-1).tolist(),
        'mse_over_raw_crlb_at_integrated_snr_db': {str(snr): variance+10**(snr/10)*average_bias_per_snr for snr in (20, 30, 40, 50)},
        'scope': 'Conditional full fixed-support GLS using exact compressed covariance. Includes bias of the uncorrected mean template; does not claim a bound on nonlinear motion or actual ship-image error.'}


def metric_risk(raw_root, exact, template, metric):
    """Risk of the actual weighted estimator in exact-noise coordinates."""
    _, rs, rv = np.linalg.svd(raw_root, full_matrices=False)
    u, s, v = np.linalg.svd(metric@template, full_matrices=False)
    if min(rs[-1]/rs[0], s[-1]/s[0]) <= 1e-12:
        return {'complete': False, 'reason': 'Unresolved full physical amplitude inverse'}
    inverse = (v.conj().T/s)@(u.conj().T@metric)
    factor = rs[:, None]*rv
    bias = factor@(inverse@exact-np.eye(template.shape[1]))
    variance = float(np.linalg.norm(factor@inverse)**2/template.shape[1])
    bias_per_snr = float(np.linalg.norm(bias)**2/(template.shape[1]*np.linalg.norm(raw_root)**2))
    return {'complete': True, 'noise_risk_over_raw_crlb': variance,
        'average_squared_bias_over_raw_crlb_per_snr': bias_per_snr,
        'mse_over_raw_crlb_at_integrated_snr_db': {str(snr): variance+10**(snr/10)*bias_per_snr for snr in (20, 30, 40, 50)},
        'scope': 'Actual stated weighted estimator; exact compressed noise propagated through its inverse. Fixed physical support and Gaussian rows only.'}


def probe_groups(rows, block):
    """Retain complete consecutive time-selected blocks, never bridge a gap."""
    if block == 1:
        return [[row] for row in rows], []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['probe_interval'], row['probe_bin']].append(row)
    accepted, rejected = [], []
    for key, selected in grouped.items():
        step = NFFT/FS*(1+1/selected[0]['guard'])
        valid = (len(selected) == block and
            [row['probe_offset'] for row in selected] == list(range(block)) and
            len({row['segment'] for row in selected}) == 1 and
            np.all(abs(np.diff([row['time_s'] for row in selected])-step) < 2e-7))
        if valid:
            accepted.append(selected)
        else:
            rejected.append({'interval': key[0], 'bin': key[1], 'rows': len(selected),
                'reason': 'Incomplete, nonconsecutive or cross-gap symbol block'})
    if not accepted:
        raise ValueError('No complete consecutive probe blocks')
    return accepted, rejected


def whiten_block(exact, template, noise, weights, deramp):
    """Apply the declared weighted coherent mean before exact whitening."""
    weights = np.asarray(weights)
    coefficient = weights*np.asarray(deramp)/weights.sum()
    covariance = np.einsum('n,nij->ij', abs(coefficient)**2, noise)
    lower = np.linalg.cholesky(covariance)
    return tuple(solve_triangular(lower, np.einsum('n,nij->ij', coefficient, values),
                                  lower=True) for values in (exact, template))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-channel', type=int, default=0, help='Zero uses every in-aperture probe')
    parser.add_argument('--range-half-width', type=int, default=16)
    parser.add_argument('--template-spectrum', choices=('average', 'per-row', 'per-block'), default='average',
        help='Compare the current mean pulse with exact per-row realised spectral information')
    parser.add_argument('--averaging-rows', type=int, default=1)
    parser.add_argument('--field-directory', type=Path,
        help='Use the saved observation centre, Doppler and deramp at each original symbol epoch')
    parser.add_argument('--compact-pulse', action='store_true',
        help='Evaluate per-block spectra using 129 retained lags and fixed Chebyshev interpolation')
    parser.add_argument('--estimator-metric', choices=('exact', 'mean', 'block'), default='exact',
        help='Also measure the actual estimator with mean or per-block circular noise covariance')
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists() or args.per_channel < 0 or args.range_half_width < 8 or args.averaging_rows < 1:
        raise ValueError('Use a new output directory, nonnegative row limit and adequate range crop')
    if args.compact_pulse and args.template_spectrum != 'per-block':
        raise ValueError('Compact response requires the per-block spectrum control')
    if args.estimator_metric != 'exact' and (not args.compact_pulse or not args.field_directory):
        raise ValueError('Metric controls require compact block kernels and saved field coordinates')
    model = json.loads(args.motion.read_text()); curve = SplineMotion(model)
    if not model['complete'] or not model['fit']['converged'] or model['original_row_kkt'] > 1e-7:
        raise ValueError('Use an original-row stationary physical motion fit')
    if not model['fixed_ne_frame'] or np.any(curve.yaw_coefficients):
        raise ValueError('Keep the declared fixed N/E frame')
    nominal_tx, nominal_rx, _ = sites(Path(model['geometry_directory']))
    tx, rx = corrected_sites(nominal_tx, nominal_rx, model.get('antenna_fit'))
    points = unique_points([np.asarray(p) for p in model['selected_points_body_m']])
    rows = defaultdict(list)
    for row in probes(args.input):
        channel = row['channel']
        if not model['interval_s'][0] <= row['time_s'] < model['interval_s'][1]:
            continue
        if args.per_channel and len(rows[channel]) >= args.per_channel:
            continue
        rows[channel].append({k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in row.items() if k not in ('measured', 'zero', 'one')})
    if sorted(rows) != list(range(5)):
        raise ValueError('Need in-aperture waveform probes for every channel')
    offsets = np.arange(-args.range_half_width, args.range_half_width+1)
    files = [args.input, args.motion, Path(__file__), *[Path(__file__).with_name(n) for n in
        ('finite_reference.py', 'geographic_movie.py', 'imaging.py', 'spline_motion.py', 'symbol_moments.py', 'block_pulse.py')]]
    if args.field_directory:
        files += [args.field_directory/f'channel-{c}.npz' for c in range(5)]
    result = {'scope': __doc__, 'complete': False,
        'source_sha256': {str(p): fingerprint(p) for p in files},
        'requested_interval_s': model['interval_s'], 'points_body_m': points.tolist(),
        'range_offsets_samples': offsets.tolist(), 'probe_limit_per_channel': args.per_channel,
        'noise_scope': 'Known frozen per-carrier variance; independent proper-complex rows. No interference-covariance, motion, location or scattering uncertainty is claimed.',
        'operator_scope': 'Production C0/C1 coefficient planes, including previous-row linear clock interpolation, frozen centre-Doppler combination, declared coherent row averaging and exact covariance whitening. Sparse time-selected blocks do not represent a full aperture.',
        'averaging_rows': args.averaging_rows,
        'observation_centre': str(args.field_directory) if args.field_directory else 'Fitted physical centre',
        'template_spectrum': args.template_spectrum,
        'compact_pulse': args.compact_pulse,
        'estimator_metric': args.estimator_metric,
        'mean_model_scope': 'Declared average or per-row realised spectral information, with the same circular mean within-symbol moment model. No arbitrary normalization of recorded data.',
        'channels': []}
    args.output.mkdir(parents=True)
    for channel, selected in sorted(rows.items()):
        groups, rejected = probe_groups(selected, args.averaging_rows)
        selected = [row for group in groups for row in group]
        selected = [{k: (v.astype(complex if np.iscomplexobj(v) else float)
            if isinstance(v, np.ndarray) else v) for k, v in row.items()} for row in selected]
        times = np.array([r['time_s'] for r in selected])
        state = state_at(None, model, curve, nominal_tx, nominal_rx, times)
        centre = state['position_ecef_m']; frame = state['body_to_ecef']
        position = centre[:, None, :]+np.einsum('nij,pj->npi', frame, points)
        velocity = state['velocity_ecef_mps'][:, None, :]+np.einsum('nij,pj->npi', state['frame_velocity'], points)
        path, gradient = excess_path(position.reshape(-1, 3), tx[channel], rx)
        path = path.reshape(len(times), len(points))
        rate = np.sum(gradient.reshape(position.shape)*velocity, axis=-1)
        centre_path, centre_gradient = excess_path(centre, tx[channel], rx)
        centre_rate = np.sum(centre_gradient*state['velocity_ecef_mps'], axis=-1)
        observation_coordinate = (centre_path+model['range_bias_m'][channel])*FS/C
        observation_omega = -2*np.pi*selected[0]['rf_hz']/C*centre_rate*NFFT/FS
        observation_path = centre_path
        if args.field_directory:
            with np.load(args.field_directory/f'channel-{channel}.npz') as data:
                mean_pulse_samples = data['pulse_samples']
                original_time = data['row_time']
                index = np.searchsorted(original_time, times)
                index = np.clip(index, 1, len(original_time)-1)
                index -= abs(original_time[index-1]-times) < abs(original_time[index]-times)
                if np.max(abs(original_time[index]-times)) > 2e-8:
                    raise ValueError('Waveform probes do not match saved observation epochs')
                observation_path = data['centre_path_m'][index]
                observation_coordinate = observation_path*FS/C
                observation_omega = data['centre_omega'][index]
        averaged = np.mean([abs(r['response'])**2*abs(r['current'])**2/r['variance']/r['information'] for r in selected], axis=0)
        block_spectra = []
        if args.template_spectrum == 'per-block':
            for first in range(0, len(selected), args.averaging_rows):
                group = selected[first:first+args.averaging_rows]
                block_spectra.append(np.average([abs(r['response'])**2*abs(r['current'])**2/
                    r['variance']/r['information'] for r in group], axis=0,
                    weights=[r['information'] for r in group]))
        kernels = None
        if args.compact_pulse:
            packed = np.zeros((len(block_spectra), NFFT), complex)
            packed[:, np.arange(-3408, 3409) % NFFT] = block_spectra
            samples = np.fft.ifft(packed, axis=1)[:, np.arange(-64, 65) % NFFT]
            kernels = BlockPulse(samples)
        raw_root = np.zeros((0, len(points)), complex)
        compressed, templates, covariance_conditions = [], [], []
        metric_blocks = []
        relative_delays = []
        block_exact, block_template, block_noise = [], [], []
        for i, row in enumerate(selected):
            rf = row['rf_hz']; k = 2*np.pi*rf/C
            origin = row['reference_origin']-row['source_offset']
            previous_origin = row['reference_origin']-row['previous_source_offset']
            coordinate = observation_coordinate[i]
            delays = (path[i]+model['range_bias_m'][channel])*FS/C
            fd = -rf/C*rate[i]
            omega, centre_omega = 2*np.pi*fd*NFFT/FS, observation_omega[i]
            phase = np.exp(-1j*k*path[i])
            planes = coefficient_planes(row['current'], row['previous'], row['response'], row['variance'],
                origin, coordinate, offsets, guard=row['guard'],
                previous_fraction=previous_origin-np.floor(previous_origin))
            a = planes[0]-1j*centre_omega*planes[1]
            noise = covariance(a, row['variance'])
            covariance_conditions.append(float(np.linalg.cond(noise)))
            q = np.column_stack([echo(row['current'], row['previous'], row['response'], origin,
                delay, doppler, guard=row['guard'])*amplitude
                for delay, doppler, amplitude in zip(delays, fd, phase)])
            raw_root = append_root(raw_root, q/np.sqrt(row['variance'])[:, None])
            block_exact.append(a@q)
            block_noise.append(noise)
            m0, m1, _, _ = moments(omega)
            spectral = (averaged if args.template_spectrum == 'average' else
                block_spectra[i//args.averaging_rows] if args.template_spectrum == 'per-block' else
                abs(row['response'])**2*abs(row['current'])**2/row['variance']/row['information'])
            pulse = (np.column_stack([circular_profile(spectral, delay, coordinate, offsets)
                                      for delay in delays]) if kernels is None else
                kernels.at(coordinate+offsets[:, None]-delays[None, :], i//args.averaging_rows))
            mean = pulse*((m0-1j*centre_omega*m1)*phase)
            block_template.append(mean)
            if len(block_exact) == args.averaging_rows:
                first = i+1-args.averaging_rows
                exact_white, template_white = whiten_block(block_exact, block_template, block_noise,
                    [r['information'] for r in selected[first:i+1]],
                    np.exp(1j*k*observation_path[first:i+1]))
                compressed.append(exact_white); templates.append(template_white)
                if args.estimator_metric != 'exact':
                    information = np.array([r['information'] for r in selected[first:i+1]])
                    true_noise = np.einsum('n,nij->ij', (information/information.sum())**2, block_noise)
                    lag = offsets[:, None]-offsets[None, :]+64
                    pulse = mean_pulse_samples if args.estimator_metric == 'mean' else samples[i//args.averaging_rows]
                    approximate = pulse[lag]
                    if args.estimator_metric == 'block':
                        approximate = approximate*np.average(1+observation_omega[first:i+1]**2/12, weights=information)
                    approximate = (approximate+approximate.conj().T)/2
                    transform = solve_triangular(np.linalg.cholesky(approximate),
                        np.linalg.cholesky(true_noise), lower=True)*np.sqrt(NFFT*information.sum())
                    metric_blocks.append(transform)
                block_exact, block_template, block_noise = [], [], []
            relative_delays.extend((delays-coordinate).tolist())
        compressed, templates = np.concatenate(compressed), np.concatenate(templates)
        records = []
        core = np.asarray(model['selected_points_body_m'][channel])
        core_columns = [int(np.argmin(np.linalg.norm(points-point, axis=1))) for point in core]
        for name, columns in [('band_core', core_columns), ('all_band_point_union', list(range(len(points))))]:
            raw = raw_root[:, columns]; exact, mean = compressed[:, columns], templates[:, columns]
            projected, rank = projected_statistics(mean, exact)
            records.append({'support': name, 'columns': columns, 'mean_template_span_rank': rank,
                'uncorrected_mean_gls_risk': amplitude_risk(raw, exact, mean),
                'implemented_metric_gls_risk': (metric_risk(raw, exact, mean, block_diag(*metric_blocks))
                    if metric_blocks else None),
                'bounds': [{'raw_relative_power_floor': floor,
                    'finite_profiles': retained_information(raw, exact, floor),
                    'mean_template_filter_span': retained_information(raw, projected, floor)}
                    for floor in (1e-8, 1e-10, 1e-12)]})
        path_out = args.output/f'channel-{channel}.npz'
        np.savez(path_out, raw_root=raw_root, exact_compressed=compressed, mean_template=templates,
                 times_s=times, points_body_m=points, range_offsets_samples=offsets)
        item = {'channel': channel, 'probes': len(selected), 'first_last_time_s': times[[0, -1]].tolist(),
            'coherent_blocks': len(groups), 'rejected_blocks': rejected,
            'relative_point_delay_samples_minmax': [min(relative_delays), max(relative_delays)],
            'maximum_covariance_condition': max(covariance_conditions),
            'arrays': str(path_out), 'arrays_sha256': fingerprint(path_out), 'supports': records}
        result['channels'].append(item)
        (args.output/'information.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'channel': channel, 'probes': len(selected),
            'core_worst_loss_db': records[0]['bounds'][0]['mean_template_filter_span']['worst_information_loss_db'],
            'union_worst_loss_db': records[1]['bounds'][0]['mean_template_filter_span']['worst_information_loss_db'],
            'seconds': time.monotonic()-started}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started)
    (args.output/'information.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
