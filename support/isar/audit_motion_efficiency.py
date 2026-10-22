"""Monte Carlo efficiency of the physical variable-projection motion estimator.

This isolates the estimator under its declared Gaussian model, with unknown
complex reflector and stationary amplitudes. It is not a real-capture accuracy
claim. Geometry/operator correctness has separate known-waveform tests.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import chi2

from fit_physical_profile import solve
from full_motion import Band
from physical_profile import PhysicalProfile
from scene_dictionary import fingerprint
from spline_motion import SplineMotion, basis
from test_rigid_motion import fixture


def experiment():
    field = fixture()
    d = field.data
    interval = [0., float(d['row_time'][-1])]
    model = {'interval_s': interval, 'path_spacing_seconds': 6., 'heading_spacing_seconds': 8.}
    count = sum(basis([8.], interval, spacing).shape[1] for spacing in (6., 8.))
    rng = np.random.default_rng(1491)
    truth = rng.normal(size=count)*.02
    truth, active = SplineMotion({**model, 'parameters': truth}).gauge_fixed_parameters()
    prior = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
        'body_to_ecef': d['body_to_ecef'],
        'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
    points = [[-70., 4., 20.], [90., -7., 20.], [15., 8., 20.]]
    block = np.floor(field.time).astype(int)
    selected = (block%2 == 0)|(block == block.max())
    band = Band(field, prior, d['transmitter_ecef_m'], d['receiver_ecef_m'], points, model, selected, range_bias_m=0.)
    target = band.design(truth, range_bias_m=1.3)[0]@np.array([.8+.3j, -.5+.4j, .3-.5j])
    stationary = band.nuisance.design@np.full(band.nuisance.design.shape[1], .2+.1j)
    mean = target+stationary
    field.y = mean.reshape(field.y.shape)
    band.clean = band.nuisance.remove(mean[band.selected])
    band.energy = float(np.vdot(band.clean, band.clean).real)
    joint = PhysicalProfile([band], np.zeros(count), fit_ranges=True)
    true_parameters = np.r_[truth[active], 1.3]
    expected_information = joint.evaluate(true_parameters)['gauss_newton']*joint.energy
    if joint.evaluate(true_parameters)['value'] > 1e-20:
        raise ValueError('Noiseless expected-information reference is not at the declared truth')
    valid = np.repeat(field.root_information > 0, field.y.shape[1])
    signal_power = float(np.mean(abs(target[band.selected&valid])**2))
    return band, mean, true_parameters, expected_information, signal_power, valid


def transverse_experiment():
    """Five bands, unknown complex amplitudes, two physical motion coordinates.

    The midpoint location is the declared coarse ambiguity anchor. It does not
    supply velocities, accelerations or per-sample phase. No AIS is involved.
    """
    fields = [fixture(c) for c in range(5)]
    interval = [0., float(fields[0].data['row_time'][-1])]
    model = {'interval_s': interval, 'path_spacing_seconds': 6., 'heading_spacing_seconds': 8.,
             'transverse_spacing_seconds': 8., 'anchor_transverse': True, 'horizontal_translation': True}
    count = sum(basis([8.], interval, spacing).shape[1] for spacing in (6., 8., 8.))
    truth = np.random.default_rng(1491).normal(size=count)*.02
    truth, active = SplineMotion({**model, 'parameters': truth}).gauge_fixed_parameters()
    reference = fields[3].data['transmitter_ecef_m']
    bands, means, valid, powers = [], [], [], []
    for c, field in enumerate(fields):
        d = field.data
        prior = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
                 'body_to_ecef': d['body_to_ecef'],
                 'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
        block = np.floor(field.time).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        band = Band(field, prior, reference, d['receiver_ecef_m'],
                    [[-70., 4., 20.], [90., -7., 20.], [15., 8., 20.]], model, selected)
        response = np.array([.8+.3j, -.5+.4j, .3-.5j])*np.exp(1j*c*.7)
        target = band.design(truth)[0]@response
        mean = target+band.nuisance.design@np.full(band.nuisance.design.shape[1], .2+.1j)
        field.y = mean.reshape(field.y.shape)
        band.clean = band.nuisance.remove(mean[band.selected])
        band.energy = float(np.vdot(band.clean, band.clean).real)
        keep = np.repeat(field.root_information > 0, field.y.shape[1])
        bands.append(band); means.append(mean); valid.append(keep)
        powers.append(float(np.mean(abs(target[band.selected&keep])**2)))
    joint = PhysicalProfile(bands, np.zeros(count))
    parameters = truth[active]
    expected = joint.evaluate(parameters)['gauss_newton']*joint.energy
    if joint.evaluate(parameters)['value'] > 1e-20:
        raise ValueError('Two-dimensional benchmark is not at the known noiseless truth')
    return bands, means, parameters, expected, float(np.mean(powers)), valid


def summarize(errors, covariance):
    errors = np.asarray(errors)
    n, p = errors.shape
    mse = np.mean(errors**2, axis=0)/np.diag(covariance)
    bias = errors.mean(axis=0)/np.sqrt(np.diag(covariance))
    lower = np.linalg.cholesky(covariance)
    standardized = np.linalg.solve(lower, errors.T).T
    second_moment = standardized.T@standardized/n
    measured_variance = np.var(errors, axis=0, ddof=1)/np.diag(covariance)
    interval = np.array([(n-1)*measured_variance/chi2.ppf(.975, n-1),
                         (n-1)*measured_variance/chi2.ppf(.025, n-1)]).T
    return {'trials': n, 'parameters': p, 'mse_over_crlb_by_parameter': mse.tolist(),
        'bias_in_crlb_standard_deviations': bias.tolist(),
        'mean_mahalanobis_error_per_parameter': float(np.trace(second_moment)/p),
        'normalized_error_second_moment_eigenvalues': np.linalg.eigvalsh(second_moment).tolist(),
        'variance_ratio_95_percent_intervals': interval.tolist(),
        'interval_scope': 'Chi-square variance intervals assume approximately Gaussian estimation errors. Bias and nonconvergence are reported separately; no failed trial is discarded.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--snr-db', type=float, nargs='+', default=[10., 20., 30.])
    parser.add_argument('--seed', type=int, default=91317)
    parser.add_argument('--skip-polish', action='store_true', help='Diagnostic only: omit the solver\'s usual stationarity refinement')
    parser.add_argument('--transverse', action='store_true',
        help='Five-channel two-dimensional physical motion benchmark; no AIS, fixed midpoint coordinate gauge')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 30:
        raise ValueError('At least 30 independent noise trials are required')
    if args.transverse:
        bands, means, truth, information, power, valid_rows = transverse_experiment()
    else:
        band, mean, truth, information, power, valid = experiment()
        bands, means, valid_rows = [band], [mean], [valid]
    base_covariance = np.linalg.inv(information)
    if np.linalg.eigvalsh(information).min() <= 0:
        raise ValueError('The declared physical benchmark has unresolved information nulls')
    rng = np.random.default_rng(args.seed)
    n = len(truth)
    matrix = np.empty((0, n))
    lower, upper = np.full(n, -5.), np.full(n, 5.)
    report = {'scope': __doc__, 'seed': args.seed, 'complete': False, 'cases': [],
        'truth': truth.tolist(), 'unit_noise_information': information.tolist(),
        'signal_power_definition': 'Mean moving-signal power per selected valid whitened complex observation; stationary return excluded.',
        'unknown_nuisance': 'Three constant complex reflector amplitudes per band and the declared constant-delay stationary dictionary.',
        'data_split': 'Alternate one-second blocks with both edges observed, same as the current physical-profile control.',
        'initialization': 'All physical correction coefficients start at zero for every trial. Range origins are fixed in the two-dimensional benchmark and estimated from zero in the original benchmark.',
        'two_dimensional_motion': args.transverse,
        'midpoint_location_anchor': 'Fixed coordinate gauge; no trajectory information is supplied by AIS.',
        'parameter_box': [-5., 5.], 'stationarity_polish_enabled': not args.skip_polish,
        'noise': 'Independent proper complex Gaussian in retained whitened coordinates; erased rows stay erased.',
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_motion_efficiency.py', 'physical_profile.py', 'fit_physical_profile.py', 'full_motion.py', 'spline_motion.py', 'test_rigid_motion.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for snr in args.snr_db:
        variance = power/10**(snr/10)
        covariance = variance*base_covariance
        errors, failures, bound_hits, iterations = [], [], [], []
        for trial in range(args.trials):
            for band, mean, valid in zip(bands, means, valid_rows):
                noise = np.sqrt(variance/2)*(rng.normal(size=len(mean))+1j*rng.normal(size=len(mean)))*valid
                observed = mean+noise
                band.field.y = observed.reshape(band.field.y.shape)
                band.clean = band.nuisance.remove(observed[band.selected])
                band.energy = float(np.vdot(band.clean, band.clean).real)
            joint = PhysicalProfile(bands, np.zeros(sum(b.shape[1] for b in bands[0].bases)), fit_ranges=not args.transverse)
            parameters, result = solve(joint, matrix, np.empty(0), np.empty(0), lower, upper,
                iterations=80, polish_stationarity=not args.skip_polish)
            errors.append(parameters-truth)
            iterations.append(result['iterations'])
            if not result['converged']: failures.append({'trial': trial, 'kkt': result['maximum_kkt_residual']})
            if np.any(abs(parameters) >= 5-1e-6): bound_hits.append(trial)
            if (trial+1)%25 == 0:
                print(json.dumps({'snr_db': snr, 'trials': trial+1, 'nonconverged': len(failures),
                    'seconds': time.perf_counter()-started}), flush=True)
        case = {'snr_db': snr, 'noise_variance': variance, 'crlb_covariance': covariance.tolist(),
            **summarize(errors, covariance), 'nonconverged_trials': failures, 'bound_hit_trials': bound_hits,
            'mean_iterations': float(np.mean(iterations))}
        report['cases'].append(case)
        np.savez(args.output.with_name(args.output.stem+f'-{snr:g}dB.npz'), errors=errors, covariance=covariance)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    report['seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
