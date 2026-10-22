"""Known-truth efficiency of shared positions with independent band phases.

Independent time-domain symbol-moment quadrature synthesizes the mean. Proper
complex white noise is added after the declared whitening operator. Physical
motion, antennas and origins are known in this component test. The Fisher bound
profiles unknown complex reflectivities and stationary clutter; it is not the
joint unknown-motion/geometry bound for the recorded ship.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import chi2

from bounded_point_profile import BoundedPointProfile, solve_points
from scene_dictionary import fingerprint
from test_attitude_motion import independent_echo
from test_channel_alignment import synthetic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trials', type=int, default=128)
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists() or args.trials < 32:
        raise ValueError('Use a fresh output and at least 32 trials')
    rng = np.random.default_rng(197364)
    truth = np.array([0., 0., .25, -.15, -.2, .1])
    alignment, origins, _ = synthetic()
    base = alignment.bands; points = base[0].points.copy()
    actual = points.copy(); actual[:, :2] += truth.reshape(-1, 2)
    coefficients = [rng.uniform(.5, 1., len(points))*np.exp(1j*rng.uniform(-np.pi, np.pi, len(points)))
                    for _ in base]
    means, jacobians = [], []
    for band, origin, coefficient in zip(base, origins, coefficients):
        mean = independent_echo(band.field, band.state, actual, coefficient, range_bias_m=origin)
        path = round(band.field.data['centre_path_m'].mean()*8e6/299792458)*299792458/8e6
        mean += (3+2j)*band.field.stationary(path)
        means.append(mean)
        target, columns = [], []
        for point, amplitude in zip(actual, coefficient):
            value, derivative = band.field.point(point, motion=band.state, derivatives=True, range_bias_m=origin)
            target.append(value.ravel())
            columns.extend((amplitude*derivative[:, :, axis]).ravel()[band.selected] for axis in range(2))
        design = band.nuisance.remove(np.column_stack(target)[band.selected])
        derivative = band.nuisance.remove(np.column_stack(columns))
        derivative -= design@np.linalg.lstsq(design, derivative, rcond=None)[0]
        jacobians.append(np.sqrt(2)*np.r_[derivative.real, derivative.imag])
    jacobian = np.concatenate(jacobians)
    unit_information = jacobian.T@jacobian
    unit_covariance = np.linalg.inv(unit_information)
    # Set a declared local-noise regime with 3 cm worst-axis CRLB standard
    # deviation; all searches start at the original unshifted coordinates.
    sigma = .03/np.sqrt(np.diag(unit_covariance).max())
    information, covariance = unit_information/sigma**2, unit_covariance*sigma**2
    def observation(noisy):
        bands = []
        for original, mean in zip(base, means):
            band = copy.copy(original); band.field = copy.copy(original.field)
            noise = (rng.normal(size=mean.shape)+1j*rng.normal(size=mean.shape))*sigma/np.sqrt(2) if noisy else 0
            band.field.y = mean+noise
            band.clean = band.nuisance.remove(band.field.y.ravel()[band.selected])
            band.energy = float(np.vdot(band.clean, band.clean).real)
            bands.append(band)
        return BoundedPointProfile(bands, points, origins)
    known = observation(False)
    reconstructed = known.evaluate(truth)
    mismatch = reconstructed['value']*known.energy/sigma**2
    if mismatch > .01:
        raise ValueError('Independent synthesis mismatch is material relative to this noise experiment')
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ('bounded_point_profile.py', 'point_profile.py', 'stationarity.py', 'test_attitude_motion.py',
         'test_channel_alignment.py', 'test_rigid_motion.py', 'field.py')]]
    result = {'complete': False, 'scope': __doc__, 'source_sha256': {
        str(path): fingerprint(path) for path in paths}, 'trials_requested': args.trials,
        'seed': 197364, 'truth': truth.tolist(), 'initial': known.initial.tolist(),
        'noise_complex_variance': sigma**2, 'crlb_covariance_m2': covariance.tolist(),
        'expected_information_per_m2': information.tolist(),
        'independent_mean_mismatch_in_noise_units': mismatch,
        'independent_band_coefficients': [[[v.real, v.imag] for v in c] for c in coefficients],
        'cases': [], 'accepted_calibration': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for trial in range(args.trials):
        model = observation(True)
        parameters, fit = solve_points(model, np.full(len(truth), -1.), np.full(len(truth), 1.))
        result['cases'].append({'trial': trial, 'parameters': parameters.tolist(),
            'error_m': (parameters-truth).tolist(), 'converged': fit['converged'],
            'kkt': fit['maximum_kkt_residual'], 'evaluations': fit['evaluations'],
            'bound_hits': int(np.isclose(abs(parameters), 1., atol=1e-7, rtol=0).sum())})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        if (trial+1)%16 == 0:
            print(json.dumps({'trials': trial+1, 'nonstationary': sum(not c['converged'] for c in result['cases']),
                'seconds': time.monotonic()-started}), flush=True)
    errors = np.asarray([case['error_m'] for case in result['cases']])
    dof = errors.size
    result['statistics'] = {'joint_mse_over_crlb': float(np.einsum('ni,ij,nj->', errors, information, errors)/dof),
        'per_axis_mse_over_crlb': (np.mean(errors**2, axis=0)/np.diag(covariance)).tolist(),
        'bias_m': errors.mean(axis=0).tolist(), 'empirical_covariance_m2': np.cov(errors, rowvar=False).tolist(),
        'efficient_normal_reference_interval_95': (chi2.ppf([.025, .975], dof)/dof).tolist(),
        'reference_scope': 'Reference distribution if independent errors are zero-mean Gaussian with covariance equal to the CRLB. This is not a confidence interval proving arbitrary recorded-data efficiency.'}
    result.update(complete=True, all_stationary=all(case['converged'] for case in result['cases']),
        seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', 'statistics': result['statistics'], 'all_stationary': result['all_stationary']}), flush=True)


if __name__ == '__main__':
    main()
