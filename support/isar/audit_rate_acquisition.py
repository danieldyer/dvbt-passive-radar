"""Large-phase-error acquisition followed by exact physical ML refinement.

Five bands observe one point with unknown complex band amplitudes and strong
stationary nuisance. Path motion is unknown; point support, orientation and
geometry are known. This isolates capture of the correct path-likelihood basin,
complementing the separate two-dimensional near-truth efficiency experiment.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from audit_motion_efficiency import summarize
from fit_physical_profile import solve
from physical_profile import PhysicalProfile
from radar_rate_initialization import rate_seed
from scene_dictionary import fingerprint
from spline_motion import basis
from test_radar_rate_initialization import experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--snr-db', nargs='+', type=float, default=[10., 20.])
    parser.add_argument('--zero-start-trials', type=int,
        help='Optional smaller paired uninitialized control; all acquired-start trials are still run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.trials<2 or (args.zero_start_trials is not None and not 2<=args.zero_start_trials<=args.trials):
        raise ValueError('At least two trials are needed for an error covariance')
    bands, initial, truth = experiment()
    model = bands[0].model
    npath = bands[0].npath
    means = [b.field.y.copy() for b in bands]
    joint = PhysicalProfile(bands, truth)
    expected = joint.evaluate(joint.initial)['gauss_newton'][:npath, :npath]*joint.energy
    # Projecting the much stronger clutter introduces round-off at roughly
    # 1e-19 relative residual power in this fixture. Keep and report that check.
    noiseless_residual = joint.evaluate(joint.initial)['value']
    if noiseless_residual > 1e-16 or np.linalg.eigvalsh(expected).min()<=0:
        raise ValueError('Invalid noiseless expected-information reference')
    powers = [float(np.mean(abs(b.design(truth)[0][b.selected, 0])**2)) for b in bands]
    power = float(np.mean(powers))
    lower = np.r_[np.full(npath, -200.), np.zeros(len(truth)-npath)]
    upper = -lower
    time_grid = np.linspace(*model['interval_s'], 129)
    matrix = np.zeros((len(time_grid), len(truth)))
    matrix[:, :npath] = basis(time_grid, model['interval_s'], model['path_spacing_seconds'], derivative=2)
    lo, hi = np.full(len(time_grid), -.15), np.full(len(time_grid), .15)
    rng = np.random.default_rng(904127)
    report = {'scope': __doc__, 'complete': False, 'seed': 904127, 'truth_parameters': truth.tolist(),
        'noiseless_relative_residual_power': noiseless_residual,
        'unit_noise_path_information': expected.tolist(), 'cases': [],
        'known_quantities': 'Orientation, target point coordinates and antenna positions; only path spline and complex moving/stationary amplitudes are estimated.',
        'noise': 'Proper complex Gaussian in original retained whitened observations; original erased rows remain erased.',
        'selection': 'The initializer and both likelihood fits see only the training half-second blocks. No trial is discarded.',
        'zero_start_trials': args.zero_start_trials or args.trials,
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_rate_acquisition.py', 'radar_rate_initialization.py', 'physical_profile.py',
             'fit_physical_profile.py', 'test_radar_rate_initialization.py')},
        'signal_power_definition': 'Mean moving-point power per selected whitened complex observation, averaged across bands.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for snr in args.snr_db:
        variance = power/10**(snr/10)
        covariance = variance*np.linalg.inv(expected)
        trials = {name: {'errors': [], 'failures': [], 'objectives': [], 'bound_hits': [], 'seconds': 0.}
                  for name in ('zero_start', 'radar_rate_start')}
        for trial in range(args.trials):
            for band, mean in zip(bands, means):
                noise = np.sqrt(variance/2)*(rng.normal(size=mean.shape)+1j*rng.normal(size=mean.shape))
                band.field.y = mean+noise*(band.field.root_information>0)[:, None]
                band.clean = band.nuisance.remove(band.field.y.ravel()[band.selected])
                band.energy = float(np.vdot(band.clean, band.clean).real)
            acquired, _ = rate_seed(bands, initial)
            for name, start in (('zero_start', initial), ('radar_rate_start', acquired)):
                if name=='zero_start' and trial >= (args.zero_start_trials or args.trials):
                    continue
                tick = time.perf_counter()
                fitted, diagnostics = solve(PhysicalProfile(bands, start), matrix, lo, hi, lower, upper, iterations=100)
                record = trials[name]
                record['seconds'] += time.perf_counter()-tick
                record['errors'].append(fitted[:npath]-truth[:npath])
                record['objectives'].append(diagnostics['objective'])
                if not diagnostics['converged']:
                    record['failures'].append({'trial': trial, 'kkt_residual': diagnostics['maximum_kkt_residual']})
                if np.any(abs(fitted[:npath])>200-1e-6) or np.any(abs(matrix@fitted)>.15-1e-6):
                    record['bound_hits'].append(trial)
            if (trial+1)%10==0:
                print(json.dumps({'snr_db': snr, 'trials': trial+1, 'seconds': time.perf_counter()-started}), flush=True)
        case = {'snr_db': snr, 'noise_variance': variance, 'crlb_covariance': covariance.tolist(), 'methods': {}}
        for name, record in trials.items():
            errors = record.pop('errors')
            np.savez(args.output.with_name(f'{args.output.stem}-{snr:g}dB-{name}.npz'), errors=errors, covariance=covariance)
            case['methods'][name] = {**summarize(errors, covariance), **record}
        report['cases'].append(case)
        report['seconds'] = time.perf_counter()-started
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
