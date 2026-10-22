"""Monte Carlo check of finite-waveform amplitude estimator bias and variance.

Uses saved forward matrices, actual reconstructed symbol statistics and fixed
physical support. Noise is independent proper complex Gaussian after exact
compression whitening. This validates a conditional linear estimator, not
ship scattering, motion recovery or unknown interference covariance.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from finite_scene_information import amplitude_risk
from scene_dictionary import fingerprint


def trial_risks(raw, exact, templates, snr_db, trials, seed):
    rng = np.random.default_rng(seed)
    inverses = []
    for template in templates:
        u, singular, right = np.linalg.svd(template, full_matrices=False)
        if singular[-1] <= singular[0]*1e-12:
            raise ValueError('Unresolved physical support; do not hide it with a pseudoinverse cutoff')
        inverses.append((right.conj().T/singular, u.conj().T))
    p = exact.shape[1]
    power = 10**(snr_db/10)/np.linalg.norm(raw)**2
    risks = [[] for _ in templates]
    for first in range(0, trials, 128):
        n = min(128, trials-first)
        coefficients = (rng.normal(size=(p, n))+1j*rng.normal(size=(p, n)))*np.sqrt(power/2)
        noise = (rng.normal(size=(len(exact), n))+1j*rng.normal(size=(len(exact), n)))/np.sqrt(2)
        data = exact@coefficients+noise
        for result, (inverse, left) in zip(risks, inverses):
            estimate = inverse@(left@data)
            result.extend((np.sum(abs(raw@(estimate-coefficients))**2, axis=0)/p).tolist())
    result = []
    for template, values in zip(templates, risks):
        values = np.asarray(values)
        theoretical = amplitude_risk(raw, exact, template)
        expected = (theoretical['noise_risk_over_raw_crlb']+
            10**(snr_db/10)*theoretical['average_squared_bias_over_raw_crlb_per_snr'])
        mean, se = float(values.mean()), float(values.std(ddof=1)/np.sqrt(trials))
        result.append({'mse_over_raw_crlb': mean, 'standard_error': se,
            'approximate_95_percent_interval': [mean-1.96*se, mean+1.96*se],
            'analytic_expectation': expected, 'deviation_standard_errors': (mean-expected)/se})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--average', type=Path, required=True)
    parser.add_argument('--per-row', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=8192)
    parser.add_argument('--snr-db', type=float, nargs='+', default=[40., 50.])
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists() or args.trials < 1000:
        raise ValueError('Use a new output and at least 1000 trials')
    a, b = [json.loads(p.read_text()) for p in (args.average, args.per_row)]
    if not a['complete'] or not b['complete']:
        raise ValueError('Both forward-operator audits must be complete')
    for key in ('source_sha256', 'requested_interval_s', 'range_offsets_samples', 'averaging_rows',
                'observation_centre', 'points_body_m'):
        if a[key] != b[key]:
            raise ValueError('Unmatched forward comparisons: '+key)
    result = {'scope': __doc__, 'complete': False, 'trials_per_channel_and_snr': args.trials,
        'snr_definition': 'Expected total raw-data SNR for independent equal-power complex reflector amplitudes.',
        'risk_definition': 'Mean squared amplitude error in raw-data Fisher coordinates, divided by physical complex amplitude count. Includes estimator bias.',
        'source_sha256': {str(p): fingerprint(p) for p in (args.average, args.per_row, Path(__file__))},
        'cases': []}
    for ca, cb in zip(a['channels'], b['channels']):
        if ca['channel'] != cb['channel'] or ca['supports'][0]['columns'] != cb['supports'][0]['columns']:
            raise ValueError('Different channel or reflector support')
        for record in (ca, cb):
            if fingerprint(Path(record['arrays'])) != record['arrays_sha256']:
                raise ValueError('Saved design changed')
        with np.load(ca['arrays']) as x, np.load(cb['arrays']) as y:
            for key in ('raw_root', 'exact_compressed', 'times_s'):
                np.testing.assert_array_equal(x[key], y[key])
            columns = ca['supports'][0]['columns']
            raw, exact = x['raw_root'][:, columns], x['exact_compressed'][:, columns]
            templates = [x['mean_template'][:, columns], y['mean_template'][:, columns], exact]
        for snr in args.snr_db:
            values = trial_risks(raw, exact, templates, snr, args.trials, 41837+ca['channel'])
            item = {'channel': ca['channel'], 'snr_db': snr,
                'estimators': dict(zip(('average_spectrum', 'per_row_spectrum', 'exact_finite'), values))}
            result['cases'].append(item)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(item), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
