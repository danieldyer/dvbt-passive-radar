"""Test feasible GLS without supplying injected truth to its covariance or fit.

The initial physical fit supplies a residual for covariance estimation. The
frozen metric is then used for a new physical fit. No per-band phase trajectory
or point-dependent whitening is introduced. Truth is used only for scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from correlated_likelihood import FourierMetric, MetricExperiment, crossvalidated_metric
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment, SCALE
from report_measured_scatterer_phases import serial
from validate_recorded_injection import metrics as comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/data/injection_20261018/separated-500m'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--width-hz', type=float, default=2.)
    parser.add_argument('--scalar', action='store_true')
    parser.add_argument('--crossvalidate', action='store_true')
    args = parser.parse_args(); root = args.root
    out = args.output or root/'correlated-likelihood'
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((root/'analytic-profile-fit/analysis.json').read_text())
    fields, rotation, rx, tx, state, mid = load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    training = [f.time < prior['training_end_s'] for f in fields]
    base = Experiment(fields, rotation, rx, tx, state, mid, training=training)
    with np.load(root/'analytic-profile-fit/observations.npz') as saved:
        data = [saved[f'data_{c}'] for c in range(5)]
        clean = [saved[f'injected_{c}'] for c in range(5)]
    # Both seed and covariance residual depend on observed summed data only.
    initial = np.asarray(prior['selected']['parameters'])
    _, _, _, prediction = base.profile(initial, data, derivative=False)
    metrics = []
    for c, (f, mask) in enumerate(zip(fields, training)):
        residual = (data[c]-prediction[c]).reshape(f.y.shape)[mask]
        if args.crossvalidate:
            metric=crossvalidated_metric(residual,np.median(np.diff(f.time)),scalar=args.scalar)
        else:
            metric=FourierMetric.estimate(residual,np.median(np.diff(f.time)),args.width_hz,scalar=args.scalar)
        metrics.append(metric)
    experiment = MetricExperiment(base, metrics)
    result = dict(scope=__doc__, initial_parameters=initial, metrics=[m.description for m in metrics], complete=False)
    def save():
        (out/'analysis.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n')
    save()
    fit, _ = experiment.fit(data, initial, 100, position_halfwidth_m=1000.)
    result['fit'] = fit
    truth = np.asarray(prior['truth_parameters'])
    error = (np.asarray(fit['parameters'])-truth)*SCALE
    result['physical_error'] = error
    result['horizontal_error_m'] = np.linalg.norm(error[:2])
    result['per_channel_template_loss'] = []
    for c in range(5):
        alpha = np.asarray(fit['coefficients_real_imag'][c]); alpha = alpha[:, 0]+1j*alpha[:, 1]
        result['per_channel_template_loss'].append(comparison(experiment.templates(np.asarray(fit['parameters']), c)@alpha, clean[c]))
    result['complete'] = True; save()
    print(json.dumps(serial({k:result[k] for k in ['horizontal_error_m','physical_error','per_channel_template_loss']})), flush=True)


if __name__ == '__main__':
    main()
