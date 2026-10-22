"""Measure iterative phase-gradient performance on retained ship BP histories."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from gpga import fit, phase_metrics, refine


def binned_metrics(time, values, information, selected, reference, seconds):
    index = np.floor((time-time[0])/seconds).astype(int)
    count = index[-1]+1
    weight = np.bincount(index, information*selected, minlength=count)
    totals = np.empty((count, values.shape[1]), complex)
    for column in range(values.shape[1]):
        value = values[:, column]*information*selected
        totals[:, column] = (np.bincount(index, value.real, minlength=count)+
                             1j*np.bincount(index, value.imag, minlength=count))
    mean = np.divide(totals, weight[:, None], out=np.zeros_like(totals), where=weight[:, None] > 0)
    return phase_metrics(np.arange(count)*seconds, mean, weight, weight > 0, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories', type=Path, required=True)
    parser.add_argument('--interval', type=float, nargs=2, required=True)
    parser.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--bandwidths-hz', type=float, nargs='+', default=[2., 1., .5, .25, .125])
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--refine-knot-seconds', type=float)
    parser.add_argument('--fit-points', type=int, nargs='+', help='Fit only these point indices; report every point')
    parser.add_argument('--holdout-seconds', type=float, default=0.,
                        help='Zero alternates individual rows; positive values alternate whole time blocks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.holdout_seconds < 0:
        raise ValueError('Holdout block duration must be nonnegative')
    for channel in args.channels:
        clock = time.perf_counter()
        with np.load(args.histories/f'channel-{channel}-histories.npz') as archive:
            selected = (archive['time'] >= args.interval[0]) & (archive['time'] < args.interval[1])
            t, z, w = (archive[key][selected] for key in ('time', 'value', 'information'))
            pixels = archive['pixels'].copy()
        train = (np.arange(len(t))%2 == 0 if args.holdout_seconds == 0 else
                 np.floor((t-args.interval[0])/args.holdout_seconds).astype(int)%2 == 0)
        fitted_points = np.arange(z.shape[1]) if args.fit_points is None else np.asarray(args.fit_points)
        result = fit(t, z[:, fitted_points], w, train, bandwidths_hz=args.bandwidths_hz,
                     iterations_per_bandwidth=args.iterations)
        phase = result.pop('phase_radians')
        result['gpga_converged'] = result['converged']
        if args.refine_knot_seconds is not None:
            phase, refinement = refine(t, z[:, fitted_points], w, train, phase,
                                       knot_seconds=args.refine_knot_seconds)
            result['smooth_refinement'] = refinement
            result['converged'] = refinement['converged']
            result['convergence_definition'] = 'Smooth phase likelihood: solver success and maximum analytic gradient at most 1e-7. GPGA fixed-point convergence is reported separately; neither establishes physical motion.'
        result['fitted_points'] = fitted_points.tolist()
        corrected = z*np.exp(-1j*phase[:, None])
        before_ref = np.sum(w[train, None]*z[train], axis=0)/w[train].sum()
        after_ref = np.sum(w[train, None]*corrected[train], axis=0)/w[train].sum()
        result['before'] = phase_metrics(t, z, w, ~train, before_ref)
        result['after'] = phase_metrics(t, corrected, w, ~train, after_ref)
        result['heldout_coherent_power_gain_db'] = float(10*np.log10(
            sum(result['after']['coherent_energy_by_point'])/sum(result['before']['coherent_energy_by_point'])))
        result['heldout_prediction_error_change_percent'] = float(100*(
            result['after']['prediction_error']/result['before']['prediction_error']-1))
        unseen = np.setdiff1d(np.arange(z.shape[1]), fitted_points)
        if len(unseen):
            result['unfitted_points'] = unseen.tolist()
            result['unfitted_point_heldout_power_gain_db'] = float(10*np.log10(
                np.sum(np.asarray(result['after']['coherent_energy_by_point'])[unseen])/
                np.sum(np.asarray(result['before']['coherent_energy_by_point'])[unseen])))
        result['binned_residuals'] = []
        for duration in (.25, 1.):
            result['binned_residuals'].append({'seconds': duration,
                'before': binned_metrics(t, z, w, ~train, before_ref, duration),
                'after': binned_metrics(t, corrected, w, ~train, after_ref, duration)})
        result.update(channel=channel, interval_s=args.interval, input=str(args.histories),
                      holdout_seconds=args.holdout_seconds, seconds=time.perf_counter()-clock)
        np.savez(args.output/f'channel-{channel}.npz', time=t, phase_radians=phase,
                 before=z, after=corrected, information=w, train=train, pixels=pixels)
        (args.output/f'channel-{channel}.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({key: result[key] for key in ('channel', 'converged', 'seconds',
              'heldout_coherent_power_gain_db', 'heldout_prediction_error_change_percent')}), flush=True)


if __name__ == '__main__':
    main()
