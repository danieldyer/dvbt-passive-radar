"""Test a constant-material-reflectivity facet against the actual flash rows.

Acquisition and fitting use the training fold only. Native-symbol predictions
freeze every geometric parameter, stationary coefficient and constant band
gain. Transverse centre motion remains an explicit AIS condition. A successful
optimiser is not evidence that the scalar facet explains the recorded ship.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from field import FS
from finite_facet import geometry, C
from fixed_range_field import stationary_profile_design
from midpoint_model import representative, discrepancy
from native_facet_scene import FacetBand, fit
from native_folds import training_mask
from recorded_injection_fields import load_fields
from recovery_experiment import NAMES
from report_measured_scatterer_phases import serial
from waveform_kernels import KernelField


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', nargs=2, type=float, default=[154., 166.])
    parser.add_argument('--length-m', type=float, default=300.)
    parser.add_argument('--quadrature', type=int, default=32)
    parser.add_argument('--max-evaluations', type=int, default=90)
    parser.add_argument('--native-evaluations', type=int, default=20)
    parser.add_argument('--root', type=Path, default=Path('docs/data/isar_20261018/native-spatial-validation'))
    parser.add_argument('--profiles', type=Path, default=Path('build/isar-20261018/current-native-full/profiles.bin'))
    args = parser.parse_args(); out, root = args.output, args.root
    out.mkdir(parents=True, exist_ok=False)
    first, last = args.interval; mid = float((first+last)/2)
    ref = json.loads((root/'ais-near-shared-point-4s/analysis.json').read_text())
    state = np.asarray(ref['prior_mean_state_enu']); state[0] += (mid-ref['midpoint_s'])*state[1]; state[2] = 0.
    fields, rotation, rx, tx, _, _ = load_fields(args.profiles, offsets=np.arange(-12, 13), fixed_epoch_s=mid, interval_s=[first, last])
    local_tx = (tx-rx)@rotation.T; g, _ = geometry(state[0], local_tx[2], np.zeros(3))
    tangent = np.array([g[1], -g[0], 0.]); tangent /= np.linalg.norm(tangent)
    direction = np.r_[g[:2]/np.dot(g[:2], g[:2]), 0.]
    exact, fast = [], []
    for c, f in enumerate(fields):
        mean = representative(f); blocks = f.pulse.row_blocks.reshape(len(f.time), 16)
        if np.any(blocks != blocks[:, 0, None]): raise ValueError('Kernel identities changed within a retained bin')
        mean = KernelField(mean, f.pulse.kernels, blocks[:, 0])
        keep = training_mask(f.time, first)
        for target, field in [(exact, f), (fast, mean)]:
            target.append(FacetBand(field, keep, stationary_profile_design(field), rx, rotation,
                state, mid, tangent, direction, quadrature=args.quadrature))
    measured = json.loads((root/'ais-near-brightness-4s/analysis.json').read_text())
    origin = float(np.median([c['fading']['parameters'][0]*C/FS for c in measured['channels']]))
    result = dict(scope=__doc__, complete=False, interval_s=[first, last], midpoint_s=mid,
        reference_state_enu=state, tangent_enu=tangent, radial_direction_enu=direction,
        parameter_names=['radial_path_offset_m', 'radial_path_rate_mps', 'heading_offset_deg',
            'counterclockwise_spin_degps', 'facet_along_centre_m', 'facet_length_m'],
        length_fixed_m=args.length_m, transverse_motion_conditioned_on_ais=True,
        geographic_localisation_certified=False, actual_ship_focus_proved=False,
        phase_nuisance='One training-only constant complex coefficient per band; no phase history', trials=[])
    def save():
        (out/'analysis.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n')
    seeds = [[origin, 0., angle, spin, 0., args.length_m]
        for angle, spin in [(0., .23), (1., .1), (-1., .35)]]
    for seed in seeds:
        trial = fit(fast, np.asarray(seed), max_evaluations=args.max_evaluations)
        result['trials'].append(trial); save(); print('Representative facet', serial(trial), flush=True)
    best = min(result['trials'], key=lambda q:q['objective'])
    if args.native_evaluations:
        best = fit(exact, np.asarray(best['parameters']), max_evaluations=args.native_evaluations)
    result['fit'] = best; save(); print('Native facet', serial(best), flush=True)
    result['prediction'], result['operator_comparison'], result['quadrature_comparison'] = [], [], []
    fig, axes = plt.subplots(5, 2, figsize=(12, 13), sharex=True, layout='constrained')
    for c, (native, mean) in enumerate(zip(exact, fast)):
        prediction = native.prediction(best['parameters'])
        h = native.template(best['parameters'], derivatives=False)
        approx = mean.template(best['parameters'], derivatives=False)
        result['operator_comparison'].append(discrepancy(h, approx))
        finer = FacetBand(native.field, native.training, native.design, rx, rotation, state,
            mid, tangent, direction, quadrature=2*args.quadrature)
        result['quadrature_comparison'].append(discrepancy(h, finer.template(best['parameters'], derivatives=False)))
        model = prediction.pop('prediction').reshape(native.field.y.shape)
        clean = prediction.pop('stationary_clean').reshape(native.field.y.shape)
        result['prediction'].append(prediction); save()
        t = native.field.time; cross = np.sum(model.conj()*clean, axis=1)
        phase = np.degrees(np.angle(cross)); strength = np.sum(abs(model)**2, axis=1)
        axes[c, 0].plot(t[native.training], phase[native.training], '.', ms=2, alpha=.5)
        axes[c, 0].plot(t[~native.training], phase[~native.training], '.', ms=3, color='C1')
        axes[c, 0].set(ylabel=f'{NAMES[c]}\nresidual phase (degrees)', ylim=(-180, 180))
        axes[c, 1].plot(t, strength); axes[c, 1].set(ylabel='Predicted profile energy',
            title=f"Excluded complex prediction: {prediction['explained_fraction']:.1%}")
        np.savez(out/f'channel-{c}-prediction.npz', time_s=t, training=native.training,
            predicted=model, stationary_clean=clean, phase_residual_deg=phase)
        print(NAMES[c], prediction, 'operator', result['operator_comparison'][-1],
            'quadrature', result['quadrature_comparison'][-1], flush=True)
    axes[-1, 0].set_xlabel('Capture time (s)'); axes[-1, 1].set_xlabel('Capture time (s)')
    fig.suptitle('Finite physical facet: frozen native-symbol predictions\nOrange samples excluded from fitting; all phases and amplitudes frozen from training')
    fig.savefig(out/'prediction.png', dpi=150); plt.close(fig)
    result['complete'] = True; save()


if __name__ == '__main__':
    main()
