"""Separate geometric conditioning, finite-template error and recorded interference.

Diagnostic controls use injection truth explicitly. They are not acquisition
benchmarks and do not alter the previous blind-to-truth recovery results.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from recorded_injection_fields import load_fields
from recovery_experiment import Experiment, SCALE
from report_measured_scatterer_phases import serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/data/injection_20261018'))
    parser.add_argument('--replay', type=Path, default=Path('build/injection-20261018'))
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root/'plan.json').read_text())
    previous = json.loads((root/'exact-profile-fit/analysis.json').read_text())
    output = root/'localisation'
    output.mkdir(exist_ok=True)
    fields, rotation, rx, tx, state, mid = load_fields(args.replay/'deterministic-baseline/profiles.bin')
    profile_fields = load_fields(args.replay/'exact-profile/profiles.bin')[0]
    masks = [f.time < previous['training_end_s'] for f in fields]
    ex = Experiment(fields, rotation, rx, tx, state, mid, training=masks)
    truth = np.asarray(previous['truth_parameters'])
    selected = np.asarray(previous['selected']['parameters'])
    gains = np.array([complex(*a) for a in plan['gains_real_imag']])
    analytic = [ex.independent_signal(truth, c, [g]) for c, g in enumerate(gains)]
    exact = [f.apply(p.data['value']-f.data['value']).ravel() for f, p in zip(fields, profile_fields)]
    mixed = [f.apply(p.data['value']).ravel() for f, p in zip(fields, profile_fields)]
    result = dict(scope=__doc__, fixed_height=True, independent_constant_band_coefficients=True,
                  unique_transmitter_centres=len(np.unique(tx, axis=0)),
                  transmitter_baseline_m=float(np.linalg.norm(tx[4]-tx[0])), controls={})
    for name, data, seed in [('analytic_point_only', analytic, selected),
                             ('finite_point_only', exact, selected),
                             ('recorded_background_truth_start', mixed, truth)]:
        fit, _ = ex.fit(data, seed, 120)
        error = (np.asarray(fit['parameters'])-truth)*SCALE
        result['controls'][name] = dict(fit=fit, physical_error=error,
                                        horizontal_error_m=float(np.linalg.norm(error[:2])))
        print(name, serial(result['controls'][name]), flush=True)
    # Local Gaussian-noise information, after profiling band coefficients,
    # stationary coefficients, velocity and acceleration. This is conditional
    # thermal information, not a claimed covariance for the recorded clutter.
    _, jacobian, _, _ = ex.profile(truth, analytic)
    nuisance = jacobian[:, 2:]
    q, _ = np.linalg.qr(nuisance, mode='reduced')
    position_jacobian = (jacobian[:, :2]-q@(q.T@jacobian[:, :2]))/SCALE[:2]
    _, singular, vh = np.linalg.svd(position_jacobian, full_matrices=False)
    sigma = 1/(np.sqrt(2)*singular)
    covariance = (vh.T*sigma**2)@vh
    result['conditional_thermal_position_information'] = dict(
        singular_values_per_m=singular, principal_sigma_m=sigma,
        principal_directions_enu=vh, covariance_en_m2=covariance,
        coordinate_sigma_en_m=np.sqrt(np.diag(covariance)))
    position = np.asarray(plan['state_enu'])[0]
    world = rx+position@rotation
    gradients = np.array([((world-t)/np.linalg.norm(world-t)
                          +(world-rx)/np.linalg.norm(world-rx))@rotation.T for t in tx])[:, :2]
    sv = np.linalg.svd(gradients, compute_uv=False)
    normal = gradients[2]/np.linalg.norm(gradients[2])
    tangent = np.array([-normal[1], normal[0]])
    error = (selected-truth)*SCALE
    result['geometry'] = dict(range_gradient_en=gradients,
                             range_gradient_condition_number=sv[0]/sv[-1],
                             normal_error_m=error[:2]@normal,
                             contour_tangent_error_m=error[:2]@tangent)
    result['training_objective'] = {}
    for name, parameters in [('truth', truth), ('selected', selected)]:
        r, _, _, _ = ex.profile(parameters, mixed, derivative=False)
        result['training_objective'][name] = dict(rss=r@r, complex_cells=len(r)//2,
                                                rss_per_complex_cell=r@r/(len(r)/2))
    fractions = np.linspace(-.5, 1.5, 41)
    costs = {}
    for name, data in [('Analytic point only', analytic), ('Finite point only', exact),
                       ('Point + recorded background', mixed)]:
        values = []
        for fraction in fractions:
            r, _, _, _ = ex.profile(truth+fraction*(selected-truth), data, derivative=False)
            values.append(r@r)
        costs[name] = np.asarray(values)
    result['objective_slice'] = dict(fractions=fractions, costs=costs,
        note='All six coordinates interpolate along this line. Band and stationary coefficients are profiled at each point.')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    axes[0].scatter([0], [0], marker='*', s=160, color='black', label='Injected truth')
    axes[0].scatter([error[0]], [error[1]], s=70, label='Point + background')
    for name, colour in [('analytic_point_only', '#009e73'), ('finite_point_only', '#d55e00')]:
        e = np.asarray(result['controls'][name]['physical_error'])
        axes[0].scatter([e[0]], [e[1]], color=colour, marker='x', s=80, label=name.replace('_', ' '))
    axis = np.linspace(-90, 90, 200)
    axes[0].plot(axis*tangent[0], axis*tangent[1], '--', color='#777777', label='Local equal-range tangent')
    axes[0].set(xlabel='East position error (m)', ylabel='North position error (m)', aspect='equal',
                title='Nearly all recorded-background error is along the contour')
    axes[0].legend(fontsize=8)
    for name, values in costs.items():
        axes[1].plot(fractions, values-values.min(), label=name)
    axes[1].axvline(0, color='black', linestyle=':')
    axes[1].axvline(1, color='black', linestyle=':')
    axes[1].set(xlabel='Truth → recorded-background solution (fraction)',
                ylabel='Training RSS above each curve minimum', title='The recorded interference moves the likelihood minimum')
    axes[1].legend(fontsize=8)
    fig.savefig(output/'diagnosis.png', dpi=160)
    (output/'analysis.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n')
    print('Localisation diagnosis written', output, flush=True)


if __name__ == '__main__':
    main()
