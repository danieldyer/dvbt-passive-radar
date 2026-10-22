"""Bound coherent acquisition loss for the separated point and diagnose refinement.

The matched-filter bound is conditional on the existing thermal metric,
stationary projection and independent constant complex band coefficients.
Truth-started and fixed-motion refits are labelled diagnostic controls.
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


def score_templates(ex, templates, data):
    energy = 0.; bound = 0.; records = []
    for c, (f, train, q) in enumerate(zip(ex.fields, ex.training, ex.q)):
        use = np.repeat(train, len(f.offsets))
        h = templates[c][use]; y = data[c][use]
        h = h-q@(q.conj().T@h); y = y-q@(q.conj().T@y)
        power = abs(np.vdot(h, y))**2/np.vdot(h, h).real
        available = np.vdot(y, y).real
        records.append(dict(channel=c, available_energy=available, captured_energy=power,
                            mismatch_loss_db=-10*np.log10(power/available)))
        energy += power; bound += available
    return dict(available_energy=bound, captured_energy=energy,
                mismatch_loss_db=-10*np.log10(energy/bound), channels=records)


def geometric_score(ex, p, data):
    return score_templates(ex, [ex.templates(p, c)[:, 0] for c in range(5)], data)


def approximate_score(ex, proposal, data):
    position = ex.state[0]
    tx = (ex.transmitters[2]-ex.origin)@ex.rotation.T
    rx = (ex.receiver-ex.origin)@ex.rotation.T
    gradient = sum((position-a)/np.linalg.norm(position-a) for a in (tx, rx))[:2]
    direction = gradient/(gradient@gradient)
    base = np.zeros(6)
    base[:2] = direction*proposal['delay_change_m']/SCALE[:2]
    templates = []
    for c, f in enumerate(ex.fields):
        dt = np.repeat(f.time-ex.midpoint, len(f.offsets))
        templates.append(ex.templates(base, c)[:, 0]*np.exp(-1j*f.k*(
            dt*proposal['path_rate_mps']+.5*dt**2*proposal['path_acceleration_mps2'])))
    return score_templates(ex, templates, data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('docs/data/injection_20261018'))
    args = p.parse_args(); root = args.root; output = root/'separated-500m'
    prior = json.loads((output/'analytic-profile-fit/analysis.json').read_text())
    original = json.loads((root/'exact-profile-fit/analysis.json').read_text())
    fields, rotation, rx, tx, state, mid = load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    ex = Experiment(fields, rotation, rx, tx, state, mid, training=[f.time < prior['training_end_s'] for f in fields])
    with np.load(output/'analytic-profile-fit/observations.npz') as archive:
        clean = [archive[f'injected_{c}'] for c in range(5)]
        mixed = [archive[f'data_{c}'] for c in range(5)]
    truth = np.asarray(prior['truth_parameters'])
    result = dict(scope=__doc__, interval_s=[142., 161.5], truth=truth, stages={}, controls={})

    def save():
        (output/'acquisition-validation.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n')

    result['stages']['oracle'] = geometric_score(ex, truth, clean)
    for i, proposal in enumerate(prior['acquisition']):
        result['stages'][f'coarse_bank_{i}'] = approximate_score(ex, proposal, clean)
        result['stages'][f'geometric_seed_{i}'] = geometric_score(ex, np.asarray(proposal['parameters']), clean)
    save()
    print('Coarse loss', serial(result['stages']), flush=True)
    fit, _ = ex.fit(clean, np.asarray(prior['acquisition'][0]['parameters']), 120, position_halfwidth_m=1000.)
    result['controls']['clean_refinement_from_acquisition'] = fit
    result['stages']['clean_refined'] = geometric_score(ex, np.asarray(fit['parameters']), clean)
    save()
    fit, _ = ex.fit(mixed, truth, 120, position_halfwidth_m=1000.)
    result['controls']['mixed_truth_start'] = fit
    result['stages']['mixed_truth_start'] = geometric_score(ex, np.asarray(fit['parameters']), clean)
    result['stages']['mixed_acquisition_start'] = geometric_score(ex, np.asarray(prior['selected']['parameters']), clean)
    print('Truth-start position error', np.linalg.norm((np.asarray(fit['parameters'])-truth)[:2]*SCALE[:2]), flush=True)
    save()
    # Known-motion controls isolate geographic imaging from joint motion fitting.
    plan = json.loads((root/'plan.json').read_text())
    gains = np.array([complex(*a) for a in plan['gains_real_imag']])
    for label, point, data in [('separated', truth, mixed), ('original', np.asarray(original['truth_parameters']), None)]:
        if data is None:
            data = [f.y.ravel()+ex.independent_signal(point, c, [gains[c]]) for c, f in enumerate(fields)]
        fit, _ = ex.fit(data, point, 100, free=[0, 1], position_halfwidth_m=1000.)
        result['controls'][label+'_known_motion'] = dict(fit=fit,
            horizontal_error_m=np.linalg.norm((np.asarray(fit['parameters'])-point)[:2]*SCALE[:2]))
        print(label, 'known motion', result['controls'][label+'_known_motion']['horizontal_error_m'], flush=True)
        save()
    # Show that a nearly unchanged phase history need not imply the same position.
    selected = np.asarray(prior['selected']['parameters'])
    fractions = np.linspace(0, 1.4, 15)
    loss_free, loss_fixed = [], []
    for fraction in fractions:
        free = truth+fraction*(selected-truth)
        fixed = truth.copy(); fixed[:2] = free[:2]
        loss_free.append(geometric_score(ex, free, clean)['mismatch_loss_db'])
        loss_fixed.append(geometric_score(ex, fixed, clean)['mismatch_loss_db'])
    result['position_slice'] = dict(fraction=fractions, all_coordinates_change_loss_db=loss_free,
                                   position_only_changes_loss_db=loss_fixed)
    result['complete'] = True
    save()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    labels = ['Coarse chirp bank', 'Exact seed', 'Clean refinement', 'Mixed refinement']
    keys = ['coarse_bank_0', 'geometric_seed_0', 'clean_refined', 'mixed_acquisition_start']
    axes[0].bar(np.arange(4), [max(0, result['stages'][k]['mismatch_loss_db']) for k in keys])
    axes[0].set(xticks=np.arange(4), xticklabels=labels, ylabel='Known-point energy loss (dB)',
                title='Conditional matched-filter bound: 0 dB')
    axes[0].tick_params(axis='x', labelrotation=15)
    error_m = np.linalg.norm((selected-truth)[:2]*SCALE[:2])
    axes[1].plot(fractions*error_m, loss_fixed, label='Velocity and acceleration held true')
    axes[1].plot(fractions*error_m, loss_free, label='Position, velocity and acceleration change together')
    axes[1].set(xlabel='Geographic position displacement from truth (m)', ylabel='Known-point energy loss (dB)',
                title='The full fit has a much shallower position ambiguity')
    axes[1].legend(fontsize=8)
    fig.savefig(output/'acquisition-validation.png', dpi=160)
    print('Acquisition verification complete', flush=True)


if __name__ == '__main__':
    main()
