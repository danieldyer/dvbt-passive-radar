"""Predict the deterministic displacement caused by this recorded background.

All coordinates, weights and nuisance subspaces are those of the 500 m test.
Truth is used for a local perturbation/causality diagnosis, not acquisition.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recorded_injection_fields import load_fields
from recovery_experiment import Experiment, SCALE
from report_measured_scatterer_phases import serial


def stack(value):
    return np.r_[value.real, value.imag]


def position_information(jacobian, nuisance=()):
    position = jacobian[:, :2]/SCALE[:2]
    if len(nuisance):
        q, _ = np.linalg.qr(jacobian[:, nuisance], mode='reduced')
        position -= q@(q.T@position)
    u, singular, vh = np.linalg.svd(position, full_matrices=False)
    if singular[-1] < singular[0]*1e-10:
        raise ValueError('Do not silently remove an unobserved position direction')
    sigma = 1/np.sqrt(2)/singular
    return dict(singular_values_per_m=singular, principal_sigma_m=sigma,
                principal_directions_en=vh), u, singular, vh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/data/injection_20261018/separated-500m'))
    args = parser.parse_args(); root = args.root
    output = root/'bias-explanation'; output.mkdir(exist_ok=True)
    prior = json.loads((root/'analytic-profile-fit/analysis.json').read_text())
    fields, rotation, rx, tx, state, mid = load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    masks = [f.time < prior['training_end_s'] for f in fields]
    ex = Experiment(fields, rotation, rx, tx, state, mid, training=masks)
    truth = np.asarray(prior['truth_parameters']); selected = np.asarray(prior['selected']['parameters'])
    with np.load(root/'analytic-profile-fit/observations.npz') as saved:
        clean = [saved[f'injected_{c}'] for c in range(5)]
        mixed = [saved[f'data_{c}'] for c in range(5)]
    background = [y-s for y,s in zip(mixed, clean)]
    tangent = np.asarray(json.loads((root/'plan.json').read_text())['separation_control']['contour_tangent_enu'])[:2]
    rclean, j0, _, _ = ex.profile(truth, clean)
    rb, _, _, _ = ex.profile(truth, background, derivative=False)
    rmix, _, amplitudes, _ = ex.profile(truth, mixed, derivative=False)
    if np.linalg.norm(rmix-rb-rclean) > 1e-7:
        raise ValueError('Profiled residual linearity failed')
    u, singular, vh = np.linalg.svd(j0, full_matrices=False)
    coefficients = u.T@rb
    linear = -vh.T@(coefficients/singular)
    # Scalar adjoint: predicted cross-range displacement as a linear function
    # of the original complex recorded range samples, before range whitening.
    sensitivity = -(tangent@(SCALE[:2,None]*vh.T[:2])/singular)@u.T
    actual = (selected-truth)*SCALE
    result = dict(scope=__doc__, parameter_epoch_s=mid, truth=truth,
        observed_physical_displacement=actual,
        linear_predicted_physical_displacement=linear*SCALE,
        observed_cross_range_m=tangent@actual[:2],
        linear_predicted_cross_range_m=tangent@(linear*SCALE)[:2],
        background_residual_energy=rb@rb, projected_tangent_energy=coefficients@coefficients,
        tangent_fraction_of_background_energy=(coefficients@coefficients)/(rb@rb),
        scaled_jacobian_singular_values=singular, scaled_mode_background_scores=coefficients,
        position_information={}, attribution=[], controls={}, homotopy=[])

    def save():
        (output/'analysis.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n')

    for label, nuisance in [('motion_known', []), ('velocity_free', [2,3]),
                             ('acceleration_free', [4,5]), ('all_motion_free', [2,3,4,5])]:
        info, pu, ps, pv = position_information(j0, nuisance)
        result['position_information'][label] = info
        if label == 'all_motion_free':
            z = pu.T@rb
            result['position_mode_background_scores'] = z
            result['position_mode_predictions_m'] = -z/ps
            result['weak_position_score_thermal_sigmas'] = z[-1]*np.sqrt(2)
    # Expected mean Jacobian at the fitted amplitudes: unlike the complete
    # residual Jacobian, its columns lie in the nuisance-orthogonal subspace.
    efficient, mean_columns, independent_amplitude_columns, global_phase = [], [], [], []
    channel_parts = []; cursor = 0
    score_stats = []
    input_cells = []
    model_crosscheck = []
    for c, (f, keep, q) in enumerate(zip(fields, masks, ex.q)):
        use = np.repeat(keep, len(f.offsets)); n = int(use.sum())
        h, derivative = ex.templates(truth, c, derivatives=True)
        h = h[use,0]; dh = derivative[use,0]
        h -= q@(q.conj().T@h); dh -= q@(q.conj().T@dh)
        dh -= h[:,None]*(h.conj()@dh)[None,:]/np.vdot(h,h).real
        efficient.append(np.r_[(-dh*amplitudes[c][0]).real, (-dh*amplitudes[c][0]).imag])
        alpha = np.linalg.lstsq(h[:,None], clean[c][use]-q@(q.conj().T@clean[c][use]), rcond=None)[0][0]
        model_crosscheck.append(np.r_[(-dh*alpha).real, (-dh*alpha).imag])
        # Keep known relative phases as a separate Fisher-information control:
        # independent real amplitudes plus one common unknown phase remain.
        raw_h, raw_dh = ex.templates(truth, c, derivatives=True)
        raw_d = raw_dh[use,0]*alpha
        raw_d -= q@(q.conj().T@raw_d)
        mean_columns.append(np.r_[raw_d.real, raw_d.imag])
        mean = raw_h[use,0]*alpha; mean -= q@(q.conj().T@mean)
        independent_amplitude_columns.append(stack(mean)); global_phase.append(stack(1j*mean))
        section = slice(cursor, cursor+2*n)
        g = sensitivity[section][:n]+1j*sensitivity[section][n:]
        g = g.reshape(int(keep.sum()), len(f.offsets))
        raw_g = (g@f.whiten.conj())*f.root_information[keep,None]
        cells = np.real(raw_g.conj()*f.data['value'][keep])
        white_total = sensitivity[section]@rb[section]
        if not np.isclose(cells.sum(), white_total, atol=1e-6, rtol=1e-8):
            raise ValueError('Original-range adjoint does not reproduce predicted displacement')
        delay = f.offsets*299792458./8e6
        record = dict(channel=c, predicted_cross_range_m=white_total,
            time_s=f.time[keep], contribution_per_time_bin_m=cells.sum(axis=1),
            delay_relative_ais_m=delay, contribution_per_range_cell_m=cells.sum(axis=0),
            ais_core_signed_contribution_m=cells[:,abs(delay)<=100].sum(),
            injection_range_signed_contribution_m=cells[:,(delay>=150)&(delay<=300)].sum())
        result['attribution'].append(record)
        input_cells.append(cells)
        channel_parts.append(section)
        cursor += 2*n
    result['clean_jacobian_crosscheck_relative_error'] = np.linalg.norm(np.vstack(model_crosscheck)-j0)/np.linalg.norm(j0)
    known_phase = np.vstack(mean_columns)
    nuisance = np.zeros((len(rb), 6))
    for c, section in enumerate(channel_parts):
        nuisance[section,c] = independent_amplitude_columns[c]
        nuisance[section,5] = global_phase[c]
    phase_q, _ = np.linalg.qr(nuisance, mode='reduced')
    known_phase -= phase_q@(phase_q.T@known_phase)
    result['position_information']['relative_band_phases_known'] = position_information(known_phase, [2,3,4,5])[0]
    effective = np.vstack(efficient)
    eu, es, ev = np.linalg.svd(effective, full_matrices=False)
    drive = eu@(eu.T@rb)
    result['exact_score_component_energy'] = drive@drive
    result['exact_score_component_fraction'] = (drive@drive)/(rb@rb)
    result['exact_score_gradient_remaining_relative'] = np.linalg.norm(effective.T@(rb-drive))/np.linalg.norm(effective.T@rb)
    # Reconstruct the canonical score-carrying component in the same complex
    # observation coordinates. All nuisance coefficients at truth are retained.
    score_only, score_removed = [], []
    for c, (f, keep, section) in enumerate(zip(fields, masks, channel_parts)):
        use = np.repeat(keep,len(f.offsets)); n = int(use.sum())
        vector = drive[section][:n]+1j*drive[section][n:]
        residual = rb[section][:n]+1j*rb[section][n:]
        nuisance_background = background[c][use]-residual
        only = clean[c].copy(); removed = mixed[c].copy()
        only[use] += nuisance_background+vector
        removed[use] -= vector
        score_only.append(only); score_removed.append(removed)
    result['normal_equation_prediction_mixed_amplitudes'] = (-ev.T@((eu.T@rb)/es))*SCALE
    save()
    np.savez(output/'range-influence.npz', **{f'channel_{c}':v for c,v in enumerate(input_cells)})
    print('Linear prediction', serial({k:result[k] for k in ['observed_physical_displacement',
        'linear_predicted_physical_displacement','linear_predicted_cross_range_m',
        'weak_position_score_thermal_sigmas','exact_score_component_fraction']}), flush=True)
    # Counterfactual data isolate the score contribution, without deleting any
    # measurement or altering weights/templates. This is not a production fix.
    for name, data, seed in [('score_removed', score_removed, selected), ('score_only_with_nuisance', score_only, truth)]:
        fit, _ = ex.fit(data, seed, 120, position_halfwidth_m=1000.)
        error = (np.asarray(fit['parameters'])-truth)*SCALE
        result['controls'][name] = dict(fit=fit, physical_error=error,
            horizontal_error_m=np.linalg.norm(error[:2]), cross_range_error_m=tangent@error[:2])
        save(); print(name, serial(result['controls'][name]), flush=True)
    for epsilon in [-1., .05, .1, .25, .5]:
        data = [s+epsilon*b for s,b in zip(clean, background)]
        fit, _ = ex.fit(data, truth, 120, position_halfwidth_m=1000.)
        error = (np.asarray(fit['parameters'])-truth)*SCALE
        result['homotopy'].append(dict(background_voltage_scale=epsilon, fit=fit,
            physical_error=error, cross_range_error_m=tangent@error[:2]))
        save(); print('Background voltage', epsilon, serial(error), flush=True)
    for name, free in [('velocity_known', [0,1,4,5]), ('acceleration_known', [0,1,2,3])]:
        fit, _ = ex.fit(mixed, truth, 120, free=free, position_halfwidth_m=1000.)
        error = (np.asarray(fit['parameters'])-truth)*SCALE
        result['controls'][name] = dict(fit=fit, physical_error=error,
            horizontal_error_m=np.linalg.norm(error[:2]), cross_range_error_m=tangent@error[:2])
        save(); print(name, serial(error), flush=True)
    result['complete'] = True
    save(); print('Bias explanation controls complete', flush=True)


if __name__ == '__main__':
    main()
