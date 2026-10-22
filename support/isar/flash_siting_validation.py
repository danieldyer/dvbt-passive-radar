"""Validate flash geometry on withheld complex data, through original rows."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from cluster_phase import crop
from compare_ais_phase import NAMES, COLOURS
from flash_geometry import Flash


def phase_origin(h, y, training):
    norm = np.sum(abs(h)**2, axis=1)
    product = np.sum(h.conj()*y, axis=1)
    def cost(phase):
        projected = np.maximum(0, (product[training]*np.exp(-1j*phase)).real)
        return -float(np.sum(projected**2/np.maximum(norm[training], 1e-300)))
    grid = np.linspace(-np.pi, np.pi, 361)[:-1]
    best = min(grid, key=cost)
    result = minimize_scalar(cost, bounds=(best-2*np.pi/360, best+2*np.pi/360), method='bounded')
    q = product*np.exp(-1j*result.x)/np.maximum(norm, 1e-300)
    return q, norm


def validate(result, options, track, utc, rotation, receiver, output, *, fields):
    rate_case = next(c for c in result['cases'] if c['width_cells']==5 and not c['include_acceleration'])
    range_case = next(c for c in result['cases'] if c['width_cells']==5 and c['include_range'])
    selected = [
        ('Nominal assignment; phase-rate fit', rate_case,
         next(r for r in rate_case['records'] if r['sites']==result['nominal_sites'])),
        ('Best assignment; phase-rate fit', rate_case,
         min(rate_case['records'], key=lambda r:r['penalized_squared_residual'])),
        ('Best assignment; range + phase fit', range_case,
         min(range_case['records'], key=lambda r:r['penalized_squared_residual']))]
    results = []
    fig, axes = plt.subplots(5, 2, figsize=(13, 13), layout='constrained')
    for label, case, fit in selected:
        problem = Flash(case['measurements'], case['covariance'], track, utc+164.5, rotation, receiver,
                        options[fit['index']], include_range=case['include_range'],
                        include_acceleration=case['include_acceleration'], **result.get('fit_options', {}))
        p, v, a, transmitters, rx = problem.state(np.array(fit['parameters']))
        rows = []
        for ch in range(5):
            with np.load(fields/f'channel-{ch}.npz') as source:
                field = crop(source, 164, 165, np.arange(-2, 3))
                elapsed = field.data['row_time']-164.5
                positions = receiver+(p+elapsed[:, None]*v+.5*elapsed[:, None]**2*a)@rotation
                velocities = (v+elapsed[:, None]*a)@rotation
                # A free envelope origin is profiled in the rate-only control,
                # exactly as in its likelihood. It does not change carrier phase.
                bias = (0. if case['include_range'] else
                        case['measurements'][ch][0]-fit['predicted_observables'][ch][0])
                h = field.trajectory(positions, velocities, transmitter=receiver+transmitters[ch]@rotation,
                                     receiver=receiver+rx@rotation, range_bias_m=bias)
                training = ((field.time-164)//.125).astype(int)%2 == 0
                q, information = phase_origin(h, field.y, training)
                test = ~training
                phase = np.angle(q)
                weight = abs(q[test])**2*information[test]
                rms = np.degrees(np.sqrt(np.average(phase[test]**2, weights=weight)))
                amplitude = np.maximum(q.real, 0)
                prediction_amplitude = np.interp(field.time, field.time[training], amplitude[training])
                # q contains only the carrier phase origin removal. Reconstruct
                # the phase-rotated template consistently for the error metric.
                original = np.sum(h.conj()*field.y, axis=1)/np.maximum(information, 1e-300)
                rotation_factor = np.sum(original[training]*q[training].conj())
                rotation_factor /= abs(rotation_factor)
                prediction = prediction_amplitude[:, None]*h*rotation_factor
                error = np.sum(abs(field.y[test]-prediction[test])**2)
                fraction = 1-error/np.sum(abs(field.y[test])**2)
                rows.append({'channel': NAMES[ch], 'test_phase_rms_deg': float(rms),
                             'test_complex_prediction_fraction': float(fraction),
                             'range_origin_m': float(bias), 'time_s': field.time.tolist(),
                             'phase_residual_deg': np.degrees(phase).tolist(), 'training': training.tolist()})
                style = '--' if label.startswith('Nominal') else '-'
                if not case['include_range']:
                    axes[ch, 0].plot(field.time[test], np.degrees(phase[test]), style, lw=1,
                                     color='#777777' if label.startswith('Nominal') else COLOURS[ch], label=label)
                axes[ch, 1].bar(label.split(';')[0]+'\n'+label.split(';')[1].strip(), rms,
                                color='#777777' if label.startswith('Nominal') else COLOURS[ch], alpha=.7)
        results.append({'label': label, 'sites': fit['sites'], 'channels': rows,
                        'scope': 'Exact archived row epochs, physical carrier/envelope trajectory, both symbol moments and recorded averaging. Only constant phase and training amplitudes profiled after geometry fitting. Prediction samples never fit the geometry.'})
    for ch in range(5):
        axes[ch, 0].set_ylabel(NAMES[ch]+' phase error (degrees)')
        axes[ch, 0].axhline(0, color='black', lw=.5)
        axes[ch, 1].set_ylabel('Withheld RMS (degrees)')
    axes[0, 0].legend(fontsize=7);axes[-1, 0].set_xlabel('Recording time (s)')
    fig.suptitle('Geometry validation on the withheld complex flash samples\n'
                 'Original row-level forward operator; no fitted per-band velocity, no phase correction learned from test samples')
    fig.savefig(output/'09-withheld-complex-validation.png', dpi=170);plt.close(fig)
    return results


def validate_common(result, options, track, utc, rotation, receiver, output, *, fields):
    """Keep the common fitted phase fixed when checking unused flash samples."""
    case = next(c for c in result['cases'] if c['width_cells']==5 and not c['include_acceleration'])
    records = sorted(result['common_phase_control']['records'], key=lambda r:r['conditional_squared_cost'])[:2]
    values = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for record in records:
        problem = Flash(case['measurements'], case['covariance'], track, utc+164.5, rotation, receiver,
                        options[record['index']], include_range=False, include_acceleration=False,
                        **result.get('fit_options', {}))
        parameters = np.array(record['parameters'])
        p, v, a, tx, rx = problem.state(parameters[:-1])
        prediction = problem.prediction(parameters[:-1])
        envelope_error = prediction[:, 0]-np.array(case['measurements'])[:, 0]
        rms = []
        for ch in range(5):
            with np.load(fields/f'channel-{ch}.npz') as source:
                field = crop(source, 164, 165, np.arange(-2, 3))
            elapsed = field.data['row_time']-164.5
            positions = receiver+(p+elapsed[:, None]*v+.5*elapsed[:, None]**2*a)@rotation
            velocities = (v+elapsed[:, None]*a)@rotation
            h = field.trajectory(positions, velocities, transmitter=receiver+tx[ch]@rotation,
                receiver=receiver+rx@rotation, range_bias_m=-envelope_error[ch])*np.exp(1j*parameters[-1])
            training = ((field.time-164)//.125).astype(int)%2 == 0
            information = np.sum(abs(h)**2, axis=1)
            q = np.sum(h.conj()*field.y, axis=1)/information
            weight = abs(q[~training])**2*information[~training]
            rms.append(float(np.degrees(np.sqrt(np.average(np.angle(q[~training])**2, weights=weight)))))
        label = '/'.join({'12013': 'O', '12015': 'E', '12014': 'R'}[s] for s in record['sites'])
        axes[0].plot(np.arange(5), rms, '.-', label=label)
        axes[1].plot(np.arange(5), envelope_error, '.-', label=label)
        values.append({'sites': record['sites'], 'fitted_midpoint_phase_rms_deg': record['phase_rms_deg'],
            'withheld_phase_rms_deg': rms, 'unfitted_envelope_error_m': envelope_error.tolist(),
            'scope': '164-165 s unused samples, exact archived row operator, common training-fit phase held fixed. Per-band range origins remain free as in this control; their unfitted physical errors are shown separately.'})
    for ax in axes:
        ax.set_xticks(np.arange(5), NAMES);ax.legend(fontsize=8)
    axes[0].set_ylabel('Withheld temporal phase RMS (degrees)')
    axes[0].set_title('Unused samples across the flash; no phase refit')
    axes[1].set_ylabel('Predicted physical path minus measured envelope (m)')
    axes[1].set_title('Absolute range was excluded from these two fits')
    fig.suptitle('Limits of the two midpoint-phase solutions\n'
                 'Fitted midpoint agreement is not a full-data geometry validation')
    fig.savefig(output/'12-common-phase-validation.png', dpi=170);plt.close(fig)
    return values
