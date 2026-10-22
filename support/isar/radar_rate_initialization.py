"""Training-only phase-gradient seed for the physical motion likelihood.

Stationary nuisance is projected from the whitened training observations.
Adjacent complex products then provide a coarse common excess-path rate. The
rate initializes a bounded physical spline; it is never applied as a separate
phase-only correction, and the subsequent likelihood has no penalty toward it.
This is an acquisition aid, not a CRLB certificate for an extended target.
"""
import numpy as np

from field import average
from full_capture_motion import constrained_curve
from motion import excess_path
from spline_motion import SplineMotion, basis


def rate_seed(bands, parameters, *, bin_seconds=2., acceleration_limit=.15, coefficient_limit=200.):
    if not (bin_seconds > 0 and acceleration_limit > 0 and coefficient_limit > 0):
        raise ValueError('Positive acquisition intervals and bounds are required')
    curve = SplineMotion({**bands[0].model, 'parameters': parameters})
    if np.max(abs(curve.parameters[curve.sections[0]])) > 1e-10:
        raise ValueError('Rate acquisition requires a zero initial path correction')
    first, last = curve.interval
    count = int(np.ceil((last-first)/bin_seconds))
    numerator, precision, epoch_sum = (np.zeros(count) for _ in range(3))
    pairs_per_band = []
    for band in bands:
        field = band.field
        state = curve.apply(band.time, band.prior, band.reference_transmitter, band.receiver)
        point = np.array([0., 0., 20.])
        position = state['position_ecef_m']+state['body_to_ecef']@point
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
        delta, _, _ = field.path_kinematics(position, velocity)
        block = field.metadata['block_rows']
        phase, _ = average(np.exp(-1j*field.k*delta), field.data['row_information'], block)
        deramp = np.divide(phase.conj(), abs(phase), out=np.ones_like(phase), where=abs(phase)>1e-12)
        # This sensitivity maps the shared physical translation coordinate to
        # each channel's path, retaining the small geometric differences.
        _, gradient = excess_path(position, field.data['transmitter_ecef_m'], field.data['receiver_ecef_m'])
        sensitivity, _ = average(np.sum(gradient*state['translation_direction'], axis=1),
                                  field.data['row_information'], block)
        clean = np.zeros_like(field.y)
        clean[band.selected_time] = band.clean.reshape(-1, field.y.shape[1])
        clean *= deramp[:, None]
        pair = np.sum(clean[1:]*clean[:-1].conj(), axis=1)
        dt = np.diff(field.time)
        selected = band.selected_time & (field.root_information>0)
        valid = selected[1:] & selected[:-1] & (dt>0)
        centre = (field.time[1:]+field.time[:-1])/2
        scale = field.k*dt*(sensitivity[1:]+sensitivity[:-1])/2
        weight = abs(pair)*valid
        group = np.floor((centre-first)/bin_seconds).astype(int)
        if np.any((group<0)|(group>=count)):
            raise ValueError('Acquisition samples leave the declared motion interval')
        numerator += np.bincount(group, weights=-weight*scale*np.angle(pair), minlength=count)
        precision += np.bincount(group, weights=weight*scale**2, minlength=count)
        epoch_sum += np.bincount(group, weights=weight*scale**2*centre, minlength=count)
        pairs_per_band.append(int(valid.sum()))
    keep = precision>0
    if keep.sum()<2:
        raise ValueError('Insufficient training-only adjacent phase products')
    time = epoch_sum[keep]/precision[keep]
    rate = numerator[keep]/precision[keep]
    spacing = curve.path_spacing
    n = curve.counts[0]
    at_middle = basis([np.mean(curve.interval)], curve.interval, spacing)[0]
    centered = (np.eye(n)-np.ones((n, 1))*at_middle)[:, :-1]
    design = basis(time, curve.interval, spacing, derivative=1)@centered
    check_time = np.linspace(first, last, int(np.ceil((last-first)*8))+1)
    acceleration = basis(check_time, curve.interval, spacing, derivative=2)@centered
    coefficients = constrained_curve(design, rate, precision[keep],
        np.r_[acceleration, centered], np.r_[np.full(len(check_time), acceleration_limit), np.full(n, coefficient_limit)])
    result = np.array(parameters, dtype=float).copy()
    result[:n] = centered@coefficients
    return result, {'scope': __doc__, 'bin_seconds': bin_seconds,
        'time_s': time.tolist(), 'measured_path_rate_mps': rate.tolist(),
        'fitted_path_rate_mps': (design@coefficients).tolist(),
        'relative_weights': (precision[keep]/precision[keep].sum()).tolist(),
        'valid_training_pairs_by_band': pairs_per_band,
        'phase_reference_point_body_m': [0., 0., 20.],
        'path_origin': 'Zero at the interval midpoint; subsequently free in the physical likelihood',
        'branch_assumption': 'Adjacent residual phase differences stay within pi; this is a local acquisition seed, not arbitrary velocity ambiguity resolution.'}
