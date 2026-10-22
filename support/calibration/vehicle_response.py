"""Extended-vehicle response and channel-phase identifiability diagnostic.

Fixed body grids replace the single common complex point coefficient. The
observed range response, exact road-following motion and one-second temporal
holdout are unchanged. Coordinates remain a conditional, unaccepted trial.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from coherent_fit import OTHER
from road_fit import load_trial
from scatterer_validation import RigidEcho


def ridge_gcv(matrix, values):
    """Complex Tikhonov fit, regularization chosen using training samples only."""
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    score = u.conj().T@values
    strengths = singular[0]**2*np.logspace(-12, 2, 141)
    influence = singular[:, None]**2/(singular[:, None]**2+strengths)
    residual = np.vdot(values, values).real-((2*influence-influence**2)*abs(score[:, None])**2).sum(axis=0)
    degrees = influence.sum(axis=0)
    criterion = residual/(len(values)-degrees)**2
    best = int(np.argmin(criterion))
    alpha = vh.conj().T@(singular*score/(singular**2+strengths[best]))
    return alpha, {'regularization_relative_to_max_gram': float(strengths[best]/singular[0]**2),
                   'effective_complex_parameters': float(degrees[best]),
                   'strength_search_boundary': best in (0, len(strengths)-1),
                   'singular_values': singular.tolist()}


def phase_information(matrix, alpha, sizes):
    """Expected phase information after all complex reflectivities and gains.

    This deliberately omits pose/location nuisance: it is optimistic about
    calibration information, not a covariance estimate for the real receiver.
    """
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    basis = u[:, singular > singular[0]*1e-8]
    signal = matrix@alpha
    gain = np.zeros((len(signal), 4), complex)
    offset = np.r_[0, np.cumsum(sizes)]
    for column, channel in enumerate(OTHER):
        gain[offset[channel]:offset[channel+1], column] = signal[offset[channel]:offset[channel+1]]
    phase = 1j*gain
    phase -= basis@(basis.conj().T@phase)
    gain -= basis@(basis.conj().T@gain)
    real_gain = np.r_[gain.real, gain.imag]
    real_phase = np.sqrt(2)*np.r_[phase.real, phase.imag]
    q, s, _ = np.linalg.svd(real_gain, full_matrices=False)
    q = q[:, s > max(s[0]*1e-10, 1e-20)]
    real_phase -= q@(q.T@real_phase)
    return {'singular_values_per_radian': np.linalg.svd(real_phase, compute_uv=False).tolist(),
            'fisher_information': (real_phase.T@real_phase).tolist(),
            'reflectivity_nuisance_rank': basis.shape[1],
            'reflectivity_rank_relative_tolerance': 1e-8,
            'scope': 'Expected local sensitivity; complex body reflectivities and positive band gains profiled. Geometry and trajectory fixed, so additional motion/position nuisance can only reduce this information.'}


def grid(kind):
    if kind == 'point':
        return np.zeros((1, 3))
    if kind == 'line':
        return np.c_[np.linspace(-4, 4, 5), np.zeros((5, 2))]
    if kind == 'plane':
        return np.array(list(itertools.product(np.linspace(-4, 4, 5), [-1.5, 0., 1.5], [0.])))
    if kind == 'volume':
        return np.array(list(itertools.product(np.linspace(-4, 4, 5), [-1.5, 0., 1.5], [-1.5, 0., 1.5])))
    raise ValueError('Unknown body grid')


def compare(train, test, values, withheld, sizes):
    alpha, regularization = ridge_gcv(train, values)
    prediction = test@alpha
    offsets = np.r_[0, np.cumsum(sizes)]
    channels = []
    for channel in range(5):
        part = slice(offsets[channel], offsets[channel+1])
        independent, _ = ridge_gcv(train[part], values[part])
        other = test[part]@independent
        channels.append({'channel': channel, 'shared_prediction_error': float(np.sum(abs(withheld[part]-prediction[part])**2)),
                         'independent_prediction_error': float(np.sum(abs(withheld[part]-other)**2)),
                         'shared_prediction_energy': float(np.sum(abs(prediction[part])**2)),
                         'independent_prediction_energy': float(np.sum(abs(other)**2))})
    return {**regularization, 'channels': channels,
            'shared_prediction_error': sum(ch['shared_prediction_error'] for ch in channels),
            'independent_prediction_error': sum(ch['independent_prediction_error'] for ch in channels),
            'phase_information': phase_information(train, alpha, sizes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'histories', 'patches', 'templates', 'routes', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--labels', nargs='+', default=['A', 'B', 'C', 'K'])
    args = parser.parse_args()
    saved = json.loads(args.fit.read_text())
    if saved.get('range_bias', False) or saved.get('amplitude_block_seconds'):
        raise ValueError('This diagnostic requires the existing constant, zero-envelope-bias control')
    joint = load_trial(saved, args.routes, args.histories, args.patches, args.templates)
    p = np.asarray(saved['state'])
    beta = np.zeros(5);beta[OTHER] = p[joint.beta_offset:joint.magnitude_offset]
    result = {'source_fit': str(args.fit), 'status': 'Offline model/identifiability diagnostic, no adopted calibration',
              'scope': 'Fixed conditional site/road trajectories; regular deterministic vehicle grids. One-second holdout fits reflectivities and band amplitudes on training only. Road/trajectory discovery used the full recording.',
              'targets': []}
    for index, (label, problem) in enumerate(zip(saved['labels'], joint.problems)):
        if label not in args.labels:
            continue
        state = np.r_[p[9+index*joint.n:9+(index+1)*joint.n], p[:9]]
        start = problem.model.seed['epoch']-problem.model.half
        selected = lambda times: (np.floor(times-start).astype(int)%2) == 0
        target = {'label': label, 'models': []}
        point_control = RigidEcho(problem, state, beta, 1, selected)
        point_matrix, _ = point_control.design(np.zeros(7))
        sizes = [len(field.y) for field in point_control.fields]
        offsets = np.r_[0, np.cumsum(sizes)]
        gains = []
        for channel in range(5):
            part = slice(offsets[channel], offsets[channel+1])
            u = point_matrix[part, 0]
            gains.append(abs(np.vdot(u, point_control.data[part])/np.vdot(u, u)))
        gains = np.array(gains)/gains[1]
        row_gain = np.repeat(gains, sizes)
        for kind in ('point', 'line', 'plane', 'volume'):
            locations = grid(kind)
            train = RigidEcho(problem, state, beta, len(locations), selected)
            test = RigidEcho(problem, state, beta, len(locations), lambda times: ~selected(times))
            parameters = np.r_[locations.ravel(), np.zeros(4)]
            a, _ = train.design(parameters);b, _ = test.design(parameters)
            if sizes != [len(field.y) for field in train.fields] or sizes != [len(field.y) for field in test.fields]:
                raise ValueError('Holdout observation dimensions changed')
            # Positive illumination/RCS scale is a fitted nuisance magnitude;
            # all bands remain in the likelihood and retain their measured phase.
            a *= row_gain[:, None];b *= row_gain[:, None]
            variants = []
            for ten_phase in (0., np.pi/2, np.pi):
                rotation = np.ones(sum(sizes), complex)
                rotation[offsets[3]:offsets[4]] = np.exp(1j*ten_phase)
                compared = compare(a*rotation[:, None], b*rotation[:, None], train.data, test.data, sizes)
                variants.append({'ten_phase_perturbation_degrees': float(np.degrees(ten_phase)), **compared})
            target['models'].append({'grid': kind, 'points_body_m': locations.tolist(),
                                     'training_positive_band_gains': gains.tolist(), 'variants': variants})
            print(label, kind, 'held-out shared errors', [round(v['shared_prediction_error'], 2) for v in variants],
                  'phase singular', np.round(variants[0]['phase_information']['singular_values_per_radian'], 5), flush=True)
        result['targets'].append(target)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
