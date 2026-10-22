"""Finite antenna perturbations under optimistic, exactly known trajectories.

No fitting prior is included in the mismatch. A synthetic nondispersive point
response uses each real band's independently measured power. This isolates
geometric sensitivity from optimizer behaviour and real scattering mismatch.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from coherent_fit import OTHER
from road_fit import load_trial


def phase_objective(phases, information):
    """Negative explained energy and exact derivative for four band phases."""
    full = np.zeros(5);full[OTHER] = phases
    u = np.exp(1j*full)
    score = information@u
    return -np.vdot(u, score).real, -2*np.imag(u.conj()*score)[OTHER]


def profile_channel_phases(correlations, model_energy):
    """Eliminate one complex target coefficient and fit shared band phases.

    Multiple deterministic starts test the small periodic phase problem. The
    returned solution is a numerical profile, not a global-optimum certificate.
    """
    information = (correlations.T/model_energy)@correlations.conj()
    scale = np.trace(information).real
    if scale <= 0:
        return np.zeros(5)
    _, vectors = np.linalg.eigh(information)
    starts = [np.zeros(5), np.angle(vectors[:, -1])]
    starts.extend(np.angle(correlations))
    solutions = [minimize(lambda x: phase_objective(x, information/scale),
                          (start-start[1])[OTHER], jac=True, method='BFGS',
                          options={'gtol': 1e-10, 'maxiter': 300}) for start in starts]
    best = min(solutions, key=lambda result: result.fun)
    phases = np.zeros(5);phases[OTHER] = best.x
    return np.angle(np.exp(1j*phases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('fit', 'histories', 'patches', 'templates', 'routes', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads(args.fit.read_text())
    if saved.get('amplitude_block_seconds') or saved.get('range_bias'):
        raise ValueError('Use the constant-reflectivity, zero-envelope-bias control for this diagnostic')
    joint = load_trial(saved, args.routes, args.histories, args.patches, args.templates)
    p = np.array(saved['state'])
    phases = np.zeros(5);phases[OTHER] = p[joint.beta_offset:joint.magnitude_offset]
    scales = np.tile([30., 30., 75.], 3)
    sites = np.array(saved['site_positions_enu_m'])+p[:9].reshape(3, 3)*[30, 30, 75]
    receiver = joint.problems[0].model.receiver_enu_m
    direction = sites-receiver
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    used = sorted(set(saved['assignment']))
    inputs = []
    for i, problem in enumerate(joint.problems):
        state = np.r_[p[9+i*joint.n:9+(i+1)*joint.n], p[:9]]
        templates = [problem.channel_model(state, c, True)[0] for c in range(5)]
        amplitudes = np.array([abs(np.vdot(u, problem.channel_data(c))/np.vdot(u, u))
                               for c, u in enumerate(templates)])*np.exp(1j*phases)
        values = [a*u for a, u in zip(amplitudes, templates)]
        inputs.append((problem, state, amplitudes, values))

    records = []
    for site in used:
        for axis in ('towards_receiver', 'east', 'north', 'up'):
            vector = -direction[site] if axis == 'towards_receiver' else np.eye(3)[('east', 'north', 'up').index(axis)]
            for metres in (-100., -30., -10., 10., 30., 100.):
                correction = np.zeros((3, 3));correction[site] = metres*vector
                known, common, independent = 0., 0., 0.
                maximum_path_change = 0.
                comparisons = []
                for problem, state, amplitudes, values in inputs:
                    changed = state.copy();changed[-9:] += correction.ravel()/scales
                    predicted = [a*problem.channel_model(changed, c, True)[0] for c, a in enumerate(amplitudes)]
                    truth = np.concatenate(values);model = np.concatenate(predicted)
                    known += np.vdot(model-truth, model-truth).real
                    gain = np.vdot(model, truth)/np.vdot(model, model)
                    common += np.vdot(gain*model-truth, gain*model-truth).real
                    comparisons.append((values, predicted))
                    for channel, (reference, moved) in enumerate(zip(values, predicted)):
                        gain = np.vdot(moved, reference)/np.vdot(moved, moved)
                        independent += np.vdot(gain*moved-reference, gain*moved-reference).real
                        time = problem.fields[channel].time
                        channels = np.full(len(time), channel)
                        path = problem.model.paths(state, time, channels, problem.sites, True)[0]
                        shifted = problem.model.paths(changed, time, channels, problem.sites, True)[0]
                        maximum_path_change = max(maximum_path_change, float(np.max(abs(shifted-path))))
                correlations = np.array([[np.vdot(m, y) for m, y in zip(model, truth)]
                                         for truth, model in comparisons])
                energies = np.array([sum(np.vdot(m, m).real for m in model)
                                     for _, model in comparisons])
                phases = profile_channel_phases(correlations, energies)
                unknown_phase = 0.
                for truth, model in comparisons:
                    m = np.concatenate([u*np.exp(1j*phase) for u, phase in zip(model, phases)])
                    y = np.concatenate(truth)
                    gain = np.vdot(m, y)/np.vdot(m, m)
                    unknown_phase += np.vdot(gain*m-y, gain*m-y).real
                records.append({'site': site, 'direction': axis, 'displacement_m': metres,
                                'maximum_excess_path_change_m': maximum_path_change,
                                'known_complex_reflectivity_squared_distance': float(known),
                                'unknown_common_target_reflectivity_squared_distance': float(common),
                                'unknown_common_target_and_channel_phase_squared_distance': float(unknown_phase),
                                'profiled_channel_phase_degrees': np.degrees(phases).tolist(),
                                'unknown_per_band_target_reflectivity_squared_distance': float(independent)})
            selected = records[-6:]
            print(site, axis, [(r['displacement_m'], round(r['unknown_common_target_reflectivity_squared_distance'], 6)) for r in selected], flush=True)
    result = {'source_fit': str(args.fit), 'used_sites': used, 'labels': saved['labels'],
              'records': records,
              'scope': 'Exact finite path/range-response perturbations, fixed known trajectories and fixed receiver. Ideal shared-phase point echoes at all independently measured band powers. Distances use the existing empirical whitened covariance; no location prior contributes. Shared channel phases use a numerical profile with deterministic multiple starts, not a certified global optimum. These are optimistic conditional sensitivities, not surveyed coordinates or a certified experimental CRLB.',
              'interpretation': 'For a proper complex unit-variance Gaussian observation, squared mean distance is KL divergence. Weak sensitivity even with known paths cannot be repaired by freeing unknown motion or scattering parameters.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
