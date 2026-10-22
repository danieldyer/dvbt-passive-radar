"""Compare shared and independent scattering using the full complex range PSF.

All linear amplitudes are eliminated exactly. One fixed complex band response
may be shared by the entire patch. Separate time blocks test that response and
the scatterer amplitudes without refitting either on validation samples.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from field import Field


def normal(design, data):
    a = design.reshape(-1, design.shape[-1])
    y = data.ravel()
    return {'gram': a.conj().T@a, 'score': a.conj().T@y,
            'energy': float(np.vdot(y, y).real), 'observations': len(y)}


def independent(equations):
    return np.array([np.linalg.solve(e['gram'], e['score']) for e in equations])


def shared(equations, *, reference=3):
    other = np.array([c for c in range(len(equations)) if c != reference])
    energy = sum(e['energy'] for e in equations)
    count = len(other)

    def unpack(p):
        gains = np.ones(len(equations), complex)
        gains[other] = np.exp(p[:count]+1j*p[count:])
        return gains

    def project(gains):
        gram = sum(abs(g)**2*e['gram'] for g, e in zip(gains, equations))
        score = sum(g.conjugate()*e['score'] for g, e in zip(gains, equations))
        amplitude = np.linalg.solve(gram, score)
        return amplitude, float(np.vdot(score, amplitude).real)

    def objective(p):
        gains = unpack(p)
        amplitude, captured = project(gains)
        cross = np.array([g.conjugate()*np.vdot(amplitude, g*(e['gram']@amplitude)-e['score'])
                          for g, e in zip(gains, equations)])[other]
        gradient = 2*np.r_[cross.real, cross.imag]/energy
        return -captured/energy, gradient

    separate = independent(equations)
    u, _, _ = np.linalg.svd(separate, full_matrices=False)
    seed = u[:, 0]/u[reference, 0]
    initial = np.r_[np.clip(np.log(abs(seed[other])), -4, 4), np.angle(seed[other])]
    bounds = [(-5., 5.)]*count+[(None, None)]*count
    trials = [minimize(objective, p, jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 30})
              for p in (np.zeros(2*count), initial)]
    # At floating-point precision the line search may report failure after the
    # independently checked stationarity gate has already passed. Keep that
    # status for audit, but do not confuse it with a non-stationary gain fit.
    stationary = [r for r in trials if np.isfinite(r.fun) and np.max(abs(r.jac)) <= 1e-7]
    best = min(stationary or trials, key=lambda result: result.fun)
    gains = unpack(best.x)
    amplitude, captured = project(gains)
    return {'gains': gains, 'amplitude': amplitude, 'captured_training_energy': captured,
            'converged': bool(np.isfinite(best.fun) and np.max(abs(best.jac)) <= 1e-7),
            'optimizer_success': bool(best.success), 'termination': str(best.message),
            'evaluations': best.nfev, 'optimality': float(np.max(abs(best.jac))),
            'gain_bound_hit': bool(np.any(abs(best.x[:count]) > 4.99))}


def prediction(equations, amplitudes):
    model_energy, cross, error, upper = 0., 0j, 0., 0.
    channels = []
    for channel, (e, a) in enumerate(zip(equations, amplitudes)):
        predicted = float(np.vdot(a, e['gram']@a).real)
        product = np.vdot(a, e['score'])
        residual = e['energy']-2*product.real+predicted
        best = float(np.vdot(e['score'], np.linalg.solve(e['gram'], e['score'])).real)
        channels.append({'channel': channel, 'prediction_error': float(residual),
                         'predicted_energy': predicted, 'independently_fitted_energy': best})
        model_energy += predicted;cross += product;error += residual;upper += best
    captured = abs(cross)**2/model_energy if model_energy > 0 else 0.
    return {'prediction_error': float(error), 'predicted_energy': model_energy,
            'matched_validation_energy': float(captured), 'independent_upper_energy': upper,
            'efficiency_loss_db': float(-10*np.log10(captured/upper)) if captured > 0 else None,
            'channels': channels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fields', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design-cache', type=Path, help='Reuse an already evaluated fixed-point forward model')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    positions = np.array([[-50., -2., 20.], [25., 5., 20.], [90., 8., 20.], [135., 13., 20.]])
    train, test, all_equations = [], [], []
    clock = time.perf_counter()
    for channel in range(5):
        with np.load(args.fields/f'channel-{channel}.npz') as loaded:
            data = dict(loaded)
        field = Field(data, data['pulse_samples'])
        if args.design_cache:
            design = np.load(args.design_cache/f'channel-{channel}-design.npy')
        else:
            design = np.stack([field.point(p) for p in positions], axis=-1)
            np.save(args.output/f'channel-{channel}-design.npy', design)
        selected = ((field.time-90)//2)%2 == 0
        train.append(normal(design[selected], field.y[selected]))
        test.append(normal(design[~selected], field.y[~selected]))
        all_equations.append(normal(design, field.y))
        print('Prepared full point model', channel, round(time.perf_counter()-clock, 2), 'seconds', flush=True)
    fit = shared(train)
    separate = independent(train)
    result = {'point_body_m': positions.tolist(), 'shared_converged': fit['converged'],
              'gain_bound_hit': fit['gain_bound_hit'], 'evaluations': fit['evaluations'], 'optimality': fit['optimality'],
              'shared_band_gains': np.c_[fit['gains'].real, fit['gains'].imag].tolist(),
              'shared_reflectivities': np.c_[fit['amplitude'].real, fit['amplitude'].imag].tolist(),
              'independent_reflectivities_real': separate.real.tolist(),
              'independent_reflectivities_imag': separate.imag.tolist(),
              'training_gram_conditions': [float(np.linalg.cond(e['gram'])) for e in train],
              'shared_held_out': prediction(test, fit['gains'][:, None]*fit['amplitude']),
              'independent_held_out': prediction(test, separate),
              'scope': 'Full complex range response and range covariance; exact original-row binning. Fixed points and motion from prior images; alternating 2-second blocks fit/test amplitudes and shared band response. No antenna fit or autofocus accepted.'}
    (args.output/'scatterer-validation.json').write_text(json.dumps(result, indent=2))
    np.savez(args.output/'normal-equations.npz', positions=positions,
             train_gram=np.array([e['gram'] for e in train]), train_score=np.array([e['score'] for e in train]),
             train_energy=np.array([e['energy'] for e in train]), test_gram=np.array([e['gram'] for e in test]),
             test_score=np.array([e['score'] for e in test]), test_energy=np.array([e['energy'] for e in test]))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
