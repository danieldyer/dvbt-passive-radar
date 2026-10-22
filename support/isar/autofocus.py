"""Constrained common-phase fitting for fixed, backprojected scatterer histories.

The complex amplitudes are nuisance parameters eliminated by least squares.
Only smooth non-affine phase is fitted. Interleaved time blocks are held out;
their samples never enter the fit. This is a diagnostic of a point-scatterer
model, not proof that an extended ship has frequency-independent reflectivity.
"""
import numpy as np
from scipy.optimize import minimize


def design(time, interval, degree):
    if degree < 2 or degree > 20 or not interval[0] < interval[1]:
        raise ValueError('Invalid phase-model order or aperture')
    u = 2*(np.asarray(time)-interval[0])/(interval[1]-interval[0])-1
    # Fix the constant/linear gauge: these terms change phase/image origin.
    return np.polynomial.legendre.legvander(u, degree)[:, 2:]


def evaluate(model, time, *, scale=1):
    return scale*design(time, model['interval_s'], model['degree'])@model['coefficients']


def fit(groups, interval, *, degree=10, initial=None):
    """Fit groups {time, value[N,P], information[N], phase_scale, train[N]}.

    phase_scale=1 fits radians; k=2*pi*RF/c fits a shared path-phase in metres.
    Information is fixed by the input matched filter, never inferred from peaks.
    """
    prepared = []
    total_energy = sum(np.sum(g['information'][:, None]*abs(g['value'])**2) for g in groups)
    total_weight = sum(np.sum(g['information'])*g['value'].shape[1] for g in groups)
    amplitude_scale = np.sqrt(total_energy/total_weight)
    weight_scale = sum(np.sum(g['information']) for g in groups)/sum(len(g['time']) for g in groups)
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0 or weight_scale <= 0:
        raise ValueError('No finite autofocus information')
    for g in groups:
        if g['value'].ndim != 2 or len(g['value']) != len(g['time']) or not np.isfinite(g['value']).all():
            raise ValueError('Invalid phase history')
        b = g['phase_scale']*design(g['time'], interval, degree)
        z, w = g['value']/amplitude_scale, g['information']/weight_scale
        train = np.asarray(g['train'], bool)&(w > 0)
        if train.sum() < 4*degree or ((~train)&(w > 0)).sum() < degree:
            raise ValueError('Insufficient training/held-out samples')
        prepared.append((b, z, w, train))

    training_weight = sum(np.sum(w[train])*z.shape[1] for b, z, w, train in prepared)

    def objective(coefficients):
        # Concentrated likelihood: the input energy is phase invariant, so
        # optimise only coherent model energy. A residual Gauss-Newton Hessian
        # would contain a large spurious curvature from unmodelled echoes/noise.
        cost, gradient = 0., np.zeros(degree-1)
        for b, z, w, train in prepared:
            b, z, w = b[train], z[train], w[train]
            rotated = z*np.exp(-1j*(b@coefficients))[:, None]
            total = np.sum(w[:, None]*rotated, axis=0)
            derivative = -1j*(w[:, None]*rotated).T@b
            cost -= .5*np.sum(abs(total)**2)/w.sum()
            gradient -= np.real(total.conj()@derivative)/w.sum()
        return cost/training_weight, gradient/training_weight

    starts = [np.zeros(degree-1)]
    if initial is not None:
        starts.append(np.asarray(initial))
    if len(groups) == 1:
        b, z, w, train = prepared[0]
        # Adjacent-history products supply an unwrap seed only. The final
        # complex fit uses training data, including its nonuniform epochs.
        measured = z[train]
        increment = np.angle(np.sum(measured[1:]*measured[:-1].conj(), axis=1))
        phase = np.r_[0., np.cumsum(increment)]
        t = groups[0]['time'][train]
        full = np.column_stack([np.ones(len(t)), t-t.mean(), b[train]])
        seed = np.linalg.lstsq(full*np.sqrt(w[train])[:, None], phase*np.sqrt(w[train]), rcond=None)[0][2:]
        starts.append(seed)
    candidates = [minimize(objective, start, jac=True, method='BFGS',
                           options={'maxiter': 200, 'gtol': 1e-8}) for start in starts]
    solution = min(candidates, key=lambda s: s.fun)
    diagnostics = []
    for g, (b, z, w, train) in zip(groups, prepared):
        phase = b@solution.x
        rotated = z*np.exp(-1j*phase)[:, None]
        amplitude = np.sum(w[train, None]*rotated[train], axis=0)/w[train].sum()
        baseline = np.sum(w[train, None]*z[train], axis=0)/w[train].sum()
        test = (~train)&(w > 0)
        before_error = np.sum(w[test, None]*abs(z[test]-baseline)**2)
        after_error = np.sum(w[test, None]*abs(rotated[test]-amplitude)**2)
        before_power = np.sum(abs(np.sum(w[test, None]*z[test], axis=0))**2)
        after_power = np.sum(abs(np.sum(w[test, None]*rotated[test], axis=0))**2)
        diagnostics.append({'channel': g.get('channel'), 'training_bins': int(train.sum()),
                            'held_out_bins': int(test.sum()),
                            'held_out_coherent_power_gain_db': float(10*np.log10(after_power/before_power)),
                            'held_out_prediction_error_reduction_percent': float(100*(before_error-after_error)/before_error),
                            'phase_rms_radians': float(np.sqrt(np.mean(phase**2)))})
    return {'interval_s': list(interval), 'degree': degree, 'coefficients': solution.x.tolist(),
            'converged': bool(solution.success), 'evaluations': solution.nfev,
            'negative_coherent_training_energy': float(solution.fun), 'diagnostics': diagnostics,
            'gauge': 'Legendre constant and linear terms fixed at zero',
            'model': 'Common non-affine phase, constant complex amplitude per selected scatterer; no image entropy objective'}
