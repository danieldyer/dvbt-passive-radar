"""Exact finite-block GLS and REML for a declared spectral covariance.

The covariance is circulant in slow time and constant within each frequency
band, with independent whitened range samples. This is a testable model, not
an assertion that ship echoes or sea clutter satisfy it. Nuisance amplitudes
are eliminated in the same covariance metric as the signal. Fourier transforms
are unitary; no data-dependent image normalization or phase adjustment occurs.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def orthogonal_nuisance(design, tolerance=1e-10):
    """Restrict a saved nuisance span to a new block without dependent columns."""
    shape = design.shape
    u, s, _ = np.linalg.svd(design.reshape(-1, shape[-1]), full_matrices=False)
    keep = s > s[0]*tolerance
    return u[:, keep].reshape(*shape[:-1], int(keep.sum()))


def sufficient_statistics(data, nuisance, labels):
    """Band sufficient statistics, with input data and nuisance in Fourier space."""
    power, cross, gram, counts = [], [], [], []
    for b in range(int(labels.max())+1):
        selected = labels == b
        y = data[selected].ravel()
        q = nuisance[selected].reshape(len(y), -1)
        power.append(float(np.vdot(y, y).real))
        cross.append(q.conj().T@y)
        gram.append(q.conj().T@q)
        counts.append(len(y))
    return dict(power=np.array(power), cross=np.array(cross), gram=np.array(gram), counts=np.array(counts))


def restricted_likelihood(log_variance, statistics):
    """Proper-complex REML negative log likelihood and log-variance gradient.

    Constants independent of covariance are omitted. The log determinant of
    the nuisance information is required: profiling its mean without that
    term biases estimated variances downwards.
    """
    precision = np.exp(-np.asarray(log_variance))
    gram = np.einsum('b,bij->ij', precision, statistics['gram'])
    rhs = precision@statistics['cross']
    factor = cho_factor(gram, lower=True, check_finite=False)
    beta = cho_solve(factor, rhs, check_finite=False)
    energy = (statistics['power']-2*np.real(statistics['cross']@beta.conj())+
        np.einsum('i,bij,j->b', beta.conj(), statistics['gram'], beta).real)
    inverse = cho_solve(factor, np.eye(len(beta)), check_finite=False)
    leverage = np.einsum('ij,bji->b', inverse, statistics['gram']).real*precision
    value = (statistics['counts']@log_variance+
        2*np.log(np.diag(factor[0]).real).sum()+precision@energy)
    gradient = statistics['counts']-leverage-precision*energy
    return float(value), gradient


def fit_variance(records):
    """Fit a few shared band powers using secondary blocks only."""
    dof = sum(r['counts'].sum()-r['gram'].shape[-1] for r in records)
    def objective(parameters):
        values = [restricted_likelihood(parameters, r) for r in records]
        return sum(v[0] for v in values)/dof, sum(v[1] for v in values)/dof
    initial = np.log(sum(r['power'] for r in records)/sum(r['counts'] for r in records))
    result = minimize(objective, initial, jac=True, method='BFGS',
        options={'gtol': 1e-8, 'maxiter': 300})
    value, gradient = objective(result.x)
    converged = bool(np.max(abs(gradient)) < 1e-7)
    return result.x, {'value_per_dof': value, 'maximum_gradient': float(np.max(abs(gradient))),
        'converged': converged, 'iterations': int(result.nit), 'message': str(result.message),
        'variance': np.exp(result.x).tolist()}


def matched_filter(template, nuisance, labels, log_variance):
    """Unit-response filter h: h^H template=1, h^H nuisance=0.

    Inputs and returned filter are in unitary Fourier coordinates. Under the
    stated covariance, its complex amplitude variance equals 1/information.
    """
    weight = np.exp(-np.asarray(log_variance)[labels])[:, None]
    q = nuisance.reshape(template.size, -1)
    weighted_q = (nuisance*weight[:, :, None]).reshape(template.size, -1)
    weighted_target = (weight*template).ravel()
    factor = cho_factor(q.conj().T@weighted_q, lower=True, check_finite=False)
    coefficient = cho_solve(factor, q.conj().T@weighted_target, check_finite=False)
    value = weighted_target-weighted_q@coefficient
    information = float(np.vdot(template, value).real)
    if not information > 0 or not np.isfinite(information):
        raise ValueError('The target has no identifiable amplitude outside the nuisance span')
    return value.reshape(template.shape)/information, information
