"""Finite-block complex GLS with range covariance in each Doppler band.

The stated covariance is block diagonal in unitary Fourier coordinates:
R_f = C_band(f), C_b >= I after the existing thermal whitening. The noise
floor is a physical model assumption, not a fitted image normalization.
Stationary means are integrated out by proper-complex REML. EM updates keep
their coefficient uncertainty and solve the covariance-floor M-step exactly.
All matrices are small range/nuisance statistics; no sample-by-sample global
covariance is built. These are support kernels, not production corrections.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def hermitian(value):
    return (value+value.conj().swapaxes(-2, -1))/2


def statistics(data, nuisance, labels):
    """Sufficient range/mean cross-products, with an invariant mean removal."""
    data, nuisance, labels = np.asarray(data), np.asarray(nuisance), np.asarray(labels)
    if (data.ndim != 2 or nuisance.shape[:2] != data.shape or labels.shape != (len(data),)
            or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0
            or not np.all(np.bincount(labels) > 0)):
        raise ValueError('Need matching Fourier observations, nuisance and contiguous band labels')
    q = nuisance.reshape(data.size, -1)
    # REML is invariant to y -> y-Q beta. Remove a potentially large stationary
    # mean before cross-products, using the complete nuisance span, not a notch.
    factor = cho_factor(q.conj().T@q, lower=True)
    clean = (data.ravel()-q@cho_solve(factor, q.conj().T@data.ravel())).reshape(data.shape)
    value = np.concatenate([clean[:, :, None], nuisance], axis=2)
    moments, counts = [], []
    r, m = value.shape[1:]
    for b in range(labels.max()+1):
        w = value[labels == b].reshape(-1, r*m)
        # H[r,s,m,n] = sum_f conj(W[f,r,m]) W[f,s,n].
        moments.append((w.conj().T@w).reshape(r, m, r, m).transpose(0, 2, 1, 3))
        counts.append(len(w))
    return {'moments': np.array(moments), 'counts': np.array(counts),
        'nuisance_rank': q.shape[1], 'ranges': r}


def expectation(covariance, records):
    """Restricted NLL and the expected complete residual range covariance."""
    covariance = hermitian(np.asarray(covariance))
    bands, r, _ = covariance.shape
    factors = [cho_factor(c, lower=True, check_finite=False) for c in covariance]
    precision = np.array([cho_solve(f, np.eye(r), check_finite=False) for f in factors])
    logdet = np.array([2*np.log(np.diag(f[0]).real).sum() for f in factors])
    count = sum(record['counts'] for record in records)
    complete = np.zeros_like(covariance)
    likelihood = float(count@logdet)
    for record in records:
        h = record['moments']
        if h.shape[:3] != (bands, r, r):
            raise ValueError('Covariance and sufficient statistics have different shapes')
        gram = np.einsum('brs,brsmn->mn', precision, h, optimize=True)
        factor = cho_factor(hermitian(gram[1:, 1:]), lower=True, check_finite=False)
        beta = cho_solve(factor, gram[1:, 0], check_finite=False)
        residual = np.r_[1., -beta]
        moment = np.outer(residual, residual.conj())
        moment[1:, 1:] += cho_solve(factor, np.eye(len(beta)), check_finite=False)
        likelihood += (2*np.log(np.diag(factor[0]).real).sum()+
            np.vdot(residual, gram@residual).real)
        complete += np.einsum('brsmn,mn->brs', h.conj(), moment, optimize=True)
    complete = hermitian(complete)
    gradient = count[:, None, None]*precision-precision@complete@precision
    return float(likelihood), complete, hermitian(gradient)


def floor_covariance(value, floor=1.):
    eigen, vectors = np.linalg.eigh(hermitian(value))
    return (vectors*np.maximum(eigen, floor)[..., None, :])@vectors.conj().swapaxes(-2, -1)


def fit_covariance(records, *, tolerance=1e-7, iterations=100):
    """EM-REML with C >= I, checked by a whitened fixed-point residual."""
    if not records:
        raise ValueError('Covariance training observations are required')
    r, bands = records[0]['ranges'], len(records[0]['counts'])
    covariance = np.tile(np.eye(r, dtype=complex), (bands, 1, 1))
    counts = sum(record['counts'] for record in records)
    dof = int(counts.sum()*r-sum(record['nuisance_rank'] for record in records))
    if dof <= 0:
        raise ValueError('No residual degrees of freedom')
    previous, trace = np.inf, []
    for iteration in range(iterations+1):
        value, complete, gradient = expectation(covariance, records)
        if value > previous+1e-9*max(1., abs(previous)):
            raise ArithmeticError('The constrained REML EM step increased the likelihood objective')
        proposed = floor_covariance(complete/counts[:, None, None])
        eigen, vectors = np.linalg.eigh(covariance)
        inverse_root = (vectors/eigen[..., None, :]**.5)@vectors.conj().swapaxes(-2, -1)
        change = inverse_root@(proposed-covariance)@inverse_root
        residual = float(np.max(abs(np.linalg.eigvalsh(hermitian(change)))))
        trace.append({'iteration': iteration, 'value_per_dof': value/dof, 'fixed_point_residual': residual})
        if residual <= tolerance or iteration == iterations:
            break
        covariance, previous = proposed, value
    return covariance, {'converged': residual <= tolerance, 'iterations': iteration,
        'fixed_point_residual': residual, 'value_per_dof': value/dof, 'dof': dof,
        'eigenvalues': np.linalg.eigvalsh(covariance).tolist(), 'trace': trace,
        'criterion': 'Relative covariance fixed-point residual in each whitened mode; at a fixed point the C>=I cone KKT conditions hold.'}


def matched_filters(templates, nuisance, labels, covariance):
    """Independent unit-response filters, sharing the same nuisance solve.

    Templates have shape (frequency, range, hypothesis). Each output is the
    constrained GLS filter for one hypothesis, not a simultaneous scene fit.
    """
    templates, nuisance = np.asarray(templates), np.asarray(nuisance)
    shape = templates.shape
    if (templates.ndim != 3 or nuisance.shape[:2] != shape[:2]
            or len(labels) != shape[0]):
        raise ValueError('Need matching Fourier template and nuisance arrays')
    q = nuisance.reshape(shape[0]*shape[1], -1)
    weighted = np.empty((*shape[:2], shape[2]+nuisance.shape[2]), dtype=complex)
    values = np.concatenate([templates, nuisance], axis=2)
    for b, covariance_b in enumerate(covariance):
        selected = labels == b
        factor = cho_factor(covariance_b, lower=True, check_finite=False)
        local = values[selected]
        weighted[selected] = cho_solve(factor, local.transpose(1, 0, 2).reshape(local.shape[1], -1),
            check_finite=False).reshape(local.shape[1], local.shape[0], local.shape[2]).transpose(1, 0, 2)
    weighted_q = weighted[:, :, shape[2]:].reshape(q.shape)
    weighted_a = weighted[:, :, :shape[2]].reshape(-1, shape[2])
    gram = hermitian(q.conj().T@weighted_q)
    beta = cho_solve(cho_factor(gram, lower=True), q.conj().T@weighted_a)
    value = weighted_a-weighted_q@beta
    information = np.sum(templates.reshape(value.shape).conj()*value, axis=0).real
    if not np.all(np.isfinite(information)) or np.any(information <= 0):
        raise ValueError('No identifiable target amplitude after nuisance elimination')
    return (value/information).reshape(shape), information


def matched_filter(template, nuisance, labels, covariance):
    """Unit complex target response and zero nuisance response, in Fourier space."""
    h, information = matched_filters(np.asarray(template)[:, :, None], nuisance, labels, covariance)
    return h[:, :, 0], float(information[0])


def shrink(covariance, strength):
    if not 0 <= strength <= 1:
        raise ValueError('Shrinkage strength must lie between zero and one')
    return np.eye(covariance.shape[-1])+strength*(covariance-np.eye(covariance.shape[-1]))
