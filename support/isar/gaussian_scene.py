"""Proper-complex Gaussian scene marginal likelihood and posterior prediction.

y = A a + n, a ~ CN(0, variance I), n ~ CN(0, I). One scene variance is
estimated per band using training data only. The determinant term is retained;
this is not just a fitted ridge residual. No singular scene modes are removed.
The scalar variance search examines all sign-bracketed minima on a log grid;
it does not certify global physical motion or correctness of the scene prior.
"""
import numpy as np
from scipy.optimize import brentq


def variance_fit(singular, projection):
    if singular[0] == 0:
        return 0., {'variance_stationarity': 0., 'candidate_minima': 1, 'zero_scene': True}
    ratio, power = (singular/singular[0])**2, abs(projection)**2
    upper = np.max(np.divide(np.maximum(power-1, 0), ratio,
        out=np.zeros_like(power), where=ratio > 0), initial=0.)
    if upper == 0:
        return 0., {'variance_stationarity': 0., 'candidate_minima': 1, 'zero_scene': True}
    def terms(log_strength):
        d = np.exp(log_strength)*ratio
        shrink = d/(1+d)
        return np.sum(np.log1p(d)-shrink*power), np.sum(shrink*(1-power/(1+d)))
    high = np.log(upper)
    low = min(-32., high-32.)
    grid = np.linspace(low, high, int(np.ceil((high-low)/.25))+1)
    gradients = np.array([terms(x)[1] for x in grid])
    candidates = [(0., 0., 0.)]
    for lo, hi, a, b in zip(grid[:-1], grid[1:], gradients[:-1], gradients[1:]):
        if a < 0 <= b:
            at = brentq(lambda x: terms(x)[1], lo, hi, xtol=1e-12)
            value, derivative = terms(at)
            candidates.append((value, np.exp(at)/singular[0]**2, derivative))
    value, derivative = terms(high)
    candidates.append((value, upper/singular[0]**2, derivative))
    _, variance, gradient = min(candidates, key=lambda v: v[0])
    return float(variance), {'variance_stationarity': float(gradient),
        'candidate_minima': len(candidates), 'zero_scene': bool(variance == 0),
        'upper_variance_bound': float(upper/singular[0]**2)}


def marginal(design, observation, *, variance=None):
    design, observation = np.asarray(design), np.asarray(observation)
    if design.ndim != 2 or observation.shape != (len(design),) or design.shape[0] < design.shape[1]:
        raise ValueError('Need at least as many observations as declared scene coefficients')
    if not np.isfinite(design).all() or not np.isfinite(observation).all():
        raise ValueError('Finite complex observations and templates are required')
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    projection = u.conj().T@observation
    fitted = None
    if variance is None:
        variance, fitted = variance_fit(singular, projection)
    if not np.isfinite(variance) or variance < 0:
        raise ValueError('The scene variance must be finite and nonnegative')
    d = variance*singular**2
    shrink = d/(1+d)
    gain = variance*singular/(1+d)
    coefficient = vh.conj().T@(gain*projection)
    # Near-coincident point columns can require large cancelling coefficients.
    # Evaluate C^-1 y in the orthogonal basis, retaining tiny attenuated modes
    # instead of subtracting those coefficients or forming 1-shrink.
    outside = observation-u@projection
    residual = outside+u@(projection/(1+d))
    covariance = (vh.conj().T*(variance/(1+d)))@vh
    # dL = 2 Re trace(Z^H dA), including covariance determinant and scene prior.
    z = (u*gain)@vh-np.outer(residual, coefficient.conj())
    value = float(np.vdot(outside, outside).real+np.sum(abs(projection)**2/(1+d))+np.log1p(d).sum())
    return {'value': value, 'coefficient': coefficient, 'residual': residual,
        'covariance': covariance, 'design_gradient': z, 'basis': u, 'shrink': shrink,
        'singular_values': singular, 'right_basis': vh,
        'variance': float(variance), 'variance_fit': fitted, 'effective_scene_rank': float(shrink.sum()),
        'condition': float(singular[0]/max(singular[-1], 1e-300))}


def motion_information(result, derivatives, *, profile_variance=True):
    """Proper-complex covariance Fisher information, without an N by N matrix.

    Derivatives have shape (observations, scene coefficients, real parameters).
    The square-root representation retains the covariance-change term. A
    training-estimated log variance is eliminated as a nuisance direction.
    This information is conditional on the stated random-scene/noise model.
    """
    u, shrink, vh = result['basis'], result['shrink'], result['right_basis']
    variance = result['variance']
    n, p, q = derivatives.shape
    if u.shape != (n, p):
        raise ValueError('Scene derivatives disagree with the marginal model')
    if variance == 0:
        return np.zeros((q, q))
    projection = (u.conj().T@derivatives.reshape(n, -1)).reshape(p, p, q)
    # C^(-1/2) sqrt(v) A = U R. Split each covariance derivative into
    # its Hermitian U/U block and the two conjugate U/perpendicular blocks.
    root = np.sqrt(variance)
    r = np.sqrt(shrink)[:, None]*vh
    local = projection*(root/np.sqrt(1+variance*result['singular_values']**2))[:, None, None]
    inside = np.einsum('apq,bp->abq', local, r.conj())
    inside += inside.conj().transpose(1, 0, 2).copy()
    outside = derivatives-(u@projection.reshape(p, -1)).reshape(n, p, q)
    outside = root*np.einsum('npq,ap->naq', outside, r.conj())
    if profile_variance:
        # The whitened log-variance derivative is U diag(shrink) U^H.
        diagonal = np.arange(p)
        cross = np.real(shrink@inside[diagonal, diagonal])
        inside[diagonal, diagonal] -= shrink[:, None]*cross[None, :]/(shrink@shrink)
    a, b = inside.reshape(-1, q), outside.reshape(-1, q)
    return np.real(a.conj().T@a+2*b.conj().T@b)
