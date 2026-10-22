"""Proper-complex Gaussian motion information, including covariance motion.

For known variance and diffuse penalty, the Slepian-Bangs information is
  2 Re(dm^H C^-1 dm) + tr(C^-1 C_i C^-1 C_j).
Unknown deterministic complex core amplitudes are eliminated by the exact
Schur complement in whitened coordinates. The covariance term is assembled
as a positive Gram matrix, not a subtraction of large cancelling traces.
This is an expected information matrix of the declared model, not optimizer
Gauss-Newton curvature or an empirical claim about the recording.
"""
import numpy as np

from marginal_scene import Covariance


def information(design, coefficients, partials, core, strength, *, variance=1.):
    """partials: observations x scene points x real physical parameters."""
    if partials.shape[:2] != design.shape or not 0 < core <= design.shape[1]:
        raise ValueError('Scene derivatives and explicit core must match the design')
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('Declare a finite positive thermal variance')
    mean_partial = np.einsum('niq,i->nq', partials[:, :core], coefficients[:core])
    count = partials.shape[-1]
    covariance_information = np.zeros((count, count))
    covariance = None if strength is None else Covariance(design[:, core:], strength)
    def whiten(value):
        return (value if covariance is None else covariance.apply_inverse(value, square_root=True))/np.sqrt(variance)
    nuisance = whiten(design[:, :core])
    q, s, _ = np.linalg.svd(nuisance, full_matrices=False)
    if s[-1] <= s[0]*1e-12:
        raise ValueError('The deterministic mean basis is rank deficient')
    whitened = whiten(mean_partial)
    efficient = whitened-q@(q.conj().T@whitened)
    mean_information = 2*np.real(efficient.conj().T@efficient)
    if covariance is not None:
        # Hold the right singular basis fixed when differentiating at this
        # physical point. Its derivative is neither needed nor discarded.
        u, s, vh = covariance.u, covariance.s, covariance.vh
        r = s/np.sqrt(strength+s**2)
        inner_features = np.empty((len(s)**2, count), complex)
        outer_features = np.empty((len(design)*len(s), count), complex)
        for j in range(count):
            derivative = (partials[:, core:, j]@vh.conj().T)/np.sqrt(strength)
            inner = u.conj().T@derivative
            outside = derivative-u@inner
            inner /= np.sqrt(1+covariance.ratio)[:, None]
            symmetric = inner*r[None, :]+r[:, None]*inner.conj().T
            inner_features[:, j] = symmetric.ravel()
            outer_features[:, j] = (outside*r).ravel()
        covariance_information = np.real(inner_features.conj().T@inner_features+
            2*(outer_features.conj().T@outer_features))
    elif design.shape[1] != core:
        raise ValueError('Remove diffuse columns in the core-only limit')
    total = mean_information+covariance_information
    return {'mean': mean_information, 'covariance': covariance_information,
        'total': (total+total.T)/2}
