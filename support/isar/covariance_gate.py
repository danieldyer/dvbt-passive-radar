"""Gaussian sufficient Fourier statistics with heteroscedastic time noise.

For means B a and time covariance D, QR of D^(-1/2) B gives U. Then
U^H D^(-1/2) y has unit noise covariance and retains the full likelihood for
a. No signal or covariance scale is estimated here. The existing Fourier
bandwidth, observation selection and declared covariance are kept unchanged.
"""
import copy

import numpy as np


def whiten_first_field(original):
    result = copy.copy(original)
    keep = original.selected
    variance = original.data['variance'][:, len(original.offsets)//2]
    sigma = np.sqrt(variance[keep])
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError('Positive finite selected-bin variances are required')
    count = len(original.q)
    if count%2 != 1 or sum(keep) < count:
        raise ValueError('Use the existing odd-dimensional Fourier gate')
    harmonics = np.arange(count)-count//2
    time = original.time[keep]-original.seed['epoch']
    basis = np.exp(2j*np.pi*time[:, None]*harmonics[None, :]/original.seed.get('duration_s', 30.))
    u, r = np.linalg.qr(basis/sigma[:, None], mode='reduced')
    if np.min(abs(np.diag(r))) <= 1e-10*np.linalg.norm(r, ord=2):
        raise ValueError('The declared Fourier subspace is rank deficient')
    result.q = np.zeros_like(original.q)
    result.q[:, keep] = u.conj().T/sigma[None, :]
    result.y = result.apply(result.data['value']).ravel()
    return result
