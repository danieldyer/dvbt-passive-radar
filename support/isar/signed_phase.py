"""Exact shared-phase update when each band has an unrestricted real strength.

For normalized correlations b, profiling the real strengths maximizes
sum Re(exp(-i phase) b)**2. Its phase is half arg(sum b**2), modulo pi.
This admits sign-changing scattering but cannot identify the pi branch of
either a point phase or a band calibration phase.
"""
import numpy as np


def maximum(correlation):
    correlation = np.asarray(correlation, complex)
    if correlation.ndim != 1 or not np.isfinite(correlation).all():
        raise ValueError('Finite one-dimensional correlations are required')
    phase = .5*np.angle(np.sum(correlation**2))
    return phase, np.real(np.exp(-1j*phase)*correlation)
