"""Exact common phase for independent nonnegative complex-tone amplitudes.

For correlations b_c divided by the square root of column energy, maximize
sum_c max(0, Re(exp(-i phi) b_c))**2. The active set changes only at the 2C
quadrature crossings. On each interval the objective is a constant plus one
second harmonic, so the endpoints and its stationary phases contain a global
maximum. No phase grid, angle unwrap or local line search is needed.
"""
import numpy as np


def maximum(correlations):
    correlations = np.asarray(correlations, complex)
    if correlations.ndim != 1 or not len(correlations) or not np.isfinite(correlations).all():
        raise ValueError('Expected finite, nonempty complex correlations')
    angle = np.angle(correlations)
    boundaries = np.sort(np.mod(np.r_[angle+np.pi/2, angle+3*np.pi/2], 2*np.pi))
    middle = (boundaries+np.r_[boundaries[1:], boundaries[0]+2*np.pi])/2
    active = np.real(np.exp(-1j*middle[:, None])*correlations) > 0
    stationary = np.angle(active@(correlations**2))/2
    candidates = np.r_[boundaries, stationary, stationary+np.pi]
    magnitudes = np.maximum(0., np.real(np.exp(-1j*candidates[:, None])*correlations))
    index = int(np.argmax(np.sum(magnitudes**2, axis=1)))
    return float(candidates[index]), magnitudes[index]
