"""Finite causal scattering responses through the recorded waveform operator.

A tap adds a constant nonnegative excess path. Its envelope and RF carrier
phase change together, while its path rate remains that of the physical point.
The complex tap amplitudes may interfere and therefore allow band-dependent
strength and phase without assigning a separate response to every band.
Weights can represent distinct paths or a quadrature of a declared causal
power-delay density. A uniform density is a testable Gaussian prior, not a
claim that the measured ship contains uniformly distributed internal paths.
"""
import numpy as np


def uniform_delays(width_m, nodes):
    """Gauss quadrature for unit power uniformly over [0, width] excess metres."""
    if not np.isfinite(width_m) or width_m < 0 or not isinstance(nodes, int) or nodes < 1:
        raise ValueError('Need a nonnegative finite delay width and a positive node count')
    if width_m == 0:
        return np.array([0.]), np.array([1.])
    x, w = np.polynomial.legendre.leggauss(nodes)
    return (x+1)*width_m/2, w/2


def response(band, motion, points, delays_m, *, weights=None, range_origin_m=None,
             antenna_offsets=None):
    delays = np.asarray(delays_m, float)
    if (delays.ndim != 1 or not len(delays) or not np.isfinite(delays).all()
            or np.any(delays < 0) or np.any(np.diff(delays) <= 0)):
        raise ValueError('Need distinct increasing nonnegative scattering delays in excess metres')
    origin = band.range_bias if range_origin_m is None else range_origin_m
    weights = np.full(len(delays), 1/len(delays)) if weights is None else np.asarray(weights)
    if (weights.shape != delays.shape or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('Positive delay weights must sum to one')
    values = [band.design(motion, points=points, range_bias_m=origin+d,
                          antenna_offsets=antenna_offsets)[0]*
              np.exp(-1j*band.field.k*d)*np.sqrt(w) for d, w in zip(delays, weights)]
    return np.column_stack(values)
