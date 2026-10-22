"""Lossless fixed-template likelihood compression, including outside energy."""
import numpy as np


def compress(design, observations):
    """QR sufficient statistics without discarding any singular direction."""
    design, observations = np.asarray(design), np.asarray(observations)
    if design.ndim != 2 or observations.shape != (len(design),) or len(design) < design.shape[1]:
        raise ValueError('Need one observation per row and at least as many rows as columns')
    q, r = np.linalg.qr(design, mode='reduced')
    z = q.conj().T@observations
    outside = float(np.linalg.norm(observations-q@z)**2)
    return {'design': r, 'data': z, 'outside': outside, 'observations': len(observations)}
