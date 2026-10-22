"""Gaussian quadrature for a smooth forward model, never for recorded data.

Each retained bin keeps its original observation and whitening. Its original
nonnegative row weights define a discrete measure; quadrature matches its first
2*n moments. Archive geometry is interpolated at those nodes. The resulting
model must still pass an original-row score check at the required precision.
No observation, motion coordinate or channel phase is resampled by this tool.
"""
import json

import numpy as np
from scipy.interpolate import CubicSpline

from field import Field


def gaussian_nodes(time, weight, count):
    """Discrete-measure Gauss rule, padded with zero weights after exact rank."""
    time, weight = np.asarray(time), np.asarray(weight)
    if (time.ndim != 1 or weight.shape != time.shape or not len(time)
            or not np.isfinite(time).all() or not np.isfinite(weight).all()
            or np.any(weight < 0) or not isinstance(count, int) or count < 1):
        raise ValueError('Finite epochs, nonnegative row weights and a positive node count are required')
    total = float(weight.sum())
    measure = weight/total if total else np.full(len(time), 1/len(time))
    origin = time[0]
    centre = float(measure@(time-origin))
    span = max(float(np.max(abs(time-origin-centre))), np.finfo(float).tiny)
    x = (time-origin-centre)/span
    q, old = np.ones(len(time)), []
    diagonal, off_diagonal = [], []
    for k in range(min(count, len(time))):
        diagonal.append(float(measure@(x*q*q)))
        residual = (x-diagonal[-1])*q
        if old:
            residual -= off_diagonal[-1]*old[-1]
        # Reorthogonalization makes tiny/rank-deficient discrete measures safe.
        for v in [*old, q]:
            residual -= v*(measure@(v*residual))
        norm = float(np.sqrt(measure@(residual*residual)))
        if norm <= 64*np.finfo(float).eps or k+1 == min(count, len(time)):
            break
        old.append(q)
        off_diagonal.append(norm)
        q = residual/norm
    matrix = np.diag(diagonal)+np.diag(off_diagonal, 1)+np.diag(off_diagonal, -1)
    nodes, vectors = np.linalg.eigh(matrix)
    weights = total*vectors[0]**2
    epochs = origin+centre+span*nodes
    # Preserve an exactly zero-weight tail while keeping the fixed bin stride.
    return np.pad(epochs, (0, count-len(epochs)), mode='edge'), np.pad(weights, (0, count-len(weights)))


def quadrature(field, count=2):
    """Evaluate a forward template at count nodes per original retained bin."""
    if count < 1 or count > field.metadata['block_rows']:
        raise ValueError('Use between one and the original number of rows per bin')
    d, data = field.data, dict(field.data)
    original, weight, block = d['row_time'], d['row_information'], field.metadata['block_rows']
    rules = [gaussian_nodes(original[i:i+block], weight[i:i+block], count)
             for i in range(0, len(original), block)]
    epochs, weights = [np.concatenate([rule[j] for rule in rules]) for j in range(2)]
    for key in ('centre_ecef_m', 'body_to_ecef', 'centre_path_m',
                'centre_velocity_ecef_mps', 'centre_omega'):
        value = d[key]
        # Subtract the constant before interpolating Earth-sized coordinates.
        data[key] = value[0]+CubicSpline(original, value-value[0], axis=0, extrapolate=False)(epochs)
    u, _, vh = np.linalg.svd(data['body_to_ecef'])
    data['body_to_ecef'] = u@vh
    data['row_time'], data['row_information'] = epochs, weights
    data['metadata_json'] = json.dumps({**field.metadata, 'block_rows': count,
        'model_approximation': __doc__, 'source_model_block_rows': block,
        'quadrature_nodes_per_bin': count})
    result = Field(data, field.pulse.samples)
    np.testing.assert_array_equal(result.y, field.y)
    np.testing.assert_allclose(weights.reshape(-1, count).sum(axis=1), d['information'], rtol=3e-14, atol=1e-14)
    return result
