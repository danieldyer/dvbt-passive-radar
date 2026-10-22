"""Short point-hypothesis arcs from immutable vehicle range/phase patches.

The archived carrier reference is an empirical polynomial, not a geographic
position. Preserve its absolute envelope and phase constant explicitly. The
same C0/C1 symbol response as the ship likelihood applies before row means.
"""
import json

import numpy as np

from field import Field, average, FS
from motion import C


def seed_path(seed, time, derivative=0):
    coefficients = np.polynomial.polynomial.polyder(seed['path'], derivative)
    return np.polynomial.polynomial.polyval((np.asarray(time)-seed['epoch'])/seed['scale'], coefficients)/seed['scale']**derivative


def observation(raw, seed, pulse, *, block=16):
    time = np.asarray(raw['row_time'])
    information = np.asarray(raw['row_information'])
    value = np.asarray(raw['row_value'])
    if (time.ndim != 1 or not len(time) or np.any(np.diff(time) <= 0)
            or information.shape != time.shape or np.any(information < 0)
            or value.shape != (len(time), len(raw['offset_samples']))
            or not isinstance(block, int) or block < 1):
        raise ValueError('Ordered row epochs, nonnegative information and matching range vectors are required')
    binned, weight = average(value, information, block)
    epoch, _ = average(time, information, block)
    epoch[weight == 0] = average(time, np.ones_like(time), block)[0][weight == 0]
    metadata = {'rf_hz': seed['rf_hz'], 'block_rows': block,
        'phase_reference': 'exp(+i k (seed_path(t) - seed_path_polynomial_constant)); empirical path, no ECEF centre',
        'phase_reference_offset_m': seed['path'][0],
        'source_seed': seed,
        'covariance': 'Conditional thermal range covariance and original row information; no empirical local noise rescaling'}
    result = {'row_time': time, 'row_information': information, 'time': epoch,
        'value': binned, 'information': weight, 'offset_samples': np.asarray(raw['offset_samples']),
        'centre_path_m': seed_path(seed, time),
        'centre_omega': -2*np.pi*seed['rf_hz']/C*.001024*seed_path(seed, time, 1),
        'pulse_samples': np.asarray(pulse), 'metadata_json': json.dumps(metadata)}
    Field(result, pulse)  # Verify shape and positive-definite range covariance.
    return result


def observables(row, seed, midpoint=None):
    """Absolute [range, rate, acceleration] and their joint covariance.

    This eliminates the unknown constant band reflectivity phase. It does not
    assert a common absolute scattering phase at different RF frequencies.
    """
    time = row['epoch_s'] if midpoint is None else midpoint
    u = (time-row['epoch_s'])/row['scale_s']; scale = row['scale_s']
    if row['degree'] != 2:
        raise ValueError('This observable conversion requires a quadratic short arc')
    transform = np.array([[C/FS, 0., u, u*u], [0., 0., 1/scale, 2*u/scale], [0., 0., 0., 2/scale**2]])
    reference = np.array([seed_path(seed, time, j) for j in range(3)])
    parameters = np.asarray(row['parameters'])
    covariance = np.asarray(row['conditional_parameter_covariance'])
    return reference+transform@parameters, transform@covariance@transform.T
