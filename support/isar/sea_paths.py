"""Numerical reference for a scalar point above one fixed reflecting plane.

This is a diagnostic hypothesis, not an adopted ship scattering correction.
The target has the same scalar response towards the four closely spaced rays.
Reflection coefficients are supplied explicitly; roughness, polarization,
curved water and refraction are not inferred or silently approximated here.
The reference baseline is always the physical transmitter-receiver distance.
"""
import numpy as np

from field import average


def geometry(position, velocity, transmitter, receiver, origin, normal):
    """Four echo paths in order direct, reflected Tx, reflected Rx, both."""
    position, velocity = np.asarray(position), np.asarray(velocity)
    transmitter, receiver, origin, normal = [np.asarray(v, float) for v in (transmitter, receiver, origin, normal)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (transmitter, receiver, origin, normal)):
        raise ValueError('Plane and antenna coordinates must be finite three-vectors')
    if not np.isclose(np.linalg.norm(normal), 1., atol=1e-12, rtol=0.):
        raise ValueError('The fixed reflecting plane needs a unit normal')
    if (position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape
            or not len(position) or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
        raise ValueError('Point positions and velocities must have matching N by 3 shapes')
    height = (position-origin)@normal
    if np.min(height) < -1e-8 or (transmitter-origin)@normal <= 0 or (receiver-origin)@normal <= 0:
        raise ValueError('The point and physical antennas must be on or above the reflecting plane')
    images = [antenna-2*normal*((antenna-origin)@normal) for antenna in (transmitter, receiver)]
    path, rate, spreading = [], [], []
    baseline = np.linalg.norm(transmitter-receiver)
    direct_product = np.linalg.norm(position-transmitter, axis=1)*np.linalg.norm(position-receiver, axis=1)
    for tx, rx in ((transmitter, receiver), (images[0], receiver), (transmitter, images[1]), tuple(images)):
        legs = [position-tx, position-rx]
        lengths = [np.linalg.norm(leg, axis=1) for leg in legs]
        if any(np.any(distance == 0) for distance in lengths):
            raise ValueError('A point cannot coincide with an antenna or its image')
        path.append(sum(lengths)-baseline)
        rate.append(sum(np.sum(leg*velocity, axis=1)/length for leg, length in zip(legs, lengths)))
        spreading.append(direct_product/(lengths[0]*lengths[1]))
    return {'excess_path_m': np.array(path), 'path_rate_mps': np.array(rate),
        'relative_spreading': np.array(spreading), 'physical_baseline_m': float(baseline)}


def components(field, state, point_body_m, origin, normal, *, range_bias_m=0.):
    """Each ray through unchanged range, C0/C1, row averaging and whitening."""
    point = np.asarray(point_body_m)
    position = state['position_ecef_m']+state['body_to_ecef']@point
    velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
    d = field.data
    paths = geometry(position, velocity, d['transmitter_ecef_m'], d['receiver_ecef_m'], origin, normal)
    values = []
    for length, rate, spreading in zip(paths['excess_path_m'], paths['path_rate_mps'], paths['relative_spreading']):
        raw = field.path_response(length-d['centre_path_m'], -field.k*.001024*rate, range_bias_m=range_bias_m)
        observed, _ = average(raw*spreading[:, None], d['row_information'], field.metadata['block_rows'])
        values.append(field.apply(observed))
    return np.array(values), paths


def point(field, state, point_body_m, origin, normal, *, tx_reflection, rx_reflection, range_bias_m=0.):
    values, paths = components(field, state, point_body_m, origin, normal, range_bias_m=range_bias_m)
    coefficients = np.asarray([1., tx_reflection, rx_reflection, tx_reflection*rx_reflection], complex)
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise ValueError('Supply two finite scalar reflection coefficients explicitly')
    return np.einsum('p,pnr->nr', coefficients, values), paths
