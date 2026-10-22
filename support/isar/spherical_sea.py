"""Specular spherical-water reference for the ship's unresolved four paths.

Exact geometric lengths and endpoint derivatives on a declared fixed sphere;
geometrical-optics divergence and scalar horizontal Fresnel reflection. This
does not model atmospheric bending, rough-surface diffuse scattering, antenna
pattern discrimination or different target responses towards the four rays.
"""
import numpy as np

from field import average


def reflected_leg(position, velocity, antenna, centre, radius):
    position, velocity, antenna, centre = [np.asarray(x, float) for x in (position, velocity, antenna, centre)]
    if (position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape
            or antenna.shape != (3,) or centre.shape != (3,) or not np.isfinite(radius) or radius <= 0
            or any(not np.isfinite(x).all() for x in (position, velocity, antenna, centre))):
        raise ValueError('Finite N by 3 point states, antenna/centre vectors and a positive radius are required')
    a, b = antenna-centre, position-centre
    ra, rb = np.linalg.norm(a), np.linalg.norm(b, axis=1)
    ha, hb = ra-radius, rb-radius
    if ha <= 0 or np.any(hb <= 0):
        raise ValueError('Specular endpoints must be strictly above the declared water sphere')
    ua, ub = a/ra, b/rb[:, None]
    cross = np.cross(ua, ub)
    sine = np.linalg.norm(cross, axis=1)
    cosine = np.clip(ub@ua, -1., 1.)
    angle = np.arctan2(sine, cosine)
    if np.any(sine < 1e-12):
        raise ValueError('Coincident radial directions need a separate normal-incidence limit')
    tangent = (ub-cosine[:, None]*ua)/sine[:, None]
    horizon_a, horizon_b = np.arccos(radius/ra), np.arccos(radius/rb)
    if np.any(angle >= horizon_a+horizon_b):
        raise ValueError('No mutually visible specular point above the spherical horizon')
    lower = np.maximum(0., angle-horizon_b)
    upper = np.minimum(angle, horizon_a)
    phi = np.clip(angle*ha/(ha+hb), lower, upper)
    def terms(x):
        other = angle-x
        la = np.hypot(ha, 2*np.sqrt(ra*radius)*np.sin(x/2))
        lb = np.hypot(hb, 2*np.sqrt(rb*radius)*np.sin(other/2))
        sa = (ha-2*ra*np.sin(x/2)**2)/la
        sb = (hb-2*rb*np.sin(other/2)**2)/lb
        ca, cb = ra*np.sin(x)/la, rb*np.sin(other)/lb
        # Equal grazing sines are equivalent to Fermat stationarity on this
        # visible branch. Unlike subtracting almost-unit grazing cosines, this
        # equation remains well resolved at small grazing angles.
        residual = sa-sb
        derivative = -ca*(1+radius*sa/la)-cb*(1+radius*sb/lb)
        return la, lb, residual, derivative, sa, sb
    for _ in range(32):
        la, lb, residual, derivative, sa, sb = terms(phi)
        converged = abs(residual) < 2e-13
        if np.all(converged):
            break
        lower = np.where(residual > 0, phi, lower)
        upper = np.where(residual <= 0, phi, upper)
        candidate = phi-residual/derivative
        good = (candidate > lower)&(candidate < upper)&np.isfinite(candidate)
        phi = np.where(converged, phi, np.where(good, candidate, (lower+upper)/2))
    la, lb, residual, derivative, sa, sb = terms(phi)
    if np.max(abs(residual)) > 1e-11 or np.any(derivative >= 0):
        raise ValueError('Specular Fermat solve did not converge on the visible convex branch')
    normal = np.cos(phi)[:, None]*ua+np.sin(phi)[:, None]*tangent
    specular = centre+radius*normal
    outbound = (position-specular)/lb[:, None]
    sin_grazing = (sa+sb)/2
    # Stationary phase on a sphere: both principal surface curvatures increase
    # the ray-tube Hessian. The second factor approaches one at grazing angles;
    # retaining it also gives the correct paraxial convex-mirror normal limit.
    distance = la*lb/(la+lb)
    divergence = 1/np.sqrt((1+2*distance/(radius*sin_grazing))*(1+2*distance*sin_grazing/radius))
    return {'length_m': la+lb, 'rate_mps': np.sum(outbound*velocity, axis=1),
        'antenna_segment_m': la, 'target_segment_m': lb,
        'specular_ecef_m': specular, 'sin_grazing': sin_grazing, 'divergence': divergence,
        'specular_sine_residual': abs(residual)}


def horizontal_reflection(sin_grazing, frequency_hz, *, relative_permittivity, conductivity_s_per_m):
    """e^(+i wt) convention, matching propagation exp(-i k L)."""
    if frequency_hz <= 0 or relative_permittivity <= 0 or conductivity_s_per_m < 0:
        raise ValueError('Positive frequency/permittivity and nonnegative conductivity are required')
    sine = np.asarray(sin_grazing)
    if np.any((sine <= 0)|(sine > 1)):
        raise ValueError('Visible grazing angles are required')
    epsilon = relative_permittivity-1j*conductivity_s_per_m/(2*np.pi*frequency_hz*8.8541878128e-12)
    root = np.sqrt(epsilon-1+sine**2)
    return (sine-root)/(sine+root)


def geometry(position, velocity, transmitter, receiver, centre, radius):
    reflected = [reflected_leg(position, velocity, antenna, centre, radius) for antenna in (transmitter, receiver)]
    direct, rates = [], []
    for antenna in (transmitter, receiver):
        vector = position-antenna
        length = np.linalg.norm(vector, axis=1)
        direct.append(length)
        rates.append(np.sum(vector*velocity, axis=1)/length)
    baseline = np.linalg.norm(transmitter-receiver)
    paths, doppler, spreading = [], [], []
    for tx_reflected, rx_reflected in ((False, False), (True, False), (False, True), (True, True)):
        choice = [tx_reflected, rx_reflected]
        lengths = [r['length_m'] if flag else d for r, flag, d in zip(reflected, choice, direct)]
        rate = [r['rate_mps'] if flag else v for r, flag, v in zip(reflected, choice, rates)]
        factor = [r['divergence'] if flag else 1. for r, flag in zip(reflected, choice)]
        paths.append(sum(lengths)-baseline)
        doppler.append(sum(rate))
        spreading.append(direct[0]*direct[1]/(lengths[0]*lengths[1])*factor[0]*factor[1])
    return {'excess_path_m': np.array(paths), 'path_rate_mps': np.array(doppler),
        'relative_spreading': np.array(spreading), 'physical_baseline_m': float(baseline), 'reflected_legs': reflected}


def components(field, state, point_body_m, centre, radius, *, relative_permittivity, conductivity_s_per_m,
          range_bias_m=0., reflection_scale=1.):
    """Four paths through the fixed observation, with explicit water parameters."""
    position = state['position_ecef_m']+state['body_to_ecef']@np.asarray(point_body_m)
    velocity = state['velocity_ecef_mps']+state['frame_velocity']@np.asarray(point_body_m)
    d = field.data
    paths = geometry(position, velocity, d['transmitter_ecef_m'], d['receiver_ecef_m'], centre, radius)
    gt, gr = [reflection_scale*horizontal_reflection(r['sin_grazing'], field.metadata['rf_hz'],
        relative_permittivity=relative_permittivity, conductivity_s_per_m=conductivity_s_per_m)
        for r in paths['reflected_legs']]
    values = []
    for length, rate, spreading, gain in zip(paths['excess_path_m'], paths['path_rate_mps'],
            paths['relative_spreading'], [1., gt, gr, gt*gr]):
        response = field.path_response(length-d['centre_path_m'], -field.k*.001024*rate, range_bias_m=range_bias_m)
        averaged, _ = average(response*(spreading*gain)[:, None], d['row_information'], field.metadata['block_rows'])
        values.append(field.apply(averaged))
    return np.array(values), paths


def point(field, state, point_body_m, centre, radius, **parameters):
    values, paths = components(field, state, point_body_m, centre, radius, **parameters)
    return values.sum(axis=0), paths
