"""Double-precision reference geometry for rigid-target bistatic imaging."""
import numpy as np

C = 299792458.
WGS84_A = 6378137.
WGS84_E2 = 6.6943799901413165e-3


def ecef(latitude, longitude, height):
    """Latitude/longitude in radians; ellipsoidal height in metres."""
    latitude, longitude, height = np.broadcast_arrays(latitude, longitude, height)
    n = WGS84_A/np.sqrt(1-WGS84_E2*np.sin(latitude)**2)
    return np.stack([(n+height)*np.cos(latitude)*np.cos(longitude),
                     (n+height)*np.cos(latitude)*np.sin(longitude),
                     (n*(1-WGS84_E2)+height)*np.sin(latitude)], axis=-1)


def enu_basis(latitude, longitude):
    """Columns are east, north, up expressed in ECEF."""
    latitude, longitude = np.broadcast_arrays(latitude, longitude)
    sl, cl = np.sin(longitude), np.cos(longitude)
    sp, cp = np.sin(latitude), np.cos(latitude)
    zero = np.zeros_like(sp)
    return np.stack([np.stack([-sl, cl, zero], axis=-1),
                     np.stack([-sp*cl, -sp*sl, cp], axis=-1),
                     np.stack([cp*cl, cp*sl, sp], axis=-1)], axis=-1)


def body_basis(latitude, longitude, heading):
    """Right-handed forward/port/up frame; heading clockwise from true north.

    This zero-roll/zero-pitch frame is an explicit initial attitude hypothesis.
    Course over ground is not a substitute for the true heading argument.
    """
    if not np.all(np.isfinite(heading)):
        raise ValueError('True heading or an explicit attitude hypothesis is required')
    s, c = np.sin(heading), np.cos(heading)
    zero, one = np.zeros_like(s), np.ones_like(s)
    body_to_enu = np.stack([np.stack([s, c, zero], axis=-1),
                            np.stack([-c, s, zero], axis=-1),
                            np.stack([zero, zero, one], axis=-1)], axis=-1)
    return enu_basis(latitude, longitude)@body_to_enu


def excess_path(point, transmitter, receiver):
    tx, rx = point-transmitter, point-receiver
    a, b = np.linalg.norm(tx, axis=-1), np.linalg.norm(rx, axis=-1)
    if np.any(a == 0) or np.any(b == 0):
        raise ValueError('Scatterer coincides with a phase centre')
    path = a+b-np.linalg.norm(transmitter-receiver, axis=-1)
    gradient = tx/a[..., None]+rx/b[..., None]
    return path, gradient


def path_difference(centre, offset, transmitter, receiver):
    """Stable exact L(centre+offset)-L(centre); no far-field approximation."""
    result = np.zeros(np.broadcast_shapes(centre.shape[:-1], offset.shape[:-1]))
    for phase_centre in (transmitter, receiver):
        vector = centre-phase_centre
        numerator = 2*np.sum(vector*offset, axis=-1)+np.sum(offset*offset, axis=-1)
        result += numerator/(np.linalg.norm(vector+offset, axis=-1)+np.linalg.norm(vector, axis=-1))
    return result
