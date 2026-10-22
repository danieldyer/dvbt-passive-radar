"""Independent scalar finite-facet control for the ship's bistatic geometry.

This is direct scattering from a coherent material line, not multipath. Exact
spherical two-leg paths are integrated before range compression. Reflectivity
is constant in material coordinates; no fitted phase history is present.
The sinc pulse is an ideal rectangular occupied spectrum, deliberately
independent of the native C0/C1/BlockPulse operator used by the real-data fit.
It is a physical diagnostic, not a full electromagnetic hull simulator.
"""
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

C = 299792458.


@dataclass
class Facet:
    position: np.ndarray
    velocity: np.ndarray
    tangent: np.ndarray
    spin_radps: float
    length_m: float

    def __post_init__(self):
        self.position, self.velocity, self.tangent = (
            np.asarray(a, float) for a in (self.position, self.velocity, self.tangent))
        if any(a.shape != (3,) or not np.isfinite(a).all()
               for a in (self.position, self.velocity, self.tangent)):
            raise ValueError('Finite three-dimensional material coordinates required')
        if (abs(self.tangent[2]) > 1e-12 or
                not np.isclose(np.linalg.norm(self.tangent), 1.) or
                not np.isfinite(self.spin_radps) or
                not np.isfinite(self.length_m) or self.length_m < 0):
            raise ValueError('Use a unit horizontal tangent, finite spin and nonnegative length')

    def frame(self, time):
        time = np.atleast_1d(time)
        angle = self.spin_radps*time
        c, s = np.cos(angle), np.sin(angle)
        e, n, _ = self.tangent
        return np.c_[c*e-s*n, s*e+c*n, np.zeros(len(time))]

    def points(self, time, coordinates):
        time, q = np.atleast_1d(time), np.atleast_1d(coordinates)
        centre = self.position+time[:, None]*self.velocity
        return centre[:, None, :]+self.frame(time)[:, None, :]*q[None, :, None]


def path(position, transmitter, receiver):
    return (np.linalg.norm(position-transmitter, axis=-1)+
            np.linalg.norm(position-receiver, axis=-1)-
            np.linalg.norm(np.asarray(transmitter)-receiver))


def geometry(position, transmitter, receiver):
    position = np.asarray(position)
    gradient = np.zeros_like(position)
    hessian = np.zeros(position.shape[:-1]+(3, 3))
    for antenna in (transmitter, receiver):
        vector = position-antenna
        distance = np.linalg.norm(vector, axis=-1)
        unit = vector/distance[..., None]
        gradient += unit
        hessian += (np.eye(3)-unit[..., :, None]*unit[..., None, :])/distance[..., None, None]
    return gradient, hessian


def local_signature(facet, transmitter, receiver):
    """Quadratic stationary-phase quantities at the material origin, t=0.

    L(q,t) = L(0,t) + a(t)q + H(t)q²/2. The expression -a'²/H is
    the additional glint-path acceleration at a=0 with locally constant H.
    For a nonzero a, use the exact stationary point/history for comparison.
    """
    g, h = geometry(facet.position, transmitter, receiver)
    u = facet.tangent
    du = facet.spin_radps*np.array([-u[1], u[0], 0.])
    a, curvature = float(g@u), float(u@h@u)
    a_rate = float(g@du+u@h@facet.velocity)
    return dict(tangent_gradient=a, curvature_per_m=curvature,
        tangent_gradient_rate_per_s=a_rate, glint_coordinate_m=-a/curvature,
        glint_speed_mps=-a_rate/curvature,
        glint_extra_acceleration_at_specular_mps2=-a_rate*a_rate/curvature,
        material_origin_acceleration_mps2=float(facet.velocity@h@facet.velocity),
        geometric_visibility_s=facet.length_m*curvature/abs(a_rate)
            if a_rate else None)


def stationary_coordinates(facet, time, transmitter, receiver):
    """Unconstrained stationary point; points outside the finite line stay flagged."""
    time = np.atleast_1d(time)
    centre = facet.points(time, [0.])[:, 0]
    u = facet.frame(time)
    g, h = geometry(centre, transmitter, receiver)
    q = -np.sum(g*u, axis=-1)/np.einsum('ni,nij,nj->n', u, h, u)
    for _ in range(6):
        g, h = geometry(centre+q[:, None]*u, transmitter, receiver)
        q -= np.sum(g*u, axis=-1)/np.einsum('ni,nij,nj->n', u, h, u)
    return q


def field(facet, time, range_offsets_m, transmitter, receiver, rf_hz, *,
          bandwidth_hz=6.65625e6, quadrature=128):
    """Compressed complex profiles relative to each material-origin path.

    Quadrature weights sum to one: total facet amplitude is the convention,
    rather than amplitude per metre. This changes no phase or image shape.
    Both the range envelope and RF phase use the identical exact optical path.
    """
    time, offsets = np.atleast_1d(time), np.asarray(range_offsets_m)
    if offsets.ndim != 1 or not len(offsets) or np.any(np.diff(offsets) <= 0):
        raise ValueError('Range coordinates must be a strictly increasing vector')
    if quadrature < 2 or rf_hz <= 0 or bandwidth_hz <= 0:
        raise ValueError('Positive RF/bandwidth and at least two quadrature points required')
    q, weight = leggauss(quadrature)
    q, weight = q*facet.length_m/2, weight/2
    centre_path = path(facet.points(time, [0.])[:, 0], transmitter, receiver)
    result = np.empty((len(time), len(offsets)), complex)
    k = 2*np.pi*rf_hz/C
    for first in range(0, len(time), 64):
        use = slice(first, first+64)
        difference = path(facet.points(time[use], q), transmitter, receiver)-centre_path[use, None]
        pulse = np.sinc((offsets[None, :, None]-difference[:, None, :])*bandwidth_hz/C)
        result[use] = np.einsum('nrq,nq,q->nr', pulse, np.exp(-1j*k*difference), weight)
    return result


def backproject(facet, time, offsets_m, profiles, pixels_enu_m, transmitters,
                receiver, rf_hz, *, flatten_glint=False, spin_radps=None):
    """Independent exact-path adjoint, on fixed midpoint east/north image axes.

    flatten_glint is an intentionally wrong control: force the integrated
    flash phase to a point-origin phase before applying the material-motion
    adjoint. It is never used to process observed radar data.
    """
    time, offsets, pixels = np.asarray(time), np.asarray(offsets_m), np.asarray(pixels_enu_m)
    tangent = facet.frame(time)
    # Convert the material frame's angle change, not its nominal hull heading,
    # to a rotation of the east/north pixel coordinates.
    angle = (facet.spin_radps if spin_radps is None else spin_radps)*time
    centre = facet.position+time[:, None]*facet.velocity
    images = []
    for tx, rf, data in zip(transmitters, rf_hz, profiles):
        origin_path = path(centre, tx, receiver)
        k = 2*np.pi*rf/C
        correction = np.ones(len(time), complex)
        if flatten_glint:
            q = stationary_coordinates(facet, time, tx, receiver)
            glint_path = path(centre+q[:, None]*tangent, tx, receiver)
            correction = np.exp(1j*k*(glint_path-origin_path))
        output = np.zeros(len(pixels), complex)
        for first in range(0, len(time), 32):
            use = slice(first, first+32)
            c, s = np.cos(angle[use, None]), np.sin(angle[use, None])
            e, n, z = pixels.T
            moved = np.stack([c*e-s*n, s*e+c*n, np.broadcast_to(z, (len(c), len(z)))], axis=-1)
            difference = path(centre[use, None, :]+moved, tx, receiver)-origin_path[use, None]
            coordinate = (difference-offsets[0])/(offsets[1]-offsets[0])
            index = np.floor(coordinate).astype(int)
            valid = (index >= 0)&(index+1 < len(offsets))
            index = np.clip(index, 0, len(offsets)-2)
            fraction = coordinate-index
            row = np.arange(len(index))[:, None]
            values = data[use][row, index]*(1-fraction)+data[use][row, index+1]*fraction
            output += np.sum(np.where(valid, values, 0)*correction[use, None]*np.exp(1j*k*difference), axis=0)
        images.append(output/len(time))
    return np.asarray(images)
