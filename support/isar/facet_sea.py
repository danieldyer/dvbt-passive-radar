"""Conditional HH physical-optics rectangle above a smooth water sphere.

Each surface element has four geometric paths. Its illumination and receive
directions, PEC current, spreading, spherical divergence and complex Fresnel
coefficients are evaluated separately for each path before integration. This
is an infinite-water sensitivity model: it does not assert that a reflecting
footprint is water or unobstructed, and it does not model roughness/diffraction.
"""
import numpy as np

from motion import C, path_difference
from specular_facet import hh_current, quadrature, unit
from spherical_sea import horizontal_reflection, reflected_leg


def response(centres, transmitter, receiver, along, vertical, length_m, height_m,
             frequency_hz, sphere_centre, sphere_radius, *, up=(0., 0., 1.),
             relative_permittivity=80., conductivity_s_per_m=4., order=(64, 12)):
    """Return four area-normalized fields, shape (path, time, frequency).

    The direct path at the rectangle centre is the common phase reference.
    All paths use the same direct centre current/spreading normalization.
    Path order: direct, sea on transmit leg, sea on receive leg, both legs.
    """
    centres = np.atleast_2d(np.asarray(centres, float))
    tx, rx = np.asarray(transmitter, float), np.asarray(receiver, float)
    along = np.broadcast_to(along, centres.shape)
    vertical = np.broadcast_to(vertical, centres.shape)
    normal = np.cross(along, vertical)
    frequency = np.atleast_1d(frequency_hz)
    if (centres.shape[1] != 3 or min(length_m, height_m) <= 0
            or np.any(frequency <= 0) or min(order) < 2
            or not np.allclose(np.sum(along*vertical, axis=-1), 0., atol=1e-12)
            or not np.allclose(np.linalg.norm(along, axis=-1), 1., atol=1e-12)
            or not np.allclose(np.linalg.norm(vertical, axis=-1), 1., atol=1e-12)):
        raise ValueError('Positive dimensions/frequencies and orthonormal surface axes are required')
    ct, cr = centres-tx, rx-centres
    if np.any(np.sum(ct*normal, axis=-1) >= 0) or np.any(np.sum(cr*normal, axis=-1) <= 0):
        raise ValueError('The surface front must be illuminated and visible')
    centre_current = hh_current(unit(ct), unit(cr), normal, up)
    if np.any(abs(centre_current) < 1e-10):
        raise ValueError('Direct centre current normalization is undefined')
    u, wu = quadrature(order[0]); v, wv = quadrature(order[1])
    coordinates = np.array(np.meshgrid(u*length_m/2, v*height_m/2, indexing='ij')).reshape(2, -1).T
    weights = (wu[:, None]*wv[None, :]).ravel()/4
    offsets = along[:, None]*coordinates[None, :, 0, None]+vertical[:, None]*coordinates[None, :, 1, None]
    points = centres[:, None]+offsets
    ntime, npoint = points.shape[:2]
    direct = [np.linalg.norm(points-antenna, axis=-1) for antenna in (tx, rx)]
    rays = [reflected_leg(points.reshape(-1, 3), np.zeros((ntime*npoint, 3)), antenna,
                         sphere_centre, sphere_radius) for antenna in (tx, rx)]
    extra = [ray['length_m'].reshape(ntime, npoint)-distance for ray, distance in zip(rays, direct)]
    reflection = [np.stack([horizontal_reflection(ray['sin_grazing'], f,
        relative_permittivity=relative_permittivity, conductivity_s_per_m=conductivity_s_per_m)
        for f in frequency], axis=-1).reshape(ntime, npoint, -1) for ray in rays]
    centre_scale = np.linalg.norm(ct, axis=-1)*np.linalg.norm(cr, axis=-1)/centre_current
    direct_delta = path_difference(centres[:, None], offsets, tx, rx)
    fields = []
    for tx_sea, rx_sea in ((False, False), (True, False), (False, True), (True, True)):
        delta = direct_delta.copy()
        factor = np.broadcast_to(weights[None, :, None]*centre_scale[:, None, None],
                                 (ntime, npoint, len(frequency))).astype(complex).copy()
        directions = []
        for leg, (antenna, reflected) in enumerate(zip((tx, rx), (tx_sea, rx_sea))):
            ray = rays[leg]
            if reflected:
                delta += extra[leg]
                endpoint = ray['specular_ecef_m'].reshape(ntime, npoint, 3)
                factor *= reflection[leg]*ray['divergence'].reshape(ntime, npoint, 1)
                distance = ray['length_m'].reshape(ntime, npoint)
            else:
                endpoint, distance = antenna, direct[leg]
            factor /= distance[:, :, None]
            directions.append(unit(points-endpoint))
        factor *= hh_current(directions[0], -directions[1], normal[:, None], up)[:, :, None]
        fields.append(np.sum(factor*np.exp(-2j*np.pi*frequency/C*delta[:, :, None]), axis=1))
    return np.array(fields)
