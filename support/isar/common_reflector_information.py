"""Antenna information with common unknown physical reflecting points.

Complex coefficients remain independent per band. Real point coordinates can
be known, common but unknown, or independent by band. The same fitted mean is
used in all three cases. Nested square-root projections preserve their order;
no search bound, motion penalty or antenna prior counts as a measurement.
"""
import numpy as np
from scipy.linalg import block_diag

from geometry_information import real_jacobian
from spline_motion import SplineMotion


def compact_roots(joint, parameters):
    """Eliminate complex coefficients locally, retaining shared point columns."""
    points = joint.bands[0].points
    if any(band.points.shape != points.shape or not np.array_equal(band.points, points)
           for band in joint.bands):
        raise ValueError('Point indices must describe identical physical coordinates across bands')
    mean, _ = real_jacobian(joint, parameters)
    motion = joint.expand(parameters)
    roots, start = [], 0
    for c, (band, response) in enumerate(zip(joint.bands, joint.responses(parameters))):
        design = band.nuisance.remove(response['target'][band.selected])
        u, singular, _ = np.linalg.svd(design, full_matrices=False)
        if singular[-1] <= singular[0]*1e-10:
            raise ValueError('Require the unchanged full-rank complex scene')
        state = SplineMotion({**band.model, 'parameters': motion}).apply(
            band.time, band.prior, band.reference_transmitter, band.receiver)
        offsets = joint.offsets(parameters, c)
        columns = []
        for coefficient, point in zip(response['coefficient'], points):
            _, derivative = band.field.point(point, motion=state, derivatives=True,
                transmitter_offset_ecef_m=offsets[0], receiver_offset_ecef_m=offsets[1],
                range_bias_m=parameters[joint.range_slice.start+c])
            columns.extend((coefficient*derivative[:, :, axis]).ravel()[band.selected]
                           for axis in range(3))
        derivative = band.nuisance.remove(np.column_stack(columns))
        derivative -= u@(u.conj().T@derivative)
        point_columns = np.sqrt(2)*np.r_[derivative.real, derivative.imag]
        stop = start+len(point_columns)
        roots.append(np.linalg.qr(np.column_stack((mean[start:stop], point_columns)), mode='r'))
        start = stop
    if start != len(mean):
        raise ValueError('Physical and point derivative rows do not match')
    return roots


def basis(matrix, cutoff, *, scale=None):
    """Use original column scales after a projection to reject numerical dust."""
    matrix = np.asarray(matrix)
    if not 0 < cutoff < 1 or not np.isfinite(matrix).all():
        raise ValueError('Require finite derivatives and a relative information cutoff')
    if scale is None:
        scale = np.linalg.norm(matrix, axis=0)
    normalized = matrix/np.where(np.asarray(scale) > 0, scale, 1.)
    u, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    keep = singular > max(1., singular.max(initial=0.))*np.sqrt(cutoff)
    return u[:, keep], int(keep.sum())


def nested_information(roots, physical_count, interest, nuisance, *, cutoff):
    """Known >= shared unknown >= independent unknown, at an identical mean."""
    mean = np.concatenate([root[:, :physical_count] for root in roots])
    point_blocks = [root[:, physical_count:] for root in roots]
    if len({block.shape[1] for block in point_blocks}) != 1:
        raise ValueError('Every band must have the same point-coordinate columns')
    shared = np.concatenate(point_blocks)
    independent = block_diag(*point_blocks)
    q, motion_rank = basis(mean[:, nuisance], cutoff)
    target = mean[:, interest]-q@(q.T@mean[:, interest])
    shared -= q@(q.T@shared)
    independent -= q@(q.T@independent)
    known = target.T@target
    independent_scale = np.linalg.norm(independent, axis=0)
    q, common_rank = basis(shared, cutoff)
    target -= q@(q.T@target)
    independent -= q@(q.T@independent)
    common = target.T@target
    q, additional_rank = basis(independent, cutoff, scale=independent_scale)
    target -= q@(q.T@target)
    return {'known': known, 'shared': common, 'independent': target.T@target,
        'motion_and_origin_rank': motion_rank, 'shared_point_rank': common_rank,
        'additional_independent_point_rank': additional_rank, 'cutoff': cutoff}
