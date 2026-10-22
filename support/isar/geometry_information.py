"""Conditional expected information for the independent-band geometry model.

This uses derivatives of the mean with fitted amplitudes held fixed, then
eliminates complex amplitudes and real position nuisance directions. The
residual-Jacobian Gauss--Newton matrix is deliberately not used as Fisher
information. Real-data evaluations remain conditional on the noise model and
plug-in scene, and are not calibrated location confidence intervals.
"""
import numpy as np

from physical_profile import observe_partials
from profile_information import from_jacobian
from spline_motion import SplineMotion


def real_jacobian(joint, parameters, *, point_coordinates=False, cutoff=1e-10):
    derivatives = []
    point_ranks = []
    motion = joint.expand(parameters)
    for c, (band, transform, response) in enumerate(zip(
            joint.bands, joint.transforms, joint.responses(parameters))):
        offset, origin = joint.offsets(parameters, c), parameters[joint.range_slice.start+c]
        target, partials = band.design(motion, derivatives=True, antenna_offsets=offset,
            range_bias_m=origin, range_derivative=True)
        design = band.nuisance.remove(target[band.selected])
        u, singular, _ = np.linalg.svd(design, full_matrices=False)
        if singular[-1] <= singular[0]*1e-10:
            raise ValueError('Do not silently change the rank of the fitted scene')
        raw = sum(a*d for a, d in zip(response['coefficient'], partials))
        del partials
        derivative = observe_partials(band, raw, True, True)@transform
        derivative -= u@(u.conj().T@derivative)
        real_derivative = np.r_[derivative.real, derivative.imag]
        if point_coordinates:
            curve = SplineMotion({**band.model, 'parameters': motion})
            state = curve.apply(band.time, band.prior, band.reference_transmitter, band.receiver)
            columns = []
            for a, point in zip(response['coefficient'], band.points):
                _, dp = band.field.point(point, motion=state, derivatives=True,
                    transmitter_offset_ecef_m=offset[0], receiver_offset_ecef_m=offset[1], range_bias_m=origin)
                columns.extend((a*dp[:, :, axis]).ravel()[band.selected] for axis in range(3))
            points = band.nuisance.remove(np.column_stack(columns))
            points -= u@(u.conj().T@points)
            real_points = np.r_[points.real, points.imag]
            norm = np.linalg.norm(real_points, axis=0)
            normalised = real_points/np.where(norm > 0, norm, 1.)
            q, s, _ = np.linalg.svd(normalised, full_matrices=False)
            keep = s > s[0]*np.sqrt(cutoff)
            real_derivative -= q[:, keep]@(q[:, keep].T@real_derivative)
            point_ranks.append(int(keep.sum()))
        derivatives.append(np.sqrt(2)*real_derivative)
    return np.concatenate(derivatives), point_ranks


def expected(joint, parameters, *, point_coordinates=False, cutoff=1e-10):
    jacobian, ranks = real_jacobian(joint, parameters, point_coordinates=point_coordinates, cutoff=cutoff)
    return jacobian.T@jacobian, ranks


def antenna_summary(joint, parameters, *, point_coordinates=False, cutoff=1e-10):
    jacobian, point_ranks = real_jacobian(joint, parameters, point_coordinates=point_coordinates, cutoff=cutoff)
    antenna = np.arange(len(parameters))[joint.antenna_slice]
    curve = SplineMotion({**joint.bands[0].model, 'parameters': joint.expand(parameters)})
    nuisance = np.r_[np.arange(joint.nmotion), np.arange(len(parameters))[joint.range_slice]]
    if joint.bands[0].model.get('fixed_ne_frame'):
        yaw = (joint.active >= curve.sections[1].start)&(joint.active < curve.sections[1].stop)
        nuisance = np.setdiff1d(nuisance, np.flatnonzero(yaw))
    marginal, ranks = from_jacobian(jacobian, antenna, nuisance, cutoff=cutoff)
    eigen, vectors = np.linalg.eigh(marginal)
    covariance = (vectors/(1+np.maximum(eigen, 0)))@vectors.T
    factor = joint.antennas.factor
    posterior = factor@covariance@factor.T
    return {'point_coordinates_profiled': point_coordinates, 'point_coordinate_ranks_by_band': point_ranks,
        'antenna_information_eigenvalues_before_prior': eigen.tolist(),
        'directions_weaker_than_prior': int(np.sum(eigen < 1)),
        'conditional_posterior_sigma_enu_m': np.sqrt(np.maximum(np.diag(posterior), 0)).reshape(-1, 3).tolist(),
        'prior_sigma_enu_m': np.sqrt(np.diag(factor@factor.T)).reshape(-1, 3).tolist(),
        'motion_and_range_elimination': ranks,
        'scope': __doc__+' No hard search-bound information is counted. Independent three-dimensional point positions per band are eliminated only when requested.'}
