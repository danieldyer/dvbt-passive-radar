"""Translation-only physical motion in a fixed geographic east/north frame.

Changing coordinates preserves the initial physical reflector locations at
the aperture midpoint and the centre trajectory at every time. It removes
time-varying attitude, rather than rotating a formed image afterwards.
"""
import copy

import numpy as np

from spline_motion import SplineMotion, basis
from imaging import motion_state
from ais import utc_seconds


def restrict_interval(model, interval):
    """Project an initial curve onto a shorter aperture, before likelihood fitting."""
    old = SplineMotion(model)
    if interval[0] < old.interval[0] or interval[1] > old.interval[1] or interval[0] >= interval[1]:
        raise ValueError('Requested aperture is outside the source motion')
    if abs(np.mean(interval)-np.mean(old.interval)) > 1e-8:
        raise ValueError('Preserve the midpoint gauge when shortening an initial curve')
    time = np.linspace(*interval, 1001)
    values = old.components(time)
    coefficients, errors = [], []
    for i, (spacing, anchored, unit) in enumerate(zip(old.spacings, old.anchored, old.units)):
        a = basis(time, interval, spacing, fixed_midpoint=anchored)*unit
        fitted = np.linalg.lstsq(a, values[:, i], rcond=None)[0]
        coefficients.append(fitted)
        errors.append(float(np.max(abs(a@fitted-values[:, i]))))
    result = {**model, 'interval_s': list(interval), 'parameters': np.concatenate(coefficients).tolist(),
        'initial_curve_projection_maximum_component_error': errors}
    result.pop('path_reference_time_s', None)
    return result


def geographic_frame(model):
    if 'linear_reference' not in model or model.get('rock_spacing_seconds') is not None:
        raise ValueError('Need a single-anchor motion with fixed roll and pitch')
    result = copy.deepcopy(model)
    curve = SplineMotion(model)
    reference = np.asarray(model['linear_reference']['body_to_ecef'])
    up = reference[:, 2]
    east = np.cross([0., 0., 1.], up)
    length = np.linalg.norm(east)
    if length < 1e-8:
        raise ValueError('Geographic east is undefined at the pole')
    east /= length
    fixed = np.column_stack([east, np.cross(up, east), up])
    yaw = curve.components([np.mean(curve.interval)])[0, 1]
    c, s = np.cos(yaw), np.sin(yaw)
    at_midpoint = reference@np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    transform = at_midpoint.T@fixed
    result['selected_points_body_m'] = [(np.asarray(points)@transform).tolist()
                                        for points in model['selected_points_body_m']]
    if 'points_body_m' in model:
        result['points_body_m'] = (np.asarray(model['points_body_m'])@transform).tolist()
    parameters = curve.parameters.copy()
    parameters[curve.sections[1]] = 0.
    result['parameters'] = parameters.tolist()
    result['linear_reference']['body_to_ecef'] = fixed.tolist()
    result.update(fixed_ne_frame=True, fitted_orientation=False, point_coordinate_frame='east_north_up',
        fixed_frame_scope=__doc__)
    return result


def rotating_control_seed(free, fixed):
    """Embed a fitted fixed-frame solution exactly in its rotating control.

    This removes an initialization disadvantage in the larger model. The
    fitted no-yaw motion and all physical reflector positions are preserved.
    """
    if free['interval_s'] != fixed['interval_s'] or not fixed.get('fixed_ne_frame'):
        raise ValueError('Need matching fixed and rotating model intervals')
    old, new = SplineMotion(free), SplineMotion(fixed)
    if old.sections != new.sections or old.anchored != new.anchored or old.anchored[1]:
        raise ValueError('The matched models must have the same unanchored yaw basis')
    reference = np.asarray(free['linear_reference']['body_to_ecef'])
    frame = np.asarray(fixed['linear_reference']['body_to_ecef'])
    a = np.concatenate(free['selected_points_body_m'])
    b = np.concatenate(fixed['selected_points_body_m'])@frame.T@reference
    angle = np.arctan2(np.sum(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]), np.sum(a[:, :2]*b[:, :2]))
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    if not np.allclose(a@rotation.T, b, rtol=0., atol=1e-8):
        raise ValueError('The two supports are not the same physical points')
    result = copy.deepcopy(free)
    parameters = new.parameters.copy()
    parameters[old.sections[1]] = angle/old.units[1]
    result.update(parameters=parameters.tolist(),
        nested_fixed_frame_seed=True, nested_seed_constant_yaw_degrees=float(np.degrees(angle)))
    return result


def point_centred_start(original, fixed, point, transmitter, receiver):
    """Project one original reflector's trajectory onto a translation-only seed.

    This changes initialization only. All fitted points, samples, likelihood
    terms and final fixed-orientation constraints remain identical. The reported
    spline projection error is not a residual in the final fitted model.
    """
    curve, constant = SplineMotion(original), SplineMotion(fixed)
    time = np.linspace(*curve.interval, 1001)
    epoch = utc_seconds(original['nominal_start_utc'])
    moving = curve.apply(time, motion_state(None, epoch+time, original), transmitter, receiver)
    centre_frame = moving['body_to_ecef'][len(time)//2]
    delta = (moving['body_to_ecef']-centre_frame)@np.asarray(point)
    values = constant.components(time)
    direction, transverse = moving['translation_direction'], moving['transverse_direction']
    values[:, 0] += np.sum(delta*direction, axis=1)/np.sum(direction**2, axis=1)
    values[:, -1] += np.sum(delta*transverse, axis=1)
    parameters = constant.parameters.copy()
    for component in (0, len(constant.sections)-1):
        design = constant.coefficient_bases(time)[component]
        parameters[constant.sections[component]] = np.linalg.lstsq(design, values[:, component], rcond=None)[0]
    result = {**fixed, 'parameters': parameters.tolist(), 'translation_seed_point_original_body_m': list(point)}
    residual = SplineMotion(result).components(time)-values
    position_error = residual[:, :1]*direction+residual[:, -1:]*transverse
    result['translation_seed_maximum_projection_error_m'] = float(np.max(np.linalg.norm(position_error, axis=1)))
    return result
