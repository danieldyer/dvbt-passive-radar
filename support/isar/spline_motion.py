"""A physical path/heading correction, shared by fitting and backprojection.

The correction changes position, attitude and their derivatives together. A
body yaw is positive towards port; its nautical heading correction is negative.
No phase-only correction is applied to an already compressed image.
"""
import numpy as np
from scipy.interpolate import BSpline

from motion import excess_path
from rigid_motion import rotation_state


def knots(interval, spacing):
    if spacing <= 0 or not interval[0] < interval[1]:
        raise ValueError('Invalid spline interval or spacing')
    # Form offsets before adding the epoch. Floating arange(start, stop) can
    # include stop after rounding, creating a fifth repeated endpoint knot and
    # an identically zero coefficient column. The endpoint is already clamped.
    span = interval[1]-interval[0]
    tolerance = 8*np.finfo(float).eps*max(1., abs(interval[0]), abs(interval[1]))
    offsets = np.arange(1, int(np.ceil(span/spacing))+1)*spacing
    interior = interval[0]+offsets[offsets < span-tolerance]
    return np.r_[np.repeat(interval[0], 4), interior, np.repeat(interval[1], 4)]


def canonical_model(description):
    """Read legacy local fits with a redundant endpoint coefficient safely.

    Only the unused endpoint column is removed; the physical curve throughout
    the open fitted aperture is preserved. Source result files are untouched.
    """
    if 'parameters' not in description:
        return description
    interval = description['interval_s']
    spacings = [description['path_spacing_seconds'], description['heading_spacing_seconds']]
    if description.get('rock_spacing_seconds') is not None:
        spacings += [description['rock_spacing_seconds']]*2
    if description.get('transverse_spacing_seconds') is not None:
        spacings.append(description['transverse_spacing_seconds'])
    counts = [len(knots(interval, spacing))-4 for spacing in spacings]
    parameters = np.asarray(description['parameters'])
    if len(parameters) == sum(counts):
        return description
    old = [4+len(np.arange(interval[0]+spacing, interval[1], spacing)) for spacing in spacings]
    if len(parameters) != sum(old) or any(a-b not in (0, 1) for a, b in zip(old, counts)):
        return description  # Constructor reports the inconsistent parameter count.
    sections = np.split(parameters, np.cumsum(old)[:-1])
    return {**description, 'parameters': np.concatenate([p[:n] for p, n in zip(sections, counts)]),
            'spline_endpoint_migration': 'Removed redundant legacy endpoint coefficients; interior curve preserved.'}


def basis(time, interval, spacing, *, fixed_midpoint=False, derivative=0, reference_time=None):
    grid = knots(interval, spacing)
    curve = BSpline(grid, np.eye(len(grid)-4), 3, extrapolate=False)
    value = curve(time, nu=derivative)
    if fixed_midpoint and derivative == 0:
        reference = np.mean(interval) if reference_time is None else reference_time
        if not interval[0] <= reference <= interval[1]:
            raise ValueError('The path reference must lie inside the fitted interval')
        value -= curve(reference)
    if not np.isfinite(value).all():
        raise ValueError('Motion extrapolation outside the fitted aperture is not permitted')
    return value


def evaluate_coefficients(time,interval,spacing,coefficients,*,derivative=0,anchor=None):
    """Evaluate one scalar curve without materialising its coefficient basis."""
    curve=BSpline(knots(interval,spacing),np.asarray(coefficients),3,extrapolate=False)
    value=curve(time,nu=derivative)
    if anchor is not None and derivative==0:
        if not interval[0]<=anchor<=interval[1]:raise ValueError('The path reference must lie inside the fitted interval')
        value=value-curve(anchor)
    if not np.isfinite(value).all():raise ValueError('Motion extrapolation outside the fitted aperture is not permitted')
    return value


class SplineMotion:
    def __init__(self, description):
        description = canonical_model(description)
        self.description = description
        self.interval = np.asarray(description['interval_s'])
        self.path_spacing = description['path_spacing_seconds']
        self.heading_spacing = description['heading_spacing_seconds']
        self.parameters = np.asarray(description['parameters'])
        npath = basis([np.mean(self.interval)], self.interval, self.path_spacing).shape[1]
        nyaw = basis([np.mean(self.interval)], self.interval, self.heading_spacing).shape[1]
        self.rock_spacing = description.get('rock_spacing_seconds')
        nrock = (0 if self.rock_spacing is None else
                 basis([np.mean(self.interval)], self.interval, self.rock_spacing).shape[1])
        self.transverse_spacing = description.get('transverse_spacing_seconds')
        ncross = (0 if self.transverse_spacing is None else
                  basis([np.mean(self.interval)], self.interval, self.transverse_spacing).shape[1])
        self.counts = [npath, nyaw]+([nrock, nrock] if nrock else [])+([ncross] if ncross else [])
        self.attitude_count = 3 if nrock else 1
        self.spacings = [self.path_spacing, self.heading_spacing]+([self.rock_spacing]*2 if nrock else [])
        self.anchored = [description.get('anchor_path', True), False]+([True]*2 if nrock else [])
        self.units = [1., np.pi/180]+([np.pi/180]*2 if nrock else [])
        if ncross:
            self.spacings.append(self.transverse_spacing)
            self.anchored.append(description.get('anchor_transverse', False))
            self.units.append(1.)
        self.sections = [slice(a, b) for a, b in zip(np.r_[0, np.cumsum(self.counts)[:-1]], np.cumsum(self.counts))]
        if self.parameters.shape != (sum(self.counts),) or not np.isfinite(self.parameters).all():
            raise ValueError('Motion coefficient count or values are invalid')
        self.path_coefficients = self.parameters[:npath]
        self.yaw_coefficients = self.parameters[npath:npath+nyaw]

    def coefficient_bases(self, time, derivative=0):
        reference = self.description.get('path_reference_time_s')
        return [basis(time, self.interval, spacing, fixed_midpoint=anchored, derivative=derivative,
                      reference_time=reference if i == 0 else None)*unit
                for i, (spacing, anchored, unit) in enumerate(zip(self.spacings, self.anchored, self.units))]

    def components(self, time, derivative=0):
        """Path metres, yaw/optional roll/pitch radians, optional transverse metres."""
        reference=self.description.get('path_reference_time_s',float(np.mean(self.interval)))
        return np.column_stack([evaluate_coefficients(time,self.interval,spacing,self.parameters[section],derivative=derivative,
            anchor=(reference if i==0 else float(np.mean(self.interval))) if anchored else None)*unit
            for i,(spacing,anchored,unit,section) in enumerate(zip(self.spacings,self.anchored,self.units,self.sections))])

    def gauge_fixed_parameters(self):
        """Eliminate the exact uniform-coefficient nulls of anchored curves."""
        p = self.parameters.copy()
        free = np.ones(len(p), bool)
        for anchored, section in zip(self.anchored, self.sections):
            if anchored:
                p[section] -= p[section.stop-1]
                free[section.stop-1] = False
        return p, np.flatnonzero(free)

    def curves(self, time, derivative=0):
        values = self.components(time, derivative)
        return values[:, 0], values[:, 1]

    def apply(self, time, state, reference_transmitter, receiver, *, derivatives=False):
        centre, velocity = state['position_ecef_m'], state['velocity_ecef_mps']
        frame, frame_rate = state['body_to_ecef'], state['frame_velocity']
        _, gradient = excess_path(centre, reference_transmitter, receiver)
        gradient_rate = np.zeros_like(gradient)
        for antenna in (reference_transmitter, receiver):
            vector = centre-antenna
            distance = np.linalg.norm(vector, axis=1)
            unit = vector/distance[:, None]
            gradient_rate += (velocity-unit*np.sum(unit*velocity, axis=1)[:, None])/distance[:, None]
        # With two translation coordinates, constrain the correction to the
        # local sea tangent plane. This is a position/velocity model, not a
        # phase-only adjustment. The older one-coordinate model is unchanged.
        if self.description.get('horizontal_translation', False):
            up, up_rate = frame[:, :, 2], frame_rate[:, :, 2]
            vertical = np.sum(gradient*up, axis=1)
            vertical_rate = np.sum(gradient_rate*up+gradient*up_rate, axis=1)
            gradient_rate = gradient_rate-up_rate*vertical[:, None]-up*vertical_rate[:, None]
            gradient = gradient-up*vertical[:, None]
        squared = np.sum(gradient**2, axis=1)
        if np.any(squared < 1e-8):
            raise ValueError('Bistatic translation direction is singular')
        direction = gradient/squared[:, None]
        direction_rate = (gradient_rate/squared[:, None]-gradient*
                          (2*np.sum(gradient*gradient_rate, axis=1)/squared**2)[:, None])
        values, rates = self.components(time), self.components(time, 1)
        path, yaw = values[:, 0], values[:, 1]
        path_rate = rates[:, 0]
        angles, angle_rates = np.zeros((len(time), 3)), np.zeros((len(time), 3))
        order = [2, 0, 1][:self.attitude_count]
        angles[:, order] = values[:, 1:1+self.attitude_count]
        angle_rates[:, order] = rates[:, 1:1+self.attitude_count]
        rotated, derivative, partial, rate_partial = rotation_state(angles, angle_rates, derivatives=derivatives)
        result = {**state,
            'position_ecef_m': centre+path[:, None]*direction,
            'translation_offset_m': path[:, None]*direction,
            'velocity_ecef_mps': velocity+path_rate[:, None]*direction+path[:, None]*direction_rate,
            'body_to_ecef': frame@rotated,
            'frame_velocity': frame_rate@rotated+frame@derivative,
            'translation_direction': direction, 'translation_direction_rate': direction_rate,
            'path_correction_m': path, 'body_yaw_correction_radians': yaw,
            'body_roll_pitch_yaw_radians': angles}
        if self.transverse_spacing is not None:
            vector = np.cross(frame[:, :, 2], gradient)
            vector_rate = np.cross(frame_rate[:, :, 2], gradient)+np.cross(frame[:, :, 2], gradient_rate)
            norm = np.linalg.norm(vector, axis=1)
            if np.any(norm < 1e-8):
                raise ValueError('Horizontal transverse translation direction is singular')
            transverse = vector/norm[:, None]
            transverse_rate = (vector_rate-transverse*np.sum(transverse*vector_rate, axis=1)[:, None])/norm[:, None]
            cross, cross_rate = values[:, -1], rates[:, -1]
            result['position_ecef_m'] += cross[:, None]*transverse
            result['translation_offset_m'] += cross[:, None]*transverse
            result['velocity_ecef_mps'] += cross_rate[:, None]*transverse+cross[:, None]*transverse_rate
            result.update(transverse_direction=transverse, transverse_direction_rate=transverse_rate,
                          transverse_correction_m=cross)
        if derivatives:
            result['attitude_partials'] = [frame@partial[i] for i in order]
            result['attitude_rate_partials'] = [frame_rate@partial[i]+frame@rate_partial[i] for i in order]
        return result
