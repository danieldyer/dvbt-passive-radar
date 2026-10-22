"""Small physical translation/attitude corrections to an explicit AIS prior.

Corrections vanish at the aperture midpoint, fixing the constant pose gauge.
Translations use a fixed midpoint body frame. Rotations are intrinsic body
roll, pitch, yaw (Rz Ry Rx), not a substitute for reported true heading.
"""
import numpy as np


def rotation(angles, *, second_derivatives=False):
    identity = np.eye(3)
    generators = np.array([[[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                           [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                           [[0, -1, 0], [1, 0, 0], [0, 0, 0]]])
    matrices, derivatives, seconds = [], [], []
    for axis, generator in enumerate(generators):
        sine = np.sin(angles[:, axis])[:, None, None]
        cosine = np.cos(angles[:, axis])[:, None, None]
        square = generator@generator
        matrices.append(identity+sine*generator+(1-cosine)*square)
        derivatives.append(cosine*generator+sine*square)
        seconds.append(-sine*generator+cosine*square)
    x, y, z = matrices
    dx, dy, dz = derivatives
    result = z@y@x, (z@y@dx, z@dy@x, dz@y@x)
    if not second_derivatives:
        return result
    dxx, dyy, dzz = seconds
    xy, xz, yz = z@dy@dx, dz@y@dx, dz@dy@x
    return (*result, ((z@y@dxx, xy, xz), (xy, z@dyy@x, yz), (xz, yz, dzz@y@x)))


def rotation_state(angles, rates, *, derivatives=True):
    """Exact Rz Ry Rx value, time derivative and their angle derivatives.

    Angles/rates are roll, pitch, yaw in radians and radians/second. The
    second partials are needed when varying an angle while holding its rate
    fixed; omitting them breaks phase/Doppler consistency during a joint fit.
    """
    parts = rotation(angles, second_derivatives=derivatives)
    value, partial = parts[:2]
    rate = sum(p*r[:, None, None] for p, r in zip(partial, rates.T))
    rate_partial = (tuple(sum(p*r[:, None, None] for p, r in zip(row, rates.T)) for row in parts[2])
                    if derivatives else ())
    return value, rate, partial, rate_partial


class RigidCorrection:
    def __init__(self, field, interval, *, degree=3, reference_frame=None,
                 translation_sigma_m=(3., 3., 1.), rotation_sigma_degrees=(1., 1., 3.)):
        if not 1 <= degree <= 4 or not interval[0] < interval[1]:
            raise ValueError('Need a positive aperture and polynomial degree 1–4')
        self.field = field
        self.time = field.data['row_time']
        self.degree = degree
        self.interval = np.array(interval, float)
        half = (interval[1]-interval[0])/2
        u = (self.time-np.mean(interval))/half
        powers = np.arange(1, degree+1)
        self.basis = u[:, None]**powers
        self.rate_basis = powers*u[:, None]**(powers-1)/half
        self.scale = np.r_[translation_sigma_m, np.radians(rotation_sigma_degrees)]
        if np.any(self.scale <= 0):
            raise ValueError('Physical regularisation scales must be positive')
        midpoint = np.argmin(abs(self.time-np.mean(interval)))
        self.reference_frame = (field.data['body_to_ecef'][midpoint]
                                if reference_frame is None else np.array(reference_frame))

    def state(self, parameters):
        coefficients = np.asarray(parameters).reshape(6, self.degree)*self.scale[:, None]
        correction = self.basis@coefficients.T
        translation = correction[:, :3]@self.reference_frame.T
        translation_rate = (self.rate_basis@coefficients[:3].T)@self.reference_frame.T
        rotated, derivatives = rotation(correction[:, 3:])
        d = self.field.data
        frame = d['body_to_ecef']@rotated
        return {'position_ecef_m': d['centre_ecef_m']+translation,
                'velocity_ecef_mps': d['centre_velocity_ecef_mps']+translation_rate,
                'body_to_ecef': frame,
                'frame_velocity': np.gradient(frame, self.time, axis=0, edge_order=2),
                'rotation_derivatives': [d['body_to_ecef']@derivative for derivative in derivatives],
                'translation_local_m': correction[:, :3], 'angles_radians': correction[:, 3:]}

    def point(self, state, body_point, *, derivatives=True):
        position = state['position_ecef_m']+state['body_to_ecef']@body_point
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@body_point
        if not derivatives:
            return self.field.trajectory(position, velocity)
        count = 6*self.degree
        position_jacobian = np.empty((len(self.time), 3, count))
        velocity_jacobian = np.empty_like(position_jacobian)
        for axis in range(3):
            columns = slice(axis*self.degree, (axis+1)*self.degree)
            direction = self.reference_frame[:, axis]*self.scale[axis]
            position_jacobian[:, :, columns] = direction[None, :, None]*self.basis[:, None, :]
            velocity_jacobian[:, :, columns] = direction[None, :, None]*self.rate_basis[:, None, :]
        for axis in range(3):
            columns = slice((axis+3)*self.degree, (axis+4)*self.degree)
            direction = (state['rotation_derivatives'][axis]@body_point)*self.scale[axis+3]
            position_jacobian[:, :, columns] = direction[:, :, None]*self.basis[:, None, :]
            velocity_jacobian[:, :, columns] = np.gradient(position_jacobian[:, :, columns],
                                                           self.time, axis=0, edge_order=2)
        return self.field.trajectory(position, velocity, position_jacobian=position_jacobian,
                                      velocity_jacobian=velocity_jacobian)

    def description(self):
        return {'degree': self.degree, 'interval_s': self.interval.tolist(),
                'coordinate_order': ['forward', 'port', 'up', 'roll', 'pitch', 'yaw'],
                'coefficient_scales': self.scale.tolist(),
                'basis': 'Powers 1..degree of (time-midpoint)/half_aperture; no constant correction',
                'scale_units': 'metres for translation; radians for rotation',
                'prior_scope': 'Declared weak physical regularisation; not measured AIS error statistics',
                'gauge': 'Constant centre and attitude fixed to the explicit AIS/course hypothesis; antenna coordinates fixed',
                'reference_frame_ecef': self.reference_frame.tolist()}
