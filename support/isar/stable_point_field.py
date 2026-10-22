"""Preserve local point displacements before adding Earth-sized coordinates.

This isolated Field variant lets the ongoing older-model fits finish with
unchanged source identities. It changes floating-point evaluation order only:
the point offset is formed as (centre-reference)+displacement, instead of
(centre+displacement)-reference. Motion is frozen in these point fits.
"""
import numpy as np

from field import Field


class StablePointField(Field):
    def __init__(self, source):
        super().__init__(source.data, source.pulse.samples)
        self.y = source.y

    def point(self, position_body_m, *, transmitter_offset_ecef_m=None, receiver_offset_ecef_m=None,
              derivatives=False, motion=None, range_bias_m=0., range_bias_derivative=False):
        d = self.data
        frame = d['body_to_ecef'] if motion is None else motion['body_to_ecef']
        centre = d['centre_ecef_m'] if motion is None else motion['position_ecef_m']
        velocity = d['centre_velocity_ecef_mps'] if motion is None else motion['velocity_ecef_mps']
        displacement = frame@np.asarray(position_body_m)
        tx = d['transmitter_ecef_m']+(0 if transmitter_offset_ecef_m is None else transmitter_offset_ecef_m)
        rx = d['receiver_ecef_m']+(0 if receiver_offset_ecef_m is None else receiver_offset_ecef_m)
        frame_rate = (motion['frame_velocity'] if motion is not None and 'frame_velocity' in motion
                      else np.gradient(frame, d['row_time'], axis=0, edge_order=2))
        offset = (centre-d['centre_ecef_m'])+displacement
        return self.trajectory(centre+displacement, velocity+frame_rate@position_body_m,
            position_jacobian=frame if derivatives else None,
            velocity_jacobian=frame_rate if derivatives else None,
            transmitter=tx, receiver=rx, range_bias_m=range_bias_m,
            range_bias_derivative=range_bias_derivative, position_offset=offset)
