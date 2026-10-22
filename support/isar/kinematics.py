"""Declared smooth ship-motion bounds, separate from the radar likelihood.

Bounds concern actual heading and the reference bistatic path, including the
AIS prior curvature. They are analysis assumptions, not measurements. The
path constraint uses the small-displacement linearization L = Lprior + d;
the final trajectory is checked again with exact bistatic geometry.
"""
import numpy as np

from imaging import physical_state
from motion import enu_basis, excess_path
from spline_motion import basis


def constraints(track, epoch, model, reference_transmitter, receiver, active, parameter_count, *,
                path_acceleration_mps2=.1, heading_rate_degrees_ps=.5, heading_acceleration_degrees_ps2=.01):
    t = np.linspace(model['interval_s'][0], model['interval_s'][1],
                    int(round(np.diff(model['interval_s'])[0]/.25))+1)
    step = .05
    def prior(offset):
        state = physical_state(track, epoch+t+offset, model['attitude_prior'])
        path = excess_path(state['position_ecef_m'], reference_transmitter, receiver)[0]
        forward = np.einsum('nji,nj->ni', enu_basis(state['latitude'], state['longitude']), state['body_to_ecef'][:, :, 0])
        heading = np.degrees(np.unwrap(np.arctan2(forward[:, 0], forward[:, 1])))
        return path, heading
    before, middle, after = prior(-step), prior(0), prior(step)
    path_acceleration = (after[0]-2*middle[0]+before[0])/step**2
    plus = (after[1]-middle[1]+180)%360-180
    minus = (before[1]-middle[1]+180)%360-180
    heading_rate = (plus-minus)/(2*step)
    heading_acceleration = (plus+minus)/step**2
    bp = basis(t, model['interval_s'], model['path_spacing_seconds'], derivative=2)
    bh = basis(t, model['interval_s'], model['heading_spacing_seconds'], derivative=2)
    dh = basis(t, model['interval_s'], model['heading_spacing_seconds'], derivative=1)
    matrices = [np.c_[bp, np.zeros_like(bh)], np.c_[np.zeros_like(bp), -dh], np.c_[np.zeros_like(bp), -bh]]
    matrix = np.zeros((3*len(t), parameter_count))
    matrix[:, :len(active)] = np.concatenate(matrices)[:, active]
    centres = np.r_[path_acceleration, heading_rate, heading_acceleration]
    radii = np.repeat([path_acceleration_mps2, heading_rate_degrees_ps, heading_acceleration_degrees_ps2], len(t))
    description = {'path_acceleration_bound_mps2': path_acceleration_mps2,
        'heading_rate_bound_degrees_ps': heading_rate_degrees_ps,
        'heading_acceleration_bound_degrees_ps2': heading_acceleration_degrees_ps2,
        'constraint_epochs_s': t.tolist(),
        'scope': __doc__, 'prior_reference_path_acceleration_mps2': path_acceleration.tolist(),
        'prior_heading_rate_degrees_ps': heading_rate.tolist(),
        'prior_heading_acceleration_degrees_ps2': heading_acceleration.tolist()}
    return matrix, -radii-centres, radii-centres, description
