"""Frozen physical source columns for explicit competing-scene nuisances.

Only a training-fitted native source supplies motion and positive visibility.
The caller profiles an unknown constant complex multiplier for each column.
The same nuisance projection must act on observations and target templates.
"""
import numpy as np
from amplitude_arc import linear_basis
from field import average
from render_injected_point import fitted_state


def component(field,rotation,receiver,render,brightness,points,index):
    c=field.metadata['channel'];band=brightness['bands'][c]
    if field.metadata['rf_hz']!=band['rf_hz']:raise ValueError('Frozen source frequency differs from observation')
    t=field.data['row_time'];state=fitted_state(t,np.asarray(render['coarse_state_enu']),render['midpoint_s'],rotation,receiver,render)
    q=np.asarray(points)[index];position=state['position_ecef_m']+state['body_to_ecef']@q
    velocity=state['velocity_ecef_mps']+state['frame_velocity']@q
    raw=field.raw_trajectory(position,velocity,position_offset=position-field.data['centre_ecef_m'])
    strength=linear_basis(t,np.asarray(band['knots_s']))@np.asarray(band['coefficients'])[index]
    phase=np.exp(1j*(band['band_phase_rad']+brightness['point_phase_rad'][index]))
    return field.apply(average(raw*strength[:,None]*phase,field.data['row_information'],field.metadata['block_rows'])[0]).ravel()
