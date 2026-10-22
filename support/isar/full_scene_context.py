"""Reopen a saved scene on its exact original waveform-bin identities."""
import json
from pathlib import Path
import numpy as np
from full_ship_field import load,select_range
from full_path_point import PointBand
from full_path_scene import SceneBand
from joint_ship_motion import JointMotion


def point_dictionary(band,points,channel,*,phase=0.):
    """Exact physical columns without a second full-size stacking buffer."""
    points=np.asarray(points);result=np.empty((*band.field.y.shape,len(points)),complex)
    for j,point in enumerate(points):
        result[:,:,j]=band.waveform(np.zeros(band.b.shape[1]),channel,False,point-band.reference_point)
    result*=np.exp(1j*phase)
    return result


def open_scene(path, *, full_field=False):
    path=Path(path);fit=json.loads(path.read_text())
    if not fit.get('complete') or 'scene_points_body_m' not in fit:
        raise ValueError('Use a complete saved physical scene')
    motion=JointMotion(Path(fit['motion']));motion.interval=fit['interval_s']
    with np.load(path.parent/'trajectory.npz') as source:saved=dict(source)
    fields,coordinates,_=load(fit['field']);bands=[]
    for c,(old,coordinate) in enumerate(zip(fields,coordinates)):
        time=saved[f'time_s_{c}'];index=np.searchsorted(old.time,time)
        np.testing.assert_array_equal(old.time[index],time)
        field=old.windowed(time[0],np.nextafter(time[-1],np.inf))
        np.testing.assert_array_equal(field.time,time)
        if not full_field and fit.get('fit_range_offsets'):
            field=select_range(field,*fit['fit_range_offsets'])
        coord={k:coordinate[k].reshape(-1,16)[index].ravel() for k in ['native_time_s','native_information']}
        for k in ['training','stationary_groups']:
            if k in coordinate:coord[k]=coordinate[k][index]
        b=PointBand(field,coord,motion,fit['point_body_m'],fit['path_knots_s'],fit['path_coefficients_m'],
            fit.get('anchor_path_time_s',162.),fit['amplitude_spacing_s'],fit.get('orientation','existing'),
            angular_rate_offset_degps=fit.get('angular_rate_offset_degps',0.),
            angular_acceleration_offset_degps2=fit.get('angular_acceleration_offset_degps2',0.))
        band=SceneBand(b,np.asarray(fit['scene_points_body_m']),fit.get('scene_window_edges_s'),fit['amplitude_spacing_s'],fit.get('scene_active_points'))
        if saved[f'amplitude_{c}'].shape!=(len(time),band.points_per_window):
            raise ValueError('Saved brightness does not match the physical scene')
        bands.append(band)
    return fit,motion,bands,saved


def prediction(band,saved,fit,channel,correction=None):
    b=band.point
    if correction is None:correction=np.zeros(b.b.shape[1])
    h=band.design(correction,channel,False)
    phi=band.phase_at_time(np.asarray(fit['scene_point_phase_radians']))
    beta=np.angle(complex(*fit['physical_gains_real_imag'][channel]))
    return np.sum(h*(saved[f'amplitude_{channel}']*np.exp(1j*(phi+beta)))[:,None,:],axis=2)
