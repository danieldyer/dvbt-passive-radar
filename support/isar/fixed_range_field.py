"""A fixed absolute range reference for unchanged original C0/C1 rows.

The observer reference is a constant physical point; AIS motion remains an
external prior, not part of these observations. Stationary clutter is then
one constant complex vector over the measured range cells, with an exactly
orthogonal design in the conditionally whitened coordinates. A retained known
carrier deramp multiplies that vector by its original-row averaged phasor.
"""
import json

import numpy as np

from field import Field,FS,average
from motion import C,excess_path
from recorded_field_context import expand


def fixed_reference(field,aperture,epoch_s,offsets,*,omega_reference=0.,preserve_carrier=False,observer_ecef_m=None):
    d=field.data;rows=d['row_time']
    if not rows[0]<=epoch_s<=rows[-1] or not np.isfinite(omega_reference):
        raise ValueError('A finite fixed reference within the original row support is required')
    old=float(np.interp(epoch_s,rows,d['centre_path_m']))
    geographic={}
    if 'centre_ecef_m' in d:
        point=np.array([np.interp(epoch_s,rows,d['centre_ecef_m'][:,j]) for j in range(3)]) if observer_ecef_m is None else np.asarray(observer_ecef_m,dtype=float)
        if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Fixed observer must be a finite ECEF position')
        path=float(excess_path(point[None,:],d['transmitter_ecef_m'],d['receiver_ecef_m'])[0][0])
        geographic=dict(centre_ecef_m=np.broadcast_to(point,d['centre_ecef_m'].shape).copy(),
            centre_velocity_ecef_mps=np.zeros_like(d['centre_velocity_ecef_mps']))
    else:
        # An empirical road patch supplies an absolute path polynomial rather
        # than a fictitious ECEF trajectory. A constant path is sufficient to
        # define the same digital observation operator.
        path=old
    old_phase_offset=float(field.metadata.get('phase_reference_offset_m',0.))
    metadata={**field.metadata,'centre_prior':'Fixed observer reference; no assumed target trajectory',
        'operator':'Fixed absolute range grid; 64-tap symmetric interpolation; fixed C0/C1 combination; unchanged original-row information-weighted means',
        'fixed_reference_epoch_s':float(epoch_s),'fixed_reference_path_m':path,
        'fixed_symbol_omega':float(omega_reference),
        'phase_reference':'exp(+i k fixed_reference_path); no additional constant offset',
        'phase_reference_offset_m':0.}
    data={**d,'value':np.zeros_like(d['value']),
        **geographic,
        'centre_path_m':np.full(len(rows),path),'centre_omega':np.full(len(rows),omega_reference),
        'metadata_json':json.dumps(metadata)}
    if preserve_carrier:
        data['carrier_reference_path_m']=np.asarray(d.get('carrier_reference_path_m',d['centre_path_m']-old_phase_offset)).copy()
        metadata['operator']+='; retain the original known carrier deramp before averaging'
        metadata['carrier_reference']='Original row-wise carrier phase; fixed range coordinate is independent'
        data['metadata_json']=json.dumps(metadata)
    else:data.pop('carrier_reference_path_m',None)
    result=expand(Field(data,field.pulse.samples),aperture,offsets)
    change=dict(reference_path_m=path,old_midpoint_path_m=old,
        envelope_origin_change_samples=(old-path)*FS/C,
        constant_phase_change_rad=0. if preserve_carrier else field.k*(path-old+old_phase_offset),
        observer_path_rate_mps=-omega_reference/(field.k*.001024))
    return result,change


def stationary_profile_design(field):
    """One constant coefficient per actually observed whitened range cell.

    D^H D = sum(root_information² |carrier_factor|²) I. The optional known
    carrier factor is averaged through original rows, never divided out.
    There are no unobserved delay columns, extrapolated FIR coefficients,
    local power normalization or Doppler mask.
    """
    d=field.data
    if (not np.all(d['centre_path_m']==d['centre_path_m'][0]) or
            not np.all(d['centre_omega']==d['centre_omega'][0])):
        raise ValueError('A stationary profile requires a fixed range grid and C0/C1 combination')
    n=len(field.offsets);factor=np.ones(len(field.time),complex)
    if 'carrier_reference_path_m' in d:
        carrier=d['carrier_reference_path_m'];phase=np.exp(1j*field.k*(carrier-carrier[0]))
        factor=average(phase,d['row_information'],field.metadata['block_rows'])[0]
    return ((field.root_information*factor)[:,None,None]*np.eye(n)[None,:,:]).reshape(field.y.size,n).astype(complex)
