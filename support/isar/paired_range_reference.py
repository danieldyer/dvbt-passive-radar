"""Choose between two saved fixed grids without changing carrier origin.

The grid schedule is fixed before the fine fit. Original epochs, information,
known waveform kernels and training folds agree exactly. Both regions retain
one constant phase calibration per band. Only the stationary range nuisance
uses separate coefficients for the two different digital sampling grids.
"""
import json
import numpy as np
from field import Field
from waveform_kernels import KernelField


def combine(outer,middle,coordinate,interval):
    for key in ['time','row_time','information','row_information','offset_samples','transmitter_ecef_m','receiver_ecef_m']:
        np.testing.assert_array_equal(outer.data[key],middle.data[key])
    np.testing.assert_array_equal(outer.pulse.samples,middle.pulse.samples)
    if outer.metadata['block_rows']!=1 or middle.metadata['block_rows']!=1:raise ValueError('Composite fields require explicit representative row identities')
    if outer.k!=middle.k:raise ValueError('Cannot mix different RF channels')
    t=outer.time;selected=(t>=interval[0])&(t<interval[1]);data=dict(outer.data)
    carrier=lambda f:np.asarray(f.data.get('carrier_reference_path_m',f.data['centre_path_m']))
    phase_origin=float(carrier(outer)[0]);value=outer.data['value']*np.exp(1j*outer.k*(phase_origin-carrier(outer)))[:,None]
    value[selected]=middle.data['value'][selected]*np.exp(1j*outer.k*(phase_origin-carrier(middle)[selected]))[:,None]
    for key in ['centre_ecef_m','centre_velocity_ecef_mps','centre_path_m','centre_omega']:
        data[key]=np.array(outer.data[key],copy=True);data[key][selected]=middle.data[key][selected]
    data['value']=value;data['carrier_reference_path_m']=np.full(len(t),phase_origin)
    meta={**outer.metadata,'centre_prior':'Two predetermined fixed range grids with one unchanged constant carrier origin',
        'range_grid_switch_interval_s':list(interval),'carrier_reference_path_m':phase_origin,'stationary_nuisance_groups':2,
        'no_independent_grid_phase_calibration':True}
    data['metadata_json']=json.dumps(meta);field=Field(data,outer.pulse.samples)
    if isinstance(outer,KernelField):
        np.testing.assert_array_equal(outer.pulse.row_blocks,middle.pulse.row_blocks)
        field=KernelField(field,outer.pulse.kernels,outer.pulse.row_blocks)
    epoch={**coordinate,'stationary_groups':selected.astype(np.int32)}
    return field,epoch
