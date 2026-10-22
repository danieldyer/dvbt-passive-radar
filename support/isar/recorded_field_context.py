"""Expand a saved field using its unchanged original C0/C1 observation map."""
import json

import numpy as np

from field import Field,average,interpolate,FS
from motion import C


def expand(field,aperture,offsets):
    d=field.data;epochs=d['row_time'];index=np.searchsorted(aperture['time_s'],epochs)
    index=np.minimum(index,len(aperture['time_s'])-1)
    before=np.maximum(index-1,0)
    index=np.where(abs(aperture['time_s'][before]-epochs)<abs(aperture['time_s'][index]-epochs),before,index)
    if np.max(abs(aperture['time_s'][index]-epochs))>2e-8:raise ValueError('Original profile epochs do not match')
    if np.any(np.diff(index)!=1):raise ValueError('Expected the original contiguous row support')
    information=aperture['information'][index]*aperture['valid'][index]
    if not np.array_equal(information,d['row_information']):raise ValueError('Original row information changed')
    meta=json.loads(str(aperture['metadata_json']));values=[]
    for start in range(0,len(epochs),512):
        rows=slice(start,min(start+512,len(epochs)));idx=index[rows]
        moments=aperture['moments'][:,idx[0]:idx[-1]+1,:]
        coordinate=d['centre_path_m'][rows]*FS/C-meta['first_range_sample']
        zero,one=interpolate(moments,coordinate,offsets)
        carrier=d.get('carrier_reference_path_m',d['centre_path_m'])
        values.append((zero-1j*d['centre_omega'][rows,None]*one)*np.exp(1j*field.k*carrier[rows,None]))
    value,weight=average(np.concatenate(values),information,field.metadata['block_rows'])
    if not np.array_equal(weight,d['information']):raise ValueError('Original bin information changed')
    return Field({**d,'value':value,'offset_samples':np.asarray(offsets)},field.pulse.samples)
