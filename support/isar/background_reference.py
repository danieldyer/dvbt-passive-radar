"""Independent stationary reference from unchanged captured C0/C1 profiles.

Use the capture's existing frozen-subtraction residual, never mix it with a
replay using different processing. The learned complex mean is passed through
exactly the field's saved interpolation, moment combination and deramp.
"""
import json

import numpy as np

from field import average,interpolate,FS
from motion import C


def profile_means(aperture,intervals,*,chunk_rows=512):
    time=aperture['time_s'];information=aperture['information']*aperture['valid']
    rows=[]
    for first,last in intervals:
        selected=np.flatnonzero((time>=first)&(time<last))
        if len(selected)<2:raise ValueError('Insufficient independent reference coverage')
        total=np.zeros((2,aperture['moments'].shape[2]),complex);weight=0.
        for begin in range(0,len(selected),chunk_rows):
            index=selected[begin:begin+chunk_rows];w=information[index]
            values=np.asarray(aperture['moments'][:,index,:],complex)
            total+=np.einsum('n,mnr->mr',w,values);weight+=float(w.sum())
        if weight<=0:raise ValueError('No valid independent reference samples')
        rows.append(dict(first_s=float(first),last_s=float(last),rows=int(len(selected)),
            valid_rows=int(np.count_nonzero(information[selected])),information=weight,mean=total/weight))
    return rows


def combine_means(rows):
    weight=np.asarray([r['information'] for r in rows])
    return np.einsum('n,nmr->mr',weight,np.stack([r['mean'] for r in rows]))/weight.sum()


def field_reference(field,mean,first_range_sample):
    """Fixed mean through the unchanged linear observation map, including C1."""
    mean=np.asarray(mean)
    if mean.ndim!=2 or mean.shape[0]!=2:raise ValueError('Require both original moments')
    d=field.data;count=len(d['row_time']);raw=np.empty((count,len(field.offsets)),complex)
    for first in range(0,count,512):
        section=slice(first,min(count,first+512));n=len(d['row_time'][section])
        constant=np.broadcast_to(mean[:,None,:],(2,n,mean.shape[1]))
        coordinate=d['centre_path_m'][section]*FS/C-first_range_sample
        zero,one=interpolate(constant,coordinate,field.offsets)
        carrier=d.get('carrier_reference_path_m',d['centre_path_m'])
        raw[section]=(zero-1j*d['centre_omega'][section,None]*one)*np.exp(1j*field.k*carrier[section,None])
    return field.apply(average(raw,d['row_information'],field.metadata['block_rows'])[0])


def identity(aperture):
    metadata=json.loads(str(aperture['metadata_json']))
    return {name:metadata[name] for name in ('source','channel','first_range_sample','last_range_sample','rows','complete')}
