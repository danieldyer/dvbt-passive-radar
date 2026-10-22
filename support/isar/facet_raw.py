"""Finite-facet template through the original archived sample operator."""
import numpy as np

from field import average
from specular_facet import hh_current,quadrature,unit


def template(field,positions,velocities,along,along_rate,up,transmitter,receiver,
             length_m,height_m,*,order=(64,8)):
    """Integrate point responses before symbol averaging and whitening.

    One constant centre-current/spreading normalization is used for the whole
    aperture. In particular the surface's time-varying coherent amplitude is
    not normalized away. No per-band delay or phase-rate correction is added.
    """
    normal=np.cross(along,up)
    middle=len(positions)//2
    rt,rr=np.linalg.norm(positions-transmitter,axis=1),np.linalg.norm(positions-receiver,axis=1)
    current=hh_current(unit(positions-transmitter),unit(receiver-positions),normal,up)
    scale=rt[middle]*rr[middle]/abs(current[middle])
    u,wu=quadrature(order[0]);v,wv=quadrature(order[1])
    raw=np.zeros((len(positions),len(field.offsets)),complex)
    for x,wx in zip(u*length_m/2,wu/2):
        for z,wz in zip(v*height_m/2,wv/2):
            point=positions+x*along+z*up
            velocity=velocities+x*along_rate
            incident,scattered=point-transmitter,receiver-point
            a,b=np.linalg.norm(incident,axis=1),np.linalg.norm(scattered,axis=1)
            weight=wx*wz*scale*hh_current(incident/a[:,None],scattered/b[:,None],normal,up)/(a*b)
            raw+=field.raw_trajectory(point,velocity,transmitter=transmitter,receiver=receiver)*weight[:,None]
    value=average(raw,field.data['row_information'],field.metadata['block_rows'])[0]
    return field.apply(value)
