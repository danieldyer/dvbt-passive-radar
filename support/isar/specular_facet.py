"""Finite rectangular physical-optics aperture, retaining spherical phase.

Uniform surface current is an explicit hypothesis. The horizontal co-polar
PEC current factor and geometric spreading are evaluated at every quadrature
point. Responses are normalized by the centre's current/spreading and area;
this is an aperture shape factor, not an absolute RCS prediction. No edge
diffraction, shadowing by other ship structures or sea reflection is included.
"""
import numpy as np
from functools import cache
from numpy.polynomial.legendre import leggauss

from motion import C, path_difference


@cache
def quadrature(order):
    nodes,weights=leggauss(order)
    nodes.flags.writeable=False;weights.flags.writeable=False
    return nodes,weights


def unit(vector):
    length = np.linalg.norm(vector,axis=-1,keepdims=True)
    if np.any(length == 0):
        raise ValueError('Nonzero directions are required')
    return vector/length


def hh_current(incident, scattered, normal, up):
    """Scalar receive projection of PO current 2 n × (ki × Ei)."""
    pi = unit(np.cross(up,incident))
    ps = unit(np.cross(up,scattered))
    current = 2*np.cross(normal,np.cross(incident,pi))
    return np.sum(ps*np.cross(scattered,np.cross(scattered,current)),axis=-1)


def response(centres, transmitter, receiver, along, vertical, length_m, height_m,
             frequency_hz, *, up=(0.,0.,1.), order=(64,12), polarization=True):
    """Complex shape and its exact carrier-wavenumber logarithmic derivative.

    Return shape is (number of times, number of frequencies). ``path_moment``
    is the complex path moment divided by field; its real part is the group
    excess path relative to the centre, where the field is nonzero. At a null
    no unique effective phase centre is defined. Coordinates share one frame.
    """
    centres = np.atleast_2d(np.asarray(centres,float))
    tx,rx = np.asarray(transmitter,float),np.asarray(receiver,float)
    a = np.broadcast_to(along,centres.shape)
    b = np.broadcast_to(vertical,centres.shape)
    frequency = np.atleast_1d(frequency_hz)
    if (centres.shape[1] != 3 or tx.shape != (3,) or rx.shape != (3,)
        or min(length_m,height_m) <= 0 or np.any(frequency <= 0)
        or min(order) < 2 or not np.allclose(np.sum(a*b,axis=1),0.,atol=1e-12)
        or not np.allclose(np.linalg.norm(a,axis=1),1.,atol=1e-12)
        or not np.allclose(np.linalg.norm(b,axis=1),1.,atol=1e-12)):
        raise ValueError('Finite centres, orthonormal axes, positive size/frequency and quadrature orders are required')
    normal = np.cross(a,b)
    txvector,rxvector = centres-tx,centres-rx
    if np.any(np.sum(normal*txvector,axis=1) >= 0) or np.any(np.sum(normal*rxvector,axis=1) >= 0):
        raise ValueError('The facet front must be illuminated and visible')
    u,wu = quadrature(order[0]);v,wv = quadrature(order[1])
    coordinate = np.array(np.meshgrid(u*length_m/2,v*height_m/2,indexing='ij')).reshape(2,-1).T
    weight = (wu[:,None]*wv[None,:]).ravel()/4
    displacement = a[:,None,:]*coordinate[None,:,0,None]+b[:,None,:]*coordinate[None,:,1,None]
    delta = path_difference(centres[:,None,:],displacement,tx,rx)
    incoming = txvector[:,None,:]+displacement
    outgoing = -(rxvector[:,None,:]+displacement)
    rt,rr = np.linalg.norm(incoming,axis=-1),np.linalg.norm(outgoing,axis=-1)
    spreading = np.linalg.norm(txvector,axis=1)[:,None]*np.linalg.norm(rxvector,axis=1)[:,None]/(rt*rr)
    if polarization:
        current = hh_current(incoming/rt[:,:,None],outgoing/rr[:,:,None],normal[:,None,:],up)
        centre_current = hh_current(unit(txvector),-unit(rxvector),normal,up)
        if np.any(abs(centre_current) < 1e-10):
            raise ValueError('Centre co-polar current is zero; normalization is undefined')
        spreading *= current/centre_current[:,None]
    weight = spreading*weight
    k = 2*np.pi*frequency/C
    tone = np.exp(-1j*delta[:,:,None]*k)
    field = np.einsum('nq,nqf->nf',weight,tone)
    moment = np.einsum('nq,nq,nqf->nf',weight,delta,tone)
    return {'relative_field':field, 'path_moment_m':np.divide(moment,field,
        out=np.full_like(moment,np.nan),where=abs(field)>1e-14),
        'minimum_path_offset_m':delta.min(axis=1), 'maximum_path_offset_m':delta.max(axis=1)}
