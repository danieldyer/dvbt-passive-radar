"""Exact time derivatives of the finite-facet PO aperture integral.

Second-order real Taylor arithmetic carries position, velocity, acceleration
through path, spreading and polarization. It avoids subtracting nearly equal
phase samples to obtain curvature inside a location optimizer.
"""
from dataclasses import dataclass

import numpy as np

from motion import C
from specular_facet import quadrature


@dataclass
class Taylor:
    __array_priority__=1000
    value: np.ndarray
    rate: np.ndarray | float = 0.
    second: np.ndarray | float = 0.

    def __add__(self, other):
        other=as_taylor(other)
        return Taylor(self.value+other.value,self.rate+other.rate,self.second+other.second)
    __radd__=__add__

    def __neg__(self):
        return Taylor(-self.value,-self.rate,-self.second)

    def __sub__(self,other):
        return self+-as_taylor(other)

    def __rsub__(self,other):
        return as_taylor(other)+-self

    def __mul__(self,other):
        other=as_taylor(other)
        return Taylor(self.value*other.value,self.rate*other.value+self.value*other.rate,
            self.second*other.value+2*self.rate*other.rate+self.value*other.second)
    __rmul__=__mul__

    def reciprocal(self):
        return Taylor(1/self.value,-self.rate/self.value**2,
            2*self.rate**2/self.value**3-self.second/self.value**2)

    def __truediv__(self,other):
        return self*as_taylor(other).reciprocal()

    def sum(self,axis=-1,keepdims=False):
        return Taylor(*(np.broadcast_to(x,np.shape(self.value)).sum(axis=axis,keepdims=keepdims)
            for x in (self.value,self.rate,self.second)))

    def sqrt(self):
        root=np.sqrt(self.value)
        return Taylor(root,self.rate/(2*root),self.second/(2*root)-self.rate**2/(4*root**3))


def as_taylor(value):
    return value if isinstance(value,Taylor) else Taylor(np.asarray(value))


def cross(a,b):
    a,b=as_taylor(a),as_taylor(b)
    av,ad,add=[np.broadcast_to(v,np.shape(a.value)) for v in (a.value,a.rate,a.second)]
    bv,bd,bdd=[np.broadcast_to(v,np.shape(b.value)) for v in (b.value,b.rate,b.second)]
    return Taylor(np.cross(av,bv),np.cross(ad,bv)+np.cross(av,bd),
        np.cross(add,bv)+2*np.cross(ad,bd)+np.cross(av,bdd))


def norm(a):
    return (a*a).sum(keepdims=True).sqrt()


def unit(a):
    return a/norm(a)


def current(incident,scattered,normal,up):
    pi,ps=unit(cross(up,incident)),unit(cross(up,scattered))
    surface=2*cross(normal,cross(incident,pi))
    return (ps*cross(scattered,cross(scattered,surface))).sum(keepdims=True)


def aperture(position,velocity,acceleration,transmitter,receiver,frequency_hz,heading_rad,
             turn_rate_rad_s,length_m,height_m,*,basis=None,order=(64,8)):
    """Complex shape and apparent path corrections [group range, rate, curvature]."""
    scalar=np.ndim(transmitter)==1
    transmitter=np.atleast_2d(transmitter)[:,None,:]
    frequency=np.atleast_1d(frequency_hz)
    frame=np.eye(3) if basis is None else np.asarray(basis)
    along=frame@np.array([np.sin(heading_rad),np.cos(heading_rad),0.])
    perpendicular=frame@np.array([np.cos(heading_rad),-np.sin(heading_rad),0.])
    along=Taylor(along,turn_rate_rad_s*perpendicular,-turn_rate_rad_s**2*along)
    up=frame[:,2]
    normal=cross(along,up)
    centre=Taylor(np.asarray(position),np.asarray(velocity),np.asarray(acceleration))
    u,wu=quadrature(order[0]);v,wv=quadrature(order[1])
    uv=np.array(np.meshgrid(u*length_m/2,v*height_m/2,indexing='ij')).reshape(2,-1).T
    offset=along*uv[:,0,None]+up*uv[:,1,None]
    points=centre+offset
    incoming,outgoing=points-transmitter,receiver-points
    rt,rr=norm(incoming),norm(outgoing)
    ct,cr=centre-transmitter,receiver-centre
    rct,rcr=norm(ct),norm(cr)
    if np.any(np.sum(normal.value*ct.value,axis=-1)>=0) or np.any(np.sum(normal.value*(-cr.value),axis=-1)>=0):
        raise ValueError('The facet front must be illuminated and visible')
    # Rationalized norm differences keep sub-wavelength changes well resolved.
    delta=(2*(ct*offset).sum(keepdims=True)+(offset*offset).sum(keepdims=True))/(rt+rct)
    delta+=(-2*(cr*offset).sum(keepdims=True)+(offset*offset).sum(keepdims=True))/(rr+rcr)
    factor=(rct*rcr/(rt*rr))*current(incoming/rt,outgoing/rr,normal,up)/current(ct/rct,cr/rcr,normal,up)
    factor*=((wu[:,None]*wv[None,:]).ravel()/4)[:,None]
    k=2*np.pi*frequency[:,None]/C
    w,dw,ddw=[np.asarray(x)[...,0] for x in (factor.value,factor.rate,factor.second)]
    d,dd,ddd=[np.asarray(x)[...,0] for x in (delta.value,delta.rate,delta.second)]
    tone=np.exp(-1j*k*d)
    field=np.sum(w*tone,axis=-1)
    rate=np.sum((dw-1j*k*dd*w)*tone,axis=-1)
    second=np.sum((ddw-2j*k*dd*dw-1j*k*ddd*w-k*k*dd*dd*w)*tone,axis=-1)
    moment=np.sum(w*d*tone,axis=-1)/field
    correction=np.column_stack([moment.real,-(rate/field).imag/k[:,0],-(second/field-(rate/field)**2).imag/k[:,0]])
    return (field[0],correction[0]) if scalar else (field,correction)
