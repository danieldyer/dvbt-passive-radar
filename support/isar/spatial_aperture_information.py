"""Conditional spatial information with explicit scattering-phase nuisances.

For independent proper-complex unit noise, F=2 Re(J^H P J), where P profiles
real per-band amplitudes and either one common phase or one phase per band.
Dividing by the known template norm reports geometry at unit integrated SNR;
it never rescales recorded data. This is a local point-position bound, not an
extended-target resolution guarantee or protection against distant aliases.
"""
import numpy as np


def information(values,derivatives,*,coherent=True):
    norm=np.array([np.vdot(h,h).real for h in values])
    if np.any(norm<=0):raise ValueError('Every included band needs a nonzero known point template')
    cross=np.array([h.ravel().conj()@d.reshape(-1,2) for h,d in zip(values,derivatives)])
    result=2*sum((d.reshape(-1,2).conj().T@d.reshape(-1,2)).real for d in derivatives)
    result-=2*np.einsum('ci,cj,c->ij',cross.real,cross.real,1/norm)
    if coherent:result-=2*np.outer(cross.imag.sum(axis=0),cross.imag.sum(axis=0))/norm.sum()
    else:result-=2*np.einsum('ci,cj,c->ij',cross.imag,cross.imag,1/norm)
    return (result+result.T)/(2*norm.sum())
