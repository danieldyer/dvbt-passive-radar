"""Additional-echo score after a fitted real-parameter Gaussian nuisance.

Templates and residual are conditionally whitened proper-complex observations;
Q has orthonormal real columns in stacked real/imaginary coordinates. Fit the
two quadratures of each additional complex echo with its retained information.
No empirical noise rescaling, per-cell normalization or range-separation rule.
"""
import numpy as np


def scores(templates,residual,q,*,relative_rank_cutoff=1e-8):
    h=np.asarray(templates,complex);r=np.asarray(residual,complex);q=np.asarray(q,float)
    if (h.ndim!=2 or r.shape!=(h.shape[0],) or q.ndim!=2 or q.shape[0]!=2*len(r) or
            not 0<relative_rank_cutoff<1):
        raise ValueError('Matching complex templates/residual and stacked real nuisance basis required')
    real=np.r_[h.real,h.imag];imag=np.r_[-h.imag,h.real]
    data=np.r_[r.real,r.imag];data=data-q@(q.T@data)
    a=q.T@real;b=q.T@imag;norm=np.sum(abs(h)**2,axis=0)
    aa=norm-np.sum(a*a,axis=0);bb=norm-np.sum(b*b,axis=0);ab=-np.sum(a*b,axis=0)
    minimum=(aa+bb-np.hypot(aa-bb,2*ab))/2
    valid=(norm>0)&(minimum>relative_rank_cutoff*norm)
    x=real.T@data;y=imag.T@data;determinant=aa*bb-ab*ab
    value=np.divide(bb*x*x-2*ab*x*y+aa*y*y,determinant,out=np.zeros_like(norm),where=valid)
    retained=np.divide(aa+bb,2*norm,out=np.zeros_like(norm),where=norm>0)
    return dict(score=value,retained_information=retained,valid=valid,
        quadrature_information=np.stack((aa,ab,bb),axis=-1))
