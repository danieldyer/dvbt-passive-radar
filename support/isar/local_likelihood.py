"""A score-preserving local quadratic after nonlinear nuisance profiling.

This is for initialization/screening only. Final comparisons must evaluate the
original complex likelihood, not treat a local Gauss-Newton approximation as
an exact measurement distribution or a physical confidence certificate.
"""
import numpy as np


def compress(residual,jacobian,selected):
    residual,jacobian=np.asarray(residual,float),np.asarray(jacobian,float)
    selected=np.asarray(selected,int)
    if (residual.ndim!=1 or jacobian.ndim!=2 or jacobian.shape[0]!=len(residual) or
            selected.ndim!=1 or len(np.unique(selected))!=len(selected) or
            np.any((selected<0)|(selected>=jacobian.shape[1]))):
        raise ValueError('Matching residual/Jacobian and distinct target coordinates required')
    nuisance=np.delete(jacobian,selected,axis=1)
    norms=np.linalg.norm(nuisance,axis=0);nonzero=norms>0
    nuisance=nuisance[:,nonzero]/norms[nonzero]
    if nuisance.shape[1]:
        u,s,_=np.linalg.svd(nuisance,full_matrices=False)
        keep=s>s[0]*1e-10;q=u[:,keep]
        r=residual-q@(q.T@residual)
        target=jacobian[:,selected]-q@(q.T@jacobian[:,selected])
        rank=int(keep.sum())
    else:r=residual;target=jacobian[:,selected];rank=0
    u,s,vh=np.linalg.svd(target,full_matrices=False)
    keep=s>s[0]*1e-10 if len(s) and s[0]>0 else np.zeros(len(s),bool)
    matrix=s[keep,None]*vh[keep];offset=u[:,keep].T@r
    constant=float(r@r-offset@offset)
    return dict(matrix=matrix,offset=offset,constant=constant,
        nuisance_rank=rank,target_rank=int(keep.sum()),singular_values=s)
