"""Profile unrestricted neighbouring-echo brightness with its range response.

This is a deterministic nuisance subspace, not normalization or a phase fit.
The fixed response span is shared by data, templates and all derivatives.
Its deliberate information loss must be measured before adopting it.
"""
import numpy as np


class RangeNuisance:
    def __init__(self, design):
        design=np.asarray(design)
        if design.ndim!=3 or design.shape[1]<=design.shape[2]:
            raise ValueError('Need more range observations than nuisance responses')
        self.shape=design.shape[:2]
        self.q,r=np.linalg.qr(design,mode='reduced')
        singular=np.linalg.svd(r,compute_uv=False)
        if np.any(singular[:,-1]<1e-8*singular[:,0]):
            raise ValueError('Neighbouring range responses are numerically dependent')
        self.condition=singular[:,0]/singular[:,-1]

    def apply(self,values):
        values=np.asarray(values);original=values.shape
        x=values.reshape(*self.shape,-1)
        coefficient=np.einsum('nrk,nrs->nks',self.q.conj(),x)
        return (x-np.einsum('nrk,nks->nrs',self.q,coefficient)).reshape(original)
