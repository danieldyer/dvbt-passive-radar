"""A nested multiband test of two selected scatterers' constant phases.

For point s in band c the empirical envelope origin r_cs and constant phase
phi_cs are separate coordinates. A common intrinsic phase per point and one
constant band phase imply

    phi_cD - phi_cB + k_c (r_cD - r_cB) = intrinsic_D - intrinsic_B.

Origins in the supplied model are measured in range samples. The observation
carrier reference cancels within a band. Independent trajectories/brightness
are unchanged; only this constant-phase restriction is tested. It is not an
alignment applied to measured phases or proof of a common physical point.
"""
import numpy as np

from field import FS
from motion import C
from physical_pair import fit_parameters


class RelativeScattererPhase:
    def __init__(self,base,first,second,*,reference_band=2):
        n=base.components;nb=len(base.bands)
        if first==second or not 0<=first<n or not 0<=second<n or not 0<=reference_band<nb:
            raise ValueError('Distinct component indices and a valid reference band required')
        self.base=base;self.first=first;self.second=second;self.reference_band=reference_band
        self.origin_indices=np.array([[2*n*c+2*s for s in (first,second)] for c in range(nb)])
        self.phase_indices=self.origin_indices+1
        removed=self.phase_indices[np.arange(nb)!=reference_band,1]
        self.kept=np.setdiff1d(np.arange(base.count),removed)
        self.count=len(self.kept);self.energy=base.energy
        self.lower=base.lower[self.kept];self.upper=base.upper[self.kept]
        self.phase_per_sample=np.array([b.field.k*C/FS for b in base.bands])
        # The restriction is affine: reuse the exact constant transformation.
        transform=np.zeros((base.count,self.count));transform[self.kept,np.arange(self.count)]=1
        reference=reference_band;rr=self.origin_indices[reference];rp=self.phase_indices[reference]
        for c in range(nb):
            if c==reference:continue
            row=self.phase_indices[c,1];r=self.origin_indices[c];p=self.phase_indices[c]
            transform[row]=transform[p[0]]+transform[rp[1]]-transform[rp[0]]
            transform[row]+=self.phase_per_sample[c]*(transform[r[0]]-transform[r[1]])
            transform[row]+=self.phase_per_sample[reference]*(transform[rr[1]]-transform[rr[0]])
        self.transform=transform;self.last=None

    def mapping(self,parameters):
        return self.transform@np.asarray(parameters)

    def seed(self,independent,reference_delay_phase=0.):
        """Acquire carrier branches with at most half-wavelength range changes.

        Every actual empirical carrier phase is preserved modulo 2pi at the
        seed. Different starts vary the reference relative envelope delay;
        other bands move to their nearest compatible carrier branch. These
        tiny range changes are still penalized by the unchanged data model.
        """
        p=np.asarray(independent).copy();reference=self.reference_band
        p[self.origin_indices[reference,1]]+=reference_delay_phase/self.phase_per_sample[reference]
        for c in range(len(self.phase_indices)):
            if c==reference:continue
            mapped=self.mapping(p[self.kept]);index=self.phase_indices[c,1]
            error=np.angle(np.exp(1j*(mapped[index]-p[index])))
            p[self.origin_indices[c,1]]+=error/self.phase_per_sample[c]
        return p[self.kept]

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        r,j=self.base.evaluate(self.mapping(parameters));j=j@self.transform
        self.last=(np.asarray(parameters).copy(),r,j)
        return r,j

    def objective(self,parameters):
        r,j=self.evaluate(parameters)
        return float(r@r),2*j.T@r

    def fit(self,initial,*,frozen_base_indices=(),max_evaluations=240):
        free=np.flatnonzero(~np.isin(self.kept,frozen_base_indices))
        return fit_parameters(self,initial,free,max_evaluations=max_evaluations)

    def prediction(self,parameters,testing):
        return self.base.prediction(self.mapping(parameters),testing)
