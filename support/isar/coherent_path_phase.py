"""Conditional common-path phase probe against a frozen complex scene.

This is a bounded acquisition diagnostic, not a physical motion update:
envelope migration is deliberately unchanged. A positive result must pass
the exact physical waveform likelihood before use in image formation.
"""
import numpy as np
from rigid_patch import profile_static


class PhaseProbe:
    def __init__(self,models,data,root,groups,selected,basis,wavenumber):
        self.models=models;self.root=root;self.groups=groups;self.selected=selected;self.basis=basis;self.k=wavenumber
        self.data=[profile_static(y,r,m,g)*m[:,None] for y,r,m,g in zip(data,root,selected,groups)]
        self.energy=sum(np.vdot(y,y).real for y in self.data)

    def objective(self,parameters,full=False):
        gradient=np.zeros_like(parameters);cost=0.;records=[]
        for c,(model,root,groups,mask,b) in enumerate(zip(self.models,self.root,self.groups,self.selected,self.basis)):
            raw=model*np.exp(-1j*self.k[c]*(b@parameters))[:,None]
            predicted=profile_static(raw,root,mask,groups)*mask[:,None];residual=predicted-self.data[c]
            value=float(np.vdot(residual,residual).real);cost+=value
            time_gradient=2*np.real(np.sum(residual.conj()*(-1j*self.k[c]*raw),axis=1))
            gradient+=b.T@time_gradient
            records.append(dict(cost=value,energy=float(np.vdot(self.data[c],self.data[c]).real)))
        return (cost/self.energy,gradient/self.energy,records) if full else (cost/self.energy,gradient/self.energy)
