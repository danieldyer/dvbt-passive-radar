"""Fast common-carrier initializer with the exact stationary projection.

The estimated correction is in metres and drives every RF band. Range,
intra-symbol Doppler and differential spatial rays are frozen only during
initialization. The exact physical waveform likelihood must subsequently
validate and refine it. No band-specific time-varying phase is introduced.
"""
import numpy as np
from rigid_patch import profile_static


class FrozenCarrier:
    def __init__(self,bands,models):
        self.records=[];self.energy=0.;self.degrees=0
        for b,model in zip(bands,models):
            selected=b.training;root=b.field.root_information;groups=b.groups
            data=b.data[True];energy=float(np.vdot(data[selected],data[selected]).real)
            q=[]
            for group in np.unique(groups):
                value=root*selected*(groups==group);norm=np.linalg.norm(value)
                if norm:q.append(value/norm)
            q=np.asarray(q)
            self.records.append(dict(model=model,product=np.sum(model.conj()*data,axis=1)*selected,
                raw_energy=np.sum(abs(model[selected])**2),q=q,k=b.field.k,basis=b.b,selected=selected,data_energy=energy))
            self.energy+=energy
            self.degrees+=(np.count_nonzero(selected&(root>0))-len(q))*model.shape[1]

    def objective(self,parameters):
        value=0.;gradient=np.zeros_like(parameters)
        for r in self.records:
            phase=np.exp(-1j*r['k']*(r['basis']@parameters))
            signal=r['model']*phase[:,None]
            mean=r['q']@signal
            product=r['product']*phase.conj()
            value+=r['data_energy']+r['raw_energy']-float(np.sum(abs(mean)**2))-2*float(np.real(product.sum()))
            nuisance=np.sum((r['q'].T@mean).conj()*signal,axis=1)
            derivative=2*r['k']*np.imag(product-nuisance)
            gradient+=r['basis'].T@derivative
        return value/self.energy,gradient/self.energy

    def information(self):
        size=self.records[0]['basis'].shape[1];fisher=np.zeros((size,size))
        for r in self.records:
            b=r['basis'];power=np.sum(abs(r['model'])**2,axis=1)*r['selected']
            local=(b.T@b.multiply(power[:,None])).toarray()
            for q in r['q']:
                mean=np.asarray(b.T@(r['model']*q[:,None])).T
                local-=np.real(mean.conj().T@mean)
            fisher+=2*r['k']**2*local
        return (fisher+fisher.T)/2
