"""Training-only cubic-phase acquisition before exact physical refinement.

Each trial changes only a common path phase. Its fixed range response is an
acquisition approximation; the final likelihood must recompute physical
delay, Doppler, bin attenuation and source positions. Independent constant
band/source coefficients are nuisance variables here, never a final band
calibration. Stationary projection changes the trial Gram matrix and is kept
exact. This is the polynomial-phase counterpart of a coarse chirp search.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from rigid_patch import profile_static


def magnitude_envelope(h,y,root,selected,groups,time,sigma_s=1.,noise_variance=1.):
    """Phase-invariant acquisition weights, never final scientific weighting.

    Training-only local matched powers estimate positive fading. Unresolved
    cross terms can bias this initializer. Invariance to a common template
    phase prevents these weights from adapting to a trial motion phase.
    """
    if not np.isfinite(sigma_s) or sigma_s<=0:raise ValueError('Acquisition smoothing needs a positive time width')
    h=np.asarray(h);time=np.asarray(time);selected=np.asarray(selected,bool)&(np.asarray(root)>0)
    if np.count_nonzero(selected)<2:raise ValueError('Acquisition needs at least two retained training epochs')
    data=profile_static(y,root,selected,groups);norm=np.sum(abs(h)**2,axis=1)
    cross=np.einsum('trp,tr->tp',h.conj(),data);noise=np.ones(len(time))
    for group in np.unique(groups):
        q=np.asarray(root)*selected*(np.asarray(groups)==group);denom=q@q
        if denom>0:noise-=q*q/denom
    power=np.divide(abs(cross)**2-noise_variance*norm*noise[:,None],norm**2,out=np.zeros_like(norm),where=norm>0)
    keep=np.flatnonzero(selected);t=time[keep];lo=np.searchsorted(t,time-4*sigma_s);hi=np.searchsorted(t,time+4*sigma_s,side='right')
    rows=np.repeat(np.arange(len(time)),hi-lo);columns=np.concatenate([np.arange(a,b) for a,b in zip(lo,hi)])
    weight=np.exp(-.5*((time[rows]-t[columns])/sigma_s)**2)
    smoother=csr_matrix((weight,(rows,columns)),shape=(len(time),len(keep)))
    denominator=smoother@norm[keep];numerator=smoother@(norm[keep]*power[keep])
    return np.sqrt(np.maximum(np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0),0))


class Band:
    def __init__(self,h,y,root,selected,groups,time,k,mid,half,mode='joint'):
        if mode not in ['joint','best-point']:raise ValueError('Unknown acquisition nuisance model')
        self.mode=mode
        selected=np.asarray(selected,bool)&(np.asarray(root)>0)
        projected=profile_static(y,root,selected,groups)[selected]
        self.h=np.asarray(h)[selected];self.root=np.asarray(root)[selected]
        norm=np.sqrt(np.sum(abs(self.h)**2,axis=(0,1)));keep=norm>0
        # Constant coefficient units only: exact GLS is invariant to this
        # column scaling. Identically zero sources carry no information.
        self.h=self.h[:,:,keep]/norm[keep][None,None,:]
        self.groups=np.asarray(groups)[selected];self.k=k
        u=(np.asarray(time)[selected]-mid)/half
        self.basis=np.column_stack([u,u*u,u*u*u])
        self.cross=np.einsum('trp,tr->tp',self.h.conj(),projected)
        self.gram=np.einsum('trp,trq->pq',self.h.conj(),self.h)
        self.energy=float(np.vdot(projected,projected).real)
        self.means=[]
        for group in np.unique(self.groups):
            q=self.root*(self.groups==group);q=q/np.linalg.norm(q)
            self.means.append((self.h*q[:,None,None]).reshape(len(q),-1))

    def evaluate(self,parameters,derivatives=False):
        if self.h.shape[-1]==0:return (0.,np.zeros(3)) if derivatives else 0.
        phase=np.exp(1j*self.k*(self.basis@parameters));rhs=phase@self.cross
        p=self.h.shape[-1];gram=self.gram.copy();means=[]
        for value in self.means:
            mean=(phase.conj()@value).reshape(-1,p);means.append(mean);gram-=mean.conj().T@mean
        gram=(gram+gram.conj().T)/2
        if self.mode=='joint':coefficient=np.linalg.solve(gram,rhs)
        else:
            power=np.divide(abs(rhs)**2,gram.diagonal().real,out=np.zeros(len(rhs)),where=gram.diagonal().real>0)
            coefficient=np.zeros(len(rhs),complex)
            if len(rhs):
                best=int(np.argmax(power));coefficient[best]=rhs[best]/gram[best,best].real
        score=float(np.vdot(rhs,coefficient).real)
        if not derivatives:return score
        drhs=1j*self.k*(self.basis.T@(phase[:,None]*self.cross))
        gradient=2*np.real(drhs@coefficient.conj())
        for value,mean in zip(self.means,means):
            derivative=(-1j*self.k*((self.basis*phase.conj()[:,None]).T@value)).reshape(3,-1,p)
            gradient+=2*np.real(np.einsum('r,jr->j',(mean@coefficient).conj(),derivative@coefficient))
        return score,gradient

    def many(self,parameters,batch=32):
        parameters=np.asarray(parameters);scores=[];p=self.h.shape[-1]
        if not p:return np.zeros(len(parameters))
        for first in range(0,len(parameters),batch):
            phase=np.exp(1j*self.k*(parameters[first:first+batch]@self.basis.T))
            rhs=phase@self.cross;gram=np.broadcast_to(self.gram,(len(phase),p,p)).copy()
            for value in self.means:
                mean=(phase.conj()@value).reshape(len(phase),-1,p)
                gram-=np.einsum('brp,brq->bpq',mean.conj(),mean)
            gram=(gram+gram.conj().transpose(0,2,1))/2
            if self.mode=='joint':
                coefficient=np.linalg.solve(gram,rhs[...,None])[...,0]
                scores.extend(np.real(np.einsum('bp,bp->b',rhs.conj(),coefficient)))
            else:
                diagonal=gram.diagonal(axis1=1,axis2=2).real
                power=np.divide(abs(rhs)**2,diagonal,out=np.zeros_like(diagonal),where=diagonal>0)
                scores.extend(np.max(power,axis=1) if p else np.zeros(len(phase)))
        return np.asarray(scores)


def acquire(scene_bands,mid,half,*,starts=8,amplitude_weights=None,mode='joint',subaperture_s=0.,source_indices=None):
    bands=[]
    for c,scene in enumerate(scene_bands):
        b=scene.point;h=scene.design(np.zeros(b.b.shape[1]),c,False)
        if source_indices is not None:h=h[:,:,source_indices]
        amplitude=(magnitude_envelope(h,b.field.y,b.field.root_information,b.training,b.groups,b.t)
            if amplitude_weights is None else np.asarray(amplitude_weights[c]))
        if source_indices is not None and amplitude_weights is not None:amplitude=amplitude[:,source_indices]
        if amplitude.shape!=h.shape[::2] or np.any(amplitude<0) or not np.isfinite(amplitude).all():raise ValueError('Acquisition brightness must be finite positive real time/point arrays')
        h=h*amplitude[:,None,:]
        edges=np.arange(mid-half,mid+half,subaperture_s) if subaperture_s>0 else np.array([mid-half])
        edges=np.r_[edges,mid+half]
        for first,last in zip(edges[:-1],edges[1:]):
            selected=b.training&(b.t>=first)&(b.t<last)
            if np.count_nonzero(selected&(b.field.root_information>0))>=2:
                bands.append(Band(h,b.field.y,b.field.root_information,selected,b.groups,b.t,b.field.k,mid,half,mode))
    grid=np.array(np.meshgrid(np.arange(-3.,3.01,.5),np.arange(-2.,2.01,.5),np.arange(-3.,3.01,.5),indexing='ij')).reshape(3,-1).T
    energy=sum(b.energy for b in bands);scores=sum(b.many(grid) for b in bands)
    seeds=[]
    for index in np.argsort(scores)[::-1]:
        if all(np.linalg.norm(grid[index]-q)>=.8 for q in seeds):seeds.append(grid[index])
        if len(seeds)==starts:break
    def objective(p):
        values=[b.evaluate(p,True) for b in bands]
        return -sum(v[0] for v in values)/energy,-sum(v[1] for v in values)/energy
    trials=[]
    for seed in seeds:
        result=minimize(objective,seed,jac=True,method='L-BFGS-B',bounds=[(-4,4),(-3,3),(-4,4)],options=dict(maxiter=100,gtol=1e-9,ftol=1e-13,maxls=30))
        trials.append(dict(coefficients_m=result.x.tolist(),objective=float(result.fun),maximum_gradient=float(max(abs(result.jac))),iterations=result.nit,converged=bool(result.success and max(abs(result.jac))<1e-6)))
    best=min(trials,key=lambda q:q['objective']);parameters=np.asarray(best['coefficients_m'])
    return parameters,dict(scope=__doc__,anchor_s=mid,half_width_s=half,grid_trials=len(grid),zero_correction_score_fraction=-objective(np.zeros(3))[0],best_score_fraction=-best['objective'],selected=best,refined_starts=trials,
        acquisition_brightness='Training-only phase-invariant matched power' if amplitude_weights is None else 'Supplied fixed acquisition brightness',
        acquisition_nuisance_model=mode,subaperture_s=subaperture_s,independent_acquisition_blocks=len(bands),
        source_indices=None if source_indices is None else list(source_indices),
        coarse_grid_coefficients_m=grid.tolist(),coarse_score_fraction=(scores/energy).tolist(),physical_refinement_required=True,final_band_calibration=False)
