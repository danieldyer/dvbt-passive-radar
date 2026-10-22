"""Exact fixed-template positive-brightness likelihood from small Gram blocks.

Only a constant target phase changes. Forming the Gram blocks once avoids
revisiting the original observation matrix during scalar phase acquisition.
No ridge, local power normalization or noise-scale fit is introduced.
"""
import numpy as np
from scipy.optimize import nnls,minimize_scalar


def solve(gram,cross,energy):
    gram=np.asarray(gram,float);cross=np.asarray(cross,float)
    values,vectors=np.linalg.eigh((gram+gram.T)/2)
    largest=max(0.,values[-1]);threshold=np.finfo(float).eps*len(values)*largest
    if values[0]<-1e-10*max(largest,1.):raise ValueError('Non-positive Gram matrix')
    keep=values>threshold
    if not keep.any():return np.zeros(len(values)),float(energy),0
    root=np.sqrt(values[keep])[:,None]*vectors[:,keep].T
    target=(vectors[:,keep].T@cross)/np.sqrt(values[keep])
    amplitude,_=nnls(root,target,maxiter=40*len(cross))
    cost=float(energy-2*cross@amplitude+amplitude@gram@amplitude)
    return amplitude,cost,int(keep.sum())


def fit(design,data,target_columns,initial_phase=0.):
    design,data=np.asarray(design),np.asarray(data);n=int(target_columns)
    if design.ndim!=2 or data.shape!=(len(design),) or not 0<n<=design.shape[1]:raise ValueError('Matching data and nonempty target block required')
    gram=design.conj().T@design;cross=design.conj().T@data;energy=float(np.vdot(data,data).real)
    def evaluate(phase):
        rotation=np.r_[np.full(n,np.exp(1j*phase)),np.ones(design.shape[1]-n)]
        g=(rotation.conj()[:,None]*gram*rotation[None,:]).real
        h=(rotation.conj()*cross).real
        amplitude,cost,rank=solve(g,h,energy)
        return cost,amplitude,rank
    grid=np.arange(32)*2*np.pi/32-np.pi;values=np.array([evaluate(p)[0] for p in grid])
    minima=np.flatnonzero((values<=np.roll(values,1))&(values<=np.roll(values,-1)))
    candidates=[(float(p),evaluate(p)[0]) for p in (initial_phase,grid[np.argmin(values)])]
    for i in minima:
        scalar=minimize_scalar(lambda p:evaluate(p)[0],bounds=(grid[i]-2*np.pi/32,grid[i]+2*np.pi/32),
            method='bounded',options=dict(xatol=1e-8))
        candidates.append((float(scalar.x),float(scalar.fun)))
    phase,_=min(candidates,key=lambda x:x[1]);cost,amplitude,rank=evaluate(phase)
    return dict(phase_rad=float(np.angle(np.exp(1j*phase))),amplitude=amplitude,cost=cost,
        numerical_rank=rank,phase_starts=len(candidates),grid_phase_rad=grid,grid_cost=values)
