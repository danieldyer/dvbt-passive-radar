"""Exact conditional positive-envelope solve with tied scatterer phases.

All envelope basis functions for one physical point share one phase. Only
real nonnegative magnitudes can vary. Eigenspace truncation is limited to
numerical null directions; it is not a statistical regularisation penalty.
"""
import numpy as np
from scipy.optimize import nnls


def solve(gram,rhs,energy,groups,phase):
    groups=np.asarray(groups,dtype=int);phase=np.asarray(phase)
    rotated=np.exp(1j*phase[groups]);matrix=np.real(rotated.conj()[:,None]*gram*rotated[None,:]);matrix=(matrix+matrix.T)/2
    target=np.real(rotated.conj()*rhs);eigen,vector=np.linalg.eigh(matrix)
    scale=max(float(eigen[-1]),np.finfo(float).tiny)
    if eigen[0]<-1e-9*scale:raise ValueError('Envelope normal matrix is not positive semidefinite')
    keep=eigen>1e-12*scale
    if not keep.any():raise ValueError('No supported envelope coefficients')
    root=np.sqrt(eigen[keep]);design=root[:,None]*vector[:,keep].T;data=(vector[:,keep].T@target)/root
    amplitude=nnls(design,data,maxiter=20*len(groups))[0];coefficient=rotated*amplitude
    residual=gram@coefficient-rhs
    cost=float(energy-2*np.vdot(coefficient,rhs).real+np.vdot(coefficient,gram@coefficient).real)
    gradient=np.bincount(groups,2*np.imag(coefficient.conj()*residual),minlength=phase.size)
    return dict(cost=cost,coefficient=coefficient,amplitude=amplitude,phase_gradient=gradient,
        numerical_rank=int(keep.sum()),active_amplitudes=int(np.count_nonzero(amplitude)),minimum_normal_eigenvalue=float(eigen[0]),
        maximum_nnls_kkt_residual=float(max(np.max(np.maximum(-((matrix@amplitude)-target),0)),np.max(abs((matrix@amplitude-target)[amplitude>0]),initial=0))))
