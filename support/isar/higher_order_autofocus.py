"""Smooth residual phase likelihood, with acquired quadratic motion held fixed.

PGA supplies an optional initial guess; the constraint is imposed DURING the
fit. Removing quadratic phase AFTER an unconstrained autofocus fit is not an
invariance of image power and can undo its gain. Only the constant phase is a
true point-history gauge here; keeping rate/curvature fixed is a model choice.
"""
import numpy as np
from scipy.optimize import minimize


class HigherOrderPhase:
    def __init__(self,time,values,information,training,degree=8):
        self.z=np.asarray(values,complex);self.weight=np.asarray(information)*np.asarray(training,bool)
        if degree<3 or np.count_nonzero(self.weight)<4*(degree-2):raise ValueError('Insufficient training support for phase order')
        u=2*(np.asarray(time)-time[0])/(time[-1]-time[0])-1
        b=np.polynomial.legendre.legvander(u,degree);low=b[:,:3];root=np.sqrt(self.weight)
        self.basis=b[:,3:]-low@np.linalg.lstsq(low*root[:,None],b[:,3:]*root[:,None],rcond=None)[0]
        self.weight/=self.weight.sum();self.energy=np.sum(self.weight*abs(self.z)**2)

    def objective(self,coefficients):
        z=self.z*np.exp(-1j*(self.basis@coefficients));mean=np.sum(self.weight*z)
        value=-abs(mean)**2/self.energy
        gradient=self.basis.T@(-2*self.weight*np.imag(z*mean.conjugate()))/self.energy
        return float(value),gradient

    def fit(self,initial_phase=None):
        seeds=[np.zeros(self.basis.shape[1])]
        if initial_phase is not None:
            root=np.sqrt(self.weight);seeds.append(np.linalg.lstsq(self.basis*root[:,None],np.asarray(initial_phase)*root,rcond=None)[0])
        fits=[minimize(self.objective,s,jac=True,method='BFGS',options={'maxiter':300,'gtol':1e-9}) for s in seeds]
        fit=min(fits,key=lambda f:f.fun)
        return self.basis@fit.x,dict(coefficients=fit.x,objective=fit.fun,maximum_gradient=np.max(abs(fit.jac)),
            numerical_success=bool(fit.success),termination=fit.message,degree=self.basis.shape[1]+2)
