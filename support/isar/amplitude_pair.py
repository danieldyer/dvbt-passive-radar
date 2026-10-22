"""Quadratic echo clusters with original-row real brightness and declared clutter.

Both echoes use the existing range/phase/symbol operator. Nonnegative linear
brightness coefficients are profiled jointly, not fitted to averaged samples.
The exact residual derivative includes the motion of those coefficients.
No minimum component separation or arbitrary phase-trajectory correction is
imposed: degeneracy is reported through the conditional information matrix.
"""
import numpy as np
from scipy.optimize import least_squares,nnls


class AmplitudeCluster:
    def __init__(self,base,components):
        if not isinstance(components,int) or components<1:raise ValueError('A positive component count is required')
        self.base=base;self.energy=base.energy;self.last=None;self.components=components

    def response_blocks(self,parameters):
        """Responses for two disjoint parameter blocks, in parameter order."""
        return [self.base.response(p) for p in np.asarray(parameters).reshape(self.components,4)]

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1]
        b=self.base;basis=b.amplitude_basis;k=basis.shape[1]
        raw,change=zip(*self.response_blocks(parameters))
        design=np.concatenate([b.reduced(r[:,:,None]*basis[:,None,:]) for r in raw],axis=1)
        projected=b.nuisance.remove(design[b.selected]);matrix=np.r_[projected.real,projected.imag]
        data=np.r_[b.clean.real,b.clean.imag]
        amplitude,_=nnls(matrix,data,maxiter=40*k);residual=data-matrix@amplitude
        active=amplitude>0;matrix=matrix[:,active];inverse=np.linalg.pinv(matrix,rcond=1e-10)
        mean=np.concatenate([b.nuisance.remove(b.reduced(d*(basis@a)[:,None,None])[b.selected])
            for d,a in zip(change,amplitude.reshape(self.components,k))],axis=1)
        mean=np.r_[mean.real,mean.imag];mean-=matrix@(inverse@mean)
        extended=np.zeros(b.field.y.size,complex)
        extended[b.selected]=b.nuisance.remove(residual[:len(b.clean)]+1j*residual[len(b.clean):])
        adjoint=b.field.raw_adjoint(extended.reshape(b.field.y.shape))
        sizes=[d.shape[2] for d in change]
        if sum(sizes)!=len(parameters):raise ValueError('Response blocks must cover all parameters')
        starts=np.r_[0,np.cumsum(sizes)]
        coefficient_gradient=np.zeros((self.components*k,len(parameters)))
        for i,d in enumerate(change):
            coefficient_gradient[i*k:(i+1)*k,starts[i]:starts[i+1]]=basis.T@np.einsum('nrp,nr->np',d.conj(),adjoint).real
        derivative=-mean-inverse.T@coefficient_gradient[active]
        result=dict(residual=residual/np.sqrt(self.energy),jacobian=derivative/np.sqrt(self.energy),
            design=design,amplitude=amplitude,mean_derivative=mean,information=2*mean.T@mean)
        self.last=(parameters.copy(),result)
        return result

    def fit(self,seed,origin_bounds,rate_bound=4.):
        bound=rate_bound*self.base.scale
        lower=np.tile([origin_bounds[0],-np.inf,-bound,-bound],self.components)
        upper=np.tile([origin_bounds[1],np.inf,bound,bound],self.components)
        fit=least_squares(lambda p:self.evaluate(p)['residual'],np.clip(seed,lower,upper),
            jac=lambda p:self.evaluate(p)['jacobian'],bounds=(lower,upper),x_scale='jac',
            ftol=2e-13,xtol=2e-13,gtol=1e-11,max_nfev=400)
        value=self.evaluate(fit.x);gradient=2*value['jacobian'].T@value['residual']
        active=((fit.x<lower+1e-5)&(gradient>0))|((fit.x>upper-1e-5)&(gradient<0))
        gradient[active]=0
        eigen=np.linalg.eigvalsh(value['information'])
        def full_gradient(p):
            evaluated=self.evaluate(p)
            return 2*evaluated['jacobian'].T@evaluated['residual']
        step=2e-5
        hessian=np.column_stack([(full_gradient(fit.x+d)-full_gradient(fit.x-d))/(2*step) for d in np.eye(len(seed))*step])
        free=~active
        curvature=np.linalg.eigvalsh(((hessian+hessian.T)/2)[np.ix_(free,free)])
        return dict(parameters=fit.x.tolist(),objective=float(value['residual']@value['residual']),
            projected_gradient=float(abs(gradient).max()),evaluations=fit.nfev,termination=str(fit.message),
            bound_indices=np.flatnonzero((fit.x<lower+1e-5)|(fit.x>upper-1e-5)).tolist(),
            information_eigenvalues=eigen.tolist(),information_rank=int(np.count_nonzero(eigen>eigen[-1]*1e-10)),
            curvature_eigenvalues=curvature.tolist(),
            conditional_covariance=np.linalg.pinv(value['information'],rcond=1e-10).tolist())

    def prediction(self,parameters,testing):
        b=self.base;testing=np.asarray(testing,bool)
        if testing.shape!=b.training.shape or np.any(testing&b.training) or not np.any(testing):raise ValueError('Prediction needs unused samples')
        value=self.evaluate(parameters);design,amplitude=value['design'],value['amplitude'];k=len(b.knots)
        components=np.stack([design[:,i*k:(i+1)*k]@amplitude[i*k:(i+1)*k] for i in range(self.components)])
        predicted=components.sum(axis=0);y=b.field.y.ravel()
        beta=b.nuisance.solve(y[b.selected]-predicted[b.selected]);clean=y-b.design@beta
        use=np.repeat(testing,len(b.field.offsets));error=float(np.sum(abs(clean[use]-predicted[use])**2))
        baseline=b.nuisance.solve(y[b.selected]);base_error=float(np.sum(abs(y[use]-b.design[use]@baseline)**2))
        return dict(heldout_error=error,heldout_baseline_error=base_error,heldout_explained_fraction=1-error/base_error,
            amplitude_coefficients=amplitude.reshape(self.components,k).tolist(),amplitude_knots_s=b.knots.tolist(),
            component_to_sum_energy_ratio=float(np.sum(abs(components[:,b.selected])**2)/max(np.sum(abs(predicted[b.selected])**2),1e-300)))


class AmplitudePair(AmplitudeCluster):
    def __init__(self,base):
        super().__init__(base,2)
