"""One shared physical echo with an explicit secondary echo in each band.

The primary has one Cartesian p/v/a state and physical antenna coordinates.
Each band has its own constant scattering phase, as in PhysicalAmplitudeArc.
A secondary quadratic path is nuisance, not an identified material point or a
phase correction to the primary. Both components have original-row positive
brightness and share the measured symbol operator and stationary nuisance.

This deliberately tests whether the proposed primary association survives
overlapping echoes. It does not itself certify a unique two-scatterer scene.
"""
import numpy as np
from scipy.optimize import least_squares

from amplitude_arc import AmplitudeArc
from amplitude_pair import AmplitudePair


def fit_parameters(model,seed,free,*,max_evaluations=300):
    """Bounded profiled least squares with explicit fixed coordinates."""
    free=np.asarray(free,dtype=int);initial=np.clip(seed,model.lower,model.upper)
    def unpack(x):
        p=initial.copy();p[free]=x;return p
    fitted=least_squares(lambda x:model.evaluate(unpack(x))[0],initial[free],
        jac=lambda x:model.evaluate(unpack(x))[1][:,free],bounds=(model.lower[free],model.upper[free]),
        x_scale='jac',ftol=2e-13,xtol=2e-13,gtol=1e-10,max_nfev=max_evaluations)
    p=unpack(fitted.x);value,gradient=model.objective(p)
    active=((p<model.lower+1e-5)&(gradient>0))|((p>model.upper-1e-5)&(gradient<0))
    projected=gradient[free].copy();projected[active[free]]=0
    return p,dict(objective=value,evaluations=fitted.nfev,termination=str(fitted.message),
        optimality=float(fitted.optimality),projected_gradient=float(abs(projected).max()),
        free_indices=free.tolist(),
        bound_indices=free[(p[free]<model.lower[free]+1e-5)|(p[free]>model.upper[free]-1e-5)].tolist(),
        scope='First-order local fit; no convergence certificate from solver termination alone.')


class PhysicalNuisancePair(AmplitudePair):
    def __init__(self,physical):
        super().__init__(physical)
        self.secondary=AmplitudeArc(physical.field,physical.training,physical.knots,stationary_paths=physical.delays)
        self.primary_count=physical.geometry.count+5

    def response_blocks(self,parameters):
        n=self.primary_count
        return [self.base.response(parameters[:n]),self.secondary.response(parameters[n:])]


class SharedPhysicalPair:
    def __init__(self,bands,*,origin_bounds=(-.8,2.8),rate_bound=4.):
        if not bands or len({b.channel for b in bands})!=len(bands):
            raise ValueError('Each channel must occur once')
        self.bands=bands;self.geometry=bands[0].geometry
        if any(b.geometry is not self.geometry for b in bands):
            raise ValueError('One shared geometry is required')
        self.primary_count=self.geometry.count+5
        self.count=self.primary_count+4*len(bands)
        self.pairs=[PhysicalNuisancePair(b) for b in bands]
        self.indices=[np.r_[np.arange(self.primary_count),self.primary_count+4*i+np.arange(4)]
                      for i in range(len(bands))]
        self.energy=sum(b.energy for b in bands);self.last=None
        lo,hi=self.geometry.bounds()
        self.lower=np.r_[lo,np.full(5,-np.inf),np.empty(4*len(bands))]
        self.upper=np.r_[hi,np.full(5,np.inf),np.empty(4*len(bands))]
        for pair,index in zip(self.pairs,self.indices):
            limit=rate_bound*pair.secondary.scale
            self.lower[index[-4:]]=[origin_bounds[0],-np.inf,-limit,-limit]
            self.upper[index[-4:]]=[origin_bounds[1],np.inf,limit,limit]

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        residuals=[];jacobians=[]
        for pair,index in zip(self.pairs,self.indices):
            value=pair.evaluate(parameters[index]);scale=np.sqrt(pair.energy/self.energy)
            residuals.append(value['residual']*scale)
            j=np.zeros((len(value['residual']),self.count));j[:,index]=value['jacobian']*scale
            jacobians.append(j)
        n=self.geometry.count
        residuals.append(parameters[:n]/np.sqrt(self.energy))
        jacobians.append(np.eye(n,self.count)/np.sqrt(self.energy))
        residual=np.concatenate(residuals);jacobian=np.concatenate(jacobians)
        self.last=(parameters.copy(),residual,jacobian)
        return residual,jacobian

    def objective(self,parameters):
        residual,jacobian=self.evaluate(parameters)
        return float(residual@residual),2*jacobian.T@residual

    def prediction(self,parameters,testing):
        return [pair.prediction(parameters[index],test) for pair,index,test in zip(self.pairs,self.indices,testing)]

    def fit(self,seed,*,free_antennas=False,max_evaluations=300):
        """Use explicit fixed coordinates; bounds are never extra observations."""
        n=self.geometry.nantenna
        free=np.arange(self.count) if free_antennas else np.arange(n,self.count)
        present={b.channel for b in self.bands}
        free=np.array([i for i in free if not (self.geometry.count<=i<self.primary_count and
            i-self.geometry.count not in present)])
        return fit_parameters(self,seed,free,max_evaluations=max_evaluations)

    def mean_derivative(self,parameters):
        """Conditional mean Jacobian after static and active real brightness."""
        rows=[]
        for pair,index in zip(self.pairs,self.indices):
            local=pair.evaluate(parameters[index])['mean_derivative']*np.sqrt(2.)
            j=np.zeros((len(local),self.count));j[:,index]=local;rows.append(j)
        return np.concatenate(rows)

    def information(self,parameters):
        """Mean-response information, with all secondary trajectories removed."""
        j=self.mean_derivative(parameters)
        def remove(values,columns):
            norms=np.linalg.norm(columns,axis=0);nonzero=norms>0
            if not np.any(nonzero):return values,0
            a=columns[:,nonzero]/norms[nonzero]
            u,s,_=np.linalg.svd(a,full_matrices=False)
            keep=s>np.finfo(float).eps*max(a.shape)*s[0]
            q=u[:,keep]
            return values-q@(q.T@values),int(sum(keep))
        # Independent midpoint phases are removed first so physical/reference
        # phase conventions cannot change the projected motion information.
        j,phase_rank=remove(j,j[:,self.geometry.count:self.primary_count])
        primary=j[:,:self.geometry.count]
        primary,secondary_rank=remove(primary,j[:,self.primary_count:])
        return dict(matrix=(primary.T@primary).tolist(),phase_rank=phase_rank,secondary_rank=secondary_rank,
            scope='Conditional thermal information after static, active real brightness, band phases and both rate/curvature and envelope of every secondary; no priors or active-bound information.')
