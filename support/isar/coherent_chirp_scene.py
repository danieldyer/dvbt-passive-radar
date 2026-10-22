"""Joint complex likelihood for a few locally quadratic echo histories.

Each echo has coupled envelope/carrier migration and one constant complex
coefficient. The existing original-symbol operator and declared stationary
nuisance are retained. These are observable path coordinates, not certified
Cartesian ship scatterers or unconstrained per-epoch phase corrections.
"""
import numpy as np
from scipy.optimize import least_squares


class ChirpScene:
    def __init__(self,arc,components):
        if components<1:raise ValueError('A nonempty echo scene is required')
        self.arc=arc;self.components=components;self.last=None

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1]
        arc=self.arc;pieces=[arc.template(p,True) for p in np.asarray(parameters).reshape(self.components,3)]
        full=np.column_stack([h for h,_ in pieces]);a=arc.nuisance.remove(full[arc.selected])
        u,s,vh=np.linalg.svd(a,full_matrices=False)
        if s[-1]<s[0]*1e-10:raise ValueError('Echo histories are unresolved')
        coefficient=vh.conj().T@((u.conj().T@arc.clean)/s)
        residual=arc.clean-a@coefficient
        derivative=arc.nuisance.remove(np.concatenate([j for _,j in pieces],axis=1)[arc.selected])
        fixed=derivative*np.repeat(coefficient,3)
        cross=np.zeros((self.components,3*self.components),complex)
        cross[np.repeat(np.arange(self.components),3),np.arange(3*self.components)]=derivative.conj().T@residual
        jacobian=-fixed+u@(u.conj().T@fixed)-u@((vh@cross)/s[:,None])
        result=dict(residual=residual,jacobian=jacobian,coefficient=coefficient,templates=full,
            condition=s[0]/s[-1],training_rss=float(np.vdot(residual,residual).real))
        self.last=(np.asarray(parameters).copy(),result)
        return result

    def fit(self,seed,origin_bounds,*,max_evaluations=100):
        arc=self.arc;limit=3*arc.scale
        lower=np.tile([origin_bounds[0],-limit,-limit],self.components)
        upper=np.tile([origin_bounds[1],limit,limit],self.components)
        scale=np.sqrt(arc.energy)
        def residual(p):
            r=self.evaluate(p)['residual'];return np.r_[r.real,r.imag]/scale
        def jacobian(p):
            j=self.evaluate(p)['jacobian'];return np.r_[j.real,j.imag]/scale
        fit=least_squares(residual,seed,jac=jacobian,bounds=(lower,upper),x_scale='jac',
            max_nfev=max_evaluations,gtol=1e-10,ftol=1e-12,xtol=1e-12)
        value=self.evaluate(fit.x)
        return dict(parameters=fit.x,success=bool(fit.success),evaluations=fit.nfev,
            gradient=float(fit.optimality),message=str(fit.message),training_rss=value['training_rss'],
            condition=value['condition'],active_bounds=np.flatnonzero(fit.active_mask),
            coefficients_real_imag=np.c_[value['coefficient'].real,value['coefficient'].imag],
            prediction=self.prediction(fit.x))

    def prediction(self,parameters):
        arc=self.arc;value=self.evaluate(parameters);moving=value['templates']@value['coefficient']
        y=arc.field.y.ravel();beta=arc.nuisance.solve(y[arc.selected]-moving[arc.selected])
        baseline=arc.nuisance.solve(y[arc.selected]);test=~arc.selected
        error=np.sum(abs(y[test]-moving[test]-arc.design[test]@beta)**2)
        reference=np.sum(abs(y[test]-arc.design[test]@baseline)**2)
        return dict(test_rss=float(error),stationary_only_test_rss=float(reference),
            explained_fraction=float(1-error/reference))
