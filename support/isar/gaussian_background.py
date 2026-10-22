"""Finite Gaussian background uncertainty, without a hard clutter projection.

After subtracting an independent mean, y = target + Z b + noise, with unit
proper-complex noise and b ~ CN(0,I). Marginalization gives C = I + Z Z^H.
The fixed square-root precision acts on both data and every target derivative.
No covariance parameter is estimated from the target interval here.
"""
import numpy as np


class GaussianBackground:
    def __init__(self,design,selected):
        self.design=np.asarray(design);self.selected=np.asarray(selected,bool)
        if self.design.ndim!=2 or self.selected.shape!=(len(design),) or not np.any(self.selected):
            raise ValueError('Background factors require a nonempty selected row set')
        if not np.isfinite(self.design).all():raise ValueError('Non-finite covariance factor')
        self.u,self.singular_values,self.vh=np.linalg.svd(self.design[self.selected],full_matrices=False)
        self.rank=int(np.count_nonzero(self.singular_values))
        s=self.singular_values
        self.loss=1-1/np.hypot(1,s)
        self.posterior_scale=s/(1+s*s)

    def remove(self,values):
        """Apply C^(-1/2); this is Hermitian but deliberately not idempotent."""
        values=np.asarray(values);scale=self.loss.reshape((-1,)+(1,)*(values.ndim-1))
        return values-self.u@(scale*(self.u.conj().T@values))

    def solve(self,values):
        """Posterior mean of b for frozen out-of-sample prediction."""
        values=np.asarray(values);scale=self.posterior_scale.reshape((-1,)+(1,)*(values.ndim-1))
        return self.vh.conj().T@(scale*(self.u.conj().T@values))


def predictive_factors(values,weights):
    """Weighted independent-block covariance plus finite reference-mean error.

This is a conditional stationary independent-block model. Its transfer must
be checked; slow drift can invalidate it. The unbiased weighted sample
covariance is multiplied by (1 + sum w^2) for a new draw minus the mean.
Independent block realizations are rows; returned factors are columns.
"""
    values=np.asarray(values);weights=np.asarray(weights,float)
    if values.ndim!=2 or weights.shape!=(len(values),) or len(weights)<2 or np.any(weights<=0):
        raise ValueError('Need at least two positive-weight independent blocks')
    weights=weights/weights.sum();mean=weights@values;concentration=weights@weights
    factor=(values-mean)*np.sqrt(weights*(1+concentration)/(1-concentration))[:,None]
    return mean,factor.T


def set_background(base,factors):
    """Configure an existing amplitude arc after mean subtraction on a copy."""
    if np.asarray(factors).shape[0]!=base.field.y.size:raise ValueError('Factor rows differ from field')
    base.design=np.asarray(factors)
    base.nuisance=GaussianBackground(base.design,base.selected)
    base.clean=base.nuisance.remove(base.field.y.ravel()[base.selected])
    base.energy=float(np.vdot(base.clean,base.clean).real)
    return base
