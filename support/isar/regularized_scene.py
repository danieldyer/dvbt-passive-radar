"""Fixed-motion complex scene inversion with explicit Gaussian ridge prior.

The prior changes the statistical model and is reported as such. Its strength
is selected using blocked training predictions, never by sharpening an image.
Known band phase factors belong in the supplied design. Optional spectrum
coefficients are positive real constants, with sum of squares equal to the
number of bands to fix the scene/spectrum scale gauge.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh
from scipy.optimize import brentq
from rigid_patch import profile_static


@dataclass
class Observation:
    gram: np.ndarray
    rhs: np.ndarray
    energy: float

    @classmethod
    def from_rows(cls,design,data,root,selected,groups,*,gram=None):
        a=profile_static(design,root,selected,groups)[selected].reshape(-1,design.shape[2])
        y=profile_static(data,root,selected,groups)[selected].ravel()
        if gram is None:gram=a.conj().T@a
        else:
            if gram.shape!=(design.shape[2],design.shape[2]):raise ValueError('Cached operator dimension changed')
            index=np.unique([0,design.shape[2]//2,design.shape[2]-1]);sample=a[:,index]
            np.testing.assert_allclose(gram[np.ix_(index,index)],sample.conj().T@sample,rtol=1e-10,atol=1e-8)
        return cls((gram+gram.conj().T)/2,a.conj().T@y,float(np.vdot(y,y).real))

    def error(self,coefficient):
        x=np.asarray(coefficient)
        return float(self.energy-2*np.vdot(x,self.rhs).real+np.vdot(x,self.gram@x).real)


def solve_fixed(observations,spectrum,strength):
    if strength<=0:raise ValueError('Ridge strength must be strictly positive')
    gram=sum(abs(g)**2*o.gram for o,g in zip(observations,spectrum))
    rhs=sum(np.conj(g)*o.rhs for o,g in zip(observations,spectrum))
    gram=gram.copy();gram.flat[::len(gram)+1]+=strength
    return cho_solve(cho_factor(gram,lower=True),rhs)


class FixedRidgePath:
    """Exact repeated ridge solves for one unchanged operator/spectrum."""
    def __init__(self,observations,spectrum):
        self.gram=sum(abs(g)**2*o.gram for o,g in zip(observations,spectrum))
        self.rhs=sum(np.conj(g)*o.rhs for o,g in zip(observations,spectrum))
        self.eigenvalue,self.vector=eigh(self.gram)
        self.projected=self.vector.conj().T@self.rhs

    def solve(self,strength):
        if strength<=0 or self.eigenvalue[0]+strength<=0:raise ValueError('Ridge normal operator is not positive definite')
        return self.vector@(self.projected/(self.eigenvalue+strength))


def score(observations,coefficient,spectrum):
    errors=[o.error(g*coefficient) for o,g in zip(observations,spectrum)]
    energy=sum(o.energy for o in observations);error=sum(errors)
    return dict(cost=error,energy=energy,explained_fraction=1-error/energy,channel_cost=errors,
        channel_explained_fraction=[1/e.energy*(e.energy-r) for e,r in zip(observations,errors)])


def positive_spectrum(quadratic,linear,radius_squared):
    """Exact diagonal quadratic on the nonnegative sphere (few RF bands).

    Enumerate active faces. On a face, g_i=b_i/(a_i+mu). Its norm
    equation is strictly convex between its sign-constrained poles, so
    there are at most two roots. Zero-linear-term hard cases are included.
    This avoids an equality-constrained optimizer's stopping tolerance.
    """
    a=np.asarray(quadratic,float);b=np.asarray(linear,float);n=len(a)
    if n>8 or radius_squared<=0:raise ValueError('Use a small physical band spectrum')
    scale=max(np.max(abs(a)),np.max(abs(b)),1e-300);a=a/scale;b=b/scale
    if np.all(b>0):
        # The full positive face has a positive-semidefinite Lagrangian
        # Hessian and therefore gives the global minimum directly.
        def norm_error(mu):
            with np.errstate(over='ignore',divide='ignore'):
                return float(np.sum((b/(a+mu))**2)-radius_squared)
        lower=np.nextafter(float(-a.min()),np.inf);upper=max(1.,lower+1.)
        while norm_error(upper)>0:upper=2*upper+1
        mu=brentq(norm_error,lower,upper,xtol=1e-15)
        value=b/(a+mu)
        return value*np.sqrt(radius_squared/np.dot(value,value))
    candidates=list(np.sqrt(radius_squared)*np.eye(n))
    nonzero=np.flatnonzero(b!=0);zero=np.flatnonzero(b==0)
    for face in range(1,1<<len(nonzero)):
        index=nonzero[[bool(face&(1<<j)) for j in range(len(nonzero))]];aa=a[index];bb=b[index]
        lower=float(np.max(-aa[bb>0],initial=-np.inf));upper=float(np.min(-aa[bb<0],initial=np.inf))
        if lower>=upper:continue
        def equation(mu):
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                return float(np.sum((bb/(aa+mu))**2)-radius_squared)
        def derivative(mu):
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                return float(-2*np.sum(bb**2/(aa+mu)**3))
        lo=np.nextafter(lower,np.inf) if np.isfinite(lower) else min(-1.,upper-1.)
        hi=np.nextafter(upper,-np.inf) if np.isfinite(upper) else max(1.,lower+1.)
        if not np.isfinite(lower):
            while equation(lo)>0:lo=2*lo-1
        if not np.isfinite(upper):
            while equation(hi)>0:hi=2*hi+1
        intervals=[(lo,hi)]
        if np.isfinite(lower) and np.isfinite(upper):
            mid=brentq(derivative,lo,hi,xtol=1e-15)
            intervals=[(lo,mid),(mid,hi)] if equation(mid)<=0 else []
        for left,right in intervals:
            if equation(left)*equation(right)>0:continue
            mu=brentq(equation,left,right,xtol=1e-15)
            value=np.zeros(n);value[index]=bb/(aa+mu)
            if np.any(value[index]<=0) or abs(np.dot(value,value)-radius_squared)>1e-6:continue
            value*=np.sqrt(radius_squared/np.dot(value,value));candidates.append(value)
    # A zero linear term permits a singular stationary multiplier. Only
    # one coordinate with that diagonal value is needed to fill the norm.
    for j in zero:
        for face in range(1<<len(nonzero)):
            index=nonzero[[bool(face&(1<<k)) for k in range(len(nonzero))]]
            denominator=a[index]-a[j]
            if np.any(denominator==0):continue
            value=np.zeros(n);value[index]=b[index]/denominator
            remaining=radius_squared-np.dot(value,value)
            if np.any(value[index]<0) or remaining<0:continue
            value[j]=np.sqrt(remaining);candidates.append(value)
    return min(candidates,key=lambda v:a@(v*v)-2*b@v)


def fit(observations,strength,*,spectrum=None,fit_spectrum=True,fit_band_phases=False,maximum_iterations=100):
    count=len(observations);g=np.ones(count) if spectrum is None else np.asarray(spectrum).copy()
    if (not fit_band_phases and (np.iscomplexobj(g) or np.any(g<0))) or np.vdot(g,g).real<=0:
        raise ValueError('Need a nonzero spectrum with the declared phase constraints')
    if fit_band_phases:g=g.astype(complex)
    g*=np.sqrt(count/np.vdot(g,g).real);previous=np.inf;trace=[]
    scale=max(sum(o.energy for o in observations),1e-300)
    for iteration in range(maximum_iterations if fit_spectrum else 1):
        x=solve_fixed(observations,g,strength)
        if fit_spectrum:
            quadratic=np.array([np.vdot(x,o.gram@x).real for o in observations])
            linear=np.array([np.vdot(x,o.rhs) for o in observations])
            if fit_band_phases:
                magnitude=positive_spectrum(quadratic,abs(linear),count)
                g=magnitude*np.exp(1j*np.angle(linear))
            else:g=positive_spectrum(quadratic,linear.real,count)
        x=solve_fixed(observations,g,strength)
        value=sum(o.error(v*x) for o,v in zip(observations,g))+strength*np.vdot(x,x).real
        if value>previous+1e-10*scale:raise ValueError('Alternating conditional ridge objective increased')
        trace.append(value/scale)
        if previous-value<1e-10*scale:
            dg=2*np.array([v*np.vdot(x,o.gram@x).real-(np.vdot(x,o.rhs) if fit_band_phases else np.vdot(x,o.rhs).real) for o,v in zip(observations,g)])/scale
            tangent=dg-np.vdot(dg,g).real/count*g
            projected=tangent if fit_band_phases else np.where(g>1e-9,tangent,np.minimum(tangent,0.))
            if max(abs(projected))<1e-7:break
        previous=value
    if fit_band_phases:
        gauge=np.exp(1j*np.angle(g[min(3,count-1)]));g/=gauge;x*=gauge
    gram=sum(abs(v)**2*o.gram for o,v in zip(observations,g));rhs=sum(np.conj(v)*o.rhs for o,v in zip(observations,g))
    scene_gradient=(gram@x+strength*x-rhs)/scale
    dg=2*np.array([v*np.vdot(x,o.gram@x).real-(np.vdot(x,o.rhs) if fit_band_phases else np.vdot(x,o.rhs).real) for o,v in zip(observations,g)])/scale
    multiplier=-np.vdot(dg,g).real/(2*count);gradient=dg+2*multiplier*g
    projected=gradient if fit_band_phases else np.where(g>1e-9,gradient,np.minimum(gradient,0.))
    return x,g,dict(strength=float(strength),iterations=len(trace),objective=trace[-1],objective_trace=trace,
        maximum_scene_gradient=float(abs(scene_gradient).max()),maximum_spectrum_kkt=float(abs(projected).max()) if fit_spectrum else None,
        spectrum_fitted=fit_spectrum,numerical_convergence=bool(max(abs(scene_gradient))<1e-8 and (not fit_spectrum or max(abs(projected))<1e-6)),
        relative_band_phases_fitted=fit_band_phases,spectrum_magnitudes=abs(g).tolist(),
        constant_band_phase_correction_radians=np.angle(g).tolist(),scene_is_regularized_estimate=True)
