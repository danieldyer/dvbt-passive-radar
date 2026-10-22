"""Phase variance of a known phasor in proper-complex unit Gaussian noise.

This is a noise-only comparison, not a correction or a fitted noise scale.
Integrating the complex Gaussian density in polar coordinates gives the
Rician phase density used below. Fit-parameter uncertainty is separate.
"""
from functools import lru_cache
import math

from scipy.integrate import quad
from scipy.special import erfc,erfcx


@lru_cache(maxsize=8192)
def phase_variance(snr):
    rho=float(snr)
    if not math.isfinite(rho) or rho<0:raise ValueError('Finite nonnegative phasor SNR required')
    if rho==0:return math.pi**2/3
    root=math.sqrt(rho);base=math.exp(-rho)/(2*math.pi)
    def integrand(theta):
        cosine=math.cos(theta);a=root*cosine
        if a<0:density=base*(1+math.sqrt(math.pi)*a*erfcx(-a))
        else:density=base+a/(2*math.sqrt(math.pi))*math.exp(-rho*math.sin(theta)**2)*erfc(-a)
        return theta*theta*density
    # Scale the first interval with the peak width so adaptive quadrature
    # cannot miss a very narrow high-SNR density near theta=0.
    cut=min(math.pi,12/math.sqrt(rho+1))
    value=quad(integrand,0,cut,epsabs=1e-13,epsrel=2e-10,limit=150)[0]
    if cut<math.pi:value+=quad(integrand,cut,math.pi,epsabs=1e-13,epsrel=2e-10,limit=150)[0]
    return 2*value
