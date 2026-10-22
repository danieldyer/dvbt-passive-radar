"""Exact rectangular-symbol Fourier moments and their Doppler derivatives."""
import numpy as np


def moments(omega):
    """Return integrals of exp(i*w*t), t*exp(i*w*t), and derivatives.

    The normalized useful-symbol time is -1/2 <= t <= 1/2. Series near zero
    avoid cancellation; the trigonometric form permits valid optimizer trials
    outside the small-Doppler expansion used by earlier support diagnostics.
    """
    w = np.asarray(omega, dtype=float)
    small = abs(w) < .1
    g, derivative = np.empty_like(w), np.empty_like(w)
    x = w[small]
    g[small] = x/12-x**3/480+x**5/53760-x**7/11612160
    derivative[small] = 1/12-x**2/160+x**4/10752-x**6/1658880
    x = w[~small]
    sine, cosine = np.sin(x/2), np.cos(x/2)
    g[~small] = (2*sine-x*cosine)/x**2
    derivative[~small] = sine/(2*x)+2*cosine/x**2-4*sine/x**3
    return np.sinc(w/(2*np.pi)), 1j*g, -g, 1j*derivative
