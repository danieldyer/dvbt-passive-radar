"""Finite Gaussian uncertainty for a stationary response at fixed target motion.

y = u*a + B*b + e, b ~ CN(0, tau2*I), e ~ CN(0, sigma2*I).
The complex moving amplitude is unknown. Restricted marginal likelihood fits
the two variance components on training observations only; sigma2 profiles
analytically, leaving one scalar variance ratio. The posterior predictor and
its conditional covariance follow by Gaussian conditioning. No weak singular
direction is discarded. This is a declared stochastic clutter model, not a
universal assertion about the recording or a data-dependent display weight.
"""
import numpy as np
from scipy.optimize import brentq


class GaussianStationary:
    def __init__(self, data, moving, stationary):
        self.y, self.moving = np.asarray(data), np.asarray(moving)
        self.u, self.s, self.vh = np.linalg.svd(stationary, full_matrices=False)
        if len(self.y) <= len(self.s)+1 or self.s[0] <= 0:
            raise ValueError('Need positive stationary support and residual observations')
        self.scale = self.s[0]**2
        self.spectrum = self.s**2/self.scale
        self.yp, self.up = self.u.conj().T@self.y, self.u.conj().T@self.moving
        self.yperp, self.uperp = self.y-self.u@self.yp, self.moving-self.u@self.up
        self.moving_norm = float(np.vdot(self.moving, self.moving).real)
        self.degrees = len(self.y)-1

    def profile(self, eta):
        if not np.isfinite(eta) or eta < 0:
            raise ValueError('Use a finite nonnegative variance ratio')
        ratio = eta*self.spectrum
        w = 1/(1+ratio)
        norm = float(np.vdot(self.uperp, self.uperp).real+np.sum(w*abs(self.up)**2))
        cross = np.vdot(self.uperp, self.yperp)+np.sum(w*self.up.conj()*self.yp)
        amplitude = cross/norm
        residual = self.yp-amplitude*self.up
        quadratic = float(np.linalg.norm(self.yperp-amplitude*self.uperp)**2+np.sum(w*abs(residual)**2))
        sigma2 = quadratic/self.degrees
        value = self.degrees*np.log(sigma2)+np.log1p(ratio).sum()+np.log(norm/self.moving_norm)
        rate = self.spectrum*w*w
        derivative = (self.degrees*(-np.sum(rate*abs(residual)**2))/quadratic
                      +np.sum(self.spectrum*w)-np.sum(rate*abs(self.up)**2)/norm)
        return {'value': float(value), 'eta': float(eta), 'sigma2': sigma2,
            'tau2': float(sigma2*eta/self.scale), 'amplitude': amplitude,
            'inverse_moving_norm_without_sigma': norm, 'derivative_eta': float(derivative),
            'derivative_log_eta': float(eta*derivative)}

    def fit(self):
        # Scan the one-dimensional likelihood, then refine every bracketed
        # local minimum. The zero-clutter model is an explicit boundary case.
        grid = np.arange(-30., 21.)
        values = [self.profile(np.exp(x))['value'] for x in grid]
        while not (values[-1] > min(values)+20 and values[-1] > values[-2] > values[-3]):
            if grid[-1] >= 80:
                raise ValueError('Stationary variance minimum is not bracketed')
            extra = np.arange(grid[-1]+1, grid[-1]+11)
            values.extend(self.profile(np.exp(x))['value'] for x in extra)
            grid = np.r_[grid, extra]
        candidates = [self.profile(0.)]
        gradients = [self.profile(np.exp(x))['derivative_log_eta'] for x in grid]
        for i in range(len(grid)-1):
            if gradients[i] < 0 and gradients[i+1] >= 0:
                root = brentq(lambda x: self.profile(np.exp(x))['derivative_log_eta'],
                    grid[i], grid[i+1], xtol=1e-10)
                candidates.append(self.profile(np.exp(root)))
        fitted = min(candidates, key=lambda r: r['value'])
        stationary = (fitted['derivative_eta'] >= -1e-7 if fitted['eta'] == 0
                      else abs(fitted['derivative_log_eta']) < 1e-4)
        return {k: v for k, v in fitted.items() if k != 'amplitude'} | {
            'amplitude_real_imag': [float(fitted['amplitude'].real), float(fitted['amplitude'].imag)],
            'stationary': bool(stationary), 'log_eta_grid': grid.tolist(),
            'restricted_objective_grid': values, 'local_candidates': len(candidates),
            'variance_scope': 'Training-only restricted marginal likelihood at frozen motion. Estimated-hyperparameter uncertainty is not included in the conditional prediction bound.'}

    def predict(self, stationary, moving, eta, *, sigma2=None):
        fitted = self.profile(eta)
        if sigma2 is None:
            sigma2 = fitted['sigma2']
        if sigma2 <= 0:
            raise ValueError('Positive noise variance is required')
        ratio = eta*self.spectrum
        w = 1/(1+ratio)
        shrink = np.divide(ratio*w, self.s, out=np.zeros_like(self.s), where=self.s > 0)
        basis = stationary@self.vh.conj().T
        amplitude = fitted['amplitude']
        prediction = moving*amplitude+basis@(shrink*(self.yp-amplitude*self.up))
        effective_moving = moving-basis@(shrink*self.up)
        amplitude_variance = sigma2/fitted['inverse_moving_norm_without_sigma']
        coefficient_variance = sigma2*eta/self.scale*w
        error = (len(moving)*sigma2+float(np.sum(abs(basis)**2*coefficient_variance[None, :]))
                 +amplitude_variance*float(np.vdot(effective_moving, effective_moving).real))
        return prediction, {'amplitude': amplitude, 'amplitude_variance': amplitude_variance,
            'expected_prediction_error': error, 'noise_variance': sigma2,
            'stationary_variance': sigma2*eta/self.scale}
