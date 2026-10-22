"""Gaussian diffuse-scene likelihood on fixed stationary contrasts.

The deterministic core is profiled; the diffuse coefficients are integrated:
  z ~ CN(A0 a, variance * (I + B B^H / strength)).
B already includes the square-root cell-area scale. Stationary projection is
fixed, independent of physical motion. Its null directions contribute neither
data nor log determinant. This is the likelihood of those retained contrasts,
not a claim that the measured residual obeys this Gaussian model.

The quadratic and its exact variable-projection derivative reuse mixed_scene.
The additional log determinant uses an SVD and original-row adjoint. The
returned Gauss-Newton matrix is only a solver preconditioner, not a Fisher
matrix or an asserted Hessian of the marginalized objective.
"""
import numpy as np

from mixed_scene import MixedMotion, profile as mixed_profile
from ridge_motion import point_derivatives


class Covariance:
    """I + B B^H / strength, without an observation-sized dense matrix."""
    def __init__(self, design, strength):
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError('Diffuse covariance needs finite positive strength')
        self.u, self.s, self.vh = np.linalg.svd(design, full_matrices=False)
        self.strength = strength
        self.ratio = self.s**2/strength
        self.logdet = float(np.log1p(self.ratio).sum())
        self.design_gradient = (self.u*(self.s/(self.s**2+strength)))@self.vh

    def apply_inverse(self, values, *, square_root=False):
        inner = self.u.conj().T@values
        scale = 1/np.sqrt(1+self.ratio) if square_root else 1/(1+self.ratio)
        scale = scale.reshape((-1,)+(1,)*(values.ndim-1))
        # Preserve tiny retained eigenvalues without subtracting two nearly
        # equal numbers inside the diffuse subspace.
        return values-self.u@inner+self.u@(scale*inner)


def volume(band, parameters, scale, core, strength, target, *, derivatives=False):
    """logdet and its real physical-coordinate gradient on original rows."""
    if strength is None:
        return {'logdet': 0., 'gradient': np.zeros(len(parameters))}
    design = band.nuisance.remove(target[band.selected, core:])*scale[core:]
    covariance = Covariance(design, strength)
    gradient = np.zeros(len(parameters))
    if derivatives:
        for j, derivative in enumerate(point_derivatives(band, parameters)):
            if j < core:
                continue
            observed = np.zeros(band.field.y.size, complex)
            observed[band.selected] = band.nuisance.remove(covariance.design_gradient[:, j-core])*scale[j]
            adjoint = band.field.raw_adjoint(observed.reshape(band.field.y.shape))
            local = np.einsum('nr,nrq->nq', adjoint.conj(), derivative)
            gradient += 2*np.real(np.concatenate([
                b.T@local[:, k]+db.T@local[:, band.components+k]
                for k, (b, db) in enumerate(zip(band.bases, band.rate_bases))]))
    return {'logdet': covariance.logdet, 'gradient': gradient}


def profile(band, parameters, scale, core, strength, *, variance=1., derivatives=False):
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('The declared thermal-noise variance must be positive')
    fitted = mixed_profile(band, parameters, scale, core, strength, derivatives=derivatives)
    covariance = volume(band, parameters, scale, core, strength, fitted['target'], derivatives=derivatives)
    quadratic = fitted['value']/variance
    degrees = (int(np.count_nonzero(band.field.root_information[band.selected_time] > 0))*
        band.field.y.shape[1]-band.nuisance.rank-core)
    if degrees <= 0:
        raise ValueError('Need residual degrees of freedom after eliminating nuisance means')
    fitted.update(value=quadratic+covariance['logdet'], quadratic=quadratic,
        logdet=covariance['logdet'], residual_degrees=degrees,
        quadratic_per_residual_degree=quadratic/degrees)
    if derivatives:
        fitted['quadratic_gradient'] = fitted['gradient']/variance
        fitted['volume_gradient'] = covariance['gradient']
        fitted['gradient'] = fitted['quadratic_gradient']+fitted['volume_gradient']
        fitted['gauss_newton'] /= variance
    return fitted


class MarginalMotion(MixedMotion):
    def __init__(self, bands, initial_motion, weights, cores, strengths, *, variances=None):
        super().__init__(bands, initial_motion, weights, cores, strengths)
        self.variances = np.ones(len(bands)) if variances is None else np.asarray(variances)
        if self.variances.shape != (len(bands),) or not np.isfinite(self.variances).all() or np.any(self.variances <= 0):
            raise ValueError('Declare one positive noise variance per channel')
        self.energy = sum(b.energy/v for b, v in zip(bands, self.variances))

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        motion = self.expand(parameters)
        value, gradient = 0., np.zeros(len(parameters))
        curvature = np.zeros((len(parameters), len(parameters)))
        diagnostics = []
        for band, scale, core, strength, variance in zip(self.bands, self.scales,
                self.cores, self.strengths, self.variances):
            fitted = profile(band, motion, scale, core, strength, variance=variance, derivatives=True)
            value += fitted['value']
            gradient += fitted['gradient'][self.active]
            curvature += fitted['gauss_newton'][np.ix_(self.active, self.active)]
            diagnostics.append({k: fitted[k] for k in ('quadratic', 'logdet', 'residual_degrees',
                'quadratic_per_residual_degree', 'condition', 'effective_rank')})
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': value/self.energy,
            'gradient': gradient/self.energy, 'gauss_newton': curvature/self.energy,
            'scene_conditions': [d['condition'] for d in diagnostics],
            'effective_scene_ranks': [d['effective_rank'] for d in diagnostics],
            'channels': diagnostics, 'evaluations': self.evaluations}
        return self.last
