"""Exact profiled-residual derivatives for the positive-amplitude arc model.

The objective and observations are unchanged. This supplies a trust-region
least-squares solver with the residual Jacobian, including the response of
fitted real amplitudes to a geometry change. The mean-response information
matrix alone omits a residual-dependent term and is not that Jacobian.
"""
import numpy as np
from scipy.optimize import least_squares, nnls


def band_residual(band, parameters):
    raw, derivative = band.response(parameters)
    basis = band.amplitude_basis
    design = band.reduced(raw[:,:,None]*basis[:,None,:])
    projected = band.nuisance.remove(design[band.selected])
    matrix = np.r_[projected.real,projected.imag]
    data = np.r_[band.clean.real,band.clean.imag]
    amplitude,_ = nnls(matrix,data,maxiter=20*len(band.knots))
    residual = data-matrix@amplitude
    active = np.flatnonzero(amplitude>0)
    jacobian = np.zeros((len(data),len(parameters)))
    if not len(active):
        return residual,jacobian
    matrix = matrix[:,active]
    inverse = np.linalg.pinv(matrix,rcond=1e-10)
    mean = band.reduced(derivative*(basis@amplitude)[:,None,None])
    mean = band.nuisance.remove(mean[band.selected])
    mean_change = np.r_[mean.real,mean.imag]
    # Apply the exact averaging/whitening adjoint once instead of building a
    # separate original-row derivative tensor for every amplitude coefficient.
    complex_residual = residual[:len(band.clean)]+1j*residual[len(band.clean):]
    extended = np.zeros(band.field.y.size,complex)
    extended[band.selected] = band.nuisance.remove(complex_residual)
    original = band.field.raw_adjoint(extended.reshape(band.field.y.shape))
    row_products = np.einsum('nrp,nr->np',derivative.conj(),original).real
    amplitude_gradient = basis[:,active].T@row_products
    # Differentiate A^T (y - A a) = 0 on the current NNLS active face.
    jacobian = -mean_change+matrix@(inverse@mean_change)
    jacobian -= inverse.T@amplitude_gradient
    return residual,jacobian


class ProfiledArcSolver:
    def __init__(self, model):
        self.model = model
        self.last = None

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):
            return self.last[1:]
        mapped,transform = self.model.mapping(parameters)
        residuals,jacobians = [],[]
        for band in self.model.bands:
            residual,jacobian = band_residual(band,mapped)
            residuals.append(residual);jacobians.append(jacobian@transform)
        n = self.model.geometry.count
        residuals.append(parameters[:n]);jacobians.append(np.eye(n,len(parameters)))
        scale = np.sqrt(self.model.energy)
        residual = np.concatenate(residuals)/scale
        jacobian = np.concatenate(jacobians)/scale
        self.last = (parameters.copy(),residual,jacobian)
        return residual,jacobian

    def fit(self, seed):
        lower,upper = self.model.geometry.bounds();n=len(lower)
        if max(np.max(lower-seed[:n]),np.max(seed[:n]-upper))>1e-7:
            raise ValueError('The initial fit violates physical parameter bounds')
        seed=np.r_[np.clip(seed[:n],lower,upper),seed[n:]]
        phase_limits=np.full(len(seed)-n,np.inf)
        return least_squares(lambda p:self.evaluate(p)[0],seed,jac=lambda p:self.evaluate(p)[1],
            bounds=(np.r_[lower,-phase_limits],np.r_[upper,phase_limits]),x_scale='jac',
            ftol=2e-13,xtol=2e-13,gtol=1e-10,max_nfev=400)
