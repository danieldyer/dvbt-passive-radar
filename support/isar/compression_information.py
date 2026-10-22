"""Fisher information retained by explicit linear ISAR preprocessing.

All results are conditional on the stated, known Gaussian covariance. Unknown
complex target amplitude is eliminated before comparing delay/motion bounds.
These are information-retention tests, not estimator-efficiency certificates.
"""
import numpy as np
from scipy.linalg import solve_triangular

from field import average


def point_information(signal, derivatives):
    """Whitened proper-complex observations; real nonlinear parameters."""
    signal = np.asarray(signal).ravel()
    derivatives = np.asarray(derivatives).reshape(len(signal), -1)
    energy = float(np.vdot(signal, signal).real)
    if not energy > 0 or not np.isfinite(derivatives).all():
        raise ValueError('Need finite derivatives and a nonzero target')
    projected = derivatives-signal[:, None]*(signal.conj()@derivatives)[None, :]/energy
    information = 2*np.real(projected.conj().T@projected)
    return energy, (information+information.T)/2


def relative_information(full, reduced):
    """Generalized Fisher eigenvalues, with only a change of parameter units."""
    full, reduced = np.asarray(full), np.asarray(reduced)
    scale = 1/np.sqrt(np.diag(full))
    lower = np.linalg.cholesky(full*scale[:, None]*scale)
    relative = solve_triangular(lower, reduced*scale[:, None]*scale, lower=True)
    relative = solve_triangular(lower, relative.T, lower=True).T
    eigenvalues = np.linalg.eigvalsh((relative+relative.T)/2)
    if eigenvalues.min() < -1e-9 or eigenvalues.max() > 1+1e-8:
        raise ValueError(f'Compression violates the information inequality: {eigenvalues}')
    return eigenvalues


def range_crop_information(frequency_radians, spectrum, offsets, delay_samples):
    """A matched-profile crop versus the full known occupied spectrum.

    spectrum is the nonnegative normalized matched-filter spectral information,
    not a signed/truncated autocorrelation FFT. Fractional delays are evaluated
    by the exact Fourier sum, independently of the pipeline's interpolator.
    """
    w, q, offsets = map(np.asarray, (frequency_radians, spectrum, offsets))
    if w.ndim != 1 or q.shape != w.shape or np.any(q < 0) or not np.isfinite(q).all() or not q.sum() > 0:
        raise ValueError('Expected finite, nonnegative spectral information')
    q = q/q.sum()
    fourier = np.exp(1j*offsets[:, None]*w)
    covariance = (fourier*q)@fourier.conj().T
    lower = np.linalg.cholesky((covariance+covariance.conj().T)/2)
    signal = np.exp(-1j*w*delay_samples)
    full_energy, full = point_information(np.sqrt(q)*signal, (-1j*w*np.sqrt(q)*signal)[:, None])
    mean = fourier@(q*signal)
    derivative = fourier@(q*signal*(-1j*w))
    white = solve_triangular(lower, np.column_stack([mean, derivative]), lower=True)
    energy, information = point_information(white[:, 0], white[:, 1:])
    return {'matched_snr_fraction': energy/full_energy,
        'delay_information_fraction': float(relative_information(full, information)[0]),
        'covariance_condition': float(np.linalg.cond(covariance))}


def temporal_information(field, raw_signal, raw_derivatives):
    """Original cropped profile rows versus the implemented weighted binning.

    The conditional range covariance is field's average pulse. Rows are
    independent; uniform within-symbol waveform energy gives Var(C0)=1,
    Var(C1)=1/12 and zero cross moment. The centre-Doppler combination's noise
    factor is carried through the fixed averaging operation explicitly.
    """
    d = field.data
    weights, block = d['row_information'], field.metadata['block_rows']
    factor = 1+np.asarray(d['centre_omega'])**2/12
    root = np.sqrt(8192*weights/factor)
    raw_white = (raw_signal@field.whiten.T)*root[:, None]
    derivative_white = np.einsum('nkp,rk->nrp', raw_derivatives, field.whiten)*root[:, None, None]
    original_energy, original = point_information(raw_white, derivative_white)
    mean, information = average(raw_signal, weights, block)
    derivative, _ = average(raw_derivatives, weights, block)
    mean_factor, _ = average(factor, weights, block)
    root = np.sqrt(np.divide(8192*information, mean_factor,
        out=np.zeros_like(information), where=mean_factor > 0))
    compressed = (mean@field.whiten.T)*root[:, None]
    derivative = np.einsum('nkp,rk->nrp', derivative, field.whiten)*root[:, None, None]
    reduced_energy, reduced = point_information(compressed, derivative)
    return {'matched_snr_fraction': reduced_energy/original_energy,
        'relative_fisher_eigenvalues': relative_information(original, reduced).tolist(),
        'original_information': original.tolist(), 'compressed_information': reduced.tolist(),
        'symbol_combination_noise_factor_minmax': [float(factor.min()), float(factor.max())]}


def scene_information(full_gram, reduced_gram, *, relative_floor=1e-10):
    """Retention for all mixtures in a declared complex scene subspace.

    This is stronger than pointwise energy retention: destructive interference
    can expose different modes. Numerically unresolved modes are counted and
    explicitly excluded from this conditional bound, never silently truncated
    in a forward model or fit.
    """
    full = (full_gram+full_gram.conj().T)/2
    reduced = (reduced_gram+reduced_gram.conj().T)/2
    eigen, vector = np.linalg.eigh(full)
    keep = eigen > eigen[-1]*relative_floor
    if not keep.any() or eigen[0] < -eigen[-1]*1e-12:
        raise ValueError('Invalid original scene information')
    whitening = vector[:, keep]/np.sqrt(eigen[keep])
    comparison = whitening.conj().T@reduced@whitening
    retained = np.linalg.eigvalsh((comparison+comparison.conj().T)/2)
    if retained[0] < -1e-7 or retained[-1] > 1+1e-6:
        raise ValueError('Scene compression violates the information inequality')
    return {'relative_information_eigenvalues': retained.tolist(),
        'retained_modes': int(keep.sum()), 'unresolved_modes': int((~keep).sum()),
        'relative_information_floor': relative_floor,
        'original_relative_information_eigenvalues': (eigen/eigen[-1]).tolist(),
        'worst_information_loss_db': float(-10*np.log10(retained[0]))}
