"""Conditional real-parameter information after profiling nuisance directions.

The input is the unregularized expected Gaussian information, not an observed
residual Hessian. Rank-cutoff sensitivity must accompany near-degenerate cases.
This does not convert model error or constrained support into calibration data.
"""
import numpy as np


def from_jacobian(jacobian, interest, nuisance, *, cutoff):
    """Square-root Schur elimination for a whitened real mean Jacobian.

    Unlike subtracting two large Gram matrices, direct orthogonal projection
    retains weak directions without catastrophic information cancellation.
    A proper complex observation should be stacked real/imaginary and scaled
    by sqrt(2) before calling this routine.
    """
    jacobian = np.asarray(jacobian, float)
    interest, nuisance = np.asarray(interest, int), np.asarray(nuisance, int)
    if jacobian.ndim != 2 or not np.isfinite(jacobian).all():
        raise ValueError('A finite real mean Jacobian is required')
    if set(interest)&set(nuisance) or not 0 < cutoff < 1:
        raise ValueError('Disjoint parameter sets and a numerical rank cutoff are required')
    target, other = jacobian[:, interest], jacobian[:, nuisance]
    norm = np.linalg.norm(other, axis=0)
    other = other/np.where(norm > 0, norm, 1.)
    u, singular, _ = np.linalg.svd(other, full_matrices=False)
    keep = singular > singular.max(initial=0.)*np.sqrt(cutoff)
    residual = target-u[:, keep]@(u[:, keep].T@target)
    return residual.T@residual, {'nuisance_parameters': len(nuisance), 'nuisance_rank': int(keep.sum()),
        'cutoff': cutoff, 'scaled_nuisance_eigenvalues': (singular[::-1]**2).tolist(),
        'method': 'Direct projection of the whitened mean Jacobian; no subtraction of large Gram matrices.'}


def profile(information, interest, nuisance, *, cutoff):
    information = np.asarray(information, float)
    interest, nuisance = np.asarray(interest, int), np.asarray(nuisance, int)
    if information.ndim != 2 or information.shape[0] != information.shape[1] or not np.isfinite(information).all():
        raise ValueError('Finite square real information is required')
    if set(interest)&set(nuisance) or not 0 < cutoff < 1:
        raise ValueError('Disjoint parameter sets and a numerical rank cutoff are required')
    h = information[np.ix_(nuisance, nuisance)]
    diagonal = np.diag(h)
    scale = np.ones(len(diagonal))
    positive = diagonal > 0
    scale[positive] = 1/np.sqrt(diagonal[positive])
    h = h*scale[:, None]*scale
    eigen, vectors = np.linalg.eigh((h+h.T)/2)
    keep = eigen > max(eigen.max(initial=0.), 0)*cutoff
    cross = information[np.ix_(interest, nuisance)]*scale
    projected = cross@vectors[:, keep]
    result = information[np.ix_(interest, interest)]-(projected/eigen[keep])@projected.T
    return (result+result.T)/2, {'nuisance_parameters': len(nuisance), 'nuisance_rank': int(keep.sum()),
        'cutoff': cutoff, 'scaled_nuisance_eigenvalues': eigen.tolist()}
