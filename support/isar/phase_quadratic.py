"""Likelihood phase synchronization with a computable global upper bound.

For whitened observations y_c = exp(i phi_c) A_c a + n_c, eliminating the
unknown complex scene a gives max z^H C z, |z_c|=1, z_c=exp(-i phi_c).
QR forms C without normal equations. The semidefinite dual is
min sum(mu), diag(mu)-C >= 0. Every feasible dual is an upper bound, even
when the nonconvex phase solution is not globally optimal. A small primal/dual
gap certifies only this declared shared-scene likelihood, not its physical fit.

The dual log-det central path follows the Max-SDR approach described by
Evers (2019), https://scholar.afit.edu/etd/2240/ . No convex-solver dependency.
"""
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize


def phase_matrix(designs, observations):
    if len(designs) != len(observations) or not designs:
        raise ValueError('Need corresponding nonempty channel designs and data')
    columns = designs[0].shape[1]
    if any(a.ndim != 2 or a.shape[1] != columns or y.shape != (len(a),)
           for a, y in zip(designs, observations)):
        raise ValueError('Expected a common complex scene dimension')
    design = np.concatenate(designs)
    q, r = np.linalg.qr(design, mode='reduced')
    singular = np.linalg.svd(r, compute_uv=False)
    if singular[-1] <= singular[0]*1e-10:
        raise ValueError('Shared scene is numerically rank deficient')
    projected = np.empty((columns, len(designs)), complex)
    first = 0
    for c, (a, y) in enumerate(zip(designs, observations)):
        projected[:, c] = q[first:first+len(a)].conj().T@y
        first += len(a)
    matrix = projected.conj().T@projected
    return (matrix+matrix.conj().T)/2, r, projected


def refine_phase(matrix, initial, reference=0):
    n = len(matrix)
    other = np.delete(np.arange(n), reference)
    initial = np.angle(initial/initial[reference])[other]

    def objective(parameters):
        angle = np.zeros(n)
        angle[other] = parameters
        z = np.exp(1j*angle)
        product = matrix@z
        return -float(np.vdot(z, product).real), -2*np.imag(z.conj()*product)[other]

    fitted = minimize(objective, initial, jac=True, method='BFGS',
        options={'gtol': 2e-12, 'maxiter': 300})
    angle = np.zeros(n)
    angle[other] = fitted.x
    z = np.exp(1j*angle)
    # Exact coordinate updates also handle an occasionally stalled BFGS solve.
    off = matrix-np.diag(np.diag(matrix))
    for _ in range(1000):
        previous = z.copy()
        for c in other:
            value = off[c]@z
            if abs(value) > 0:
                z[c] = value/abs(value)
        if np.linalg.norm(z-previous) < 1e-12:
            break
    return z


def dual_bound(matrix, tolerance=1e-9):
    """Strictly feasible log-det dual path, small dense matrices only."""
    n = len(matrix)
    diagonal = np.full(n, np.max(np.sum(abs(matrix), axis=1))+1.)
    identity = np.eye(n)
    trace = []
    last_inverse = None
    tau = 1.
    while tau*n > tolerance/2:
        for iteration in range(100):
            slack = np.diag(diagonal)-matrix
            lower = np.linalg.cholesky(slack)
            inverse = cho_solve((lower, True), identity)
            gradient = 1-tau*inverse.diagonal().real
            hessian = tau*abs(inverse)**2
            step = np.linalg.solve(hessian, -gradient)
            decrement = -gradient@step
            if decrement <= max(1e-20, tau*1e-12):
                break
            value = diagonal.sum()-2*tau*np.log(lower.diagonal().real).sum()
            scale = 1.
            for _ in range(60):
                trial = diagonal+scale*step
                try:
                    factor = np.linalg.cholesky(np.diag(trial)-matrix)
                    changed = trial.sum()-2*tau*np.log(factor.diagonal().real).sum()
                    if changed <= value+1e-4*scale*(gradient@step):
                        break
                except np.linalg.LinAlgError:
                    pass
                scale *= .5
            else:
                raise ValueError('Semidefinite dual line search failed')
            diagonal = trial
        last_inverse = tau*cho_solve((np.linalg.cholesky(np.diag(diagonal)-matrix), True), identity)
        trace.append({'barrier': tau, 'iterations': iteration+1,
            'maximum_central_gradient': float(abs(1-last_inverse.diagonal().real).max())})
        tau *= .2
    # An explicit numerical margin makes the published upper bound feasible.
    smallest = float(np.linalg.eigvalsh(np.diag(diagonal)-matrix)[0])
    diagonal += max(0., -smallest)+64*np.finfo(float).eps*(1+np.linalg.norm(matrix, 2))
    return diagonal, last_inverse, trace


def solve(matrix, *, reference=0, tolerance=1e-8, seed=7812):
    matrix = np.asarray(matrix, complex)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2
            or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.conj().T, atol=1e-12)):
        raise ValueError('Need a finite Hermitian phase matrix')
    if not 0 <= reference < len(matrix):
        raise ValueError('Invalid phase reference')
    # The diagonal is constant on the feasible set and carries no phase information.
    constant = float(np.trace(matrix).real)
    off = matrix-np.diag(np.diag(matrix))
    scale = float(np.linalg.norm(off, 2))
    if scale == 0:
        return np.ones(len(matrix), complex), {'captured_energy': constant,
            'upper_bound': constant, 'absolute_gap': 0., 'globally_certified': True,
            'phase_identifiable': False}
    normalized = off/scale
    diagonal, covariance, trace = dual_bound(normalized, tolerance=tolerance)
    eigen, vectors = np.linalg.eigh((covariance+covariance.conj().T)/2)
    root = vectors*np.sqrt(np.maximum(eigen, 0))
    rng = np.random.default_rng(seed)
    starts = [np.ones(len(matrix), complex), np.exp(1j*np.angle(vectors[:, -1]))]
    for _ in range(64):
        trial = root@(rng.normal(size=len(matrix))+1j*rng.normal(size=len(matrix)))
        starts.append(np.exp(1j*np.angle(trial)))
    candidates = [refine_phase(normalized, z, reference) for z in starts]
    phase = max(candidates, key=lambda z: np.vdot(z, normalized@z).real)
    phase /= phase[reference]
    energy = float(np.vdot(phase, matrix@phase).real)
    upper = float(constant+scale*diagonal.sum())
    gap = max(0., upper-energy)
    gradient = 2*np.imag(phase.conj()*(normalized@phase))
    return phase, {'captured_energy': energy, 'upper_bound': upper,
        'absolute_gap': gap, 'relative_gap_to_phase_dependent_scale': gap/scale,
        'globally_certified': bool(gap/scale <= tolerance*2),
        'phase_identifiable': None,
        'maximum_scaled_phase_gradient': float(abs(gradient).max()),
        'dual_minimum_slack_eigenvalue': float(np.linalg.eigvalsh(np.diag(diagonal)-normalized)[0]*scale),
        'relaxed_primal_eigenvalues': eigen.tolist(), 'central_path': trace,
        'scope': 'Numerical semidefinite upper bound for the fixed-magnitude, shared-complex-scene phase likelihood. Does not certify the scene, absolute phase reference or robustness to scattering mismatch.'}
