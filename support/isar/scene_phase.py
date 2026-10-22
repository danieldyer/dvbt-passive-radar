"""Profile reflector amplitudes and stationary clutter at every phase trial.

Freezing an ill-conditioned scene can spuriously anchor a phase correction to
large, unobservable stationary components. This variable-projection objective
re-solves the complete complex linear problem at each candidate phase. Constant
and linear phase are removed as declared alignment gauges, not estimated here.
"""
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from spline_motion import basis
from stationary import Stationary
from stationarity import check, polish


def phase_basis(time, interval, selected, spacing):
    spline = basis(time, interval, spacing)
    alignment = np.c_[np.ones(len(time)), (time-np.mean(interval))/np.ptp(interval)]
    spline -= alignment@np.linalg.lstsq(alignment[selected], spline[selected], rcond=None)[0]
    _, singular, vh = np.linalg.svd(spline[selected], full_matrices=False)
    keep = singular > singular[0]*1e-10
    return spline@(vh[keep].T/singular[keep])*np.sqrt(selected.sum())


class ScenePhase:
    def __init__(self, design, data, background, time, selected, ranges, interval, spacing=3.):
        if data.ndim != 2 or design.shape[0] != data.size:
            raise ValueError('Expected time/range data and flattened physical dictionary')
        self.time, self.selected = np.asarray(time), np.asarray(selected)
        self.basis = phase_basis(time, interval, selected, spacing)
        mask = np.zeros(data.shape, bool)
        mask[np.ix_(selected, ranges)] = True
        self.row_time = np.nonzero(mask)[0]
        self.design = design[mask.ravel()]
        self.nuisance = Stationary(background, mask.ravel())
        self.data = self.nuisance.remove(data[mask])
        self.energy = float(np.vdot(self.data, self.data).real)

    def project(self, parameters):
        phase = self.basis@parameters
        raw = np.exp(1j*phase[self.row_time, None])*self.design
        design = self.nuisance.remove(raw)
        coefficient, _, rank, singular = np.linalg.lstsq(design, self.data, rcond=1e-10)
        residual = self.data-design@coefficient
        return coefficient, residual, raw@coefficient, rank, singular

    def objective(self, parameters):
        coefficient, residual, prediction, rank, singular = self.project(parameters)
        row_gradient = -2*np.real(residual.conj()*(1j*prediction))
        gradient = self.basis[self.row_time].T@row_gradient/self.energy
        return float(np.vdot(residual, residual).real/self.energy), gradient

    def fit(self, maximum_phase_radians=.8, iterations=120):
        if not np.isfinite(maximum_phase_radians) or maximum_phase_radians <= 0 or iterations <= 0:
            raise ValueError('Need a positive finite phase search bound and iteration limit')
        initial = np.zeros(self.basis.shape[1])
        lo, hi = np.full(len(self.time), -maximum_phase_radians), np.full(len(self.time), maximum_phase_radians)
        lower, upper = np.full(len(initial), -np.inf), np.full(len(initial), np.inf)
        result = minimize(self.objective, initial, jac=True, method='SLSQP',
            constraints=[LinearConstraint(self.basis, lo, hi)],
            options={'maxiter': iterations, 'ftol': 1e-12})
        parameters = result.x
        value, gradient = self.objective(parameters)
        residual, violation = check(parameters, gradient, self.basis, lo, hi, lower, upper)
        refinement = None
        if residual > 1e-8 and violation < 1e-7:
            parameters, refinement = polish(self.objective, parameters, np.eye(len(parameters)),
                self.basis, lo, hi, lower, upper)
        value, gradient = self.objective(parameters)
        residual, violation = check(parameters, gradient, self.basis, lo, hi, lower, upper)
        coefficient, _, _, rank, singular = self.project(parameters)
        phase = self.basis@parameters
        return phase, coefficient, {'objective': value, 'converged': bool(residual < 1e-7 and violation < 1e-7),
            'maximum_kkt_residual': residual, 'maximum_constraint_violation': violation,
            'phase_bound_radians': maximum_phase_radians, 'phase_bound_hit': bool(abs(phase).max() > maximum_phase_radians-1e-6),
            'maximum_phase_degrees': float(np.degrees(abs(phase).max())),
            'iterations': result.nit, 'termination': str(result.message), 'stationarity_polish': refinement,
            'scene_rank': int(rank), 'scene_columns': len(coefficient),
            'scene_condition': float(singular[0]/singular[-1]), 'parameters': parameters.tolist()}
