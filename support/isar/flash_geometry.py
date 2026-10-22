"""Short-arc multistatic siting likelihood with explicit reflector nuisance.

Input sufficient local summaries are envelope delay, phase-derived path rate
and path acceleration, together with their JOINT conditional covariance. A
constant unknown complex reflectivity in each band removes absolute carrier
phase, but not its time derivative. Receiver and transmitter coordinates enter
the same exact excess-path geometry. Site priors are reported separately from
measured information.
"""
import itertools
import numpy as np
from scipy.linalg import block_diag, solve_triangular
from scipy.optimize import least_squares

from antenna_motion import Antennas
from motion import ecef


def observables(position, velocity, acceleration, transmitters, receiver):
    """Excess path, its first derivative and second derivative, metres / s."""
    transmitter = np.asarray(transmitters)
    position, velocity, acceleration = map(np.asarray, (position, velocity, acceleration))
    # Analytic Euclidean norm (no complex conjugation) also permits exact
    # complex-step derivatives of the real geometric model.
    path = -np.sqrt(np.sum((transmitter-receiver)**2, axis=1))
    path = path.astype(np.result_type(position, velocity, acceleration, transmitter, receiver))
    rate = np.zeros_like(path);second = rate.copy()
    for antenna in (transmitter, np.broadcast_to(receiver, transmitter.shape)):
        vector = position-antenna
        distance = np.sqrt(np.sum(vector**2, axis=1))
        direction = vector/distance[:, None]
        projected = direction@velocity
        path += distance
        rate += projected
        second += direction@acceleration+(velocity@velocity-projected**2)/distance
    return np.column_stack([path, rate, second])


def local_ais_polynomial(track, utc, rotation, receiver):
    """Smooth Cartesian prior near the flash, avoiding rounded Unix-time steps.

    The fit is degree five over ±4 s, verified on an independent dense grid.
    It is solely a numerical representation of the existing AIS interpolation.
    Its first and second derivatives drive the geometry consistently.
    """
    def position(offset):
        lat, lon = track.curve((utc-track.epoch)+offset).T
        return (ecef(lat, lon, track.reference_height_ellipsoid_m)-receiver)@rotation.T
    grid = np.linspace(-4, 4, 33)
    coefficients = np.polynomial.polynomial.polyfit(grid, position(grid), 5)
    check = np.linspace(-3.97, 3.97, 100)
    error = np.max(np.linalg.norm(np.polynomial.polynomial.polyval(check, coefficients).T-position(check), axis=1))
    if error > 1e-5:
        raise ValueError('Local Cartesian AIS representation exceeds 10 micrometres')
    return coefficients, float(error)


def assignments(options):
    return list(itertools.product(*[[o for o in options if o['channel'] == c] for c in range(5)]))


class Flash:
    def __init__(self, measured, covariance, track, utc, rotation, receiver, assignment,
                 *, include_range=True, include_acceleration=True, free_antennas=True,
                 receiver_sigma=(15., 15., 7.5), receiver_up_bounds_m=None, ais_reference=None):
        self.measured = np.asarray(measured)
        self.track, self.utc = track, utc
        self.rotation, self.receiver = np.asarray(rotation), np.asarray(receiver)
        self.transmitters = np.array([o['transmitter_ecef_m'] for o in assignment])
        self.local_transmitters = (self.transmitters-self.receiver)@self.rotation.T
        self.ais_reference = ais_reference
        if ais_reference is None:
            self.ais_coefficients, self.ais_representation_error_m = local_ais_polynomial(track, utc, self.rotation, self.receiver)
        else:
            self.ais_representation_error_m = ais_reference.maximum_error_m
        self.sites = [o['site_id'] for o in assignment]
        self.antennas = Antennas(rotation, self.sites, receiver_sigma=receiver_sigma)
        self.free_antennas = free_antennas
        self.nantenna = 18 if free_antennas else 0
        self.columns = np.array(([0] if include_range else [])+[1]+([2] if include_acceleration else []))
        covariance = np.asarray(covariance)[:, self.columns[:, None], self.columns[None, :]]
        self.whiten = [solve_triangular(np.linalg.cholesky(v), np.eye(len(self.columns)), lower=True)
                       for v in covariance]
        self.nuisance_sigma = np.array([150., 150., 20., .5, .5, .1, .1, .1, .05, 1.])
        self.count = self.nantenna+len(self.nuisance_sigma)
        self.lower = np.full(self.count, -3.)
        self.upper = np.full(self.count, 3.)
        if free_antennas and receiver_up_bounds_m is not None:
            low, high = receiver_up_bounds_m
            if not low < high or receiver_sigma[2] <= 0:
                raise ValueError('Receiver height bounds need a positive vertical prior scale')
            self.lower[17], self.upper[17] = low/receiver_sigma[2], high/receiver_sigma[2]

    def bounds(self):
        return self.lower, self.upper

    def remaining_cost_decrease(self, fit):
        """Local Gauss-Newton decrease with outward active-bound directions removed."""
        x = fit.x[:self.count]
        active = ((x < self.lower+1e-6)&(fit.grad > 0)) | ((x > self.upper-1e-6)&(fit.grad < 0))
        free = ~active
        normal = fit.jac[:, free].T@fit.jac[:, free]
        return float(fit.grad[free]@np.linalg.solve(normal, fit.grad[free]))

    def state(self, parameters):
        q = np.asarray(parameters[self.nantenna:])*self.nuisance_sigma
        reference = getattr(self, 'ais_reference', None)
        if reference is None:
            p = np.polynomial.polynomial.polyval(q[-1], self.ais_coefficients)+q[:3]
            v = np.polynomial.polynomial.polyval(q[-1], np.polynomial.polynomial.polyder(self.ais_coefficients, 1))+q[3:6]
            a = np.polynomial.polynomial.polyval(q[-1], np.polynomial.polynomial.polyder(self.ais_coefficients, 2))+q[6:9]
        else:
            p, v, a = [reference(q[-1], i)+q[3*i:3*i+3] for i in range(3)]
        if self.free_antennas:
            offsets = self.antennas.factor@parameters[:18]
            tx = self.local_transmitters+offsets[:15].reshape(5, 3)
            rx = offsets[15:]
        else:
            tx, rx = self.local_transmitters, np.zeros(3)
        return p, v, a, tx, rx

    def prediction(self, parameters):
        return observables(*self.state(parameters))

    def data_residual(self, parameters):
        error = self.prediction(parameters)-self.measured
        return np.concatenate([w@e[self.columns] for w, e in zip(self.whiten, error)])

    def residual(self, parameters):
        return np.r_[self.data_residual(parameters), parameters]

    def fit(self, seed=None):
        initial = np.zeros(self.count) if seed is None else np.asarray(seed)
        return least_squares(self.residual, initial, jac='cs', bounds=self.bounds(),
                             max_nfev=150, ftol=1e-12, xtol=1e-12, gtol=1e-9)

    def describe(self, fit):
        ndata = len(self.data_residual(fit.x))
        jacobian = fit.jac[:ndata]
        result = {'sites': self.sites, 'parameters': fit.x.tolist(),
                  'data_squared_residual': float(np.sum(fit.fun[:ndata]**2)),
                  'prior_squared_residual': float(np.sum(fit.fun[ndata:]**2)),
                  'penalized_squared_residual': float(np.sum(fit.fun**2)),
                  'first_order_optimality': float(fit.optimality), 'termination': fit.message,
                  'evaluations': fit.nfev, 'converged':bool(fit.success),
                  'bound_hits': int(np.count_nonzero((fit.x<self.lower+1e-4)|(fit.x>self.upper-1e-4))),
                  'bound_parameter_indices':np.flatnonzero((fit.x<self.lower+1e-4)|(fit.x>self.upper-1e-4)).tolist(),
                  'predicted_observables': self.prediction(fit.x).tolist(),
                  'reflector_offset_enu_m': (fit.x[self.nantenna:self.nantenna+3]*self.nuisance_sigma[:3]).tolist(),
                  'velocity_correction_enu_mps': (fit.x[self.nantenna+3:self.nantenna+6]*self.nuisance_sigma[3:6]).tolist(),
                  'acceleration_correction_enu_mps2': (fit.x[self.nantenna+6:self.nantenna+9]*self.nuisance_sigma[6:9]).tolist(),
                  'relative_ais_time_offset_s': float(fit.x[-1])}
        result['ais_numerical_representation_error_m'] = self.ais_representation_error_m
        result['estimated_remaining_gauss_newton_squared_cost_decrease'] = self.remaining_cost_decrease(fit)
        if self.free_antennas:
            antenna = jacobian[:, :18];nuisance = jacobian[:, 18:]
            u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
            q = u[:, s > s[0]*1e-8]
            projected = antenna-q@(q.T@antenna)
            singular = np.linalg.svd(projected, compute_uv=False)
            values = np.r_[singular**2, np.zeros(max(0, 18-len(singular)))]
            result.update(antenna_fit=self.antennas.describe(fit.x[:18]),
                          data_information_eigenvalues_after_free_reflector_nuisance=values.tolist(),
                          data_information_eigenvalues_with_reflector_nuisance_fixed=np.linalg.eigvalsh(antenna.T@antenna)[::-1].tolist(),
                          nuisance_data_rank=int(q.shape[1]),
                          posterior_covariance=np.linalg.inv(fit.jac.T@fit.jac).tolist())
        return result
