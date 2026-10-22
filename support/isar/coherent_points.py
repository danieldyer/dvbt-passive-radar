"""Joint physical point coordinates and shared scattering phases.

Motion, sites, height and envelope origins stay fixed. Nonnegative band/point
strengths are profiled by NNLS. A common reference-epoch carrier pivot is only
a parameterization: every band receives the same point-phase factor, so it
does not change the physical model or infer an extra calibration.
"""
import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import nnls

from motion import excess_path


class CoherentPoints:
    def __init__(self, bands, points, range_origins, *, reference=3):
        self.bands, self.points, self.origins = bands, np.asarray(points), range_origins
        self.reference = reference
        self.other = [c for c in range(len(bands)) if c != reference]
        self.count = len(points)
        self.energy = sum(b.energy for b in bands)
        ref = bands[reference]
        middle = len(ref.field.data['row_time'])//2
        self.reference_position = ref.state['position_ecef_m'][middle]
        self.reference_frame = ref.state['body_to_ecef'][middle]
        self.reference_tx = ref.field.data['transmitter_ecef_m']
        self.reference_rx = ref.field.data['receiver_ecef_m']
        self.reference_k = ref.field.k
        self.reference_path = excess_path(self.reference_position+self.points@self.reference_frame.T,
            self.reference_tx, self.reference_rx)[0]
        self.last, self.evaluations = None, 0
        self.prepared, self.geometry_evaluations = None, 0

    def coordinates(self, parameters):
        points = self.points.copy()
        points[:, :2] += parameters[:2*self.count].reshape(-1, 2)
        return points

    def pivot(self, points):
        path, gradient = excess_path(self.reference_position+points@self.reference_frame.T,
            self.reference_tx, self.reference_rx)
        return self.reference_k*(path-self.reference_path), self.reference_k*(gradient@self.reference_frame)[:, :2]

    def initial(self, points, point_phases, band_phases):
        pivot, _ = self.pivot(np.asarray(points))
        return np.r_[(np.asarray(points)[:, :2]-self.points[:, :2]).ravel(),
            np.asarray(point_phases)-pivot, np.asarray(band_phases)[self.other]]

    def prepare(self, coordinates):
        coordinates = np.asarray(coordinates)
        if self.prepared is not None and np.array_equal(coordinates, self.prepared['coordinates']):
            return self.prepared
        points = self.coordinates(coordinates)
        pivot, pivot_gradient = self.pivot(points)
        entries = []
        for band, origin in zip(self.bands, self.origins):
            values, partials = [], []
            for j, point in enumerate(points):
                raw, derivative = band.field.point(point, motion=band.state, derivatives=True, range_bias_m=origin)
                values.append(raw.ravel())
                partials.append(derivative[:, :, :2].reshape(-1, 2)+1j*raw.ravel()[:, None]*pivot_gradient[j])
            target = np.column_stack(values)
            design = band.nuisance.remove(target[band.selected])
            q, r = qr(design, mode='economic', check_finite=False)
            singular = np.linalg.svd(r, compute_uv=False)
            if singular[-1] <= singular[0]*1e-10:
                raise ValueError('Moving physical points made the scene rank deficient')
            rhs = q.conj().T@band.clean
            entries.append({'target': target, 'design': design, 'r': r, 'rhs': rhs,
                'outside': float(np.linalg.norm(band.clean-q@rhs)**2),
                'derivative': np.column_stack(partials)[band.selected],
                'condition': float(singular[0]/singular[-1])})
        self.geometry_evaluations += 1
        self.prepared = {'coordinates': coordinates.copy(), 'pivot': pivot, 'bands': entries}
        return self.prepared

    def evaluate(self, parameters, *, jacobian=False):
        parameters = np.asarray(parameters)
        if (self.last is not None and np.array_equal(parameters, self.last['parameters'])
                and (not jacobian or 'residual_jacobian' in self.last)):
            return self.last
        prepared = self.prepare(parameters[:2*self.count])
        band_phase = np.zeros(len(self.bands)); band_phase[self.other] = parameters[3*self.count:]
        point_phase = parameters[2*self.count:3*self.count]+prepared['pivot']
        gradient = np.zeros_like(parameters)
        value, outputs, residual_parts, jacobian_parts = 0., [], [], []
        expected_information = np.zeros((len(parameters), len(parameters))) if jacobian else None
        for c, (band, entry) in enumerate(zip(self.bands, prepared['bands'])):
            target, design, r, rhs = [entry[key] for key in ('target', 'design', 'r', 'rhs')]
            phase = np.exp(1j*(point_phase+band_phase[c]))
            a = r*phase
            real_a, real_y = np.r_[a.real, a.imag], np.r_[rhs.real, rhs.imag]
            strength, _ = nnls(real_a, real_y, maxiter=20*self.count)
            coefficient = phase*strength
            residual = band.clean-design@coefficient
            kkt = real_a.T@(real_a@strength-real_y)
            kkt[(strength == 0)&(kkt > 0)] = 0.
            if abs(kkt).max()/self.energy > 1e-8:
                raise ValueError('Positive strength solve has not converged')
            value += np.vdot(residual, residual).real
            # Residual is already orthogonal to stationary nuisance. Its inner
            # product with raw derivatives therefore equals the projected one.
            derivative = entry['derivative']
            gradient[:2*self.count] -= 2*np.real((residual.conj()@derivative)*np.repeat(coefficient, 2))
            local = 2*np.imag((residual.conj()@design)*coefficient)
            gradient[2*self.count:3*self.count] += local
            if c in self.other:
                gradient[3*self.count+self.other.index(c)] = local.sum()
            if jacobian:
                # Exact separable-least-squares residual derivative on the
                # current NNLS active set. The second projection term matters
                # away from a noiseless solution; an amplitude-frozen Jacobian
                # would give the correct gradient but the wrong residual map.
                fixed = np.zeros((len(residual), len(parameters)), complex)
                fixed[:, :2*self.count] = derivative*np.repeat(coefficient, 2)
                fixed[:, 2*self.count:3*self.count] = 1j*design*coefficient
                if c in self.other:
                    fixed[:, 3*self.count+self.other.index(c)] = (1j*design@coefficient)
                fixed = band.nuisance.remove(fixed)
                real_fixed = np.r_[fixed.real, fixed.imag]
                active = strength > 0.
                real_design = np.r_[(design*phase).real, (design*phase).imag][:, active]
                qa, ra = qr(real_design, mode='economic', check_finite=False)
                cross = np.zeros((self.count, len(parameters)))
                indices = np.arange(2*self.count)
                cross[indices//2, indices] = np.real((residual.conj()@derivative)*np.repeat(phase, 2))
                point_cross = -np.imag((residual.conj()@design)*phase)
                cross[np.arange(self.count), 2*self.count+np.arange(self.count)] = point_cross
                if c in self.other:
                    cross[:, 3*self.count+self.other.index(c)] = point_cross
                derivative_residual = -real_fixed+qa@(qa.T@real_fixed)
                expected_information += 2*(derivative_residual.T@derivative_residual)
                if active.any():
                    derivative_residual -= qa@solve_triangular(ra.T, cross[active], lower=True, check_finite=False)
                jacobian_parts.append(derivative_residual)
                residual_parts.append(np.r_[residual.real, residual.imag])
            outputs.append({'target': target, 'coefficient': coefficient,
                'residual': residual, 'condition': entry['condition']})
        self.evaluations += 1
        self.last = {'parameters': parameters.copy(), 'value': float(value/self.energy),
            'gradient': gradient/self.energy, 'bands': outputs,
            'band_phase_radians': band_phase.tolist(), 'point_phase_radians': point_phase.tolist()}
        if jacobian:
            self.last.update(real_residual=np.concatenate(residual_parts)/np.sqrt(self.energy),
                residual_jacobian=np.concatenate(jacobian_parts)/np.sqrt(self.energy),
                expected_information=expected_information)
        return self.last

    def objective(self, parameters):
        result = self.evaluate(parameters)
        return result['value'], result['gradient']
