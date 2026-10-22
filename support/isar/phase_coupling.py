"""Test a common reflector-phase model without tying band magnitudes together.

The fixed physical dictionaries are QR-compressed without changing their
likelihood. Nonnegative band/point magnitudes are profiled at every candidate;
one point phase is shared across bands, with one constant band phase offset.
Repeated phase indices allow changing strengths without changing those phases.
This is a scattering hypothesis to validate, not automatic phase calibration.
"""
import numpy as np
from scipy.optimize import minimize, nnls

from positive_phase import maximum as positive_phase_maximum
from signed_phase import maximum as signed_phase_maximum


class PhaseCoupling:
    def __init__(self, designs, observations, reference=3, phase_indices=None, point_coordinates_m=None,
                 nonnegative=True):
        self.reference = reference
        self.nonnegative = nonnegative
        self.coordinate_maximum = positive_phase_maximum if nonnegative else signed_phase_maximum
        self.other = [c for c in range(len(designs)) if c != reference]
        self.columns = designs[0].shape[1]
        self.phase_indices = np.arange(self.columns) if phase_indices is None else np.asarray(phase_indices)
        if self.phase_indices.shape != (self.columns,) or not np.issubdtype(self.phase_indices.dtype, np.integer):
            raise ValueError('One integer reflector-phase index is required per amplitude column')
        self.points = int(self.phase_indices.max())+1
        if not np.array_equal(np.unique(self.phase_indices), np.arange(self.points)):
            raise ValueError('Reflector-phase indices must be contiguous and nonnegative')
        self.spatial = None
        if point_coordinates_m is not None:
            coordinates = np.asarray(point_coordinates_m, float)
            if coordinates.shape != (self.points, 2) or not np.all(np.isfinite(coordinates)):
                raise ValueError('Spatial phase control needs two finite horizontal coordinates per reflector')
            self.spatial = coordinates/100.
        self.energy = sum(np.vdot(y, y).real for y in observations)
        self.systems, self.independent = [], []
        for design, observation in zip(designs, observations):
            q, r = np.linalg.qr(design, mode='reduced')
            singular = np.linalg.svd(r, compute_uv=False)
            if singular[-1] <= singular[0]*1e-10:
                raise ValueError('Rank deficient point dictionary')
            rhs = q.conj().T@observation
            outside = np.vdot(observation-q@rhs, observation-q@rhs).real
            self.systems.append((r, rhs, outside))
            self.independent.append(np.linalg.solve(r, rhs))
        self.independent = np.array(self.independent)
        self.last = None

    def phases(self, parameters):
        band = np.zeros(len(self.systems))
        end = self.points+len(self.other)
        band[self.other] = parameters[self.points:end]
        phase = band[:, None]+parameters[self.phase_indices]
        if self.spatial is not None:
            slopes = np.zeros((len(self.systems), 2))
            slopes[self.other] = parameters[end:].reshape(-1, 2)
            phase += slopes@self.spatial[self.phase_indices].T
        return np.exp(1j*phase), band

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        phases, _ = self.phases(parameters)
        cost, gradient, coefficients, inner = 0., np.zeros(len(parameters)), [], []
        for c, ((r, rhs, outside), phase) in enumerate(zip(self.systems, phases)):
            design = r*phase
            real_design, real_rhs = np.r_[design.real, design.imag], np.r_[rhs.real, rhs.imag]
            magnitude = (nnls(real_design, real_rhs, maxiter=20*self.columns)[0] if self.nonnegative
                         else np.linalg.lstsq(real_design, real_rhs, rcond=None)[0])
            coefficient = phase*magnitude
            residual = rhs-r@coefficient
            cost += np.vdot(residual, residual).real+outside
            local = 2*np.imag(residual.conj()@(r*coefficient))
            gradient[:self.points] += np.bincount(self.phase_indices, weights=local, minlength=self.points)
            if c in self.other:
                index = self.other.index(c)
                gradient[self.points+index] = local.sum()
                if self.spatial is not None:
                    first = self.points+len(self.other)+2*index
                    gradient[first:first+2] = local@self.spatial[self.phase_indices]
            linear = real_design.T@(real_design@magnitude-real_rhs)
            if self.nonnegative:
                linear[(magnitude == 0)&(linear > 0)] = 0
            inner.append(float(abs(linear).max()/self.energy))
            coefficients.append(coefficient)
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': float(cost/self.energy),
            'gradient': gradient/self.energy, 'coefficients': np.array(coefficients),
            'linear_kkt_residual': max(inner)}
        if max(inner) > 1e-8:
            raise ValueError('Inner real amplitude solve did not converge')
        return self.last

    def objective(self, parameters):
        result = self.evaluate(parameters)
        return result['value'], result['gradient']

    def initial(self, anchor):
        reference = self.independent[anchor]
        phase = np.angle(np.bincount(self.phase_indices, weights=reference.real, minlength=self.points)+
            1j*np.bincount(self.phase_indices, weights=reference.imag, minlength=self.points))
        band = np.angle(np.sum(self.independent*reference.conj(), axis=1))
        phase += band[self.reference]
        band -= band[self.reference]
        return np.r_[phase, band[self.other], np.zeros(2*len(self.other) if self.spatial is not None else 0)]

    def coordinate_refine(self, parameters, *, sweeps):
        """Monotone exact point updates, including currently invisible phases.

        Each point has one coefficient per band. Repeated amplitude-block
        columns are deliberately excluded: their conditional solve is different.
        This is initialization/escape for the full profile optimizer, not a
        global-optimality certificate for the complete scattering model.
        """
        if not np.array_equal(self.phase_indices, np.arange(self.columns)):
            raise ValueError('Exact coordinate updates require one amplitude column per reflector')
        parameters = np.asarray(parameters).copy()
        energy = np.array([np.sum(abs(r)**2, axis=0) for r, _, _ in self.systems])
        root = np.sqrt(energy)
        initial = self.evaluate(parameters)
        coefficients = initial['coefficients'].copy()
        trace = [initial['value']]
        for _ in range(sweeps):
            phases, band_phases = self.phases(parameters)
            # The fixed part includes any declared spatial phase gradient.
            fixed_phase = phases*np.exp(-1j*parameters[:self.points])
            residuals = [rhs-r@a for (r, rhs, _), a in zip(self.systems, coefficients)]
            for j in range(self.points):
                correlation = np.array([np.vdot(r[:, j], residual+r[:, j]*a[j])
                    for (r, _, _), residual, a in zip(self.systems, residuals, coefficients)])
                phase, strength = self.coordinate_maximum(correlation*fixed_phase[:, j].conj()/root[:, j])
                updated = np.exp(1j*phase)*fixed_phase[:, j]*strength/root[:, j]
                for c, (r, _, _) in enumerate(self.systems):
                    residuals[c] += r[:, j]*(coefficients[c, j]-updated[c])
                coefficients[:, j] = updated
                parameters[j] = phase
            # All band phases can be updated together, then restore the exact
            # common-phase gauge so the declared reference band stays at zero.
            change = np.array([np.angle(np.vdot(r@a, rhs))
                for (r, rhs, _), a in zip(self.systems, coefficients)])
            parameters[:self.points] += change[self.reference]
            end = self.points+len(self.other)
            parameters[self.points:end] = (band_phases+change-change[self.reference])[self.other]
            fitted = self.evaluate(parameters)  # Joint NNLS can only improve the coordinate amplitudes.
            if fitted['value'] > trace[-1]+1e-11:
                raise ValueError('Exact coordinate update increased the profiled objective')
            coefficients = fitted['coefficients'].copy()
            trace.append(fitted['value'])
            if trace[-2]-trace[-1] < 1e-12:
                break
        return parameters, {'sweeps': len(trace)-1, 'objective': trace,
            'scope': 'Exact pointwise conditional maxima and exact band-phase updates; joint global optimality is not asserted.'}

    def fit(self, *, coordinate_sweeps=0):
        trials = []
        def optimize(start):
            return minimize(self.objective, start, jac=True, method='L-BFGS-B',
                options={'gtol': 1e-10, 'ftol': 1e-15, 'maxiter': 2000, 'maxcor': 40, 'maxls': 40})
        def record(result, anchor, coordinates):
            final = self.evaluate(result.x)
            trials.append({'anchor': anchor, 'objective': final['value'],
                'maximum_gradient': float(abs(final['gradient']).max()), 'iterations': result.nit,
                'termination': str(result.message), 'parameters': result.x.tolist(),
                'initialization': 'coordinates' if coordinates else 'plain',
                'coordinate_updates': coordinates})
        for anchor in range(len(self.systems)):
            start = self.initial(anchor)
            record(optimize(start), anchor, [])
            if coordinate_sweeps:
                start, detail = self.coordinate_refine(start, sweeps=coordinate_sweeps)
                coordinates = [detail]
                result = optimize(start)
                # The local solver may have exposed another inactive point.
                start, detail = self.coordinate_refine(result.x, sweeps=coordinate_sweeps)
                coordinates.append(detail)
                record(optimize(start), anchor, coordinates)
        best = min(trials, key=lambda t: t['objective'])
        parameters = np.array(best['parameters'])
        final = self.evaluate(parameters)
        _, band = self.phases(parameters)
        step = 1e-5
        hessian = np.column_stack([(self.objective(parameters+step*d)[1]-
            self.objective(parameters-step*d)[1])/(2*step) for d in np.eye(len(parameters))])
        eigen = np.linalg.eigvalsh((hessian+hessian.T)/2)
        result = {'converged': best['maximum_gradient'] < 1e-7,
            'objective': final['value'], 'maximum_gradient': best['maximum_gradient'],
            'linear_kkt_residual': final['linear_kkt_residual'], 'trials': trials,
            'reference_channel': self.reference, 'parameters': parameters.tolist(),
            'coordinate_sweeps_requested': coordinate_sweeps,
            'phase_indices': self.phase_indices.tolist(),
            'channel_phase_radians': band.tolist(), 'curvature_eigenvalues': eigen.tolist(),
            'point_phase_radians': parameters[:self.points].tolist(),
            'coefficients_real_imag': np.stack([final['coefficients'].real, final['coefficients'].imag], axis=-1).tolist(),
            'independent_coefficients_real_imag': np.stack([self.independent.real, self.independent.imag], axis=-1).tolist(),
            'independent_objective': float(sum(s[2] for s in self.systems)/self.energy),
            'strengths': 'nonnegative' if self.nonnegative else 'unrestricted real',
            'phase_ambiguity': 'common reference phase fixed' if self.nonnegative else
                'Each band and point phase is identifiable only modulo pi; signed strengths absorb pi shifts.',
            'scope': __doc__ if self.nonnegative else
                'Shared point phases with unrestricted real band strengths. Sign-changing scattering control; '
                'band phases have independent pi ambiguities and do not establish absolute calibration.'}
        if self.spatial is not None:
            slopes = np.zeros((len(self.systems), 2))
            slopes[self.other] = parameters[self.points+len(self.other):].reshape(-1, 2)/100.
            result.update(spatial_phase_gradient_radians_per_m=slopes.tolist(),
                spatial_scope='Static affine phase over horizontal reflector coordinates, relative to the reference band. This is a diagnostic of point-dependent channel phase, not an antenna-position correction: it does not include geometric envelope changes or time variation.')
        return final['coefficients'], result
