"""Joint subpixel scatterer positions and constant channel range/phase offsets.

The strongest training-selected point anchors the body-coordinate origin.
Only relative positions are varied. All channels use the same scene and the
same physical motion; per-channel complex gains remain explicit parameters.
"""
import numpy as np
from scipy.optimize import minimize


class CoherentScene:
    def __init__(self, alignment):
        if alignment.rate_scale is not None:
            raise ValueError('Validate channel path-rate correction separately before varying the scene')
        self.alignment = alignment
        self.points = alignment.bands[0].points
        self.nalignment = 3*alignment.count
        self.count = self.nalignment+2*(len(self.points)-1)

    def split(self, parameters):
        points = self.points.copy()
        points[1:, :2] += np.asarray(parameters[self.nalignment:]).reshape(-1, 2)
        bias, gains = self.alignment.unpack(parameters[:self.nalignment])
        return points, bias, gains

    def project(self, parameters, *, derivatives=False):
        points, biases, gains = self.split(parameters)
        designs, projected, position_partials, range_partials = [], [], [], []
        for band, bias, gain in zip(self.alignment.bands, biases, gains):
            values, xyz, dr = [], [], []
            for point in points:
                result = band.field.point(point, motion=band.state, derivatives=derivatives,
                    range_bias_m=bias, range_bias_derivative=derivatives)
                values.append((result[0] if derivatives else result).ravel())
                if derivatives:
                    jacobian = result[1].reshape(-1, 4)[band.selected]
                    xyz.append(band.nuisance.remove(jacobian[:, :2]))
                    dr.append(band.nuisance.remove(jacobian[:, 3]))
            design = np.column_stack(values)
            designs.append(design)
            projected.append(gain*band.nuisance.remove(design[band.selected]))
            if derivatives:
                position_partials.append(gain*np.stack(xyz, axis=1))
                range_partials.append(gain*self.alignment.scale*np.column_stack(dr))
        amplitudes = np.linalg.lstsq(np.concatenate(projected),
            np.concatenate([b.clean for b in self.alignment.bands]), rcond=None)[0]
        return amplitudes, designs, projected, position_partials, range_partials

    def objective(self, parameters):
        amplitudes, _, designs, xyz, dr = self.project(parameters, derivatives=True)
        gradient = np.zeros(self.count)
        calibration = gradient[:self.nalignment].reshape(3, self.alignment.count)
        spatial = gradient[self.nalignment:].reshape(-1, 2)
        value = 0.
        for channel, (band, design, position_partial, range_partial) in enumerate(zip(
                self.alignment.bands, designs, xyz, dr)):
            prediction = design@amplitudes
            residual = band.clean-prediction
            value += np.vdot(residual, residual).real
            spatial -= 2*np.real(np.einsum('n,npj,p->pj', residual.conj(),
                                            position_partial[:, 1:], amplitudes[1:]))
            if channel != self.alignment.reference:
                index = self.alignment.other.index(channel)
                calibration[:, index] = -2*np.real(residual.conj()@np.column_stack([
                    range_partial@amplitudes, prediction, 1j*prediction]))
        return float(value/self.alignment.energy), gradient/self.alignment.energy

    def fit(self, initial_alignment, *, position_radius_m=2., iterations=180):
        initial = np.r_[initial_alignment, np.zeros(self.count-self.nalignment)]
        n, scale = self.alignment.count, self.alignment.scale
        bounds = ([(-75/scale, 75/scale)]*n+[(-3., 3.)]*n+[(None, None)]*n+
                  [(-position_radius_m, position_radius_m)]*(self.count-self.nalignment))
        trace = []
        result = minimize(self.objective, initial, jac=True, bounds=bounds, method='L-BFGS-B',
            callback=lambda p: trace.append(float(self.objective(p)[0])),
            options={'maxiter': iterations, 'gtol': 1e-8, 'ftol': 1e-14, 'maxcor': 40, 'maxls': 30})
        gradient = result.jac.copy()
        for i, (lo, hi) in enumerate(bounds):
            if (lo is not None and result.x[i] <= lo+1e-7 and gradient[i] > 0) or (
                hi is not None and result.x[i] >= hi-1e-7 and gradient[i] < 0):
                gradient[i] = 0
        points, biases, gains = self.split(result.x)
        return result.x, {'objective': float(result.fun), 'converged': bool(result.success and abs(gradient).max() < 1e-7),
            'maximum_projected_gradient': float(abs(gradient).max()), 'iterations': result.nit,
            'termination': str(result.message), 'trace': trace, 'points_body_m': points.tolist(),
            'range_bias_m': biases.tolist(), 'gain_amplitude': abs(gains).tolist(),
            'gain_phase_degrees': np.degrees(np.angle(gains)).tolist(),
            'position_radius_m': position_radius_m,
            'position_bound_hit': bool(np.any(abs(result.x[self.nalignment:]) >= position_radius_m-1e-5)),
            'range_bound_hit': bool(np.any(abs(biases) >= 75-1e-5))}

    def validation(self, parameters, interval):
        amplitudes, designs, _, _, _ = self.project(parameters)
        _, _, gains = self.split(parameters)
        results = [band.validation(design, gain*amplitudes, interval) for band, design, gain in
                   zip(self.alignment.bands, designs, gains)]
        cross = sum(complex(*r['prediction_cross_real_imag']) for r in results)
        energy = sum(r['prediction_energy'] for r in results)
        upper = sum(r['independent_test_energy'] for r in results)
        captured = abs(cross)**2/energy
        return {'channels': results, 'prediction_error': sum(r['prediction_error'] for r in results),
            'stationary_only_error': sum(r['stationary_only_error'] for r in results),
            'matched_energy': float(captured), 'independent_test_energy': float(upper),
            'conditional_coherent_loss_db': float(-10*np.log10(captured/upper))}
