"""Controlled temporal-scattering tests with fixed geometry and target phase.

All durations use the same observations and stationary nuisance spaces. Only
training samples determine scattering coefficients. The validation projection
removes stationary nuisance from both data and the frozen prediction; it never
fits a target phase, gain or coefficient on validation data.
"""
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import nnls

from stationary import Stationary


def block_design(design, blocks, count):
    """One complex point coefficient per temporal block, without interpolation."""
    blocks = np.asarray(blocks)
    if blocks.shape != (len(design),) or np.any(blocks < 0) or np.any(blocks >= count):
        raise ValueError('Invalid temporal block assignment')
    columns = blocks[:, None]*design.shape[1]+np.arange(design.shape[1])
    output = np.zeros((len(design), count*design.shape[1]), complex)
    output[np.arange(len(design))[:, None], columns] = design
    return output


class TemporalScene:
    def __init__(self, bands, designs, gains, interval):
        self.bands, self.designs, self.gains = bands, designs, np.asarray(gains)
        self.interval = np.asarray(interval, float)
        self.points = designs[0].shape[1]
        if not all(d.shape[1] == self.points for d in designs):
            raise ValueError('A common scene requires identical physical support')
        self.tests = [Stationary(b.nuisance.design, ~b.selected) for b in bands]
        self.train_y = np.concatenate([b.clean for b in bands])
        self.test_y = np.concatenate([q.remove(b.field.y.ravel()[~b.selected]) for b, q in zip(bands, self.tests)])
        self.train_energy = float(np.vdot(self.train_y, self.train_y).real)
        self.test_energy = float(np.vdot(self.test_y, self.test_y).real)

    def partition(self, duration):
        if duration <= 0:
            raise ValueError('Aperture duration must be positive')
        count = int(np.ceil(np.diff(self.interval)[0]/duration-1e-10))
        indices = []
        for b in self.bands:
            time = np.repeat(b.field.time, b.field.y.shape[1])
            index = np.floor((time-self.interval[0])/duration).astype(int)
            if np.any(index < 0) or np.any(index >= count):
                raise ValueError('Observations leave the declared aperture')
            indices.append(index)
        return count, indices

    def fit(self, duration, *, phase_reference=None, independent_channels=False):
        count, blocks = self.partition(duration)
        designs = []
        for b, d, gain, block in zip(self.bands, self.designs, self.gains, blocks):
            designs.append(b.nuisance.remove(block_design(gain*d[b.selected], block[b.selected], count)))
        reference = None
        if phase_reference is not None:
            phase_reference = np.asarray(phase_reference)
            if phase_reference.shape != (self.points,) or np.any(abs(phase_reference) == 0):
                raise ValueError('Need one nonzero training phase reference per shared point')
            reference = np.tile(phase_reference/abs(phase_reference), count)
        conditions, ranks, gradients = [], [], []
        def solve(a, y):
            if reference is None:
                amplitude, _, rank, singular = lstsq(a, y, cond=1e-10, lapack_driver='gelsd')
                gradient = abs(a.conj().T@(a@amplitude-y)).max()/max(np.vdot(y, y).real, 1.)
            else:
                a = a*reference
                real_a, real_y = np.r_[a.real, a.imag], np.r_[y.real, y.imag]
                # QR compresses the data before the bounded linear solve. It
                # preserves the objective and avoids squaring its condition.
                q, r = np.linalg.qr(real_a, mode='reduced')
                rhs = q.T@real_y
                positive, _ = nnls(r, rhs, maxiter=10*r.shape[1])
                g = r.T@(r@positive-rhs)
                g[(positive == 0)&(g > 0)] = 0
                gradient = abs(g).max()/max(np.vdot(y, y).real, 1.)
                amplitude = reference*positive
                singular = np.linalg.svd(r, compute_uv=False)
                rank = np.sum(singular > singular[0]*1e-10)
            ranks.append(int(rank))
            conditions.append(float(singular[0]/singular[-1]) if singular[-1] > 0 else None)
            gradients.append(float(gradient))
            return amplitude
        if independent_channels:
            coefficients = [solve(d, b.clean) for b, d in zip(self.bands, designs)]
        else:
            common = solve(np.concatenate(designs), self.train_y)
            coefficients = [common]*len(self.bands)
        train_prediction = np.concatenate([a@c for a, c in zip(designs, coefficients)])
        train_error = float(np.sum(abs(self.train_y-train_prediction)**2))
        del designs, train_prediction
        checks, prediction, information = [], [], []
        for band, target, gain, block, coefficient, nuisance in zip(
                self.bands, self.designs, self.gains, blocks, coefficients, self.tests):
            test = ~band.selected
            alpha = coefficient.reshape(count, self.points)[block[test]]
            raw = gain*np.sum(target[test]*alpha, axis=1)
            model = nuisance.remove(raw)
            y = nuisance.remove(band.field.y.ravel()[test])
            prediction.append(model)
            information.append(float(np.vdot(model, model).real/np.vdot(raw, raw).real))
            time = np.repeat(band.field.time, band.field.y.shape[1])[test]
            phase = []
            for first in np.arange(self.interval[0], self.interval[1], .5):
                keep = (time >= first)&(time < first+.5)
                power = float(np.vdot(model[keep], model[keep]).real)
                cross = np.vdot(model[keep], y[keep])
                phase.append({'time_s': float(first+.25), 'phase_degrees': float(np.degrees(np.angle(cross))),
                              'prediction_energy': power, 'cross_real_imag': [float(cross.real), float(cross.imag)]})
            rms = np.sqrt(np.average([s['phase_degrees']**2 for s in phase], weights=[s['prediction_energy'] for s in phase]))
            checks.append({'prediction_phase_rms_degrees': float(rms), 'phase_by_half_second': phase})
        model = np.concatenate(prediction)
        energy = float(np.vdot(model, model).real)
        cross = np.vdot(model, self.test_y)
        error = float(np.sum(abs(self.test_y-model)**2))
        score = 2*cross.real-energy
        result = {'duration_s': duration, 'blocks': count,
            'coefficient_model': ('independent positive amplitudes, fixed shared phases' if reference is not None else 'independent complex by channel')
                if independent_channels else ('positive amplitudes, fixed phases' if reference is not None else 'shared complex'),
            'training_error': train_error, 'validation_error': error, 'stationary_only_validation_energy': self.test_energy,
            'heldout_explained_energy': float(score), 'heldout_explained_fraction': float(score/self.test_energy),
            'matched_energy': float(abs(cross)**2/energy), 'prediction_energy': energy,
            'maximum_linear_stationarity_residual': max(gradients), 'ranks': ranks, 'design_conditions': conditions,
            'minimum_target_information_retained': min(information), 'channels': checks,
            'coefficients_real_imag': [np.stack([c.real, c.imag], axis=-1).reshape(count, self.points, 2).tolist() for c in coefficients],
            'scope': 'Fixed motion/support/gains; stationary nuisance profiled identically for each duration. Explained energy = 2 Re(prediction^H data) - ||prediction||^2 on held-out whitened observations. It penalizes prediction error without a test-data target-phase fit. This is conditional predictive evidence, not a global theoretical SNR bound.'}
        return result
