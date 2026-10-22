"""A delayed moving-path component with consistent range/phase/Doppler.

The extra path is d + v(t-t0) relative to a fixed physical centre trajectory.
Its constant carrier phase is absorbed into one complex amplitude, identically
for every range crop. This allows a measured component to predict leakage into
another crop without fitting a new phase or amplitude there.
"""
import numpy as np

from field import average
from spline_motion import SplineMotion
from stationary import Stationary


class DelayedPath:
    def __init__(self, band, parameters, *, include_main=True):
        self.band = band
        self.time = band.field.data['row_time']
        self.epoch = float(band.field.time.mean())
        state = SplineMotion({**band.model, 'parameters': parameters}).apply(
            self.time, band.prior, band.reference_transmitter, band.receiver)
        point = np.array([0., 0., 20.])
        offset = (band.prior['position_ecef_m']-band.field.data['centre_ecef_m']+
            state['translation_offset_m']+state['body_to_ecef']@point)
        self.delta, self.omega, _ = band.field.path_kinematics(
            state['position_ecef_m']+state['body_to_ecef']@point,
            state['velocity_ecef_mps']+state['frame_velocity']@point, position_offset=offset)
        self.main = band.design(parameters)[0]
        design = np.column_stack([band.nuisance.design, self.main]) if include_main else band.nuisance.design
        self.nuisance = Stationary(design, band.selected)
        self.clean = self.nuisance.remove(band.field.y.ravel()[band.selected])
        self.energy = float(np.vdot(self.clean, self.clean).real)
        self.initial = np.array([500., 0.])
        self.last, self.evaluations = None, 0

    def response(self, parameters, *, derivatives=False):
        delay, rate = parameters
        field = self.band.field
        dt = self.time-self.epoch
        delta = self.delta+delay+rate*dt
        omega = self.omega-field.k*.001024*rate
        raw = field.path_response(delta, omega, range_bias_m=self.band.range_bias, derivatives=derivatives)
        gauge = np.exp(1j*field.k*delay)
        if derivatives:
            raw, path, doppler, envelope = raw
            derivative = np.stack([envelope, path*dt[:, None]-field.k*.001024*doppler], axis=-1)*gauge
        value, _ = average(raw*gauge, field.data['row_information'], field.metadata['block_rows'])
        value = field.apply(value).ravel()
        if not derivatives:
            return value
        derivative, _ = average(derivative, field.data['row_information'], field.metadata['block_rows'])
        return value, np.column_stack([field.apply(derivative[:, :, j]).ravel() for j in range(2)])

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        raw, derivative = self.response(parameters, derivatives=True)
        a = self.nuisance.remove(raw[self.band.selected])
        da = self.nuisance.remove(derivative[self.band.selected])
        norm = float(np.vdot(a, a).real)
        amplitude = np.vdot(a, self.clean)/norm
        residual = self.clean-a*amplitude
        fixed = da*amplitude
        jacobian = (-fixed+np.outer(a, a.conj()@fixed)/norm-
            np.outer(a, da.conj().T@residual)/norm)
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(),
            'value': float(np.vdot(residual, residual).real)/self.energy,
            'gradient': 2*np.real(jacobian.conj().T@residual)/self.energy,
            'gauss_newton': 2*np.real(jacobian.conj().T@jacobian)/self.energy,
            'scene_conditions': [1.], 'amplitude': amplitude, 'target': raw,
            'information': norm, 'evaluations': self.evaluations}
        return self.last

    def objective(self, parameters):
        value = self.evaluate(parameters)
        return value['value'], value['gradient']

    def acquire(self, delays, rates):
        """Phase-only coarse search; final fitting restores envelope migration.

        All scores use training samples and the same nuisance projection. The
        coarse approximation is an initializer, never a scientific correction.
        """
        field, band = self.band.field, self.band
        time = field.time[band.selected_time]-self.epoch
        phase = np.exp(-1j*field.k*np.asarray(rates)[:, None]*time)
        q = self.nuisance.q.reshape(len(time), field.y.shape[1], -1)
        y = self.clean.reshape(len(time), field.y.shape[1])
        scores = []
        for delay in delays:
            a = self.response([delay, 0.]).reshape(field.y.shape)[band.selected_time]
            cross = phase.conj()@np.sum(a.conj()*y, axis=1)
            projected = phase@np.einsum('trp,tr->tp', q.conj(), a)
            information = np.sum(abs(a)**2)-np.sum(abs(projected)**2, axis=1)
            scores.append(abs(cross)**2/np.maximum(information, np.finfo(float).tiny))
        scores = np.array(scores)
        best = np.unravel_index(np.argmax(scores), scores.shape)
        return np.array([delays[best[0]], rates[best[1]]]), scores
