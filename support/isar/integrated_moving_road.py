"""Original-row moving response for the saved vehicle-calibration time bins.

Only the forward response changes. Measured values, fixed whitening, Doppler
gate and train/test selection are retained exactly. The pulse remains the
existing interval-averaged response: this isolates temporal integration.
"""
import numpy as np

from field import C, FS


class IntegratedMovingField:
    def __init__(self, original, bins):
        self.original, self.bins = original, bins
        self.time, self.y = bins.time, original.y
        self.seed, self.k, self.offsets = original.seed, original.k, original.offsets
        self.seed_path = np.polynomial.polynomial.polyval(
            (self.time-self.seed['epoch'])/self.seed.get('scale', 15.), self.seed['path'])

    def template(self, path, jacobian=None, range_bias=0., bias_derivative=False):
        path = np.asarray(path)
        if path.shape != self.time.shape:
            raise ValueError('Evaluate the moving path at original row epochs')
        field = self.original
        lag = self.offsets[None, :]-(path+range_bias-self.seed_path)[:, None]*FS/C
        pulse = field.pulse(lag)
        phase = np.exp(-1j*self.k*(path-self.seed_path+self.seed['path'][0]))
        value = self.bins.mean(phase[:, None]*pulse)
        u = field.apply(value).ravel()
        if jacobian is None:
            return u, None
        jacobian = np.asarray(jacobian)
        if jacobian.ndim != 2 or len(jacobian) != len(path):
            raise ValueError('Path derivatives must use original row epochs')
        slope = -FS/C*phase[:, None]*field.pulse(lag, 1)
        derivative = slope-1j*self.k*phase[:, None]*pulse
        raw = derivative[:, :, None]*jacobian[:, None, :]
        if bias_derivative:
            raw = np.concatenate((raw, slope[:, :, None]), axis=2)
        n = raw.shape[-1]
        binned = self.bins.mean(raw)
        transformed = field.q@binned.reshape(self.bins.count, -1)
        du = np.einsum('ab,fbp->fap', field.whiten,
                      transformed.reshape(len(field.q), len(self.offsets), n)).reshape(-1, n)
        return u, du
