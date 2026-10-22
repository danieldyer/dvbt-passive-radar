"""Compact, fixed known-waveform kernels for coherent observation blocks.

Each integer-lag interval uses a degree-10 Chebyshev expansion of the same
64-tap interpolation used by the scalar pulse reference. Coefficients are
computed once from known autocorrelations. No target data or fitted phases
enter the interpolation. Derivatives differentiate the stored polynomial.
"""
import numpy as np
from scipy.fft import dct


class BlockPulse:
    def __init__(self, samples, *, degree=10, radius=32):
        samples = np.asarray(samples, dtype=complex)
        if samples.ndim != 2 or samples.shape[1] != 129 or not np.isfinite(samples).all():
            raise ValueError('Need finite autocorrelations at integer lags -64 through 64')
        if degree < 6 or degree > 24 or radius < 1 or radius > 32:
            raise ValueError('Unsupported pulse interpolation order or support')
        self.samples = samples
        self.radius, self.degree = radius, degree
        nodes = np.cos(np.pi*(np.arange(degree+1)+.5)/(degree+1))
        positions = np.arange(-radius, radius)[:, None]+.5+.5*nodes
        delta = positions.reshape(-1, 1)-np.arange(-64, 65)
        window = np.i0(8*np.sqrt(np.maximum(0, 1-(delta/32)**2)))/np.i0(8)
        weights = np.sinc(delta)*window*(abs(delta) <= 32)
        weights /= weights.sum(axis=1)[:, None]
        values = (samples@weights.T).reshape(len(samples), 2*radius, degree+1)
        coefficients = dct(values, type=2, axis=-1)/(degree+1)
        coefficients[..., 0] /= 2
        self.coefficients = np.moveaxis(coefficients, -1, 0)
        self.derivative_coefficients = 2*np.polynomial.chebyshev.chebder(self.coefficients, axis=0)

    def at(self, lag, index, derivative=0):
        lag, index = np.broadcast_arrays(np.asarray(lag, dtype=float), np.asarray(index, dtype=int))
        if (not np.isfinite(lag).all() or np.any(abs(lag) > self.radius) or
                np.any(index < 0) or np.any(index >= len(self.samples))):
            raise ValueError('Pulse query outside its retained support or block identity')
        if derivative not in (0, 1):
            raise ValueError('Only pulse values and first derivatives are required')
        cell = np.minimum(np.floor(lag).astype(int), self.radius-1)
        coordinate = 2*(lag-cell)-1
        coefficients = self.coefficients if derivative == 0 else self.derivative_coefficients
        a = np.zeros(lag.shape, complex); b = np.zeros_like(a)
        for c in coefficients[:0:-1]:
            a, b = c[index, cell+self.radius]+2*coordinate*a-b, a
        return coefficients[0, index, cell+self.radius]+coordinate*a-b


class RowPulse:
    """Bind an immutable block kernel to each original observation epoch."""
    def __init__(self, kernels, row_blocks, mean_samples):
        self.kernels = kernels
        self.row_blocks = np.asarray(row_blocks, dtype=int)
        self.samples = np.asarray(mean_samples)

    def __call__(self, lag, derivative=0):
        lag = np.asarray(lag)
        if lag.ndim != 2 or lag.shape[0] != len(self.row_blocks):
            raise ValueError('Pulse rows must retain their original observation identities')
        return self.kernels.at(lag, self.row_blocks[:, None], derivative)
