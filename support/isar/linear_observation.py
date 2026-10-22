"""Orthogonal sufficient statistics for a fixed complex linear observation.

QR of [A y] retains the residual energy outside the column space of A. This
reduces storage without normal equations, a rank cutoff or an approximation.
The original effective observation count must still be used for GCV.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr


@dataclass
class LinearObservation:
    design: np.ndarray
    data: np.ndarray
    orthogonal_energy: float
    observations: int

    @classmethod
    def compress(cls, design, data, nuisance_rank=0):
        if design.ndim != 2 or data.shape != (len(design),):
            raise ValueError('Expected one complex data vector and its design')
        if not 0 <= nuisance_rank < len(data):
            raise ValueError('Invalid removed nuisance dimension')
        augmented = np.empty((len(data), design.shape[1]+1), dtype=complex, order='F')
        augmented[:, :-1], augmented[:, -1] = design, data
        r = qr(augmented, mode='r', overwrite_a=True, check_finite=True)[0]
        rows = min(design.shape)
        return cls(r[:rows, :-1].copy(), r[:rows, -1].copy(),
                   float(np.sum(abs(r[rows:, -1])**2)), len(data)-nuisance_rank)

    @property
    def energy(self):
        return float(np.vdot(self.data, self.data).real+self.orthogonal_energy)

    def error(self, coefficient, phase=1.):
        residual = phase*(self.design@coefficient)-self.data
        return float(np.vdot(residual, residual).real+self.orthogonal_energy)
