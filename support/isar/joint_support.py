"""Greedy common-location selection with independent complex band responses.

Each step maximizes the exact reduction of the summed whitened residual energy
after adding one physical location. It is a multiband orthogonal least-squares
step, not a globally optimal sparse reconstruction. No band equalization or
common scattering phase is imposed. Input dictionaries may differ by band.
"""
import numpy as np


class BandProjection:
    def __init__(self, design, data, capacity, tolerance):
        self.design, self.data = design, data
        self.basis = np.empty((len(data), capacity), complex, order='F')
        self.count = 0
        self.tolerance = tolerance
        self.norm = np.einsum('ij,ij->j', design.conj(), design).real
        self.remaining = self.norm.copy()
        self.correlation = (design.T@data.conj()).conj()

    def score(self):
        usable = self.remaining > self.tolerance*self.norm
        return np.divide(abs(self.correlation)**2, self.remaining,
                         out=np.zeros_like(self.remaining), where=usable)

    def append(self, index):
        column = self.design[:, index].copy()
        q = self.basis[:, :self.count]
        # Reorthogonalization prevents subtraction loss in coherent dictionaries.
        for _ in range(2):
            column -= q@(q.conj().T@column)
        energy = float(np.vdot(column, column).real)
        if energy <= self.tolerance*self.norm[index]:
            return 0.
        column /= np.sqrt(energy)
        self.basis[:, self.count] = column
        self.count += 1
        correlation = (self.design.T@column.conj()).conj()
        coefficient = np.vdot(column, self.data)
        self.correlation -= correlation*coefficient
        self.remaining = np.maximum(self.remaining-abs(correlation)**2, 0.)
        return float(abs(coefficient)**2)


def select(designs, data, count, *, tolerance=1e-12):
    """Yield (support, step diagnostics), using training observations only.

The tolerance bounds loss of precision in residualized column energies; it is
not a sparsity penalty. Entirely dependent columns are excluded. A column can
be dependent in one band and informative in another.
    """
    if not designs or len(designs) != len(data) or not 0 < tolerance < 1:
        raise ValueError('Need aligned band designs/data and a numerical tolerance')
    columns = designs[0].shape[1]
    if not 0 < count <= columns or any(a.ndim != 2 or a.shape[1] != columns or
            y.shape != (len(a),) for a, y in zip(designs, data)):
        raise ValueError('Invalid dictionary shape or requested support size')
    bands = [BandProjection(a, y, count, tolerance) for a, y in zip(designs, data)]
    support = []
    for _ in range(count):
        score = sum(b.score() for b in bands)
        score[support] = -np.inf
        index = int(np.argmax(score))
        if not np.isfinite(score[index]) or score[index] <= 0:
            break
        actual = sum(b.append(index) for b in bands)
        support.append(index)
        discrepancy = abs(actual-score[index])/max(actual, score[index], np.finfo(float).tiny)
        if discrepancy > 2e-5:
            raise ValueError(f'Orthogonal selection energy has lost precision: {discrepancy:g}')
        yield support.copy(), {'selected_index': index, 'predicted_energy_reduction': float(score[index]),
            'actual_energy_reduction': actual, 'selection_relative_error': float(discrepancy),
            'band_ranks': [b.count for b in bands]}
