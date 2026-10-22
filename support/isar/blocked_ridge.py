"""Choose a scene penalty by temporal prediction, without reading outer data.

Ordinary GCV approximates sample-wise omission. It need not choose a useful
penalty for a strongly coherent dictionary when complete time blocks are absent.
Here the inner block prediction chooses an absolute Tikhonov strength. The same
strength is then used when all outer-training samples are refitted.
"""
import numpy as np


class RidgePath:
    def __init__(self, design, data):
        u, self.singular, self.vh = np.linalg.svd(design, full_matrices=False)
        self.projected = u.conj().T@data

    def coefficients(self, strengths):
        strengths = np.atleast_1d(strengths)
        if np.any(strengths <= 0) or not np.isfinite(strengths).all():
            raise ValueError('Ridge strengths must be finite and positive')
        return self.vh.conj().T@(self.singular[:, None]*self.projected[:, None]/
            (self.singular[:, None]**2+strengths))


def tune(training, validation, relative_strengths):
    path = RidgePath(training.design, training.data)
    strengths = path.singular[0]**2*np.asarray(relative_strengths)
    coefficient = path.coefficients(strengths)
    residual = validation.design@coefficient-validation.data[:, None]
    scores = np.sum(abs(residual)**2, axis=0)+validation.orthogonal_energy
    index = int(np.argmin(scores))
    return float(strengths[index]), {'relative_strengths': np.asarray(relative_strengths).tolist(),
        'absolute_strengths': strengths.tolist(), 'inner_prediction_errors': scores.tolist(),
        'chosen_index': index, 'grid_edge': index in (0, len(strengths)-1),
        'inner_training_observations': training.observations,
        'inner_validation_observations': validation.observations}
