"""Common point coordinates with independent Gaussian scenes in each band.

Reuse the existing exact original-row marginal likelihood, including its
determinant and training-estimated per-band scene variance. Geometry is shared;
no common reflection phase or channel calibration is imposed. The statistical
scene assumption is explicit and requires predictive checking on real data.
"""
import copy

import numpy as np

from marginal_points import MarginalPoints


class SharedMarginalPoints:
    def __init__(self, bands, parameters, points):
        self.parts = []
        for original in bands:
            band = copy.copy(original); band.points = np.array(points, copy=True)
            self.parts.append(MarginalPoints(band, parameters))
        self.initial = self.parts[0].initial.copy()
        self.energy = sum(part.energy for part in self.parts)
        self.last, self.evaluations = None, 0

    def coordinates(self, parameters):
        return self.parts[0].coordinates(parameters)

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(parameters, self.last['parameters']):
            return self.last
        local = [part.evaluate(parameters) for part in self.parts]
        weights = np.array([part.energy for part in self.parts])/self.energy
        self.evaluations += 1
        self.last = {'parameters': np.array(parameters, copy=True),
            'value': float(sum(weight*value['value'] for weight, value in zip(weights, local))),
            'gradient': sum(weight*value['gradient'] for weight, value in zip(weights, local)),
            'gauss_newton': sum(weight*value['gauss_newton'] for weight, value in zip(weights, local)),
            'scene_conditions': [value['scene_conditions'][0] for value in local],
            'scene_variances': [value['scene_variance'] for value in local],
            'effective_scene_ranks': [value['effective_scene_rank'] for value in local]}
        return self.last

    def objective(self, parameters):
        value = self.evaluate(parameters)
        return value['value'], value['gradient']

    def validation(self, parameters):
        return [part.validation(parameters) for part in self.parts]
