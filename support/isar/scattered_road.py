"""Frequency-dependent reflecting points on one translating road target.

The reference band's reflecting point defines the trajectory gauge. Other
bands may use constant offsets in a fixed road-aligned frame. Every channel
still uses exact bistatic path, phase and envelope geometry, with one complex
gain. This is a bounded scene hypothesis, not independently measured vehicle
dimensions or permission to interpret its offsets as antenna corrections.
"""
from types import SimpleNamespace

import numpy as np

from shared_target_geometry import RoadTarget, path_partials


class OffsetProblem:
    def __init__(self, original, channels, reference, limits):
        self.original, self.base_count = original, original.model.count
        self.channels = [c for c in range(channels) if c != reference]
        self.count = self.base_count+3*len(self.channels)
        self.limits = np.asarray(limits, float)
        if self.limits.shape != (3,) or np.any(self.limits <= 0):
            raise ValueError('Declare positive reflecting-point offset bounds in metres')
        self.range_origins = np.array(original.range_origins, copy=True)
        self.model = SimpleNamespace(count=self.count, bounds=self.bounds)

    def bounds(self):
        lower, upper = self.original.model.bounds()
        extent = np.tile(self.limits, len(self.channels))
        return np.r_[lower, -extent], np.r_[upper, extent]

    def auxiliary(self, parameters, jacobian=False):
        values = self.original.auxiliary(parameters[:self.base_count], jacobian=jacobian)
        return [np.pad(value, ((0, 0), (0, self.count-self.base_count))) for value in values] if jacobian else values


class ScatteredRoad(RoadTarget):
    def __init__(self, original, origins, *, reference=1, limits=(3., 2., 2.)):
        if not 0 <= reference < len(original.training):
            raise ValueError('The reference band must be observed')
        self.original, self.base_count, self.reference = original, original.nmotion, reference
        problem = OffsetProblem(original.problem, len(original.training), reference, limits)
        initial = np.r_[original.initial, np.zeros(problem.count-original.nmotion)]
        super().__init__(problem, original.training, original.validation, initial,
                         original.transmitters, original.antennas)
        self.origins = np.asarray(origins).copy()
        if self.origins.shape != (len(self.training),):
            raise ValueError('Require one fixed common origin per band')
        trajectory = original.problem.model
        epoch = np.array([trajectory.seed['epoch']])
        base, basis = trajectory.length_basis(epoch)
        distance = (base+basis@original.initial[:trajectory.radial_count])[0]
        tangent = trajectory.road.centre(np.array([distance]), 1)[0].copy()
        tangent[2] = 0.; tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0], 0.])
        self.axes = np.column_stack((tangent, normal, [0., 0., 1.]))
        self.sections = {c: slice(self.base_count+3*i, self.base_count+3*(i+1))
                         for i, c in enumerate(problem.channels)}

    def offsets(self, parameters):
        result = np.zeros((len(self.training), 3))
        for c, section in self.sections.items():
            result[c] = parameters[section]
        return result

    def template(self, motion, antenna, channel, field):
        original = self.original
        position, basis, _ = original.problem.model.positions(
            motion[:self.base_count], field.time, original.problem.sites)
        jacobian = np.zeros((len(position), 3, self.nmotion))
        jacobian[:, :, :self.base_count] = basis
        if channel in self.sections:
            section = self.sections[channel]
            position += self.axes@motion[section]
            jacobian[:, :, section] = self.axes
        mapping = self.mappings[channel]; offset = (mapping@antenna).reshape(2, 3)
        path, partial = path_partials(position, jacobian,
            self.transmitters[channel]+offset[0], offset[1])
        derivative = np.c_[partial[:, :self.nmotion], partial[:, self.nmotion:]@mapping]
        return field.template(path, derivative, range_bias=self.origins[channel])
