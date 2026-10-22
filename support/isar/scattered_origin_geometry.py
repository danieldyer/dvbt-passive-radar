"""Shared antenna geometry and range origins with per-band vehicle points.

Only the physical vehicle mean is enlarged. Observations, noise units, static
projection and existing motion priors are unchanged. Reflecting-point extents
are sensitivity assumptions, not measured vehicle dimensions.
"""
import numpy as np

from scattered_road import ScatteredRoad
from shared_origin_geometry import OriginRoad, SharedOriginGeometry
from shared_target_geometry import SharedGeometry, path_partials


class ScatteredOriginRoad(OriginRoad):
    """Use the band-specific physical point before applying a common origin."""
    def template(self, motion, antenna, origins, channel, field):
        source = self.original
        trajectory = source.original.problem
        position, basis, _ = trajectory.model.positions(
            motion[:source.base_count], field.time, trajectory.sites)
        jacobian = np.zeros((len(position), 3, self.nmotion))
        jacobian[:, :, :source.base_count] = basis
        if channel in source.sections:
            section = source.sections[channel]
            position += source.axes@motion[section]
            jacobian[:, :, section] = source.axes
        mapping = source.mappings[channel]
        offset = (mapping@antenna).reshape(2, 3)
        path, partial = path_partials(position, jacobian,
            source.transmitters[channel]+offset[0], offset[1])
        derivative = np.c_[partial[:, :self.nmotion], partial[:, self.nmotion:]@mapping]
        value, local = field.template(path, derivative, range_bias=origins[channel], bias_derivative=True)
        result = np.zeros((len(value), self.nmotion+self.antennas.count+self.nband), complex)
        result[:, :-self.nband] = local[:, :-1]
        result[:, -self.nband+channel] = local[:, -1]
        return value, result


class ScatteredOriginGeometry(SharedOriginGeometry):
    def __init__(self, original, *, limits=(3., 2., 2.), reference=1):
        roads = [ScatteredOriginRoad(ScatteredRoad(road.original,
            original.initial[original.range_slice], limits=limits, reference=reference))
            for road in original.roads]
        SharedGeometry.__init__(self, original.ship, roads,
            initial_ship=original.initial[:original.ship_count])
        self.range_slice = self.ship.range_slice
        self.base_columns = np.r_[np.arange(self.ship_count), *[
            np.arange(section.start, section.start+road.original.base_count)
            for section, road in zip(self.road_slices, self.roads)]]
        self.initial[self.base_columns] = original.initial

    def expand_bounds(self, bounds):
        matrix, lo, hi, lower, upper = bounds
        expanded = np.zeros((len(matrix), len(self.initial)))
        expanded[:, self.base_columns] = matrix
        bottom, top = np.zeros(len(self.initial)), np.zeros(len(self.initial))
        for section, road in zip(self.road_slices, self.roads):
            bottom[section], top[section] = road.original.problem.model.bounds()
        bottom[self.base_columns], top[self.base_columns] = lower, upper
        return expanded, np.array(lo, copy=True), np.array(hi, copy=True), bottom, top


class FrozenScatteredRoad:
    def __init__(self, original, origins, *, limits=(3., 2., 2.), reference=1):
        physical = ScatteredRoad(original, origins, limits=limits, reference=reference)
        self.road = ScatteredOriginRoad(physical)
        self.origins = np.array(origins, copy=True)
        self.problem, self.nmotion, self.energy = physical.problem, physical.nmotion, physical.energy

    def evaluate(self, motion, antenna):
        return self.road.evaluate(motion, antenna, self.origins)

    def predict(self, motion, antenna):
        return self.road.predict(motion, antenna, self.origins)
