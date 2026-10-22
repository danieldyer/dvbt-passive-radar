"""Joint physical motion, transmitter/receiver geometry and envelope origins.

The unknown complex reflector amplitudes are profiled independently per band.
Locations use the existing correlated site/phase-centre prior. The physical
motion reference stays fixed when changing a transmitter hypothesis: only the
propagation geometry changes. This keeps coordinate changes out of the data.
"""
import numpy as np

from physical_profile import PhysicalProfile, profile
from validate_motion import validate


class GeometryProfile(PhysicalProfile):
    def __init__(self, bands, initial_motion, antennas, *, base_offsets=None, range_origins=None):
        super().__init__(bands, initial_motion)
        self.antennas = antennas
        self.antenna_slice = slice(self.nmotion, self.nmotion+antennas.count)
        self.range_slice = slice(self.antenna_slice.stop, self.antenna_slice.stop+len(bands))
        self.base_offsets = np.zeros((len(bands), 2, 3)) if base_offsets is None else np.asarray(base_offsets)
        if self.base_offsets.shape != (len(bands), 2, 3):
            raise ValueError('Each band needs one transmitter and one receiver offset')
        origins = [b.range_bias for b in bands] if range_origins is None else range_origins
        if np.shape(origins) != (len(bands),):
            raise ValueError('Need one initial envelope origin per band')
        self.initial = np.r_[self.initial, np.zeros(antennas.count), origins]
        self.response_parameters, self.response_cache = None, None
        self.transforms = []
        for c in range(len(bands)):
            transform = np.zeros((len(self.full_initial)+7, len(self.initial)))
            transform[self.active, np.arange(self.nmotion)] = 1.
            transform[len(self.full_initial):len(self.full_initial)+6, self.antenna_slice] = antennas.mapping(c)
            transform[-1, self.range_slice.start+c] = 1.
            self.transforms.append(transform)

    def offsets(self, parameters, channel):
        return self.base_offsets[channel]+self.antennas.offsets(parameters[self.antenna_slice], channel)

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        motion, locations = self.expand(parameters), parameters[self.antenna_slice]
        value = .5*float(locations@locations)
        gradient = np.zeros(len(parameters))
        gradient[self.antenna_slice] = locations
        hessian = np.zeros((len(parameters), len(parameters)))
        hessian[self.antenna_slice, self.antenna_slice] = np.eye(self.antennas.count)
        conditions = []
        for c, (band, transform) in enumerate(zip(self.bands, self.transforms)):
            local = profile(band, motion, range_origin=parameters[self.range_slice.start+c],
                antenna_offsets=self.offsets(parameters, c), derivatives=True)
            value += local['value']
            gradient += transform.T@local['gradient']
            hessian += transform.T@local['gauss_newton']@transform
            conditions.append(local['condition'])
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': value/self.energy,
            'gradient': gradient/self.energy, 'gauss_newton': hessian/self.energy,
            'scene_conditions': conditions, 'evaluations': self.evaluations}
        return self.last

    def responses(self, parameters):
        if self.response_parameters is None or not np.array_equal(parameters, self.response_parameters):
            motion = self.expand(parameters)
            self.response_cache = [profile(band, motion, range_origin=parameters[self.range_slice.start+c],
                antenna_offsets=self.offsets(parameters, c)) for c, band in enumerate(self.bands)]
            self.response_parameters = np.array(parameters, copy=True)
        return self.response_cache

    def validation(self, parameters):
        checks = []
        for band, local in zip(self.bands, self.responses(parameters)):
            checks.append(validate(band.field, local['target'], band.nuisance, band.selected_time,
                band.model['interval_s'], coefficients=local['coefficient'], stationary_mode='project'))
        return checks
