"""One antenna geometry shared by independently modelled moving targets.

Vehicle observations retain their existing range/time whitening. Each vehicle
has a road-constrained trajectory and independent complex band reflectivities.
Only physical TX/RX coordinates are shared with the ship. Likelihoods are added
in their original noise units; the antenna prior is included once. Road/motion
priors remain explicitly separate from measured-data information.
"""
import numpy as np


def path_partials(position, position_jacobian, transmitter, receiver):
    """Excess path and derivatives: target coordinates, TX xyz, RX xyz."""
    a, b, direct = position-transmitter, position-receiver, transmitter-receiver
    na, nb, nd = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(direct)
    if min(na.min(), nb.min(), nd) <= 0:
        raise ValueError('Coincident target/antenna coordinates')
    ua, ub, ud = a/na[:, None], b/nb[:, None], direct/nd
    derivative = np.concatenate((np.einsum('ni,nij->nj', ua+ub, position_jacobian),
                                -ua-ud, -ub+ud), axis=1)
    return na+nb-nd, derivative


def complex_gain(y, u, derivative):
    """Exact real-parameter residual derivative after profiling one complex gain."""
    norm = np.vdot(u, u).real
    if norm <= 1e-20:
        raise ValueError('Target has no support in the retained observations')
    amplitude = np.vdot(u, y)/norm
    da = (derivative.conj().T@y-amplitude*2*np.real(u.conj()@derivative))/norm
    residual = amplitude*u-y
    jacobian = amplitude*derivative+u[:, None]*da
    mean = amplitude*derivative
    mean -= u[:, None]*(u.conj()@mean/norm)
    return residual, jacobian, amplitude, mean


class RoadTarget:
    """A physical vehicle model in the fixed receiver-local ENU frame.

    The nominal transmitter array is supplied by the same licensed hypothesis
    as the ship. It is not reconstructed from the vehicle's old fitted site
    corrections. Receiver and per-channel corrections in that old fit are
    likewise not silently inherited.
    """
    def __init__(self, problem, training, validation, initial_motion, transmitters_enu, antennas):
        self.problem, self.training, self.validation = problem, training, validation
        self.initial = np.asarray(initial_motion).copy()
        self.transmitters = np.asarray(transmitters_enu)
        self.antennas = antennas
        self.nmotion = len(self.initial)
        if self.nmotion != problem.model.count or len(training) != len(self.transmitters):
            raise ValueError('Road trajectory and channel dimensions disagree')
        self.origins = np.asarray(problem.range_origins).copy()
        self.energy = sum(np.vdot(field.y, field.y).real for field in training)
        self.mappings = []
        for c in range(len(training)):
            indices = np.r_[np.arange(3*c, 3*c+3), np.arange(antennas.count-3, antennas.count)]
            self.mappings.append(antennas.factor[indices])

    def template(self, motion, antenna, channel, field):
        position, jacobian, _ = self.problem.model.positions(motion, field.time, self.problem.sites)
        mapping = self.mappings[channel]
        offset = (mapping@antenna).reshape(2, 3)
        path, partials = path_partials(position, jacobian,
            self.transmitters[channel]+offset[0], offset[1])
        derivative = np.c_[partials[:, :self.nmotion], partials[:, self.nmotion:]@mapping]
        return field.template(path, derivative, range_bias=self.origins[channel])

    def evaluate(self, motion, antenna, *, include_motion_prior=True):
        count = self.nmotion+self.antennas.count
        value, gradient, curvature = 0., np.zeros(count), np.zeros((count, count))
        information = []
        coefficients = []
        for c, field in enumerate(self.training):
            u, du = self.template(motion, antenna, c, field)
            residual, jacobian, amplitude, mean = complex_gain(field.y, u, du)
            value += float(np.vdot(residual, residual).real)
            gradient += 2*np.real(jacobian.conj().T@residual)
            curvature += 2*np.real(jacobian.conj().T@jacobian)
            information.append(np.sqrt(2)*np.r_[mean.real, mean.imag])
            coefficients.append(amplitude)
        data_value = value
        if include_motion_prior:
            residual = np.concatenate(self.problem.auxiliary(motion))
            derivative = np.concatenate(self.problem.auxiliary(motion, jacobian=True))
            value += .5*float(residual@residual)
            gradient[:self.nmotion] += derivative.T@residual
            curvature[:self.nmotion, :self.nmotion] += derivative.T@derivative
        return {'value': value, 'data_value': data_value, 'motion_prior_value': value-data_value,
            'gradient': gradient, 'gauss_newton': curvature,
            'expected_real_jacobian': np.concatenate(information), 'coefficients': coefficients}

    def predict(self, motion, antenna):
        results = []
        for c, (train, test) in enumerate(zip(self.training, self.validation)):
            u, du = self.template(motion, antenna, c, train)
            _, _, amplitude, _ = complex_gain(train.y, u, du)
            v, _ = self.template(motion, antenna, c, test)
            prediction = amplitude*v
            norm = float(np.vdot(prediction, prediction).real)
            results.append({'prediction_error': float(np.vdot(test.y-prediction, test.y-prediction).real),
                'data_energy': float(np.vdot(test.y, test.y).real),
                'matched_energy': float(abs(np.vdot(prediction, test.y))**2/norm) if norm else 0.,
                'coefficient_real_imag': [float(amplitude.real), float(amplitude.imag)]})
        return results


class SharedGeometry:
    """Sum the ship and vehicle likelihoods with one common antenna parameter block."""
    def __init__(self, ship, roads, *, initial_ship=None):
        self.ship, self.roads = ship, roads
        self.antenna_slice = ship.antenna_slice
        self.ship_count = len(ship.initial)
        self.initial = np.r_[ship.initial if initial_ship is None else initial_ship,
                             *[road.initial for road in roads]]
        self.energy = float(ship.energy+sum(road.energy for road in roads))
        self.road_slices = []
        start = self.ship_count
        for road in roads:
            if not np.array_equal(road.antennas.factor, ship.antennas.factor):
                raise ValueError('Targets must share the same physical antenna covariance coordinates')
            self.road_slices.append(slice(start, start+road.nmotion))
            start += road.nmotion
        self.last, self.evaluations = None, 0

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(parameters, self.last['parameters']):
            return self.last
        ship = self.ship.evaluate(parameters[:self.ship_count])
        value = ship['value']*self.ship.energy
        gradient = np.zeros(len(parameters))
        curvature = np.zeros((len(parameters), len(parameters)))
        gradient[:self.ship_count] = ship['gradient']*self.ship.energy
        curvature[:self.ship_count, :self.ship_count] = ship['gauss_newton']*self.ship.energy
        diagnostics = []
        for road, local in zip(self.roads, self.road_slices):
            result = road.evaluate(parameters[local], parameters[self.antenna_slice])
            columns = np.r_[np.arange(len(parameters))[local],
                            np.arange(len(parameters))[self.antenna_slice]]
            value += result['value']
            gradient[columns] += result['gradient']
            curvature[np.ix_(columns, columns)] += result['gauss_newton']
            diagnostics.append({k: result[k] for k in ('data_value', 'motion_prior_value')})
        self.evaluations += 1
        self.last = {'parameters': np.array(parameters, copy=True), 'value': value/self.energy,
            'gradient': gradient/self.energy, 'gauss_newton': curvature/self.energy,
            'scene_conditions': ship['scene_conditions'], 'road_values': diagnostics,
            'raw_ship_value_including_one_antenna_prior': ship['value']*self.ship.energy,
            'evaluations': self.evaluations}
        return self.last

    def objective(self, parameters):
        result = self.evaluate(parameters)
        return result['value'], result['gradient']
