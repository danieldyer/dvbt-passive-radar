"""Shared physical ship motion with a common multichannel scattering scene."""
import numpy as np

from field import average
from scatterers import normal, shared
from spline_motion import SplineMotion
from validate_motion import validate


class CoherentMotion:
    def __init__(self, bands, gains, initial, *, profile_channel_gains=False, reference_channel=3,
                 antennas=None, fit_range_origins=False):
        self.bands, self.gains = bands, np.asarray(gains)
        if not all(np.array_equal(b.points, bands[0].points) for b in bands):
            raise ValueError('The coherent model requires shared physical scattering points')
        curve = SplineMotion({**bands[0].model, 'parameters': initial})
        self.full_initial, self.active = curve.gauge_fixed_parameters()
        self.nmotion = len(self.active)
        self.antennas = antennas
        self.nantenna = 0 if antennas is None else antennas.count
        self.antenna_slice = slice(self.nmotion, self.nmotion+self.nantenna)
        self.range_slice = slice(self.nmotion+self.nantenna, None)
        self.fit_ranges = fit_range_origins
        self.initial = np.r_[self.full_initial[self.active], np.zeros(self.nantenna),
                             [b.range_bias for b in bands] if self.fit_ranges else []]
        self.energy = sum(b.energy for b in bands)
        self.last = None
        self.profile_gains, self.reference = profile_channel_gains, reference_channel
        self.gain_diagnostics = None

    def expand(self, parameters):
        full = np.zeros_like(self.full_initial)
        full[self.active] = parameters[:self.nmotion]
        return full

    def range_origins(self, parameters):
        return np.asarray(parameters[self.range_slice] if self.fit_ranges else [b.range_bias for b in self.bands])

    def project(self, parameters, *, derivatives=False):
        full = self.expand(parameters)
        designs, raw, partials = [], [], []
        for c, band in enumerate(self.bands):
            offsets = None if self.antennas is None else self.antennas.offsets(parameters[self.antenna_slice], c)
            design, derivative = band.design(full, derivatives=derivatives, antenna_offsets=offsets,
                range_bias_m=self.range_origins(parameters)[c], range_derivative=self.fit_ranges)
            raw.append(design)
            designs.append(band.nuisance.remove(design[band.selected]))
            partials.append(derivative)
        gains = self.gains
        if self.profile_gains:
            fit = shared([normal(design, band.clean) for design, band in zip(designs, self.bands)], reference=self.reference)
            if not fit['converged'] or fit['optimality'] > 1e-7 or fit['gain_bound_hit']:
                raise ValueError(f'Inner channel-gain fit has not converged: optimality={fit["optimality"]:.6g}, '
                                 f'converged={fit["converged"]}, gain_bound_hit={fit["gain_bound_hit"]}')
            gains = fit['gains']
            self.gain_diagnostics = {key: fit[key] for key in ('converged', 'optimality', 'gain_bound_hit')}
        designs = [gain*design for gain, design in zip(gains, designs)]
        amplitude = np.linalg.lstsq(np.concatenate(designs), np.concatenate([b.clean for b in self.bands]), rcond=None)[0]
        return amplitude, designs, raw, partials, gains

    def objective(self, parameters):
        amplitude, designs, _, partials, gains = self.project(parameters, derivatives=True)
        cost, gradient = 0., np.zeros(len(parameters))
        if self.antennas is not None:
            locations = parameters[self.antenna_slice]
            cost += .5*(locations@locations)
            gradient[self.antenna_slice] = locations
        for c, (band, gain, design, derivative) in enumerate(zip(self.bands, gains, designs, partials)):
            residual = band.clean-design@amplitude
            cost += np.vdot(residual, residual).real
            local = band.profile_gradient(derivative, gain*amplitude, residual,
                antenna_derivatives=self.antennas is not None, range_derivative=self.fit_ranges)
            gradient[:self.nmotion] += local[self.active]
            if self.antennas is not None:
                gradient[self.antenna_slice] += self.antennas.mapping(c).T@local[len(self.full_initial):len(self.full_initial)+6]
            if self.fit_ranges:
                gradient[self.nmotion+self.nantenna+c] += local[-1]
        result = float(cost/self.energy), gradient/self.energy
        self.last = np.asarray(parameters).copy(), *result
        return result

    def information(self, parameters, *, profile_point_coordinates=False):
        amplitude, designs, _, partials, gains = self.project(parameters, derivatives=True)
        jacobians, scene_jacobians = [], []
        for c, (band, gain, derivatives) in enumerate(zip(self.bands, gains, partials)):
            raw = sum(gain*a*d for a, d in zip(amplitude, derivatives))
            def observation(value):
                value, _ = average(value, band.field.data['row_information'], band.field.metadata['block_rows'])
                return band.field.apply(value).ravel()[band.selected]
            columns = []
            for j, (b, db) in enumerate(zip(band.bases, band.rate_bases)):
                for k in range(b.shape[1]):
                    value = raw[:, :, j]*b[:, k, None]+raw[:, :, band.components+j]*db[:, k, None]
                    columns.append(observation(value))
            local = np.column_stack(columns)[:, self.active]
            if self.antennas is not None:
                antenna = np.column_stack([observation(raw[:, :, 2*band.components+i]) for i in range(6)])
                local = np.c_[local, antenna@self.antennas.mapping(c)]
            if self.fit_ranges:
                envelope = np.zeros((len(local), len(self.bands)), complex)
                envelope[:, c] = observation(raw[:, :, -1])
                local = np.c_[local, envelope]
            jacobians.append(band.nuisance.remove(local))
            if profile_point_coordinates:
                # Sensitivity control only: allow every shared point's three
                # coordinates to move, including the otherwise fixed anchor.
                # This avoids mistaking precise assumed scatterer locations for
                # information supplied by the received data about the antennas.
                curve = SplineMotion({**band.model, 'parameters': self.expand(parameters)})
                state = curve.apply(band.time, band.prior, band.reference_transmitter, band.receiver)
                offsets = np.zeros((2, 3)) if self.antennas is None else self.antennas.offsets(parameters[self.antenna_slice], c)
                columns = []
                for a, point in zip(amplitude, band.points):
                    _, derivative = band.field.point(point, motion=state, derivatives=True,
                        transmitter_offset_ecef_m=offsets[0], receiver_offset_ecef_m=offsets[1],
                        range_bias_m=self.range_origins(parameters)[c])
                    columns.extend((gain*a*derivative[:, :, axis]).ravel()[band.selected] for axis in range(3))
                scene_jacobians.append(band.nuisance.remove(np.column_stack(columns)))
        jacobian, design = np.concatenate(jacobians), np.concatenate(designs)
        if self.profile_gains:
            gain_columns = np.zeros((len(design), len(self.bands)-1), complex)
            offset, column = 0, 0
            for channel, band_design in enumerate(designs):
                if channel != self.reference:
                    gain_columns[offset:offset+len(band_design), column] = band_design@amplitude
                    column += 1
                offset += len(band_design)
            design = np.c_[design, gain_columns]
        jacobian -= design@np.linalg.lstsq(design, jacobian, rcond=None)[0]
        if profile_point_coordinates:
            scene = np.concatenate(scene_jacobians)
            scene -= design@np.linalg.lstsq(design, scene, rcond=None)[0]
            # Coordinates are real parameters. A complex least-squares
            # projection would incorrectly remove twice as many directions.
            real_scene, real_jacobian = np.r_[scene.real, scene.imag], np.r_[jacobian.real, jacobian.imag]
            real_jacobian -= real_scene@np.linalg.lstsq(real_scene, real_jacobian, rcond=1e-10)[0]
            return 2*(real_jacobian.T@real_jacobian)
        return 2*np.real(jacobian.conj().T@jacobian)

    def antenna_information(self, parameters, *, profile_point_coordinates=False):
        """Conditional geometry information, eliminating motion and delay origins."""
        if self.antennas is None:
            raise ValueError('No variable antennas in this model')
        information = self.information(parameters, profile_point_coordinates=profile_point_coordinates)
        antenna = np.arange(len(parameters))[self.antenna_slice]
        nuisance = np.setdiff1d(np.arange(len(parameters)), antenna)
        h = information[np.ix_(nuisance, nuisance)]
        eigen, vectors = np.linalg.eigh(h)
        keep = eigen > max(eigen.max(), 0)*1e-10
        inverse = (vectors[:, keep]/eigen[keep])@vectors[:, keep].T
        cross = information[np.ix_(antenna, nuisance)]
        marginal = information[np.ix_(antenna, antenna)]-cross@inverse@cross.T
        values, axes = np.linalg.eigh((marginal+marginal.T)/2)
        covariance = (axes/(1+np.maximum(values, 0)))@axes.T
        physical = self.antennas.factor@covariance@self.antennas.factor.T
        return {'antenna_information_eigenvalues_before_prior': values.tolist(),
            'point_coordinates_profiled': profile_point_coordinates,
            'motion_and_range_nuisance_rank': int(keep.sum()), 'motion_and_range_parameter_count': len(nuisance),
            'conditional_posterior_sigma_enu_m': np.sqrt(np.maximum(0, np.diag(physical))).reshape(-1, 3).tolist(),
            'prior_sigma_enu_m': np.sqrt(np.diag(self.antennas.factor@self.antennas.factor.T)).reshape(-1, 3).tolist(),
            'scope': 'Local Gaussian information conditional on a shared discrete scattering scene, thermal covariance, AIS midpoint pose and tower assignments. Common complex amplitudes, complex band gains, physical motion and envelope origins are eliminated; point coordinates are additionally eliminated only when requested. This is an optimistic sensitivity calculation, not calibrated antenna-position uncertainty.'}

    def validation(self, parameters, *, test_masks=None, stationary_mode='predict'):
        amplitude, _, designs, _, gains = self.project(parameters)
        if test_masks is None:
            test_masks = [None]*len(self.bands)
        return [validate(b.field, design, b.nuisance, b.selected_time, b.model['interval_s'],
                         coefficients=gain*amplitude, test_time=test, stationary_mode=stationary_mode)
                for b, gain, design, test in zip(self.bands, gains, designs, test_masks)]
