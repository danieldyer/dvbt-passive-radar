"""Short road trajectory through the exact archived complex observation model.

This likelihood never treats an independently selected weak-band peak as a
range measurement. Every band profiles its complex target amplitude jointly
with fixed-delay stationary nuisance, retaining all training range cells.
"""
import numpy as np

from field import average
from shared_target_geometry import complex_gain


def trajectory(road, state, elapsed):
    s0, speed, acceleration, lateral, height = state
    t = np.asarray(elapsed)
    s = s0+speed*t+.5*acceleration*t*t
    rate = speed+acceleration*t
    normal, dnormal = road.frame(s)
    tangent, second, third = [road.centre(s, order) for order in (1, 2, 3)]
    q = np.linalg.norm(tangent[:, :2], axis=1)
    dot = np.sum(tangent[:, :2]*second[:, :2], axis=1)
    turn = np.column_stack([-second[:, 1], second[:, 0], np.zeros(len(t))])
    d2normal = np.column_stack([-third[:, 1], third[:, 0], np.zeros(len(t))])/q[:, None]
    d2normal -= 2*turn*(dot/q**3)[:, None]
    d2normal -= normal*((np.sum(second[:, :2]**2+tangent[:, :2]*third[:, :2], axis=1))/q**2)[:, None]
    d2normal += 3*normal*(dot**2/q**4)[:, None]
    position = road.centre(s)+lateral*normal+np.array([0., 0., height])
    direction = tangent+lateral*dnormal
    curvature = second+lateral*d2normal
    velocity = direction*rate[:, None]
    basis = np.column_stack([np.ones(len(t)), t, .5*t*t])
    derivative = np.column_stack([np.zeros(len(t)), np.ones(len(t)), t])
    position_jacobian = np.concatenate([direction[:, :, None]*basis[:, None, :], normal[:, :, None],
                                        np.broadcast_to([0., 0., 1.], position.shape)[:, :, None]], axis=2)
    velocity_jacobian = np.concatenate([curvature[:, :, None]*rate[:, None, None]*basis[:, None, :]+
        direction[:, :, None]*derivative[:, None, :], (dnormal*rate[:, None])[:, :, None],
        np.zeros((len(t), 3, 1))], axis=2)
    return position, velocity, position_jacobian, velocity_jacobian


def template(field, road, state, epoch, transmitter, receiver, motion_scale,
             transmitter_mapping, receiver_mapping, *, range_origin=0., origin_derivative=False):
    position, velocity, dp, dv = trajectory(road, state, field.data['row_time']-epoch)
    dp *= np.asarray(motion_scale)[None, None, :]; dv *= np.asarray(motion_scale)[None, None, :]
    directions, lengths, rate_hessians = [], [], []
    for antenna in (transmitter, receiver):
        vector = position-antenna; distance = np.linalg.norm(vector, axis=1)
        unit = vector/distance[:, None]
        directions.append(unit); lengths.append(distance)
        rate_hessians.append((velocity-unit*np.sum(unit*velocity, axis=1)[:, None])/distance[:, None])
    direct = transmitter-receiver; baseline = np.linalg.norm(direct); direct /= baseline
    gradient = sum(directions)
    delay = sum(lengths)-baseline-field.data['centre_path_m']
    rate = np.sum(gradient*velocity, axis=1)
    path_jacobian = np.concatenate([np.einsum('ni,nij->nj', gradient, dp),
        (-directions[0]-direct)@transmitter_mapping+(-directions[1]+direct)@receiver_mapping], axis=1)
    rate_jacobian = np.concatenate([np.einsum('ni,nij->nj', gradient, dv)+
        np.einsum('ni,nij->nj', sum(rate_hessians), dp),
        -rate_hessians[0]@transmitter_mapping-rate_hessians[1]@receiver_mapping], axis=1)
    omega_factor = -field.k*.001024
    raw, dpath, domega, envelope = field.path_response(delay, omega_factor*rate,
        range_bias_m=range_origin, derivatives=True)
    derivative = dpath[:, :, None]*path_jacobian[:, None, :]+domega[:, :, None]*(omega_factor*rate_jacobian[:, None, :])
    if origin_derivative:
        derivative = np.concatenate([derivative, (30*envelope)[:, :, None]], axis=2)
    binned = average(raw, field.data['row_information'], field.metadata['block_rows'])[0]
    jacobian = average(derivative, field.data['row_information'], field.metadata['block_rows'])[0]
    return field.apply(binned).ravel(), (np.einsum('ab,tbj->taj', field.whiten, jacobian)*field.root_information[:, None, None]).reshape(-1, derivative.shape[-1])


class RawRoadGeometry:
    def __init__(self, geometry, likelihoods):
        self.geometry, self.likelihoods = geometry, likelihoods
        self.initial = geometry.initial.copy()
        self.last = None
        self.lower, self.upper = geometry.lower.copy(), geometry.upper.copy()
        # The raw model needs the complete two-second path to stay on the
        # connected road. Keep a conservative travel margin at its endpoints.
        for i, (arc, road) in enumerate(zip(geometry.arcs, geometry.roads)):
            j = geometry.start+5*i
            half = (arc['last_s']-arc['first_s'])/2+.03
            vmax = max(abs(geometry.lower[j+1]), abs(geometry.upper[j+1]))*geometry.scale[1]
            margin = vmax*half+2*half*half+1e-3
            centre = geometry.initial[j]*geometry.scale[0]
            self.lower[j] = max(margin, centre-150.)/geometry.scale[0]
            self.upper[j] = min(road.length-margin, centre+150.)/geometry.scale[0]
            if self.lower[j] >= self.upper[j]:
                raise ValueError('Road arc lacks connected support for its declared velocity/acceleration bounds')

    def band_template(self, parameters, index, channel):
        g = self.geometry
        tx, rx, origins, states = g.split(parameters[:len(g.initial)])
        arc, road = g.arcs[index], g.roads[index]
        likelihood = self.likelihoods[index][channel]
        epoch = (arc['first_s']+arc['last_s'])/2
        if g.free_antennas:
            txmap, rxmap = g.antennas.factor[3*channel:3*channel+3], g.antennas.factor[15:]
        else:
            txmap = rxmap = np.empty((3, 0))
        value, derivative = template(likelihood.field, road, states[index], epoch, tx[channel], rx, g.scale,
            txmap, rxmap, range_origin=origins[channel], origin_derivative=g.free_origins)
        columns = np.r_[np.arange(g.start+5*index, g.start+5*(index+1)), np.arange(g.antenna_count)]
        if g.free_origins:
            columns = np.r_[columns, g.antenna_count+channel]
        return value, derivative, columns

    @staticmethod
    def profile(y, h, derivative):
        return complex_gain(y, h, derivative)

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(parameters, self.last[0]):
            return self.last[1:]
        g = self.geometry
        tx, rx, origins, states = g.split(parameters[:len(g.initial)])
        residuals, jacobians, information = [], [], []
        for index, (arc, road, state, bands) in enumerate(zip(g.arcs, g.roads, states, self.likelihoods)):
            epoch = (arc['first_s']+arc['last_s'])/2
            for ch, likelihood in enumerate(bands):
                value, derivative, columns = self.band_template(parameters, index, ch)
                h = likelihood.nuisance.remove(value[likelihood.selected])
                dh = likelihood.nuisance.remove(derivative[likelihood.selected])
                residual, jacobian, _, mean = self.profile(likelihood.clean, h, dh)
                full = np.zeros((len(residual)*2, len(parameters)))
                full[:, columns] = np.sqrt(2)*np.r_[jacobian.real, jacobian.imag]
                residuals.append(np.sqrt(2)*np.r_[residual.real, residual.imag]); jacobians.append(full)
                expected = np.zeros_like(full)
                expected[:, columns] = np.sqrt(2)*np.r_[mean.real, mean.imag]
                information.append(expected)
        self.data_rows = sum(len(r) for r in residuals)
        self.expected_jacobian = np.concatenate(information)
        # Same physical priors as the summary control, included once.
        if g.free_antennas:
            residuals.append(parameters[:18]); jacobians.append(np.eye(len(parameters))[:18])
        for index, state in enumerate(states):
            base = g.start+5*index
            residuals.append(np.array([state[3]/5., (state[4]-2.)/5., state[2]/4.]))
            jacobian = np.zeros((3, len(parameters)))
            jacobian[0, base+3] = g.scale[3]/5.; jacobian[1, base+4] = g.scale[4]/5.; jacobian[2, base+2] = g.scale[2]/4.
            jacobians.append(jacobian)
        self.last = np.array(parameters, copy=True), np.concatenate(residuals), np.concatenate(jacobians)
        return self.last[1:]

    def diagnostics(self, parameters):
        """Local numerical convergence and data-only antenna information.

        The expected mean Jacobian profiles complex target/static coefficients.
        Using the residual Jacobian as Fisher information would spuriously
        count derivatives of fitted noise as geometrical information.
        """
        residual, jacobian = self.evaluate(parameters)
        step = np.linalg.lstsq(jacobian, -residual, rcond=1e-10)[0]
        # This is a local unconstrained GN diagnostic, not a global guarantee.
        decrease = .5*(residual@residual-np.linalg.norm(residual+jacobian@step)**2)
        result = {'data_squared_residual': float(residual[:self.data_rows]@residual[:self.data_rows]),
            'prior_squared_residual': float(residual[self.data_rows:]@residual[self.data_rows:]),
            'unconstrained_local_gauss_newton_decrease': float(max(0., decrease)),
            'maximum_local_gauss_newton_step': float(abs(step).max()),
            'information_scope': 'Conditional complex Gaussian mean information after target and stationary amplitude elimination; road motion and shared range origins also profiled. No location prior included.'}
        if self.geometry.free_antennas:
            antenna = self.expected_jacobian[:, :18]
            nuisance = self.expected_jacobian[:, 18:]
            u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
            q = u[:, s > s[0]*1e-10]
            profiled = antenna-q@(q.T@antenna)
            eigen, directions = np.linalg.eigh(profiled.T@profiled)
            result.update(data_information_eigenvalues_after_nuisance=eigen[::-1].tolist(),
                data_information_directions= directions[:, ::-1].T.tolist(),
                physically_free_antenna_dimensions=int(np.linalg.matrix_rank(self.geometry.antennas.factor)),
                data_modes_stronger_than_unit_location_prior=int(np.count_nonzero(eigen > 1.)),
                data_modes_above_one_percent_of_unit_location_prior=int(np.count_nonzero(eigen > .01)))
        return result

    def prediction(self, parameters):
        g = self.geometry
        tx, rx, origins, states = g.split(parameters[:len(g.initial)])
        output = []
        for index, (arc, road, state, bands) in enumerate(zip(g.arcs, g.roads, states, self.likelihoods)):
            epoch = (arc['first_s']+arc['last_s'])/2
            channels = []
            for ch, likelihood in enumerate(bands):
                h, _, _ = self.band_template(parameters, index, ch)
                train, test = likelihood.selected, ~likelihood.selected
                clean_h = likelihood.nuisance.remove(h[train])
                _, _, amplitude, _ = self.profile(likelihood.clean, clean_h, np.empty((len(clean_h), 0), complex))
                y = likelihood.field.y.ravel()
                beta = likelihood.nuisance.solve(y[train]-amplitude*h[train])
                residual = y[test]-likelihood.design[test]@beta-amplitude*h[test]
                base = likelihood.nuisance.solve(y[train])
                baseline = np.sum(abs(y[test]-likelihood.design[test]@base)**2)
                error = np.vdot(residual, residual).real
                channels.append({'channel': ch, 'error': float(error), 'baseline_error': float(baseline),
                    'prediction_fraction': float(1-error/baseline), 'target_coefficient': [amplitude.real, amplitude.imag]})
            output.append({'label': arc['label'], 'channels': channels})
        return output
