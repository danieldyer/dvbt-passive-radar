"""Finite material-line hypothesis through the unchanged native observation map.

One common constant-velocity centre and rigid angular rate drive all bands.
The coherent line is integrated before native-symbol averaging and whitening.
Each band has only one constant complex coefficient plus stationary nuisance.
This diagnostic conditions transverse centre motion on AIS; it does not claim
full geographic motion observability or a universal electromagnetic hull model.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

from field import average
from shared_target_geometry import complex_gain
from stationary import Stationary


SCALE = np.array([25., .1, 1., .1, 100., 100.])
LOWER = np.array([-112., -1., -8., -.5, -150., 80.])
UPPER = np.array([112., 1., 8., .5, 150., 350.])


class FacetBand:
    def __init__(self, field, training, stationary_design, receiver, rotation,
                 state_enu, midpoint_s, tangent_enu, direction_enu, *, quadrature=32):
        self.field, self.training = field, np.asarray(training, bool)
        self.selected = np.repeat(self.training, len(field.offsets))
        self.design = np.asarray(stationary_design)
        self.nuisance = Stationary(self.design, self.selected)
        self.clean = self.nuisance.remove(field.y.ravel()[self.selected])
        self.energy = float(np.vdot(self.clean, self.clean).real)
        self.receiver, self.rotation, self.state = map(np.asarray, (receiver, rotation, state_enu))
        self.midpoint, self.tangent, self.direction = midpoint_s, np.asarray(tangent_enu), np.asarray(direction_enu)
        self.nodes, self.weights = leggauss(quadrature); self.weights /= 2
        self.last = None

    def material_state(self, parameters, coordinate):
        """Analytic position/velocity and all six physical parameter derivatives."""
        offset, rate, heading, spin, along, length = parameters
        dt = self.field.data['row_time']-self.midpoint
        deg = np.pi/180
        angle = (heading+spin*dt)*deg
        c, s = np.cos(angle), np.sin(angle)
        e, n, _ = self.tangent
        u = np.c_[c*e-s*n, s*e+c*n, np.zeros(len(dt))]
        j = np.c_[-u[:, 1], u[:, 0], np.zeros(len(dt))]
        q = along+length*coordinate/2
        rotated, normal = u*q, j*q
        local = self.state[0]+dt[:, None]*self.state[1]+(offset+rate*dt)[:, None]*self.direction+rotated
        local_velocity = self.state[1]+rate*self.direction+spin*deg*normal
        position = (self.receiver-self.field.data['centre_ecef_m'])+local@self.rotation
        velocity = local_velocity@self.rotation
        dp = np.stack([np.broadcast_to(self.direction, local.shape), dt[:, None]*self.direction,
            deg*normal, (dt*deg)[:, None]*normal, u, u*coordinate/2], axis=-1)
        dv = np.stack([np.zeros_like(local), np.broadcast_to(self.direction, local.shape),
            -spin*deg**2*rotated, deg*normal-(dt*spin*deg**2)[:, None]*rotated,
            spin*deg*j, spin*deg*j*coordinate/2], axis=-1)
        return position, velocity, np.einsum('nij,ik->nkj', dp, self.rotation), np.einsum('nij,ik->nkj', dv, self.rotation)

    def template(self, parameters, *, derivatives=True):
        f = self.field; raw = np.zeros((len(f.data['row_time']), len(f.offsets)), complex)
        jacobian = np.zeros(raw.shape+(6,), complex) if derivatives else None
        for node, weight in zip(self.nodes, self.weights):
            offset, velocity, dp, dv = self.material_state(parameters, node)
            value = f.raw_trajectory(f.data['centre_ecef_m']+offset, velocity, position_offset=offset,
                **(dict(position_jacobian=dp, velocity_jacobian=dv) if derivatives else {}))
            if derivatives:
                h, j = value; raw += weight*h; jacobian += weight*j
            else: raw += weight*value
        reduced = f.apply(average(raw, f.data['row_information'], f.metadata['block_rows'])[0]).ravel()
        if not derivatives: return reduced
        mean = average(jacobian, f.data['row_information'], f.metadata['block_rows'])[0]
        derivative = np.einsum('nrj,sr,n->nsj', mean, f.whiten, f.root_information).reshape(-1, 6)
        return reduced, derivative

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(parameters, self.last[0]): return self.last[1]
        h, j = self.template(parameters)
        h = self.nuisance.remove(h[self.selected]); j = self.nuisance.remove(j[self.selected])
        residual, jacobian, coefficient, mean = complex_gain(self.clean, h, j)
        result = dict(residual=residual, jacobian=jacobian, coefficient=coefficient, mean_jacobian=mean)
        self.last = np.asarray(parameters).copy(), result
        return result

    def prediction(self, parameters):
        term = self.evaluate(parameters); h = self.template(parameters, derivatives=False)
        predicted = h*term['coefficient']; y = self.field.y.ravel()
        beta = self.nuisance.solve(y[self.selected]-predicted[self.selected])
        baseline = self.nuisance.solve(y[self.selected]); test = ~self.selected
        clean = y-self.design@beta
        error = np.sum(abs(clean[test]-predicted[test])**2)
        reference = np.sum(abs(y[test]-self.design[test]@baseline)**2)
        return dict(test_rss=float(error), stationary_only_test_rss=float(reference),
            explained_fraction=float(1-error/reference),
            coefficient_real_imag=[float(term['coefficient'].real), float(term['coefficient'].imag)],
            prediction=predicted, stationary_clean=clean)


def fit(bands, seed, *, free=(0, 1, 2, 3, 4), max_evaluations=100):
    free, seed = np.asarray(free), np.asarray(seed)
    energy = sum(b.energy for b in bands)
    def evaluate(p):
        q = seed.copy(); q[free] = p*SCALE[free]
        terms = [b.evaluate(q) for b in bands]
        r = np.concatenate([np.r_[v['residual'].real, v['residual'].imag] for v in terms])/np.sqrt(energy)
        j = np.concatenate([np.r_[v['jacobian'].real, v['jacobian'].imag] for v in terms])[:, free]*SCALE[free]/np.sqrt(energy)
        return r, j
    result = least_squares(lambda p:evaluate(p)[0], seed[free]/SCALE[free], jac=lambda p:evaluate(p)[1],
        bounds=(LOWER[free]/SCALE[free], UPPER[free]/SCALE[free]), x_scale='jac',
        max_nfev=max_evaluations, ftol=1e-12, xtol=1e-12, gtol=1e-9)
    p = seed.copy(); p[free] = result.x*SCALE[free]
    r, j = evaluate(result.x)
    gradient = 2*j.T@r
    lower = np.isclose(p[free], LOWER[free], rtol=0., atol=1e-7)
    upper = np.isclose(p[free], UPPER[free], rtol=0., atol=1e-7)
    gradient = np.where(lower, np.minimum(gradient, 0.), np.where(upper, np.maximum(gradient, 0.), gradient))
    projected = float(np.max(abs(gradient)))
    return dict(parameters=p, objective=float(r@r), success=bool(result.success and projected < 1e-6),
        optimizer_termination=str(result.message), maximum_scaled_projected_gradient=projected,
        optimizer_bound_scaled_optimality=float(result.optimality),
        evaluations=int(result.nfev), active_bounds=free[result.active_mask != 0], free_indices=free,
        profiled_residual_jacobian_singular_values=np.linalg.svd(j, compute_uv=False))
