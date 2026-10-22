"""Conditional, held-time comparison of point and compact vehicle scattering.

The site, road and trajectory trial is held fixed. One or two rigidly attached
points have frequency-independent complex reflectivities and share positive
band magnitudes. This tests a possible model deficiency; it is not a new
calibration, nor permission to adopt its input coordinates.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from coherent_fit import OTHER
from geometry import unit
from range_fit import Field
from road_fit import load_trial


def project_points(data, design, derivative=None):
    """Variable projection of complex reflectivities, with an exact Jacobian."""
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    keep = singular > singular[0]*1e-10
    inverse = (vh.conj().T[:, keep]/singular[keep])@u[:, keep].conj().T
    amplitude = inverse@data
    error = design@amplitude-data
    if derivative is None:
        return np.sqrt(2)*error, amplitude
    applied = np.einsum('nsp,s->np', derivative, amplitude)
    normal_inverse = (vh.conj().T[:, keep]/singular[keep]**2)@vh[keep]
    da = -normal_inverse@np.einsum('nsp,n->sp', derivative.conj(), error)-inverse@applied
    jacobian = np.sqrt(2)*(applied+design@da)
    return np.sqrt(2)*error, jacobian, amplitude


def body_frames(model, state, times):
    """Orthonormal vehicle frame following the fitted road trajectory."""
    t = (times-model.seed['epoch'])/model.half
    base, b = model.length_basis(times)
    length = base+b@state[:model.radial_count]
    lateral = state[model.radial_count]+t*state[model.radial_count+1]
    normal, dnormal = model.road.frame(length)
    ds = (np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(model.coarse))+
          model.spline(t, 1)@state[:model.radial_count])/model.half
    velocity = (model.road.centre(length, 1)+lateral[:, None]*dnormal)*ds[:, None]
    velocity += normal*state[model.radial_count+1]/model.half
    if np.any(np.linalg.norm(velocity, axis=1) < .1):
        raise ValueError('Rigid body heading is not identified for a stopped target')
    along = unit(velocity)
    side = unit(np.c_[-along[:, 1], along[:, 0], np.zeros(len(times))])
    return np.stack([along, side, np.cross(along, side)], axis=2)


class RigidEcho:
    def __init__(self, problem, state, beta, count, selected=None, half_extent=(4., 1.5, 1.5)):
        self.problem, self.state, self.beta = problem, state, beta
        self.points = count
        self.half_extent = np.asarray(half_extent)
        self.count = 3*count+4
        self.fields, self.geometry = [], []
        for ch, field in enumerate(problem.fields):
            if selected is not None:
                field = Field(field.data, field.pulse, field.seed, selected=selected(field.time))
            self.fields.append(field)
            x, _, sites = problem.model.positions(state, field.time, problem.sites, True)
            tx = sites[problem.model.channel_sites[ch]]+problem.model.channel_offsets[ch]
            rx = problem.model.receiver(ch)
            self.geometry.append((x, body_frames(problem.model, state, field.time), tx, rx))
        self.data = np.concatenate([f.y for f in self.fields])

    def bounds(self):
        bound = np.tile(self.half_extent, self.points)
        return np.r_[-bound, np.full(4, -5.)], np.r_[bound, np.full(4, 5.)]

    def design(self, p, jacobian=False):
        magnitudes = np.ones(5)
        magnitudes[OTHER] = np.exp(p[3*self.points:])
        result, derivatives = [], []
        for ch, (field, (x, frame, tx, rx)) in enumerate(zip(self.fields, self.geometry)):
            columns = []
            derivative = np.zeros((len(field.y), self.points, self.count), complex) if jacobian else None
            gain = magnitudes[ch]*np.exp(1j*self.beta[ch])
            for point in range(self.points):
                location = x+frame@p[3*point:3*point+3]
                a, b = location-tx, location-rx
                path = np.linalg.norm(a, axis=1)+np.linalg.norm(b, axis=1)-np.linalg.norm(tx-rx)
                dp = np.einsum('ni,nij->nj', unit(a)+unit(b), frame) if jacobian else None
                value, local = field.template(path, dp)
                columns.append(gain*value)
                if jacobian:
                    derivative[:, point, 3*point:3*point+3] = gain*local
                    if ch != 1:
                        index = int(np.flatnonzero(OTHER == ch)[0])
                        derivative[:, point, 3*self.points+index] = gain*value
            result.append(np.column_stack(columns))
            if jacobian:
                derivatives.append(derivative)
        return np.concatenate(result), np.concatenate(derivatives) if jacobian else None

    def residual(self, p, jacobian=False):
        matrix, derivative = self.design(p, jacobian)
        result = project_points(self.data, matrix, derivative)
        value = result[1] if jacobian else result[0]
        return np.r_[value.real, value.imag]

    def fit(self, start, iterations):
        return least_squares(self.residual, start, jac=lambda p:self.residual(p, True),
                             bounds=self.bounds(), x_scale='jac', max_nfev=iterations,
                             ftol=1e-8, xtol=1e-9, gtol=1e-5)


def validation(train, test, state):
    u, _ = train.design(state)
    v, _ = test.design(state)
    _, amplitude = project_points(train.data, u)
    predicted = v@amplitude
    offset = 0
    channels = []
    for ch in range(5):
        n = len(train.fields[ch].y)
        # Train and test gates have the same harmonic count, even when their
        # selected raw samples differ. No held-out amplitude is fitted here.
        if n != len(test.fields[ch].y):
            raise ValueError('Different train/test observation dimensions')
        sl = slice(offset, offset+n)
        _, independent = project_points(train.data[sl], u[sl])
        independent_prediction = v[sl]@independent
        channels.append({'channel': ch,
            'shared_prediction_error': float(np.sum(abs(test.data[sl]-predicted[sl])**2)),
            'independent_prediction_error': float(np.sum(abs(test.data[sl]-independent_prediction)**2)),
            'shared_predicted_energy': float(np.sum(abs(predicted[sl])**2)),
            'independent_predicted_energy': float(np.sum(abs(independent_prediction)**2))})
        offset += n
    singular = np.linalg.svd(u, compute_uv=False)
    return {'shared_prediction_error': sum(q['shared_prediction_error'] for q in channels),
            'independent_prediction_error': sum(q['independent_prediction_error'] for q in channels),
            'channels': channels, 'design_condition_number': float(singular[0]/singular[-1]),
            'complex_reflectivities': np.c_[amplitude.real, amplitude.imag].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('fit', 'histories', 'patches', 'templates', 'routes', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--starts', type=int, default=6)
    parser.add_argument('--iterations', type=int, default=150)
    args = parser.parse_args()
    saved = json.loads(args.fit.read_text())
    joint = load_trial(saved, args.routes, args.histories, args.patches, args.templates)
    state = np.array(saved['state'])
    beta = np.zeros(5)
    beta[OTHER] = state[joint.beta_offset:joint.magnitude_offset]
    results = []
    clock = time.monotonic()
    rng = np.random.default_rng(14619)
    for index, (label, problem) in enumerate(zip(saved['labels'], joint.problems)):
        if args.labels and label not in args.labels:
            continue
        local = np.r_[state[9+index*joint.n:9+(index+1)*joint.n], state[:9]]
        first = problem.model.seed['epoch']-problem.model.half
        train_blocks = lambda times:np.floor(times-first).astype(int)%2 == 0
        test_blocks = lambda times:~train_blocks(times)
        magnitudes = []
        for ch in range(5):
            u, _ = problem.channel_model(local, ch, True)
            magnitudes.append(abs(np.vdot(u, problem.channel_data(ch))/np.vdot(u, u)))
        logs = np.clip(np.log(np.array(magnitudes)[OTHER]/magnitudes[1]), -4.9, 4.9)
        target = {'label': label, 'models': []}
        point_state = None
        for count in (1, 2):
            train = RigidEcho(problem, local, beta, count, train_blocks)
            test = RigidEcho(problem, local, beta, count, test_blocks)
            starts = [np.r_[np.zeros(3*count), logs]]
            if count == 2:
                starts[0][:3] = point_state[:3]
                starts[0][3:6] = np.clip(point_state[:3]+[1., .3, .2], -train.half_extent*.99, train.half_extent*.99)
                starts[0][6:] = point_state[3:]
            for _ in range(args.starts-1):
                starts.append(np.r_[rng.uniform(-.8, .8, (count, 3)).ravel()*np.tile(train.half_extent, count), logs])
            fits = []
            for start in starts:
                fit = train.fit(start, args.iterations)
                fits.append(fit)
                print(label, count, 'cost', round(fit.cost, 2), 'eval', fit.nfev,
                      'elapsed', round(time.monotonic()-clock, 1), flush=True)
            fit = min(fits, key=lambda f:f.cost)
            if count == 1:
                point_state = fit.x.copy()
            target['models'].append({'points': count, 'training_cost': float(fit.cost),
                'state': fit.x.tolist(), 'converged': bool(fit.success),
                'optimality': float(fit.optimality), 'evaluations': fit.nfev,
                'half_extent_m': train.half_extent.tolist(), 'held_out': validation(train, test, fit.x)})
        results.append(target)
        output = {'source_fit': str(args.fit), 'targets': results,
            'scope': 'Conditional model diagnostic: fixed trial calibration/trajectory discovered using the complete recording. Alternating one-second blocks fit and test point offsets, reflectivities and positive band gains. This is not independent validation of the site calibration.',
            'model': 'One or two rigid points within an 8 by 3 by 3 metre body; each point has one complex reflectivity across all five bands. Exact bistatic paths, observed waveform templates and identical linear observation operators for model and data.',
            'reference': 'https://doi.org/10.1049/rsn2.12263'}
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
