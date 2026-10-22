"""Retain the Gaussian sufficient subspace for moving and stationary responses.

The existing moving-target Fourier projection is retained unchanged. Only its
orthogonal complement in the declared stationary dictionary is added. With
white input noise this union is white, and is sufficient for every mean in
the union. This is a conditional linear-model result, not an assertion that
the recorded interference is Gaussian or that arbitrary motion lies in it.
"""
import argparse
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from integrated_stationary_road import IntegratedField, IntegratedRoad, RecordedBins
from phase_history import C, FS
from scene_dictionary import fingerprint
from shared_target_geometry import RoadTarget, complex_gain
from stationary_road_control import StationaryRoad
from validate_shared_geometry import fit_target


class SufficientProjection:
    """Union of a time-only orthonormal projection and a fixed mean dictionary."""
    def __init__(self, temporal, dictionary, *, cutoff=1e-10):
        self.temporal = np.asarray(temporal)
        nt, self.nr, nc = dictionary.shape
        if self.temporal.shape[1] != nt or not 0 < cutoff < 1:
            raise ValueError('Invalid sufficient-subspace dimensions or cutoff')
        np.testing.assert_allclose(self.temporal@self.temporal.conj().T,
            np.eye(len(self.temporal)), rtol=0., atol=2e-10)
        complement = np.asarray(dictionary).copy()
        # Reproject once to control roundoff before the small-column SVD.
        for _ in range(2):
            complement -= np.einsum('tq,qrk->trk', self.temporal.conj().T,
                np.einsum('qt,trk->qrk', self.temporal, complement))
        u, s, _ = np.linalg.svd(complement.reshape(nt*self.nr, nc), full_matrices=False)
        scale = np.linalg.norm(dictionary.reshape(nt*self.nr, nc), ord=2)
        keep = s > cutoff*scale
        extra = u[:, keep].reshape(nt, self.nr, sum(keep))
        # Small singular vectors can regain a component in the Fourier
        # subspace through SVD roundoff. Reproject the normalized vectors,
        # then orthonormalize them; otherwise added noise statistics correlate.
        for _ in range(2):
            extra -= np.einsum('tq,qrk->trk', self.temporal.conj().T,
                np.einsum('qt,trk->qrk', self.temporal, extra))
        self.extra = np.linalg.qr(extra.reshape(nt*self.nr, sum(keep)), mode='reduced')[0]
        self.singular_values = s
        self.cutoff = cutoff
        self.dictionary_relative_loss = float(np.linalg.norm(
            dictionary-self.adjoint(self.apply(dictionary)))/max(scale, 1e-300))

    def apply(self, white):
        white = np.asarray(white)
        shape = white.shape[2:]
        matrix = white.reshape(white.shape[0], self.nr, -1)
        first = np.einsum('qt,trk->qrk', self.temporal, matrix).reshape(len(self.temporal)*self.nr, -1)
        extra = self.extra.conj().T@matrix.reshape(-1, matrix.shape[-1])
        return np.concatenate((first, extra)).reshape((-1,)+shape)

    def adjoint(self, value):
        value = np.asarray(value)
        count = len(self.temporal)*self.nr
        shape = value.shape[1:]
        first = value[:count].reshape(len(self.temporal), self.nr, -1)
        result = np.einsum('tq,qrk->trk', self.temporal.conj().T, first)
        columns = int(np.prod(shape)) if shape else 1
        result += (self.extra@value[count:].reshape(self.extra.shape[1], columns)).reshape(result.shape)
        return result.reshape((self.temporal.shape[1], self.nr)+shape)


class UnionField:
    """The original bin/range whitening followed by a fixed union projection."""
    def __init__(self, original, binned, cutoff=1e-10):
        self.original, self.time = original, original.time
        self.selected = original.selected
        self.sigma = np.sqrt(original.data['variance'][self.selected, len(original.offsets)//2])
        if np.any(self.sigma <= 0):
            raise ValueError('Positive declared bin variances are required')
        temporal = original.q[:, self.selected]*self.sigma[None, :]
        self.white_stationary = self.white(binned)
        self.projector = SufficientProjection(temporal, self.white_stationary, cutoff=cutoff)
        self.y = self.apply(original.data['value'])

    def white(self, values):
        values = np.asarray(values)
        shape = values.shape[2:]
        result = np.einsum('ab,tbk->tak', self.original.whiten,
            values[self.selected].reshape(sum(self.selected), len(self.original.offsets), -1))
        result /= self.sigma[:, None, None]
        return result.reshape(result.shape[:2]+shape)

    def apply(self, values):
        return self.projector.apply(self.white(values))

    def raw_template(self, path, jacobian=None, range_bias=0.):
        field = self.original
        lag = field.offsets[None, :]-(path+range_bias-field.seed_path)[:, None]*FS/C
        pulse = field.pulse(lag)
        phase = np.exp(-1j*field.k*(path-(field.seed_path-field.seed['path'][0])))
        value = phase[:, None]*pulse
        if jacobian is None:
            return value, None
        derivative = phase[:, None]*(-1j*field.k*pulse-FS/C*field.pulse(lag, 1))
        return value, derivative[:, :, None]*jacobian[:, None, :]

    def template(self, path, jacobian=None, range_bias=0.):
        value, derivative = self.raw_template(path, jacobian, range_bias)
        return self.apply(value), None if derivative is None else self.apply(derivative)


class UnionRoad(StationaryRoad):
    def __init__(self, previous, binned):
        def convert(field, values):
            union = UnionField(field.original, values, field.cutoff)
            return IntegratedField(SimpleNamespace(original=union, time=field.time,
                delays=field.delays, cutoff=field.cutoff), values)
        train = [convert(f, b) for f, b in zip(previous.training, binned)]
        test = [convert(f, b) for f, b in zip(previous.validation, binned)]
        RoadTarget.__init__(self, previous.problem, train, test, previous.initial,
            previous.transmitters, previous.antennas)
        self.energy = previous.energy


def coefficients(road, motion, antenna, c):
    train = road.training[c]
    u, du = road.template(motion, antenna, c, train)
    _, _, amplitude, _ = complex_gain(train.y, u, du)
    raw, _ = road.raw_template(motion, antenna, c, train)
    return amplitude, train.coefficients(train.original.y-amplitude*raw)


def compare_prediction(gated, union, motion, antenna, c, amplitude, beta):
    """One frozen prediction scored on two declared, identical test operators."""
    records = {}
    for name, road in [('original_gate', gated), ('augmented_gate', union)]:
        field = road.validation[c]
        u, _ = road.raw_template(motion, antenna, c, field)
        residual = field.original.y-amplitude*u-field.stationary@beta
        records[name] = {'prediction_error': float(np.vdot(residual, residual).real),
            'data_energy': float(np.vdot(field.original.y, field.original.y).real),
            'complex_observations': len(residual)}
    return records


def local_information(road, motion, antenna, c):
    """Compare local Gaussian information before/after the fixed union operator."""
    field = road.training[c]
    original = field.original
    # Ask the same physical model for the full, whitened bin values and their
    # derivatives. No projection, fitted phase adjustment or new weights.
    class FullField:
        time = original.time
        def template(self, path, jacobian, range_bias):
            u, du = original.raw_template(path, jacobian, range_bias)
            return original.white(u).ravel(), original.white(du).reshape(-1, jacobian.shape[1])
    full, dfull = RoadTarget.template(road, motion, antenna, c, FullField())
    compressed, derivative = road.raw_template(motion, antenna, c, field)
    background = original.white_stationary.reshape(len(full), -1)
    def efficient(u, du, b):
        nuisance = np.column_stack((b, u))
        basis, s, _ = np.linalg.svd(nuisance, full_matrices=False)
        basis = basis[:, s > s[0]*1e-10]
        value = du-basis@(basis.conj().T@du)
        return np.sqrt(2)*np.r_[value.real, value.imag]
    jfull = efficient(full, dfull, background)
    jsmall = efficient(compressed, derivative, field.stationary)
    _, s, vh = np.linalg.svd(jfull, full_matrices=False)
    keep = s > s[0]*1e-8
    modes = vh[keep].T/s[keep]
    scaled = jsmall@modes
    ratios = np.linalg.eigvalsh(scaled.T@scaled)
    return {'full_complex_observations': len(full), 'retained_complex_observations': len(compressed),
        'added_complex_observations': original.projector.extra.shape[1],
        'stationary_dictionary_relative_projection_error': original.projector.dictionary_relative_loss,
        'motion_and_antenna_information_rank': int(sum(keep)), 'relative_jacobian_rank_cutoff': 1e-8,
        'information_fraction_eigenvalues': ratios.tolist(),
        'full_noise_model_scope': 'The same empirical time variances and range covariance on saved 20-ms bin averages; not a raw-row covariance certificate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('geometry-fit', 'ship-fit', 'road-fit', 'control', 'integrated', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--iterations', type=int, default=180)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    integrated, fitted, saved = [json.loads(p.read_text()) for p in (args.integrated, args.geometry_fit, args.road_fit)]
    if not integrated['complete']:
        raise ValueError('Finish the integrated stationary comparison first')
    for group in ('sources', 'implementation_sha256'):
        for source, digest in integrated['identity'][group].items():
            if fingerprint(Path(source)) != digest:
                raise ValueError('Integrated stationary input changed: '+source)
    for path in (args.geometry_fit, args.ship_fit, args.road_fit, args.control):
        if integrated['identity']['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('Use the identical physical geometry and observations')
    control = json.loads(args.control.read_text())
    joint, _, _, labels, _ = load_experiment(args.ship_fit, args.road_fit, args.fields,
        args.geometry, fitted['identity']['labels'], fitted['identity']['nodes'])
    antenna = np.asarray(fitted['parameters'])[joint.antenna_slice]
    sources = [args.integrated, args.geometry_fit, args.ship_fit, args.road_fit, args.control]
    implementation = [Path(__file__), Path(__file__).with_name('integrated_stationary_road.py'),
        Path(__file__).with_name('stationary_road_control.py')]
    identity = {'sources': {str(p): fingerprint(p) for p in sources}, 'iterations': args.iterations,
        'implementation_sha256': {str(p): fingerprint(p) for p in implementation}}
    result = {'identity': identity, 'scope': __doc__, 'complete': False, 'targets': [],
        'accepted_calibration': False, 'production_changed': False}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume changed sufficient-subspace experiment')
    def save():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.output)
    started = time.monotonic()
    for record in integrated['targets']:
        label = record['label']
        if any(t['label'] == label for t in result['targets']):
            continue
        previous = StationaryRoad(joint.roads[labels.index(label)], cutoff=control['identity']['cutoff'])
        binned = []
        for c, field in enumerate(previous.training):
            with np.load(Path(saved['input_patches'])/f'{label}-{c}.npz') as raw:
                bins = RecordedBins(field.original, raw['row_time'], raw['row_information'])
                binned.append(bins.stationary(field.original, field.delays, previous.origins[c]))
        gated, union = IntegratedRoad(previous, binned), UnionRoad(previous, binned)
        fit = record['fit']
        initial = np.asarray(fit['starts'][fit['selected_start']]['parameters'])
        revised = fit_target(union, antenna, [initial, previous.initial], args.iterations)
        motion = np.asarray(revised['starts'][revised['selected_start']]['parameters'])
        comparisons = []
        for c in range(5):
            a = coefficients(gated, initial, antenna, c)
            b = coefficients(union, motion, antenna, c)
            comparisons.append({'channel': c,
                'gated_estimator': compare_prediction(gated, union, initial, antenna, c, *a),
                'augmented_estimator': compare_prediction(gated, union, motion, antenna, c, *b),
                'conditional_information': local_information(union, motion, antenna, c)})
        summary = {}
        for space in ('original_gate', 'augmented_gate'):
            before, after = [sum(r[model][space]['prediction_error'] for r in comparisons)
                for model in ('gated_estimator', 'augmented_estimator')]
            summary[space] = {'gated_error': before, 'augmented_error': after,
                'prediction_error_reduction_percent': 100*(1-after/before)}
        item = {'label': label, 'fit': revised, 'channels': comparisons, 'summary': summary,
            'selected_fit_stationary': revised['starts'][revised['selected_start']]['fit']['converged']}
        result['targets'].append(item); save()
        print(json.dumps({'label': label, 'stationary': item['selected_fit_stationary'],
            'summary': summary, 'seconds': time.monotonic()-started}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started); save()


if __name__ == '__main__':
    main()
