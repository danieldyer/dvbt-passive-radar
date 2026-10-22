"""Information retained by recorded vehicle time averaging and fixed weights.

The reference is the original cropped range-profile rows, with independent
proper-complex row noise and the existing mean range covariance. Row variance
uses archived information and the standard uniform-symbol first-moment noise
factor. This conditional model is not a certificate for real clutter/noise,
the finite within-symbol covariance, or prior range-profile extraction.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from field import C, FS
from fit_shared_target_geometry import load_experiment
from integrated_stationary_road import RecordedBins
from scene_dictionary import fingerprint
from shared_target_geometry import path_partials


def point_fisher(gram):
    """Eliminate one unknown complex amplitude from an augmented mean Gram."""
    gram = (gram+gram.conj().T)/2
    energy = gram[0, 0].real
    if energy <= 0:
        raise ValueError('Require nonzero signal information')
    value = 2*np.real(gram[1:, 1:]-np.outer(gram[1:, 0], gram[0, 1:])/energy)
    return (value+value.T)/2


def retention(full, compressed, cutoff):
    norms = np.sqrt(np.maximum(np.diag(full), 0.))
    active = norms > norms.max()*1e-12
    scale = norms[active]
    a = full[np.ix_(active, active)]/scale[:, None]/scale
    b = compressed[np.ix_(active, active)]/scale[:, None]/scale
    values, vectors = np.linalg.eigh(a)
    keep = values > values[-1]*cutoff
    if not keep.any() or values[0] < -values[-1]*1e-9:
        raise ValueError('Unresolved or indefinite original Fisher matrix')
    root = vectors[:, keep]/np.sqrt(values[keep])
    reduced = root.T@b@root
    eigen = np.linalg.eigvalsh((reduced+reduced.T)/2)
    if eigen.min() < -1e-7 or eigen.max() > 1+2e-5:
        raise ValueError('Averaging violates the information inequality')
    return {'relative_cutoff': cutoff, 'retained_local_modes': int(keep.sum()),
        'excluded_local_modes': int(len(norms)-keep.sum()),
        'retention_eigenvalues': eigen.tolist(),
        'worst_information_loss_db': float(-10*np.log10(max(eigen.min(), 1e-30)))}


class AveragingInformation:
    """Stream physical means/derivatives through the fixed weighted average."""
    def __init__(self, count, ranges, parameters, selected):
        self.selected = np.asarray(selected, bool)
        if self.selected.shape != (count,):
            raise ValueError('Selection must use recorded bins')
        self.weight = np.zeros(count)
        self.variance_sum = np.zeros(count)
        self.sum = np.zeros((count, ranges, parameters+1), complex)
        self.raw_gram = np.zeros((parameters+1, parameters+1), complex)

    def add(self, bins, weight, variance, signal, derivative):
        bins, weight, variance = np.asarray(bins), np.asarray(weight), np.asarray(variance)
        if (np.any(weight <= 0) or np.any(variance <= 0)
                or not np.isfinite(weight).all() or not np.isfinite(variance).all()):
            raise ValueError('Require positive finite row weights and variances')
        matrix = np.concatenate((signal[:, :, None], derivative), axis=2)
        white = matrix[self.selected[bins]]/np.sqrt(variance[self.selected[bins]])[:, None, None]
        white = white.reshape(-1, matrix.shape[-1])
        self.raw_gram += white.conj().T@white
        np.add.at(self.weight, bins, weight)
        np.add.at(self.variance_sum, bins, weight*weight*variance)
        np.add.at(self.sum, bins, weight[:, None, None]*matrix)

    def finish(self, empirical_variance=None):
        select = self.selected
        if np.any(self.weight[select] <= 0):
            raise ValueError('A selected bin has no original observations')
        mean = self.sum[select]/self.weight[select, None, None]
        variance = self.variance_sum[select]/self.weight[select]**2
        white = (mean/np.sqrt(variance)[:, None, None]).reshape(-1, mean.shape[-1])
        gram = white.conj().T@white
        ratio = float(gram[0, 0].real/self.raw_gram[0, 0].real)
        if not 0 < ratio <= 1+1e-10:
            raise ValueError('Averaging increased conditional target information')
        full, reduced = point_fisher(self.raw_gram), point_fisher(gram)
        result = {'matched_snr_retention': ratio,
            'matched_snr_loss_db': float(-10*np.log10(ratio)),
            'raw_information': full.tolist(), 'binned_information': reduced.tolist(),
            'local_information': [retention(full, reduced, cutoff) for cutoff in (1e-8, 1e-10)],
            'selected_rows_weight': float(self.weight[select].sum()), 'selected_bins': int(select.sum())}
        if empirical_variance is not None:
            empirical = np.asarray(empirical_variance)[select]
            if np.any(empirical <= 0) or not np.isfinite(empirical).all():
                raise ValueError('Require positive empirical bin variances')
            energy = np.sum(abs(mean[:, :, 0])**2, axis=1)
            optimum = np.sum(energy/variance)
            fixed = np.sum(energy/empirical)**2/np.sum(energy*variance/empirical**2)
            result['fixed_empirical_metric_snr_loss_db'] = float(-10*np.log10(fixed/optimum))
            result['empirical_over_conditional_variance_percentiles'] = np.percentile(
                empirical/variance, [1, 50, 99]).tolist()
        return result


def audit(road, motion, antenna, origin, channel, raw):
    field = road.training[channel]
    bins = RecordedBins(field, raw['row_time'], raw['row_information'])
    weights = np.asarray(raw['row_information'])[bins.keep]
    mapping = road.mappings[channel]; offset = (mapping@antenna).reshape(2, 3)
    position, basis, _ = road.problem.model.positions(motion, bins.time, road.problem.sites)
    path, partial = path_partials(position, basis, road.transmitters[channel]+offset[0], offset[1])
    derivative = np.c_[partial[:, :road.nmotion], partial[:, road.nmotion:]@mapping]
    seed_time = (bins.time-field.seed['epoch'])/field.seed.get('scale', 15.)
    seed = np.polynomial.polynomial.polyval(seed_time, field.seed['path'])
    speed = np.polynomial.polynomial.polyval(seed_time,
        np.polynomial.polynomial.polyder(field.seed['path']))/field.seed.get('scale', 15.)
    omega = -field.k*speed*.001024
    variance = (1+omega*omega/12)/(8192*weights)
    # A constant carrier-phase derivative is exactly in the unknown-gain
    # nuisance. Subtract it before forming Grams to avoid large cancellations
    # for weak antenna directions. This does not change efficient information.
    centre = np.average(derivative, axis=0, weights=weights)
    result = AveragingInformation(bins.count, len(field.offsets), derivative.shape[1]+1, field.selected)
    for begin in range(0, len(path), 512):
        take = slice(begin, begin+512)
        lag = field.offsets[None, :]-(path[take]+origin-seed[take])[:, None]*FS/C
        pulse = field.pulse(lag)
        phase = np.exp(-1j*field.k*(path[take]-seed[take]+field.seed['path'][0]))
        signal = (phase[:, None]*pulse)@field.whiten.T
        slope = (-FS/C*phase[:, None]*field.pulse(lag, 1))@field.whiten.T
        physical = (-1j*field.k*signal[:, :, None]*(derivative[take]-centre)[:, None, :]
                    +slope[:, :, None]*derivative[take, None, :])
        physical = np.concatenate((physical, slope[:, :, None]), axis=2)
        result.add(bins.index[take], weights[take], variance[take], signal, physical)
    value = result.finish(field.data['variance'][:, len(field.offsets)//2])
    value.update(channel=channel, original_rows=int(np.sum(field.selected[bins.index])),
        symbol_moment_noise_factor_minmax=[float((1+omega*omega/12).min()), float((1+omega*omega/12).max())])
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--labels', nargs='+', default=['A', 'O', 'K', 'Q'])
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve completed/partial audits')
    source, saved = [json.loads(path.read_text()) for path in (args.source, args.road_fit)]
    if not source['complete'] or not source['fit']['converged'] or source['original_row_maximum_kkt_residual'] > 1e-7:
        raise ValueError('Use a stationary common-origin reference')
    identity = source['identity']; inherited = identity['legacy_identity']
    for path in (args.ship_fit, args.road_fit):
        if fingerprint(path) != inherited['sources'][str(path)]:
            raise ValueError('Mismatched geometry source')
    joint, _, _, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], inherited['nodes'])
    if dropped or any(label not in labels for label in args.labels):
        raise ValueError('Use the same disjoint target hypotheses')
    parameters = np.asarray(source['parameters']); antenna = parameters[joint.antenna_slice]
    origins = np.asarray(source['range_origins_m']); patches = Path(saved['input_patches'])
    paths = [args.source, args.ship_fit, args.road_fit, Path(__file__),
        Path(__file__).with_name('integrated_stationary_road.py'),
        *[patches/f'{label}-{c}.npz' for label in args.labels for c in range(5)]]
    output = {'complete': False, 'scope': __doc__, 'targets': [],
        'source_sha256': {str(path): fingerprint(path) for path in paths},
        'geometry_changed': False, 'production_changed': False,
        'metric_scope': 'Observed empirical weights held fixed. Conditional SNR loss uses the stipulated row covariance; it is not an unconditional risk calculation for estimating weights from the same samples.',
        'nuisance_scope': 'Independent complex target amplitude removed per band. No static or moving interference nuisance is included in this time-averaging audit.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        i = labels.index(label); road = joint.roads[i]; motion = parameters[joint.road_slices[i]]
        channels = []
        for c in range(5):
            with np.load(patches/f'{label}-{c}.npz') as raw:
                channels.append(audit(road, motion, antenna, origins[c], c, raw))
        output['targets'].append({'label': label, 'channels': channels})
        args.output.write_text(json.dumps(output, indent=2)+'\n')
        print(json.dumps({'label': label, 'max_binning_loss_db': max(c['matched_snr_loss_db'] for c in channels),
            'max_fixed_metric_loss_db': max(c['fixed_empirical_metric_snr_loss_db'] for c in channels),
            'seconds': time.monotonic()-started}), flush=True)
    output.update(complete=True, seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
