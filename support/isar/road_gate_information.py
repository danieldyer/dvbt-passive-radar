"""Conditional information discarded by the offline vehicle Doppler gate.

Compare identical selected time bins and range samples before/after the fixed
Fourier projection, using original-row integrated physical responses and the
same declared empirical covariance. Unknown complex target gains are removed.
This does not certify the covariance, clutter model or prior time averaging.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from finite_scene_information import retained_information
from covariance_gate import whiten_first_field
from fit_shared_target_geometry import load_experiment
from integrated_moving_road import IntegratedMovingField
from integrated_stationary_road import RecordedBins
from scene_dictionary import fingerprint
from shared_origin_geometry import OriginRoad
from shared_target_geometry import RoadTarget


class SelectedWhitening:
    """Diagonal time whitening without a dense diagonal matrix."""
    def __init__(self, selected, variance):
        self.selected = np.asarray(selected, bool)
        self.sigma = np.sqrt(np.asarray(variance)[self.selected])
        if not len(self.sigma) or np.any(~np.isfinite(self.sigma)) or np.any(self.sigma <= 0):
            raise ValueError('Require selected bins with positive finite variances')

    def __len__(self):
        return len(self.sigma)

    def __matmul__(self, value):
        value = np.asarray(value)
        if len(value) != len(self.selected):
            raise ValueError('Use the original time axis before selection')
        scale = self.sigma.reshape((-1,)+(1,)*(value.ndim-1))
        return value[self.selected]/scale


def full_field(original):
    result = copy.copy(original)
    result.q = SelectedWhitening(original.selected,
        original.data['variance'][:, len(original.offsets)//2])
    result.y = result.apply(result.data['value']).ravel()
    return result


def efficient(signal, derivative):
    value = derivative-signal[:, None]*(signal.conj()@derivative)[None, :]/np.vdot(signal, signal).real
    return np.sqrt(2)*np.r_[value.real, value.imag]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--labels', nargs='+', default=['A', 'O', 'K', 'Q'])
    p.add_argument('--whiten-first', action='store_true',
        help='Form the same Fourier signal subspace after covariance whitening.')
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve completed audits')
    source, saved = [json.loads(path.read_text()) for path in (args.source, args.road_fit)]
    if not source['complete'] or not source['fit']['converged'] or source['original_row_maximum_kkt_residual'] > 1e-7:
        raise ValueError('Require an original-row stationary common-origin fit')
    identity = source['identity']; inherited = identity['legacy_identity']
    for path in (args.ship_fit, args.road_fit):
        if inherited['sources'][str(path)] != fingerprint(path):
            raise ValueError('Mismatched geometry source')
    legacy, _, _, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], inherited['nodes'])
    if dropped or any(label not in labels for label in args.labels):
        raise ValueError('Use the same disjoint road hypotheses')
    parameters = np.asarray(source['parameters']); antenna = parameters[legacy.antenna_slice]
    origins = np.asarray(source['range_origins_m']); patches = Path(saved['input_patches'])
    paths = [args.source, args.ship_fit, args.road_fit, Path(__file__),
        *[Path(__file__).with_name(name) for name in ('integrated_moving_road.py',
          'integrated_stationary_road.py', 'shared_origin_geometry.py', 'finite_scene_information.py',
          'covariance_gate.py')],
        *[patches/f'{label}-{c}.npz' for label in args.labels for c in range(5)]]
    result = {'complete': False, 'scope': __doc__, 'targets': [],
        'source_sha256': {str(path): fingerprint(path) for path in paths},
        'accepted_calibration': False, 'production_changed': False,
        'projection': 'whiten-first' if args.whiten_first else 'legacy Fourier before whitening'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        index = labels.index(label); road = legacy.roads[index]
        motion = parameters[legacy.road_slices[index]]
        channels = []
        for c, base in enumerate(road.training):
            with np.load(patches/f'{label}-{c}.npz') as data:
                bins = RecordedBins(base, data['row_time'], data['row_information'])
            narrow = IntegratedMovingField(whiten_first_field(base) if args.whiten_first else base, bins)
            full = IntegratedMovingField(full_field(base), bins)
            model = OriginRoad(road)
            a, da = model.template(motion, antenna, origins, c, full)
            b, db = model.template(motion, antenna, origins, c, narrow)
            full_root, reduced = efficient(a, da), efficient(b, db)
            norms = np.linalg.norm(full_root, axis=0)
            scale = np.where(norms > 0, norms, 1.)
            checks = [retained_information(full_root/scale, reduced/scale, cutoff)
                      for cutoff in (1e-8, 1e-10, 1e-12)]
            snr = float(np.vdot(b, b).real/np.vdot(a, a).real)
            if not 0 < snr <= 1+1e-10:
                raise ValueError('Projection increased conditional matched-filter information')
            channels.append({'channel': c, 'full_observations': len(a), 'gated_observations': len(b),
                'matched_snr_retention': snr, 'matched_snr_loss_db': float(-10*np.log10(snr)),
                'information': checks})
        record = {'label': label, 'channels': channels}
        result['targets'].append(record)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'label': label,
            'max_snr_loss_db': max(c['matched_snr_loss_db'] for c in channels),
            'max_local_parameter_information_loss_db': max(c['information'][1]['worst_information_loss_db'] for c in channels),
            'seconds': time.monotonic()-started}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
