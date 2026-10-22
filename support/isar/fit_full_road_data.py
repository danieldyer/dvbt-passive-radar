"""Compare Doppler-gated and full-bin vehicle likelihoods with static nuisance.

Use the same unknown stationary delay basis in both controls. Its response is
averaged at original epochs; the moving mean-epoch model has a separate small
error audit. Sites and common range origins stay fixed. Full waveform and
stationary-profiled moving predictions are reported on identical full-bin
held-out observations.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from field import C, FS
from fit_shared_target_geometry import load_experiment
from integrated_moving_road import IntegratedMovingField
from integrated_stationary_road import RecordedBins
from profiled_road_field import ProfiledRoadField
from road_gate_information import full_field
from scene_dictionary import fingerprint
from shared_origin_geometry import OriginRoad
from shared_origin_transfer import FrozenOriginRoad
from shared_target_geometry import RoadTarget
from validate_shared_geometry import fit_target
from wide_road_templates import widen
from geographic_io import read_profiles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--labels', nargs='+', default=['A', 'O', 'K', 'Q'])
    p.add_argument('--iterations', type=int, default=180)
    p.add_argument('--wide-psf', type=Path,
        help='Measured extended pulse export covering every requested target interval.')
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve existing experiments')
    source, saved = [json.loads(path.read_text()) for path in (args.source, args.road_fit)]
    if not source['complete'] or not source['fit']['converged'] or source['original_row_maximum_kkt_residual'] > 1e-7:
        raise ValueError('Require a stationary common-origin source')
    identity = source['identity']; inherited = identity['legacy_identity']
    for path in (args.ship_fit, args.road_fit):
        if fingerprint(path) != inherited['sources'][str(path)]:
            raise ValueError('Mismatched fitted source')
    joint, _, _, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], inherited['nodes'])
    if dropped or any(label not in labels for label in args.labels):
        raise ValueError('Require disjoint known target hypotheses')
    parameters = np.asarray(source['parameters']); antenna = parameters[joint.antenna_slice]
    origins = np.asarray(source['range_origins_m']); patches = Path(saved['input_patches'])
    wide = read_profiles(args.wide_psf)[1] if args.wide_psf else None
    paths = [args.source, args.ship_fit, args.road_fit, Path(__file__),
        *[Path(__file__).with_name(name) for name in ('profiled_road_field.py', 'road_gate_information.py',
          'shared_origin_geometry.py', 'integrated_moving_road.py', 'integrated_stationary_road.py',
          'validate_shared_geometry.py', 'fit_physical_profile.py')],
        *[patches/f'{label}-{c}.npz' for label in args.labels for c in range(5)],
        *([args.wide_psf, Path(__file__).with_name('wide_road_templates.py')] if args.wide_psf else [])]
    result = {'complete': False, 'scope': __doc__, 'targets': [],
        'source_sha256': {str(path): fingerprint(path) for path in paths},
        'accepted_calibration': False, 'production_changed': False,
        'limitation': 'Conditional empirical covariance and road associations; unmodelled moving interference is not declared Gaussian by this comparison. Common origins and antenna coordinates are frozen.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        index = labels.index(label); road = joint.roads[index]
        wide_check = None
        if wide is not None:
            road, wide_check = widen(road, wide)
        fields = {'legacy': [[], []], 'full': [[], []]}; clocks = []
        for c, train in enumerate(road.training):
            with np.load(patches/f'{label}-{c}.npz') as raw:
                bins = RecordedBins(train, raw['row_time'], raw['row_information'])
            seed = np.polynomial.polynomial.polyval((bins.time-train.seed['epoch'])/train.seed.get('scale', 15.), train.seed['path'])
            delays = np.arange(np.floor(seed.min()*FS/C)+train.offsets.min()-5,
                np.ceil(seed.max()*FS/C)+train.offsets.max()+6)*C/FS
            lag = [train.offsets.min()-(delays.max()-seed.min())*FS/C,
                   train.offsets.max()-(delays.min()-seed.max())*FS/C]
            if min(lag[0], -lag[1]) < -getattr(train.pulse, 'support', 32):
                raise ValueError('Static dictionary leaves verified pulse support')
            binned = bins.stationary(train, delays)
            clocks.append(bins)
            for split, base in enumerate((train, road.validation[c])):
                for name in fields:
                    raw_field = base if name == 'legacy' else full_field(base)
                    fields[name][split].append(ProfiledRoadField(raw_field, delays, binned))
        candidates = {name: RoadTarget(road.problem, train, test, road.initial,
            road.transmitters, road.antennas) for name, (train, test) in fields.items()}
        models = {name: OriginRoad(value) for name, value in candidates.items()}
        starts = [parameters[joint.road_slices[index]], road.initial]
        records = []
        for name, candidate in candidates.items():
            fit = fit_target(FrozenOriginRoad(candidate, origins), antenna, starts, args.iterations)
            selected = fit['starts'][fit['selected_start']]; motion = np.asarray(selected['parameters'])
            prediction = []
            for c, training in enumerate(candidate.training):
                u = models[name].template(motion, antenna, origins, c, training)[0]
                amplitude = np.vdot(u, training.y)/np.vdot(u, u).real
                raw_u = models[name].template(motion, antenna, origins, c, training.original)[0]
                beta = training.coefficients(training.original.y-amplitude*raw_u)
                test = candidates['full'].validation[c]
                raw_v = models['full'].template(motion, antenna, origins, c, test.original)[0]
                moving = test.remove(amplitude*raw_v)
                error = test.original.y-amplitude*raw_v-test.stationary@beta
                projected = test.y-moving
                exact = copy.copy(test)
                exact.original = IntegratedMovingField(test.original, clocks[c])
                exact.time = clocks[c].time
                exact_v = models['full'].template(motion, antenna, origins, c, exact)[0]
                mean_v = test.remove(raw_v)
                fraction = abs(np.vdot(mean_v, exact_v))**2/(np.vdot(mean_v, mean_v).real*np.vdot(exact_v, exact_v).real)
                norm = np.vdot(moving, moving).real
                prediction.append({'full_waveform_error': float(np.vdot(error, error).real),
                    'moving_error_after_identical_test_static_projection': float(np.vdot(projected, projected).real),
                    'moving_matched_energy': float(abs(np.vdot(moving, test.y))**2/norm) if norm else 0.,
                    'original_row_moving_template_loss_db': float(-10*np.log10(min(1., fraction))),
                    'training_static_rank': training.basis.shape[1], 'test_static_rank': test.basis.shape[1]})
            record = {'operator': name, 'fit': fit, 'prediction': prediction,
                      'selected_fit_stationary': selected['fit']['converged']}
            records.append(record)
            print(json.dumps({'label': label, 'operator': name,
                'stationary': selected['fit']['converged'], 'kkt': selected['fit']['maximum_kkt_residual'],
                'seconds': time.monotonic()-started}), flush=True)
        comparisons = {}
        for key in ('full_waveform_error', 'moving_error_after_identical_test_static_projection', 'moving_matched_energy'):
            values = [sum(c[key] for c in r['prediction']) for r in records]
            comparisons[key+'_change_percent'] = 100*(values[1]/values[0]-1)
        result['targets'].append({'label': label, 'trials': records, 'comparison': comparisons,
                                  'wide_template_check': wide_check})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    result.update(complete=True, seconds=time.monotonic()-started,
        all_selected_fits_stationary=all(r['selected_fit_stationary'] for t in result['targets'] for r in t['trials']))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
