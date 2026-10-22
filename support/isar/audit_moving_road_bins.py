"""Measure mean-epoch approximation against original-row vehicle responses.

Motion, sites, origins, measured observations and fixed covariance stay fixed.
No road reassignment or calibration correction is fitted in this audit.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from integrated_moving_road import IntegratedMovingField
from integrated_stationary_road import RecordedBins
from scene_dictionary import fingerprint
from shared_origin_geometry import OriginRoad
from shared_target_geometry import RoadTarget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--labels', nargs='+', default=['A', 'O', 'K', 'Q'])
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve existing audits')
    source, saved = [json.loads(path.read_text()) for path in (args.source, args.road_fit)]
    if not source['complete'] or not source['fit']['converged'] or source['original_row_maximum_kkt_residual'] > 1e-7:
        raise ValueError('Use a completed stationary common-origin fit')
    identity = source['identity']; inherited = identity['legacy_identity']
    for path in (args.ship_fit, args.road_fit):
        if inherited['sources'][str(path)] != fingerprint(path):
            raise ValueError('Source calibration differs: '+str(path))
    legacy, _, _, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], inherited['nodes'])
    if dropped or any(label not in labels for label in args.labels):
        raise ValueError('Use existing disjoint target hypotheses')
    patches = Path(saved['input_patches'])
    paths = [args.source, args.ship_fit, args.road_fit,
        *[patches/f'{label}-{c}.npz' for label in args.labels for c in range(5)],
        *[Path(__file__).with_name(name) for name in ('audit_moving_road_bins.py',
          'integrated_moving_road.py', 'integrated_stationary_road.py', 'shared_origin_geometry.py',
          'shared_target_geometry.py')]]
    parameters = np.asarray(source['parameters']); antenna = parameters[legacy.antenna_slice]
    origins = np.asarray(source['range_origins_m'])
    result = {'complete': False, 'scope': __doc__, 'targets': [],
        'source_sha256': {str(path): fingerprint(path) for path in paths},
        'limitation': 'Interval-averaged pulse and empirical projected covariance remain conditioned upon. This checks only the temporal forward response for the fixed recorded-data fits, not a full CRLB or site calibration.',
        'accepted_calibration': False, 'production_changed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        index = labels.index(label); road = legacy.roads[index]
        motion = parameters[legacy.road_slices[index]]
        train, test, checks = [], [], []
        for c, (a, b) in enumerate(zip(road.training, road.validation)):
            with np.load(patches/f'{label}-{c}.npz') as raw:
                bins = RecordedBins(a, raw['row_time'], raw['row_information'])
                reconstruction = bins.mean(raw['row_value'][bins.keep])
                error = np.linalg.norm(reconstruction-a.data['value'])/np.linalg.norm(reconstruction)
                if error > 1e-10 or not np.array_equal(a.time, b.time):
                    raise ValueError('Original row average does not reproduce fitted bins')
            train.append(IntegratedMovingField(a, bins)); test.append(IntegratedMovingField(b, bins))
            checks.append({'channel': c, 'rows': len(bins.time), 'bins': bins.count,
                           'recorded_bin_reconstruction_relative_error': float(error)})
        original = OriginRoad(road)
        integrated = OriginRoad(RoadTarget(road.problem, train, test, road.initial,
                                          road.transmitters, road.antennas))
        cases, predictions = [], [[], []]
        for c in range(5):
            model = []
            for variant in (original, integrated):
                a = variant.template(motion, antenna, origins, c, variant.training[c])[0]
                b = variant.template(motion, antenna, origins, c, variant.validation[c])[0]
                coefficient = np.vdot(a, road.training[c].y)/np.vdot(a, a)
                predicted = coefficient*b
                norm = float(np.vdot(predicted, predicted).real)
                error = road.validation[c].y-predicted
                model.append((a, b))
                predictions[len(model)-1].append({
                    'prediction_error': float(np.vdot(error, error).real),
                    'matched_energy': float(abs(np.vdot(predicted, road.validation[c].y))**2/norm) if norm else 0.})
            for split in range(2):
                a, b = model[0][split], model[1][split]
                energy_a, energy_b = np.vdot(a, a).real, np.vdot(b, b).real
                overlap = np.vdot(a, b)
                fraction = abs(overlap)**2/(energy_a*energy_b)
                cases.append({'channel': c, 'split': 'train' if split == 0 else 'test',
                    'template_relative_difference': float(np.linalg.norm(a-b)/np.sqrt(energy_b)),
                    'conditional_matched_filter_loss_db': float(-10*np.log10(min(1., fraction))),
                    'complex_gain_ratio_real_imag': [float((overlap/energy_a).real), float((overlap/energy_a).imag)]})
        totals = [{key: sum(ch[key] for ch in prediction) for key in ('prediction_error', 'matched_energy')}
                  for prediction in predictions]
        record = {'label': label, 'operator_checks': checks, 'template_checks': cases,
            'midpoint_prediction': predictions[0], 'integrated_prediction': predictions[1],
            'prediction_error_change_percent': 100*(totals[1]['prediction_error']/totals[0]['prediction_error']-1),
            'matched_energy_change_db': float(10*np.log10(totals[1]['matched_energy']/totals[0]['matched_energy'])),
            'worst_template_loss_db': max(case['conditional_matched_filter_loss_db'] for case in cases)}
        result['targets'].append(record)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({key: record[key] for key in ('label', 'prediction_error_change_percent',
                                                     'matched_energy_change_db', 'worst_template_loss_db')}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
