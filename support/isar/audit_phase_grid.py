"""Known-truth control for channel-phase bias from off-grid ship reflectors.

Synthetic complex observations use the capture's physical epochs, geometry,
weights and nuisance operator, with known shared reflector phases and band
phases. No measured ship samples enter these synthetic observations. Both
correct and quantized locations are tested with the same interior holdouts.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from field import Field
from full_motion import Band
from imaging import physical_state, sites
from phase_coupling import PhaseCoupling
from stationary import Stationary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'support', 'alignment', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    args = parser.parse_args()
    model, support, alignment = [json.loads(path.read_text()) for path in (args.motion, args.support, args.alignment)]
    points = np.array(support['points_body_m'][:args.count])
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    bands, quantized, tests = [], [], []
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(model['interval_s'][0]+.02, model['interval_s'][1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        block = np.floor(field.time-model['interval_s'][0]).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        band = Band(field, prior, tx[3], rx, points, model, selected, range_bias_m=model['range_bias_m'][c])
        bands.append(band)
        quantized.append(band.design(model['parameters'])[0])
        tests.append(Stationary(band.nuisance.design, ~band.selected))
    rng = np.random.default_rng(2547)
    displacement = np.c_[rng.uniform(-1, 1, size=(len(points), 2)), np.zeros(len(points))]
    phase = rng.uniform(-np.pi, np.pi, len(points))
    band_phase = np.array([.3, -.4, .6, 0., .9])
    coefficients = rng.uniform(.3, 1.3, (5, len(points)))*np.exp(1j*(phase+band_phase[:, None]))
    report = {'scope': __doc__, 'source_motion': str(args.motion), 'source_support': str(args.support),
        'points': len(points), 'random_seed': 2547, 'true_channel_phase_radians': band_phase.tolist(),
        'complete': False, 'trials': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in (0., .1, .5, 1.):
        truth = (quantized if size == 0 else
            [b.design(model['parameters'], points=points+size*displacement)[0] for b in bands])
        observations = [a@c for a, c in zip(truth, coefficients)]
        clean = [b.nuisance.remove(y[b.selected]) for b, y in zip(bands, observations)]
        for name, design in [('quantized', quantized), ('correct_locations', truth)]:
            if size == 0 and name == 'correct_locations':
                continue
            fitting = [b.nuisance.remove(a[b.selected]) for b, a in zip(bands, design)]
            joint = PhaseCoupling(fitting, clean)
            fitted, result = joint.fit()
            error, energy = 0., 0.
            for b, a, y, c, q in zip(bands, design, observations, fitted, tests):
                prediction = q.remove(a[~b.selected]@c)
                reference = q.remove(y[~b.selected])
                error += np.sum(abs(reference-prediction)**2)
                energy += np.vdot(reference, reference).real
            phase_error = np.degrees(np.angle(np.exp(1j*(np.array(result['channel_phase_radians'])-band_phase))))
            entry = {'maximum_coordinate_offset_m': size, 'model': name,
                'phase_error_degrees': phase_error.tolist(), 'heldout_error_fraction': float(error/energy),
                'fit': result}
            report['trials'].append(entry)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k: v for k, v in entry.items() if k != 'fit'}), flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
