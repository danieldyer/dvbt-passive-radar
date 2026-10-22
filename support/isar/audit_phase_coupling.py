"""Validate constant channel phase offsets against independent band scattering.

Physical motion, point locations and envelope origins remain fixed. Each band
retains its own positive point strengths; only reflector phases are shared.
Odd interior one-second blocks are withheld. Shorter halves are conditional
calibration-stability controls, not independent validation of the motion seed.
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
from temporal_scene import block_design
from validate_motion import validate, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'alignment', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[16, 32, 64])
    parser.add_argument('--support', type=Path, help='Use fixed point coordinates from this model while retaining --motion parameters')
    parser.add_argument('--split-halves', action='store_true')
    parser.add_argument('--coordinate-sweeps', type=int, default=0)
    parser.add_argument('--signed-strengths', action='store_true',
        help='Diagnostic real strengths with either sign; every channel phase then has an exact pi ambiguity')
    parser.add_argument('--amplitude-block-seconds', type=float, default=0.,
        help='Allow nonnegative strengths to change at this cadence; phases remain constant')
    args = parser.parse_args()
    if not np.isfinite(args.amplitude_block_seconds) or args.amplitude_block_seconds < 0:
        parser.error('Amplitude block duration must be finite and nonnegative')
    model, alignment = [json.loads(path.read_text()) for path in (args.motion, args.alignment)]
    if not any(model.get(key, {}).get('converged') for key in ('physical_profile', 'coherent_motion')):
        raise ValueError('Complete the physical fit before this fixed-motion control')
    if args.support:
        model['points_body_m'] = json.loads(args.support.read_text())['points_body_m']
    if max(args.counts) > len(model['points_body_m']):
        raise ValueError('Requested more points than the physical support contains')
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    interval = model['interval_s']
    intervals = [interval]
    if args.split_halves:
        middle = float(np.mean(interval))
        intervals += [[interval[0], middle], [middle, interval[1]]]
    result = {'source_motion': str(args.motion), 'source_support': str(args.support or args.motion),
        'amplitude_block_seconds': args.amplitude_block_seconds,
        'strengths': 'unrestricted real' if args.signed_strengths else 'nonnegative',
        'scope': __doc__, 'complete': False, 'trials': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for span in intervals:
        bands, designs = [], []
        for c in range(5):
            with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
                field = Field(dict(f), f['pulse_samples']).windowed(span[0]+.02, span[1]-.02)
            prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
            block = np.floor(field.time-span[0]).astype(int)
            selected = (block%2 == 0)|(block == block.max())
            band = Band(field, prior, tx[3], rx, model['points_body_m'], model, selected,
                range_bias_m=model['range_bias_m'][c])
            bands.append(band)
            designs.append(band.design(model['parameters'])[0])
        for count in args.counts:
            targets = [d[:, :count] for d in designs]
            phase_indices = None
            if args.amplitude_block_seconds:
                blocks = int(np.ceil((span[1]-span[0])/args.amplitude_block_seconds))
                targets = [block_design(d, np.repeat(np.floor((b.field.time-span[0])/
                    args.amplitude_block_seconds).astype(int), b.field.y.shape[1]), blocks)
                    for b, d in zip(bands, targets)]
                phase_indices = np.tile(np.arange(count), blocks)
            training = [b.nuisance.remove(d[b.selected]) for b, d in zip(bands, targets)]
            coupling = PhaseCoupling(training, [b.clean for b in bands], reference=model['reference_channel'],
                phase_indices=phase_indices, nonnegative=not args.signed_strengths)
            coefficient, fit = coupling.fit(coordinate_sweeps=args.coordinate_sweeps)
            variants = {}
            for key, responses in [('independent', coupling.independent), ('coupled', coefficient)]:
                checks = [validate(b.field, d, b.nuisance, b.selected_time, span,
                    coefficients=a, stationary_mode='project') for b, d, a in zip(bands, targets, responses)]
                variants[key] = {'channels': checks, 'summary': summarize(checks)}
            trial = {'interval_s': span, 'count': count, 'fit': fit, 'validation': variants}
            result['trials'].append(trial)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'interval': span, 'count': count, 'converged': fit['converged'],
                'gradient': fit['maximum_gradient'], 'training': fit['objective'],
                'independent_training': fit['independent_objective'],
                'phase_degrees': np.degrees(fit['channel_phase_radians']).tolist(),
                'test_error': {key: value['summary']['prediction_error'] for key, value in variants.items()}}), flush=True)
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
