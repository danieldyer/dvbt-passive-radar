"""Test first-half/second-half transfer of constant channel phase corrections."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from channel_alignment import Alignment, AlignmentBand
from field import Field
from imaging import physical_state, sites
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--scene', type=Path, help='A completed shared subpixel trial; freeze its points for gain-transfer testing')
    p.add_argument('--path-rates', action='store_true', help='Fit a constant range/phase-consistent path-rate residual in each nonreference channel')
    args = p.parse_args()
    source = json.loads(args.alignment.read_text())
    if args.scene:
        refined = json.loads(args.scene.read_text())
        if refined['input_alignment'] != str(args.alignment):
            raise ValueError('The scene belongs to another alignment trial')
        source['points_body_m'] = refined['fit']['points_body_m']
    model = json.loads(Path(source['input_motion']).read_text())
    curve = SplineMotion(model)
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    fields, states = [], []
    for channel in range(5):
        with np.load(Path(source['input_fields'])/f'channel-{channel}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        state = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        states.append(curve.apply(field.data['row_time'], state, tx[3], rx))
        fields.append(field)
    result = {'source_alignment': str(args.alignment), 'scene': str(args.scene) if args.scene else None,
        'scope': __doc__, 'trials': [],
        'limitation': 'Conditional calibration stability: input motion and selected spatial support use the full aperture training bins. Only gain estimation is confined to one half; reported validation uses untouched opposite-half odd bins.'}
    start = time.perf_counter()
    for half in (0, 1):
        selected, tests = [], []
        for field in fields:
            even = np.arange(len(field.time))%2 == 0
            side = (field.time >= np.mean(curve.interval)).astype(int)
            selected.append(even&(side == half))
            tests.append((~even)&(side != half))
        bands = [AlignmentBand(f, s, source['points_body_m'], mask) for f, s, mask in zip(fields, states, selected)]
        alignment = Alignment(bands, path_rate_scale_mps=.01 if args.path_rates else None,
                              phase_epoch_s=float(np.mean(curve.interval)))
        parameters, fit = alignment.fit(vary_range=args.path_rates)
        stage = {'training_half': half, 'parameters': parameters.tolist(), 'fit': fit,
            'nominal': alignment.validation(np.zeros(alignment.parameter_count), curve.interval, test_masks=tests),
            'validation': alignment.validation(parameters, curve.interval, test_masks=tests)}
        result['trials'].append(stage)
        print(json.dumps({'half': half, 'seconds': time.perf_counter()-start,
            'phase_degrees': fit['gain_phase_degrees'], 'loss': stage['validation']['conditional_coherent_loss_db']}), flush=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
