"""Residual phase audit with a static distributed scene instead of eight points.

Scene coefficients stay frozen at their training solution. This tests a common
phase screen on held-out times and complementary whitened range components.
It does not apply a correction to original samples or to images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from audit_residual_phase import audit_channel
from field import Field
from full_motion import Band
from imaging import physical_state, sites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'scene', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--knot-seconds', type=float, default=1.)
    parser.add_argument('--variants', nargs='+', choices=['shared', 'independent', 'registered'], default=['independent', 'registered'])
    parser.add_argument('--reuse-templates', action='store_true', help='Repeat only the phase solve with saved exact templates; the original QR error and row times are rechecked')
    args = parser.parse_args()
    scene, source = [json.loads(p.read_text()) for p in (args.scene, args.alignment)]
    motion = json.loads(Path(scene['source_motion']).read_text())
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=motion['reference_height_ellipsoid_m'], mmsi=563052200)
    gains = np.array([complex(*g) for g in motion['channel_gains_real_imag']])
    variant_index = dict(shared=0, independent=1, registered=2)
    result = {'source_scene':str(args.scene), 'scope':__doc__, 'original_row_phase_application_validated':False,
        'source_scene_sha256':hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        'phase_gauge':'Full-template-information-weighted mean and linear phase removed; same gauge in both range folds.',
        'limitation':'Conditional on motion and scene learned on this aperture. The phase screen is applied to retained 17.4 ms fields, not original rows.',
        'variants':{v:{'channels':[]} for v in args.variants}}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reuse_templates:
        previous = json.loads((args.output/'audit.json').read_text())
        if Path(previous['source_scene']).resolve() != args.scene.resolve():
            raise ValueError('Saved templates come from another scene')
        if previous.get('source_scene_sha256', result['source_scene_sha256']) != result['source_scene_sha256']:
            raise ValueError('Scene coefficients have changed since these templates were saved')
    started = time.perf_counter()
    for c in range(5):
        with np.load(Path(source['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(motion['interval_s'][0]+.02, motion['interval_s'][1]-.02)
        prior = physical_state(track, utc_seconds(motion['nominal_start_utc'])+field.data['row_time'], motion['attitude_prior'])
        selected = (np.arange(len(field.time))%2 == 0 if scene['holdout_seconds'] == 0 else
            np.floor((field.time-motion['interval_s'][0])/scene['holdout_seconds']).astype(int)%2 == 0)
        band = Band(field, prior, tx[3], rx, scene['points_body_m'], motion, selected, range_bias_m=motion['range_bias_m'][c])
        design = None if args.reuse_templates else band.design(motion['parameters'])[0]
        print(json.dumps({'channel':c, 'stage':'Saved templates' if args.reuse_templates else 'Exact templates ready', 'seconds':time.perf_counter()-started}), flush=True)
        for variant in args.variants:
            trial = scene['trials'][variant_index[variant]]
            coefficient = np.array(trial['channels'][c]['coefficient_real_imag'])
            coefficient = coefficient[:, 0]+1j*coefficient[:, 1]
            phase = 1. if trial['phase_registration'] is None else np.exp(1j*np.radians(trial['phase_registration']['relative_phase_degrees'][c]))
            if args.reuse_templates:
                with np.load(args.output/f'{variant}-channel-{c}.npz') as f:
                    if not np.array_equal(f['time'],field.time):
                        raise ValueError('Saved template row epochs disagree')
                    template = f['template']
            else:
                template = (gains[c]*phase*(design@coefficient)).reshape(field.y.shape)
            channel, arrays = audit_channel(band, template, knot_seconds=args.knot_seconds, profile_background=True)
            expected = trial['channels'][c]['prediction_error']
            observed = channel['trials'][0]['before']['prediction_error']
            relative = abs(observed/expected-1)
            if relative > 2e-8:
                raise ValueError(f'Regenerated template and original QR prediction disagree: {relative:g}')
            channel['prediction_error_relative_check'] = relative
            result['variants'][variant]['channels'].append(channel)
            np.savez(args.output/f'{variant}-channel-{c}.npz', **arrays, template=template)
            result['seconds'] = time.perf_counter()-started
            (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        del design, band, field


if __name__ == '__main__':
    main()
