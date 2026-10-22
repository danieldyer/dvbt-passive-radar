"""Compare yaw-free motion candidates under identical heldout projections.

Each candidate is scored with stationary nuisance alone, and then with the
same fixed delayed-return nuisance. Main-target complex strengths and Gaussian
variance are fitted using training rows; heldout target strengths are frozen.
The two projection families are reported separately, never compared as if
they retained the same data or degrees of freedom.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from companion_motion import add_companion
from field import Field, FS
from fit_marginal_motion import make_bands
from marginal_motion import MarginalMotion
from motion import C
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'source', 'control', 'fit', 'acquisition', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier validation results')
    paths = {'source': args.source, 'without_companion': args.control, 'with_companion': args.fit}
    models = {k: json.loads(v.read_text()) for k, v in paths.items()}
    source = models['source']
    for model in models.values():
        if not model['fit']['converged'] or not model.get('fixed_ne_frame') or np.any(SplineMotion(model).yaw_coefficients):
            raise ValueError('Only completed stationary yaw-free fits enter this comparison')
        for key in ('interval_s', 'selected_points_body_m', 'geometry_directory', 'range_bias_m',
                'nominal_start_utc', 'path_spacing_seconds', 'heading_spacing_seconds'):
            if model[key] != source[key]:
                raise ValueError(f'The models differ in {key}')
    acquired = json.loads(args.acquisition.read_text())
    if not acquired['complete'] or acquired['source_sha256'] != fingerprint(args.source):
        raise ValueError('Companion paths belong to a different source motion')
    fields = []
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as d:
            fields.append(Field(dict(d), d['pulse_samples']))
    bands = make_bands(fields, source)
    result = {'scope': __doc__, 'models': {k: {'path': str(v), 'sha256': fingerprint(v),
        'converged': models[k]['fit']['converged'], 'maximum_kkt_residual': models[k]['fit']['maximum_kkt_residual']}
        for k, v in paths.items()}, 'acquisition': str(args.acquisition), 'acquisition_sha256': fingerprint(args.acquisition),
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('companion_validation.py', 'companion_motion.py', 'delayed_path.py', 'gaussian_scene.py',
             'marginal_motion.py', 'validate_motion.py', 'field.py')}, 'families': {}}
    for family in ('stationary', 'stationary_and_companion'):
        if family == 'stationary_and_companion':
            result['information_cost'] = [add_companion(band, source['parameters'],
                [t for t in acquired['trials'] if t['channel'] == c]) for c, band in enumerate(bands)]
        values = {}
        for name, model in models.items():
            joint = MarginalMotion(bands, model['parameters'])
            validation = joint.validation(joint.initial)
            values[name] = {'summary': summarize(validation), 'channels': validation}
        control, fitted = [values[k]['summary'] for k in ('without_companion', 'with_companion')]
        result['families'][family] = {'candidates': values,
            'companion_vs_control_error_change_percent': 100*(fitted['prediction_error']/control['prediction_error']-1),
            'companion_vs_control_matched_change_db': 10*np.log10(fitted['matched_energy']/control['matched_energy'])}
    # A range-only leakage diagnostic; no time projection or target fit here.
    delay = np.linspace(350., 750., 401)
    leakage = []
    for field in fields:
        template = field.pulse(field.offsets[None, :]-delay[:, None]*FS/C)
        central = field.pulse(field.offsets)
        raw = np.sum(abs(template)**2, axis=1)/np.sum(abs(central)**2)
        white = np.sum(abs(template@field.whiten.T)**2, axis=1)/np.sum(abs(field.whiten@central)**2)
        leakage.append({'raw_energy_ratio_db': (10*np.log10(raw)).tolist(),
            'whitened_energy_ratio_db': (10*np.log10(white)).tolist()})
    result['range_crop_leakage'] = {'extra_path_m': delay.tolist(), 'channels': leakage,
        'scope': 'Equal-amplitude delayed versus centred point, 17-sample crop, with and without declared thermal range whitening. No slow-time projection; not measured ship SNR or a physical identification of the companion.'}
    time = np.linspace(*source['interval_s'], 901)
    result['motion_curves'] = {'time_s': time.tolist(), 'path_m': {name: SplineMotion(m).components(time)[:, 0].tolist()
        for name, m in models.items()}, 'path_rate_mps': {name: SplineMotion(m).components(time, 1)[:, 0].tolist()
        for name, m in models.items()}, 'yaw_degrees': {name: np.degrees(SplineMotion(m).components(time)[:, 1]).tolist()
        for name, m in models.items()}}
    result['complete'] = True
    result['calibration_validated'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({family: {k: v for k, v in value.items() if k != 'candidates'}
        for family, value in result['families'].items()}), flush=True)


if __name__ == '__main__':
    main()
