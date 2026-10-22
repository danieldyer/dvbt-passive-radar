"""Compare antenna information with known, shared and separate ship points.

The mean and independent complex band amplitudes come from one completed
shared-support fit. Unknown three-dimensional points, motion and common band
range origins are eliminated without counting search bounds as information.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from antenna_motion import Antennas
from audit_shared_geometry_information import summary
from common_reflector_information import compact_roots, nested_information
from field import Field
from fit_shared_ne_points import bands_for
from geometry_profile import GeometryProfile
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'fields', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Keep completed information audits')
    experiment_path = args.experiment/'experiment.json'
    experiment = json.loads(experiment_path.read_text())
    if not experiment['complete'] or experiment['selected_source_band'] is None:
        raise ValueError('Finish the shared-support fits first')
    for source, expected in experiment['source_sha256'].items():
        if fingerprint(Path(source)) != expected:
            raise ValueError('Fitted input changed: '+source)
    trial = next(t for t in experiment['trials'] if t['source_band'] == experiment['selected_source_band'])
    if not trial['fit']['converged'] or trial['fit']['maximum_kkt_residual'] > 1e-7:
        raise ValueError('Require a stationary training-selected support')
    model = json.loads(Path(experiment['source_motion']).read_text())
    model['selected_points_body_m'] = [copy.deepcopy(trial['points_body_m']) for _ in range(5)]
    prior = model['site_local_fit']
    antennas = Antennas(prior['ecef_to_enu'], prior['channel_sites'],
        site_sigma=prior['site_prior_sigma_enu_m'], antenna_sigma=prior['antenna_prior_sigma_enu_m'],
        receiver_sigma=prior['receiver_prior_sigma_enu_m'])
    fields, paths = [], [args.fields/f'channel-{c}.npz' for c in range(5)]
    for path in paths:
        if fingerprint(path) != experiment['source_sha256'].get(str(path)):
            raise ValueError('Information and fit use different measured fields')
        with np.load(path) as d:
            fields.append(Field(dict(d), d['pulse_samples']))
    joint = GeometryProfile(bands_for(model, fields), model['parameters'], antennas)
    value = joint.evaluate(joint.initial)
    np.testing.assert_allclose(value['value'], trial['fit']['objective'], rtol=0, atol=1e-10)
    roots = compact_roots(joint, joint.initial)
    curve = SplineMotion(model)
    yaw = ((joint.active >= curve.sections[1].start)&(joint.active < curve.sections[1].stop))
    interest = np.arange(len(joint.initial))[joint.antenna_slice]
    nuisance = np.r_[np.setdiff1d(np.arange(joint.nmotion), np.flatnonzero(yaw)),
                     np.arange(len(joint.initial))[joint.range_slice]]
    sources = [experiment_path, Path(__file__), *[Path(__file__).with_name(n) for n in
        ('common_reflector_information.py', 'geometry_information.py', 'geometry_profile.py',
         'physical_profile.py', 'field.py', 'fit_shared_ne_points.py')]]
    result = {'complete': False, 'scope': __doc__, 'source_sha256': {
        str(path): fingerprint(path) for path in sources},
        'selected_source_band': trial['source_band'], 'fit_objective': value['value'],
        'root_shapes': [list(root.shape) for root in roots], 'cases': [],
        'accepted_calibration': False,
        'limitation': 'Conditional local information at a fitted scene and frozen fitted motion. The support fit holds point heights fixed, but this audit treats all three point coordinates as unknown. Joint scene/motion/antenna stationarity and real-data covariance validity are not established by this calculation.'}
    for cutoff in (1e-8, 1e-10, 1e-12):
        case = nested_information(roots, len(joint.initial), interest, nuisance, cutoff=cutoff)
        for first, second in (('known', 'shared'), ('shared', 'independent')):
            scale = max(np.linalg.norm(case[first]), np.finfo(float).tiny)
            if np.linalg.eigvalsh(case[first]-case[second]).min() < -scale*1e-12:
                raise ValueError('Additional nuisance spuriously added information')
        for name in ('known', 'shared', 'independent'):
            info = case[name]
            case[name] = {**summary(info, antennas.factor), 'information_before_prior': info.tolist()}
        result['cases'].append(case)
        print(json.dumps({'cutoff': cutoff, 'weak_directions': {
            name: case[name]['directions_weaker_than_declared_prior']
            for name in ('known', 'shared', 'independent')}, 'seconds': time.monotonic()-started}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
