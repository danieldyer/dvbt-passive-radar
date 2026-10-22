"""Evaluate geometry information for completed yaw-free licensed-site fits."""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import utc_seconds
from antenna_motion import Antennas
from field import Field
from full_motion import Band
from geometry_information import antenna_summary
from geometry_profile import GeometryProfile
from imaging import motion_state, sites
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def load_joint(fitted, fields_directory, geometry_directory):
    model = json.loads(Path(fitted['source_motion']).read_text())
    curve = SplineMotion(model)
    tx, rx, _ = sites(geometry_directory)
    prior = fitted['site_local_fit']
    antennas = Antennas(prior['ecef_to_enu'], prior['channel_sites'],
        site_sigma=prior['site_prior_sigma_enu_m'], antenna_sigma=prior['antenna_prior_sigma_enu_m'],
        receiver_sigma=prior['receiver_prior_sigma_enu_m'])
    epoch, bands, offsets = utc_seconds(model['nominal_start_utc']), [], []
    for c, option in enumerate(fitted['site_assignment']):
        with np.load(fields_directory/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        state = motion_state(None, epoch+field.data['row_time'], model)
        selected = np.floor((field.time-curve.interval[0])/.5).astype(int)%2 == 0
        bands.append(Band(field, state, tx[model['reference_channel']], rx,
            model['selected_points_body_m'][c], model, selected, range_bias_m=model['range_bias_m'][c]))
        offsets.append([np.asarray(option['transmitter_ecef_m'])-field.data['transmitter_ecef_m'], np.zeros(3)])
    return GeometryProfile(bands, curve.parameters, antennas, base_offsets=offsets,
        range_origins=fitted['initial_range_bias_m'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('experiment', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--geometry-directory', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--indices', nargs='+', type=int)
    args = p.parse_args()
    experiment = json.loads(args.experiment.read_text())
    if not experiment['complete']:
        raise ValueError('Complete the licensed-site comparison before selecting the best fit')
    selected = [0, experiment['training_selected_index']] if args.indices is None else args.indices
    result = {'scope': __doc__, 'experiment': str(args.experiment),
        'experiment_sha256': fingerprint(args.experiment), 'trials': [],
        'implementation_sha256': {str(path): fingerprint(path) for path in
            [Path(__file__), Path(__file__).with_name('geometry_information.py'),
             Path(__file__).with_name('profile_information.py')]}}
    for index in dict.fromkeys(selected):
        trial = next(t for t in experiment['trials'] if t['index'] == index)
        path = Path(trial['path'])
        fitted = json.loads(path.read_text())
        joint = load_joint(fitted, Path(experiment['identity']['fields']), args.geometry_directory)
        parameters = np.array(fitted['joint_parameters'])
        records = []
        for unknown in (False, True):
            for cutoff in (1e-8, 1e-10, 1e-12):
                record = antenna_summary(joint, parameters, point_coordinates=unknown, cutoff=cutoff)
                records.append(record)
                print(json.dumps({'assignment': index, 'unknown_points': unknown, 'cutoff': cutoff,
                    'weak_directions': record['directions_weaker_than_prior']}), flush=True)
        result['trials'].append({'index': index, 'path': str(path), 'sha256': fingerprint(path),
            'site_assignment': trial['sites'], 'information': records})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
