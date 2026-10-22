"""Exact ship phase signatures of licensed alternative transmitter sites.

These are predictions under fixed ship motion, not fits to received samples.
Constant phase per reflector is eliminated explicitly. The remaining common
rate and reflector-dependent phase test what a local rate correction can model.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from ais import read_csv, utc_seconds
from imaging import physical_state, sites
from motion import C, excess_path
from spline_motion import SplineMotion
sys.path.append(str(Path(__file__).resolve().parents[1]/'calibration'))
from transmitter_config import candidate_acma_ids, configuration


def licensed_options(directory):
    geometry = json.loads((directory/'geometry-prior.json').read_text())
    licensed = json.loads((directory/'acma-transmitters.json').read_text())
    _, rx, main = sites(directory)
    centres, rotation = np.array(main['site_positions_enu_m']), np.array(geometry['ecef_to_enu'])
    site_ids, base_heights = ['12013', '12015', '12014'], [160., 110., 119.]
    calls = ['HSV6', 'SBS7', 'GTV8', 'ATV11', 'ABC12']
    allowed = candidate_acma_ids(geometry)
    config = configuration(geometry)
    options = []
    for device in licensed['devices']:
        if device['CALL_SIGN'] not in calls:
            continue
        channel = calls.index(device['CALL_SIGN'])
        if device['SITE_ID'] not in allowed[channel]:
            continue
        index = site_ids.index(device['SITE_ID'])
        site = licensed['sites'][device['SITE_ID']]
        lat, lon = np.radians([float(site['LATITUDE']), float(site['LONGITUDE'])])
        up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        alternate = rx+centres[index]@rotation
        if 'site_configuration' not in geometry:
            alternate += (float(device['HEIGHT'])-base_heights[index])*up
        signal = config['signals'][channel]
        slug = config['transmitters'][index]['id']
        options.append({'channel': channel, 'call': device['CALL_SIGN'],
            'site_id': device['SITE_ID'], 'station_type': device['STATION_TYPE'],
            'configured_role':next(role for role,value in signal['transmitter_site_roles'].items() if value==slug),
            'transmitter_ecef_m': alternate.tolist()})
    expected = {(channel, site) for channel, choices in enumerate(allowed) for site in choices}
    found = [(option['channel'], option['site_id']) for option in options]
    if set(found) != expected or len(found) != len(expected):
        raise ValueError('Configured candidates must each match exactly one licensed device')
    options.sort(key=lambda option:(option['channel'], allowed[option['channel']].index(option['site_id'])))
    return options


def assigned_rays(directory,assignment='00002'):
    """Read one explicit licensed assignment: 0 Ornata, 1 Eyre, 2 Ridge."""
    if len(assignment)!=5 or any(c not in '012' for c in assignment):
        raise ValueError('Need five explicit site digits in Seven/SBS/Nine/Ten/ABC order')
    prior=json.loads((directory/'geometry-prior.json').read_text());options=licensed_options(directory)
    ids={'0':'12013','1':'12015','2':'12014'};tx=[]
    for channel,digit in enumerate(assignment):
        match=[o for o in options if o['channel']==channel and o['site_id']==ids[digit]]
        if len(match)!=1:raise ValueError('Assignment includes an unlicensed primary/backup choice')
        tx.append(match[0]['transmitter_ecef_m'])
    return np.asarray(tx),np.asarray(prior['receiver_ecef_m'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'support', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    model, support = [json.loads(p.read_text()) for p in (args.motion, args.support)]
    directory = args.geometry_directory
    tx, rx, main = sites(directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    curve = SplineMotion(model)
    time = np.linspace(*curve.interval, 601)
    prior = physical_state(track, utc_seconds(model['nominal_start_utc'])+time, model['attitude_prior'])
    state = curve.apply(time, prior, tx[3], rx)
    points = np.r_[np.zeros((1, 3)), np.array(support['points_body_m'])]
    position = state['position_ecef_m'][:, None, :]+np.einsum('nij,pj->npi', state['body_to_ecef'], points)
    calls, rf = ['HSV6', 'SBS7', 'GTV8', 'ATV11', 'ABC12'], [177.5e6, 184.5e6, 191.625e6, 219.5e6, 226.5e6]
    line = np.c_[np.ones(len(time)), time-time.mean()]
    report = {'scope': __doc__, 'source_motion': str(args.motion), 'source_support': str(args.support),
        'interval_s': curve.interval.tolist(), 'site_options': []}
    for c, call in enumerate(calls):
        nominal = excess_path(position, tx[c], rx)[0]
        for option in licensed_options(directory):
            if option['channel'] != c:
                continue
            alternate = np.array(option['transmitter_ecef_m'])
            delta = excess_path(position, alternate, rx)[0]-nominal
            coefficients = np.linalg.lstsq(line, delta[:, 0], rcond=None)[0]
            centred = delta-delta.mean(axis=0)
            residual = centred-(time-time.mean())[:, None]*coefficients[1]
            spatial = centred-centred[:, :1]
            mean_spatial = delta.mean(axis=0)-delta[:, 0].mean()
            k = 2*np.pi*rf[c]/C
            result = {'channel': c, 'call': call, 'site_id': option['site_id'], 'station_type': option['station_type'],
                'tx_displacement_m': float(np.linalg.norm(alternate-tx[c])),
                'centre_path_rate_difference_mps': float(coefficients[1]),
                'centre_phase_rate_degrees_ps': float(np.degrees(-k*coefficients[1])),
                'maximum_residual_phase_after_common_rate_degrees': float(np.degrees(k*abs(residual).max())),
                'maximum_point_dependent_time_phase_degrees': float(np.degrees(k*abs(spatial).max())),
                'static_point_relative_phase_span_degrees': np.degrees(-k*np.array([mean_spatial.min(), mean_spatial.max()])).tolist(),
                'time_s': time.tolist(), 'centre_phase_after_constant_removal_degrees': np.degrees(-k*centred[:, 0]).tolist(),
                'centre_path_change_m': [float(delta[:, 0].min()), float(delta[:, 0].max())]}
            report['site_options'].append(result)
            print(json.dumps({key: value for key, value in result.items() if key not in
                ('time_s', 'centre_phase_after_constant_removal_degrees')}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
