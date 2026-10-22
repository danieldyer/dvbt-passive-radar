"""Materialize metadata site priors and audit their change from a saved geometry.

Saved capture phase/range references and earlier fit results remain immutable.
NED changes use the tangent frame at each incumbent site, positive downwards.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from geometry import enu_rotation, prior
from site_hypotheses import licensed_geometry
from terrain import Terrain, geodetic
from transmitter_config import ROOT, candidate_indices, nominal_indices


def displacement(old, new):
    lat, lon, height = map(float, geodetic(old))
    east, north, up = enu_rotation(lat, lon)@(new-old)
    return {'incumbent_latitude_deg':lat, 'incumbent_longitude_deg':lon,
            'incumbent_height_ellipsoid_m':height,
            'north_m':float(north), 'east_m':float(east), 'down_m':float(-up)}


def update(metadata_path, previous, terrain_directory, output):
    if output.resolve() == previous.resolve():
        raise ValueError('Use a new directory; saved geometry is an immutable data reference')
    metadata = json.loads(metadata_path.read_text())
    centres, geometry = prior(terrain_directory, metadata)
    old = json.loads((previous/'geometry-prior.json').read_text())
    main = json.loads((previous/'sites-main.json').read_text())
    receiver, rotation = np.array(geometry['receiver_ecef_m']), np.array(geometry['ecef_to_enu'])
    if np.linalg.norm(receiver-np.array(old['receiver_ecef_m'])) > 1e-8:
        raise ValueError('This transmitter-only update must preserve the incumbent receiver')
    existing = output/'geometry-prior.json'
    if existing.exists() and json.loads(existing.read_text()) != geometry:
        raise ValueError('Output contains different physical priors; choose a new directory')
    output.mkdir(parents=True, exist_ok=True)
    (output/'geometry-prior.json').write_text(json.dumps(geometry, indent=2)+'\n')
    shutil.copyfile(previous/'acma-transmitters.json', output/'acma-transmitters.json')
    assignment = nominal_indices(geometry)
    _, offsets, channels = licensed_geometry(output, terrain_directory, assignment)
    configured = {'hypothesis':'configured-primary', 'channel_sites':assignment,
                  'channels':channels, 'site_positions_enu_m':centres.tolist(),
                  'channel_offsets_enu_m':offsets.tolist(),
                  'status':'User-supplied physical priors; active sites and phase centres not fitted'}
    (output/'sites-main.json').write_text(json.dumps(configured, indent=2)+'\n')
    old_rotation = np.array(old['ecef_to_enu'])
    baseline = np.array(old['receiver_ecef_m'])+np.array(main['site_positions_enu_m'])@old_rotation
    updated = receiver+centres@rotation
    old_terrain = Terrain(terrain_directory, old)
    changes = []
    for before, after, site in zip(baseline, updated, geometry['transmitters']):
        change = displacement(before, after)
        lat, lon = change['incumbent_latitude_deg'], change['incumbent_longitude_deg']
        ground, geoid = map(float, (old_terrain.elevation(lat, lon), old_terrain.undulation(lat, lon)))
        agl = change['incumbent_height_ellipsoid_m']-ground-geoid
        change.update({'id':site['id'], 'new':site, 'incumbent_agl_m':agl,
                       'incumbent_terrain_egm96_m':ground,
                       'agl_change_m':site['antenna_agl_m']-agl,
                       'terrain_change_m':site['terrain_egm96_m']-ground,
                       'geoid_change_m':site['geoid_undulation_m']-geoid})
        # Also distinguish the older rounded JSON design point from the actual
        # licensed base centre used by the most recent optimiser.
        index = len(changes)
        old_json = np.array(old['receiver_ecef_m'])+np.array(old['enu_transmitters_m'])[index]@old_rotation
        change['from_older_json_design_point'] = displacement(old_json, after)
        changes.append(change)
    devices = json.loads((previous/'acma-transmitters.json').read_text())['devices']
    per_channel = []
    for channel, allowed in enumerate(candidate_indices(geometry)):
        signal = metadata['signals'][channel]
        for index in allowed:
            site = geometry['transmitters'][index]
            device = next(d for d in devices if d['CALL_SIGN']==signal['callsign'] and d['SITE_ID']==site['acma_site_id'])
            lat, lon, _ = geodetic(baseline[index])
            old_height = float(device['HEIGHT'])
            old_position = baseline[index]+enu_rotation(lat, lon)[2]*(old_height-changes[index]['incumbent_agl_m'])
            per_channel.append({'channel':signal['id'], 'site':site['id'],
                'licensed_old_agl_m':old_height, **displacement(old_position, updated[index])})
    paths = [metadata_path, previous/'geometry-prior.json', previous/'sites-main.json',
             previous/'acma-transmitters.json', ROOT/metadata['transmitter_terrain']['raster'],
             terrain_directory/'us_nga_egm96_15.tif', terrain_directory/'S38E145.npy']
    report = {'scope':__doc__, 'baseline_directory':str(previous),
        'new_geometry_directory':str(output), 'site_changes':changes,
        'channel_changes_from_licensed_heights':per_channel,
        'terrain':metadata['transmitter_terrain'],
        'receiver_change_m':float(np.linalg.norm(receiver-old['receiver_ecef_m'])),
        'candidate_assignments':list(itertools.product(*candidate_indices(geometry))),
        'candidate_count':int(np.prod([len(v) for v in candidate_indices(geometry)])),
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (output/'transmitter-update.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=Path('recordings/live_mt_dandenong.json'))
    parser.add_argument('--previous', type=Path, default=Path('docs/data/calibration_20261016_receiver'))
    parser.add_argument('--terrain', type=Path, default=Path('build/calibration-20261013'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/calibration_20261016_sites'))
    args = parser.parse_args()
    report = update(args.metadata, args.previous, args.terrain, args.output)
    for site in report['site_changes']:
        print(site['id'], {key:round(site[key], 4) for key in ('north_m','east_m','down_m','agl_change_m','terrain_change_m')})
    print('Candidate assignments:', report['candidate_count'])


if __name__ == '__main__':
    main()
