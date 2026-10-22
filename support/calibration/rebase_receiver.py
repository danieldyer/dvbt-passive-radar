"""Change a physical receiver prior without moving transmitters or saved data.

The destination is a new geometry directory. Original geometry, fitted target
states and archived range/phase references are never rewritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from geometry import receiver_prior


def rebase(prior, main, receiver, rotation, description):
    old_receiver = np.asarray(prior['receiver_ecef_m'])
    old_rotation = np.asarray(prior['ecef_to_enu'])
    def positions(values):
        absolute = old_receiver+np.asarray(values)@old_rotation
        return ((absolute-receiver)@rotation.T).tolist()
    updated = copy.deepcopy(prior)
    updated['receiver'] = description
    updated['receiver_ecef_m'] = receiver.tolist()
    updated['ecef_to_enu'] = rotation.tolist()
    updated['enu_transmitters_m'] = positions(prior['enu_transmitters_m'])
    # Prior target trajectories were fitted in the previous ENU frame. They are
    # not part of this antenna configuration and must not appear to be refitted.
    centres = {key: copy.deepcopy(main[key]) for key in
               ('hypothesis', 'channel_sites', 'channels')}
    centres['site_positions_enu_m'] = positions(main['site_positions_enu_m'])
    centres['channel_offsets_enu_m'] = (np.asarray(main['channel_offsets_enu_m'])@
                                      old_rotation@rotation.T).tolist()
    return updated, centres


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('geometry', 'metadata', 'terrain', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.geometry.resolve():
        raise ValueError('Use a new output directory; saved phase references are immutable')
    prior_path, main_path = [args.geometry/name for name in ('geometry-prior.json', 'sites-main.json')]
    prior, sites = [json.loads(p.read_text()) for p in (prior_path, main_path)]
    receiver, rotation, description = receiver_prior(args.terrain, json.loads(args.metadata.read_text()))
    updated, centres = rebase(prior, sites, receiver, rotation, description)
    old_rotation, old_receiver = np.array(prior['ecef_to_enu']), np.array(prior['receiver_ecef_m'])
    original = old_receiver+(np.array(sites['site_positions_enu_m'])[sites['channel_sites']]+
                             np.array(sites['channel_offsets_enu_m']))@old_rotation
    changed = receiver+(np.array(centres['site_positions_enu_m'])[centres['channel_sites']]+
                        np.array(centres['channel_offsets_enu_m']))@rotation
    error = float(np.max(np.abs(changed-original)))
    if error > 3e-9:
        raise ValueError('Rebasing changed an absolute transmitter position')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (('geometry-prior.json', updated), ('sites-main.json', centres)):
        (args.output/name).write_text(json.dumps(value, indent=2)+'\n')
    licensed = args.geometry/'acma-transmitters.json'
    shutil.copyfile(licensed, args.output/licensed.name)
    provenance = {'scope': __doc__, 'receiver': description,
        'receiver_change_in_old_enu_m': ((receiver-old_receiver)@old_rotation.T).tolist(),
        'maximum_absolute_transmitter_change_m': error,
        'archived_data_reference_geometry': str(args.geometry),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (args.metadata, prior_path, main_path, licensed)}}
    (args.output/'receiver-update.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
