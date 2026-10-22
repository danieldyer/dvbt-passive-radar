"""Independent multiscatterer phase measurements on the fixed observed grid.

The previously selected A/B/C/D cells are retained. Each channel and each
component has its own measured phase rate and curvature; no common trajectory,
antenna fit, AIS velocity constraint or multipath model is introduced.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from compare_ais_phase import path_geometry, NAMES
from fit_fixed_grid_phases import prepare_fixed
from fit_shared_echo_cluster import subset_prediction
from report_measured_scatterer_phases import load_fields, serial
from shared_echo_cluster import SharedEchoCluster
from ship_site_signature import licensed_options


def main():
    root = Path('docs/data/isar_20261017')
    source = root/'shared-echo-cluster-flash-four-seconds/fits.json'
    old = json.loads(source.read_text())
    first, last = old['interval_s']
    out = root/'fixed-grid-multiple-phases'
    out.mkdir(exist_ok=True)
    output = out/'fits.json'
    if output.exists():
        raise ValueError('Completed and partial measurements must be retained; use a fresh output')
    geometry = Path('docs/data/calibration_20261016_sites')
    fields = load_fields(Path('build/isar-20261015/full-fields'), geometry)
    bands, partition, changes = prepare_fixed(fields, geometry, first, last,
        stationary=True, observation_directory=out/'observations')
    prior = json.loads((geometry/'geometry-prior.json').read_text())
    options = licensed_options(geometry)
    reference_tx = np.asarray(next(o['transmitter_ecef_m'] for o in options
        if o['channel'] == 2 and o['site_id'] == '12013'))
    receiver = np.asarray(prior['receiver_ecef_m'])
    result = dict(scope=__doc__, complete=False, accepted_calibration=False,
        production_changed=False, interval_s=[first, last], labels=old['labels'],
        selected_origins_samples=old['selected_origins_samples'],
        origin_radius_samples=old['origin_radius_samples'], independent=[],
        phase_policy='Independent per-band and per-component physical quadratic path changes. AIS only supplies the nominal interband geometric difference. Old fits initialize the search, not a phase constraint.',
        nuisance_policy='One time-constant complex profile on actual fixed range cells. Data and moving templates have the same projection. Positive brightness is fitted at 250 ms knots. All coefficients are frozen on unused 125 ms blocks.',
        source_sha256={str(f['source']):f['source_sha256'] for f in fields})
    paths = [Path(__file__), source, geometry/'geometry-prior.json',
        geometry/'sites-main.json', geometry/'acma-transmitters.json',
        *[Path('support/isar')/(n+'.py') for n in
          ('fit_fixed_grid_phases', 'fixed_range_field', 'shared_echo_cluster',
           'amplitude_pair', 'amplitude_arc', 'coherent_arc', 'field', 'stationary')],
        *(out/'observations').glob('channel-*.npz')]
    snapshots = out/'sources'
    snapshots.mkdir(exist_ok=True)
    for path in paths:
        result['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == '.py':
            shutil.copyfile(path, snapshots/path.name)
    started = time.perf_counter()

    def save():
        result['seconds'] = time.perf_counter()-started
        output.write_text(json.dumps(serial(result), indent=2, allow_nan=False))

    save()
    n = len(old['labels'])
    for c, band in enumerate(bands):
        model = SharedEchoCluster([band], old['selected_origins_samples'],
            origin_radius=old['origin_radius_samples'])
        rows = band.field.data['row_time']
        d = fields[c]['data']
        position = np.column_stack([np.interp(rows, d['row_time'], d['centre_ecef_m'][:, j]) for j in range(3)])
        velocity = np.column_stack([np.interp(rows, d['row_time'], d['centre_velocity_ecef_mps'][:, j]) for j in range(3)])
        reference_path, _ = path_geometry(position, velocity, reference_tx, receiver)
        reference_path -= np.interp(band.epoch, rows, reference_path)
        nominal_seed = np.linalg.lstsq(band.base.basis, reference_path, rcond=None)[0]
        entry = dict(channel=c, name=NAMES[c], trials=[])
        result['independent'].append(entry)
        save()
        for start, previous in enumerate(old['independent'][c]['trials']):
            p = np.asarray(previous['parameters']).copy()
            p[:2*n:2] += changes[c]['envelope_origin_change_samples']
            p[1:2*n:2] += changes[c]['constant_phase_change_rad']
            p[2*n:] += np.tile(nominal_seed, n)
            fit = model.fit(p, 220)
            fit['start'] = start
            fit['prediction'] = subset_prediction(model, np.asarray(fit['parameters']), [partition[c]])
            entry['trials'].append(fit)
            save()
            print(NAMES[c], start, 'cost', fit['cost'], 'gradient', fit['maximum_projected_gradient'],
                'bounds', fit['bound_indices'], 'condition', fit['active_brightness_condition'],
                'energy ratio', fit['component_to_sum_energy_ratio'], flush=True)
        entry['best_index'] = int(np.argmin([f['cost'] for f in entry['trials']]))
        save()
    result['complete'] = True
    save()
    print('complete', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
