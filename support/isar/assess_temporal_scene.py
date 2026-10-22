"""Test temporal amplitude and phase changes before adding physical motion states."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from antenna_motion import Antennas
from coherent_motion import CoherentMotion
from field import Field
from full_motion import Band
from imaging import physical_state, sites
from temporal_scene import TemporalScene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'motion', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--durations', type=float, nargs='+', default=[15., 10., 5., 3., 1.])
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.alignment.read_text())
    model = json.loads(args.motion.read_text())
    if np.any(model.get('channel_path_rate_bias_mps', 0.)):
        raise ValueError('This comparison does not yet apply channel path-rate registration')
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch, bands = utc_seconds(model['nominal_start_utc']), []
    for c in range(5):
        with np.load(Path(source['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(model['interval_s'][0]+.02, model['interval_s'][1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        bands.append(Band(field, prior, tx[3], rx, model['points_body_m'], model,
            np.arange(len(field.time))%2 == 0, range_bias_m=model['range_bias_m'][c]))
    antennas = None
    if model.get('antenna_fit'):
        a = model['antenna_fit']
        antennas = Antennas(a['ecef_to_enu'], a['channel_sites'], site_sigma=a['site_prior_sigma_enu_m'],
            antenna_sigma=a['antenna_prior_sigma_enu_m'], receiver_sigma=a['receiver_prior_sigma_enu_m'])
    gains = np.array([complex(*g) for g in model['channel_gains_real_imag']])
    joint = CoherentMotion(bands, gains, model['parameters'], antennas=antennas)
    parameters = joint.initial.copy()
    if antennas is not None:
        parameters[joint.antenna_slice] = model['antenna_fit']['standardised_parameters']
    amplitude, _, designs, _, _ = joint.project(parameters)
    if model.get('amplitude_model'):
        amplitude = np.exp(1j*np.asarray(model['amplitude_model']['point_phase_radians']))
    experiment = TemporalScene(bands, designs, gains, model['interval_s'])
    result = {'motion': str(args.motion), 'alignment': str(args.alignment), 'scope': __doc__, 'trials': [],
        'caution': 'Support and motion were fitted on full-aperture even bins. This evaluates frozen models on odd bins and does not establish independent heading or antenna calibration.'}
    started = time.perf_counter()
    for duration in [float(np.diff(model['interval_s'])[0]), *args.durations]:
        for mode in ('complex', 'amplitude', 'independent_amplitude', 'independent'):
            fit = experiment.fit(duration, phase_reference=amplitude if 'amplitude' in mode else None,
                                 independent_channels='independent' in mode)
            result['trials'].append(fit)
            result['seconds'] = time.perf_counter()-started
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({k:fit[k] for k in ('duration_s', 'coefficient_model', 'heldout_explained_fraction',
                'maximum_linear_stationarity_residual', 'design_conditions')}|{'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
