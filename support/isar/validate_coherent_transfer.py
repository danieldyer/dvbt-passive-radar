"""Conditional half-aperture transfer after coherent motion/location fitting.

Scattering amplitudes, and optionally band gains, are learned on even bins in
one half. Prediction uses only odd bins in the opposite half. Motion, geometry,
range origins and point support were selected using even bins across the full
aperture, so this is a conditional transfer diagnostic, not an independent
end-to-end calibration validation. Stationary background is profiled in the
validation half with the same orthogonal projection applied to both observations
and frozen target predictions; its information loss is reported. No ship motion,
channel gain or scattering amplitude is fitted on validation samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from antenna_motion import Antennas
from coherent_motion import CoherentMotion
from field import Field
from full_motion import Band
from imaging import physical_state, sites
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'motion', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    source, model = json.loads(args.alignment.read_text()), json.loads(args.motion.read_text())
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    antennas = None
    if model.get('antenna_fit'):
        a = model['antenna_fit']
        antennas = Antennas(a['ecef_to_enu'], a['channel_sites'], site_sigma=a['site_prior_sigma_enu_m'],
            antenna_sigma=a['antenna_prior_sigma_enu_m'], receiver_sigma=a['receiver_prior_sigma_enu_m'])
    fields, priors = [], []
    for c in range(5):
        with np.load(Path(source['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(model['interval_s'][0]+.02, model['interval_s'][1]-.02)
        fields.append(field)
        priors.append(physical_state(track, epoch+field.data['row_time'], model['attitude_prior']))
    gains = np.array([complex(*g) for g in model['channel_gains_real_imag']])
    output = {'input_motion': str(args.motion), 'source_alignment': str(args.alignment), 'scope': __doc__, 'trials': []}
    for half in (0, 1):
        bands, tests = [], []
        for c, (field, prior) in enumerate(zip(fields, priors)):
            even = np.arange(len(field.time))%2 == 0
            side = (field.time >= np.mean(model['interval_s'])).astype(int)
            selected = even&(side == half)
            tests.append((~even)&(side != half))
            bands.append(Band(field, prior, tx[3], rx, model['points_body_m'], model, selected,
                              range_bias_m=model['range_bias_m'][c]))
        for refit_gains in (False, True):
            joint = CoherentMotion(bands, gains, model['parameters'], antennas=antennas,
                                   profile_channel_gains=refit_gains)
            parameters = joint.initial.copy()
            if antennas is not None:
                parameters[joint.antenna_slice] = model['antenna_fit']['standardised_parameters']
            channels = joint.validation(parameters, test_masks=tests, stationary_mode='project')
            fitted_gain = joint.project(parameters)[-1]
            record = {'training_half': half, 'refit_gains': refit_gains,
                'channel_gain_phase_degrees': np.degrees(np.angle(fitted_gain)).tolist(),
                'summary': summarize(channels), 'channels': channels}
            output['trials'].append(record)
            print(json.dumps({k:record[k] for k in ('training_half', 'refit_gains', 'summary')}), flush=True)
            args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
