"""Audit phase-gradient autofocus against an amplitude-aware full-range model.

The scalar phase history is the matched sufficient statistic of the whitened
range vector. Even time bins train the phase; odd bins test it. Additional
folds train on alternating whitened range components and test the others.
The prior scene/motion used all training ranges, so this remains conditional
validation, not an untouched experiment in target discovery.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from amplitude_motion import AmplitudeMotion
from field import Field
from fit_gpga import binned_metrics
from full_motion import Band
from gpga import fit, phase_metrics, refine
from imaging import physical_state, sites
from matched_phase import history, refine_profiled
from stationary import Stationary


def score(field, background, template, phase, selected, ranges):
    test_time = ~selected
    mask = np.zeros(field.y.shape, bool)
    mask[np.ix_(test_time, ranges)] = True
    nuisance = Stationary(background, mask.ravel())
    data = nuisance.remove(field.y[mask])
    model = nuisance.remove((template*np.exp(1j*phase[:, None]))[mask])
    data, model = data.reshape(test_time.sum(), -1), model.reshape(test_time.sum(), -1)
    z, weight = history(data, model)
    time = field.time[test_time]
    metrics = binned_metrics(time, z, weight, np.ones(len(time), bool), np.ones(1, complex), 1.)
    energy = float(np.sum(abs(model)**2))
    product = np.vdot(model, data)
    return {'prediction_error': float(np.sum(abs(data-model)**2)),
        'matched_energy': float(abs(product)**2/energy),
        'explained_energy': float(2*product.real-energy),
        'one_second_phase_metrics': metrics,
        'test_stationary_rank': nuisance.rank,
        'target_information_retained': float(energy/np.sum(abs(template[mask])**2))}


def audit_channel(band, template, *, knot_seconds=1., profile_background=True):
    field = band.field
    beta = band.nuisance.solve(field.y.ravel()[band.selected]-template.ravel()[band.selected])
    clean = field.y-(band.nuisance.design@beta).reshape(field.y.shape)
    _, all_information = history(clean, template)
    ranges = np.arange(field.y.shape[1])
    channel = {'channel': band.field.metadata['channel'], 'trials': []}
    screens = []
    for name, train_range, test_range in [('all', ranges, ranges), ('even to odd', ranges[::2], ranges[1::2]),
                                          ('odd to even', ranges[1::2], ranges[::2])]:
        z, information = history(clean, template, train_range)
        coarse = fit(field.time, z, information, band.selected_time, bandwidths_hz=(1., .5, .25), iterations_per_bandwidth=20)
        phase, detail = refine(field.time, z, information, band.selected_time, coarse.pop('phase_radians'),
                               knot_seconds=knot_seconds, gauge_information=all_information)
        fixed_background = detail
        if profile_background:
            mask = np.zeros(field.y.shape, bool)
            mask[np.ix_(band.selected_time, train_range)] = True
            phase, detail = refine_profiled(field.time, field.y, template, band.nuisance.design, mask, phase,
                knot_seconds=knot_seconds, gauge_information=all_information)
        before = score(field, band.nuisance.design, template, np.zeros(len(phase)), band.selected_time, test_range)
        after = score(field, band.nuisance.design, template, phase, band.selected_time, test_range)
        entry = {'ranges': name, 'gpga_converged': coarse['converged'], 'refinement': detail,
            'initial_fixed_background_refinement': fixed_background if profile_background else None,
            'before': before, 'after': after,
            'matched_gain_db': float(10*np.log10(after['matched_energy']/before['matched_energy'])),
            'maximum_adjacent_bin_phase_change_degrees': float(np.degrees(abs(np.diff(phase)).max())),
            'phase_rms_degrees': float(np.degrees(np.sqrt(np.average(phase**2, weights=all_information)))),
            'phase_radians': phase.tolist()}
        channel['trials'].append(entry)
        screens.append(phase)
        print(json.dumps({'channel': channel['channel'], 'ranges': name, 'converged': detail['converged'],
            'gain_db': entry['matched_gain_db'], 'phase_rms_before': before['one_second_phase_metrics']['phase_rms_degrees_by_point'],
            'phase_rms_after': after['one_second_phase_metrics']['phase_rms_degrees_by_point']}), flush=True)
    channel['range_fold_disagreement_rms_degrees'] = float(np.degrees(np.sqrt(np.average((screens[1]-screens[2])**2, weights=all_information))))
    return channel, {"time": field.time, "information": all_information, "phase_screens": screens}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'motion', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--knot-seconds', type=float, default=1.)
    p.add_argument('--profile-background', action='store_true', help='Refine using the full complex residual, re-solving stationary nuisance at every phase trial')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, source = json.loads(args.motion.read_text()), json.loads(args.alignment.read_text())
    if model.get('antenna_fit') or not model.get('amplitude_model'):
        raise ValueError('Use an amplitude-aware nominal-location fit')
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch, bands = utc_seconds(model['nominal_start_utc']), []
    for c in range(5):
        with np.load(Path(source['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(model['interval_s'][0]+.02, model['interval_s'][1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        bands.append(Band(field, prior, tx[3], rx, model['points_body_m'], model,
            np.arange(len(field.time))%2 == 0, range_bias_m=model['range_bias_m'][c]))
    gains = np.array([complex(*g) for g in model['channel_gains_real_imag']])
    phases = np.exp(1j*np.asarray(model['amplitude_model']['point_phase_radians']))
    joint = AmplitudeMotion(bands, model['parameters'], gains, phases, model['amplitude_model']['duration_s'],
        independent_magnitudes=not model['amplitude_model'].get('shared_across_channels', True),
        fit_path_rates=bool(model.get('channel_path_rate_bias_mps')))
    fitted = joint.project(joint.initial)
    result = {'input_motion': str(args.motion), 'scope': __doc__, 'channels': [],
        'background_profiled_during_phase_fit': args.profile_background,
        'phase_gauge': 'Full-template-information-weighted mean and linear phase removed using training times; identical gauge for both range folds.',
        'original_row_phase_application_validated': False,
        'limitation': 'This estimates a phase screen on retained 17.4 ms complex fields. It is not yet applied to the original sample operator, image, or production data.'}
    for c, (band, design, block) in enumerate(zip(bands, fitted['designs'], joint.blocks)):
        amplitude = fitted['magnitudes_by_channel'][c]*phases
        field = band.field
        template = (gains[c]*np.sum(design*amplitude[np.repeat(block, field.y.shape[1])], axis=1)).reshape(field.y.shape)
        channel, arrays = audit_channel(band, template, knot_seconds=args.knot_seconds,
                                        profile_background=args.profile_background)
        np.savez(args.output/f'channel-{c}.npz', **arrays)
        result['channels'].append(channel)
        (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
