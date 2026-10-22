"""Export a stationary shared-target fit for physical image formation.

The fitted trajectory retains its nominal reference geometry. Actual TX/RX
offsets combine the licensed-site assignment and its fitted local correction.
Band gains remain explicitly provisional; this export is not calibration.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from audit_joint_sites import load_joint
from imaging import corrected_sites
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import summarize


PHYSICAL_KEYS = ('nominal_start_utc', 'attitude_prior', 'reference_height_ellipsoid_m',
    'reference_channel', 'geometry_directory', 'channel_gains_real_imag', 'interval_s',
    'path_spacing_seconds', 'heading_spacing_seconds', 'rock_spacing_seconds',
    'transverse_spacing_seconds', 'horizontal_translation', 'anchor_path', 'anchor_transverse',
    'path_reference_time_s', 'linear_reference', 'fixed_reference_heading_deg', 'fixed_ne_frame',
    'max_ais_gap_s', 'yaw_convention', 'roll_pitch_fitted', 'point_coordinate_frame', 'initial_transverse_m')


def exported_model(profile, parameters):
    template = profile.bands[0].model
    model = {key: copy.deepcopy(template[key]) for key in PHYSICAL_KEYS if key in template}
    model['parameters'] = profile.expand(parameters).tolist()
    model['range_bias_m'] = np.asarray(parameters)[profile.range_slice].tolist()
    model['selected_points_body_m'] = [band.points.tolist() for band in profile.bands]
    offsets = np.array([profile.offsets(parameters, c) for c in range(len(profile.bands))])
    np.testing.assert_allclose(offsets[:, 1], np.broadcast_to(offsets[0, 1], offsets[:, 1].shape), atol=1e-12, rtol=0)
    rotation = profile.antennas.rotation
    model['antenna_fit'] = {'transmitter_offsets_enu_m': (offsets[:, 0]@rotation.T).tolist(),
        'receiver_offset_enu_m': (offsets[0, 1]@rotation.T).tolist(), 'ecef_to_enu': rotation.tolist(),
        'reference': 'Offsets from nominal field geometry, including licensed-assignment offsets; not site-local offsets.',
        'accepted_calibration': False}
    model['site_local_fit'] = profile.antennas.describe(np.asarray(parameters)[profile.antenna_slice])
    model.update(phase_centres_validated=False, channel_calibration_validated=False,
        scope=__doc__, complete=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'ship-fit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--compare', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing motion export')
    fit = json.loads(args.fit.read_text())
    if not fit['complete'] or not fit['fit']['converged'] or fit['original_row_maximum_kkt_residual'] > 1e-7:
        raise ValueError('Export a completed fit that passed original-row stationarity')
    for group in ('sources', 'implementation_sha256'):
        for file, digest in fit['identity']['legacy_identity'][group].items():
            if fingerprint(Path(file)) != digest:
                raise ValueError('A fit input changed: '+file)
    for hashes in (fit['identity']['implementation_sha256'],
                   fit.get('numerical_refinement', {}).get('implementation_sha256', {})):
        for file, digest in hashes.items():
            if fingerprint(Path(file)) != digest:
                raise ValueError('The fit implementation changed: '+file)
    if fit['identity']['legacy_identity']['sources'].get(str(args.ship_fit)) != fingerprint(args.ship_fit):
        raise ValueError('Use the original ship-fit source')
    ship = json.loads(args.ship_fit.read_text())
    profile = load_joint(ship, args.fields, args.geometry)
    parameters = np.array(fit['parameters'])[:len(profile.initial)]
    model = exported_model(profile, parameters)
    if not model['fixed_ne_frame'] or np.any(SplineMotion(model).yaw_coefficients):
        raise ValueError('This export requires fixed N/E with exactly zero yaw')
    # Verify absolute propagated locations separately from the motion gauge.
    tx = np.array([band.field.data['transmitter_ecef_m'] for band in profile.bands])
    rx = profile.bands[0].field.data['receiver_ecef_m']
    actual_tx, actual_rx = corrected_sites(tx, rx, model['antenna_fit'])
    for c in range(len(tx)):
        np.testing.assert_allclose(actual_tx[c], tx[c]+profile.offsets(parameters, c)[0], atol=1e-9, rtol=0)
        np.testing.assert_allclose(actual_rx, rx+profile.offsets(parameters, c)[1], atol=1e-9, rtol=0)
    checks = profile.validation(parameters)
    after = summarize(checks)
    np.testing.assert_allclose(after['prediction_error'], fit['validation']['ship_summary']['prediction_error'], rtol=1e-12)
    comparisons = []
    for path in args.compare:
        other = json.loads(path.read_text())
        if other['interval_s'] != model['interval_s'] or other.get('antenna_fit'):
            raise ValueError('Matched comparison requires this aperture and nominal antennas')
        curve = SplineMotion(other); full, active = curve.gauge_fixed_parameters()
        np.testing.assert_array_equal(active, profile.active)
        q = profile.initial.copy(); q[:profile.nmotion] = full[active]
        q[profile.antenna_slice] = 0.; q[profile.range_slice] = other['range_bias_m']
        validation = profile.validation(q); summary = summarize(validation)
        comparisons.append({'source': str(path), 'sha256': fingerprint(path), 'summary': summary,
            'channels': validation,
            'ship_matched_energy_change_db': float(10*np.log10(after['matched_energy']/summary['matched_energy'])),
            'prediction_error_reduction_percent': float(100*(1-after['prediction_error']/summary['prediction_error'])),
            'scope': 'Same original rows, masks, covariance, stationary nuisance and eight-point per-band dictionary. Only motion and range origins change; this is not a comparison of different scene estimators.'})
    files = [args.fit, args.ship_fit, Path(__file__), *args.compare]
    model.update(source_fit=str(args.fit), source_fit_sha256=fingerprint(args.fit), fit=fit['fit'],
        original_row_kkt=fit['original_row_maximum_kkt_residual'], after_summary=after,
        original_row_validation_after=checks, comparison= comparisons,
        source_sha256={str(p): fingerprint(p) for p in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'range_bias_m': model['range_bias_m'],
        'comparisons': [{k: v for k, v in c.items() if k not in ('summary', 'channels')} for c in comparisons]}), flush=True)


if __name__ == '__main__':
    main()
