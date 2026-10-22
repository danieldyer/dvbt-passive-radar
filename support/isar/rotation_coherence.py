"""Conditional phase-loss budget for an unmodelled constant yaw rate.

This is a hypothetical point-response calculation, not a yaw fit to the ship.
The saved translation, antenna coordinates and all existing images stay fixed.
Known centre translation, constant point amplitude and white noise isolate the
phase loss of a fixed-orientation template. Actual headings are not inferred.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from ais import utc_seconds
from imaging import motion_state, sites
from motion import C
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def path_change(centre, frame, point, angle, transmitter, receiver):
    """Exact difference between rotating and fixed-point bistatic paths.

    Difference of squared distances avoids cancellation of Earth-scale path
    coordinates. The identical direct-reference distance cancels algebraically.
    Positive angle is a right-handed rotation about the local up axis.
    """
    point = np.asarray(point)
    c, s = np.cos(angle), np.sin(angle)
    moved = np.column_stack((c*point[0]-s*point[1], s*point[0]+c*point[1],
                             np.full(len(angle), point[2])))
    original = centre+np.einsum('nij,j->ni', frame, point)
    delta = np.einsum('nij,nj->ni', frame, moved-point)
    change = np.zeros(len(centre))
    for antenna in (transmitter, receiver):
        before = original-antenna
        after = before+delta
        change += (2*np.sum(before*delta, axis=1)+np.sum(delta**2, axis=1))/(
            np.linalg.norm(before, axis=1)+np.linalg.norm(after, axis=1))
    return change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    model = json.loads(args.motion.read_text())
    curve = SplineMotion(model)
    if not model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients):
        raise ValueError('Use the fixed-N/E, zero-yaw reference control')
    tx, rx, configuration = sites(args.geometry)
    interval = curve.interval
    middle, duration = float(np.mean(interval)), float(np.ptp(interval))
    nodes, weights = np.polynomial.legendre.leggauss(256)
    time = middle+nodes*duration/2
    weights /= 2
    epoch = utc_seconds(model['nominal_start_utc'])
    state = curve.apply(time, motion_state(None, epoch+time, model), tx[model['reference_channel']], rx)
    centre, frame = state['position_ecef_m'], state['body_to_ecef']
    midpoint = curve.apply(np.array([middle]), motion_state(None, np.array([epoch+middle]), model),
                           tx[model['reference_channel']], rx)
    origin, basis = midpoint['position_ecef_m'][0], midpoint['body_to_ecef'][0]
    # One common horizontal point direction for every band, chosen to expose
    # the largest first-order phase sensitivity in the Ten reference geometry.
    gradient = sum((origin-a)/np.linalg.norm(origin-a) for a in (tx[3], rx))
    local = basis.T@gradient
    direction = np.r_[local[1], -local[0], 0.]/np.linalg.norm(local[:2])
    field_paths = [args.fields/f'channel-{c}.npz' for c in range(len(tx))]
    frequency = []
    for channel, path in enumerate(field_paths):
        with np.load(path) as field:
            metadata = json.loads(str(field['metadata_json']))
            if metadata['channel'] != channel:
                raise ValueError('Unexpected field/channel order')
            np.testing.assert_allclose(field['transmitter_ecef_m'], tx[channel], atol=1e-6, rtol=0.)
            np.testing.assert_allclose(field['receiver_ecef_m'], rx, atol=1e-6, rtol=0.)
            frequency.append(metadata['rf_hz'])
    rates = np.linspace(0., .02, 401)
    records = []
    for radius in (50., 150.):
        point = radius*direction
        for channel, rf in enumerate(frequency):
            def fraction(rate):
                delta = path_change(centre, frame, point, np.radians(rate)*(time-middle), tx[channel], rx)
                return float(abs(weights@np.exp(-2j*np.pi*rf/C*delta))**2)
            values = np.array([fraction(rate) for rate in rates])
            target = 10**(-.1)
            crossing = np.flatnonzero(values < target)
            if not len(crossing):
                raise ValueError('Extend the rate scan to bracket the first 1 dB crossing')
            j = int(crossing[0])
            threshold = brentq(lambda rate: fraction(rate)-target, rates[j-1], rates[j], xtol=1e-12)
            point_ecef = origin+basis@point
            point_gradient = sum((point_ecef-a)/np.linalg.norm(point_ecef-a) for a in (tx[channel], rx))
            lever = float(point_gradient@(basis@np.cross([0., 0., 1.], point)))
            linear = np.sinc(rf/C*lever*np.radians(rates)*duration)**2
            before_crossing = rates <= threshold
            discrepancy = np.max(abs(10*np.log10(values[before_crossing]/linear[before_crossing])))
            records.append({'channel': channel, 'name': configuration['channels'][channel]['channel'],
                'rf_hz': float(rf), 'radius_m': radius, 'point_in_reference_frame_m': point.tolist(),
                'phase_only_snr_fraction': values.tolist(), 'linear_sinc_fraction': linear.tolist(),
                'one_db_rate_deg_per_s': float(threshold), 'one_db_total_angle_deg': float(threshold*duration),
                'phase_only_loss_at_001_deg_per_s_db': float(-10*np.log10(fraction(.01))),
                'linear_exact_discrepancy_before_one_db': float(discrepancy)})
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'source': str(args.motion), 'source_sha256': fingerprint(args.motion),
        'geometry_sha256': {str(p): fingerprint(p) for p in
            [args.geometry/'geometry-prior.json', args.geometry/'sites-main.json']},
        'frequency_metadata_sources_sha256': {str(p): fingerprint(p) for p in field_paths},
        'implementation_sha256': {str(Path(__file__)): fingerprint(Path(__file__))},
        'interval_s': interval.tolist(), 'rates_deg_per_s': rates.tolist(), 'cases': records,
        'point_direction_scope': 'Same horizontal lever-arm direction for all bands, maximizing the Ten midpoint first-order yaw sensitivity. Illustrative orientation, not inferred ship attitude.',
        'window': f'Uniform rectangular {duration:g}-second aperture, integrated by 256-node Gauss quadrature.',
        'limitations': 'Phase-only template correlation for a hypothetical constant-amplitude point after known centre translation. No inferred rotation, range-envelope correction, empirical noise bound or image-gain claim.',
        'yaw_fitted': False, 'physical_model_changed': False, 'videos_changed': False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'rotation-coherence-budget.json').write_text(json.dumps(report, indent=2)+'\n')
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), layout='constrained')
    for axis, radius in zip(axes, (50., 150.)):
        for record in records:
            if record['radius_m'] != radius:
                continue
            loss = -10*np.log10(np.maximum(record['phase_only_snr_fraction'], 1e-12))
            axis.plot(rates, loss, label=record['name'])
        axis.axhline(1., color='#777777', linestyle='--', linewidth=1.)
        axis.set(title=f'Example point {radius:g} m from rotation centre', xlabel='Unmodelled yaw rate (degrees/s)',
                 ylabel='Phase-only matched-filter loss (dB)', ylim=(0, 10), xlim=(0, .02))
    axes[0].legend(fontsize=9)
    figure.suptitle(f'{duration:g}-second fixed-orientation filter; centre translation perfectly known\n'
                     'Hypothetical sensitivity calculation — no rotation fitted to this recording')
    figure.savefig(args.output/'rotation-coherence-budget.png', dpi=160)
    plt.close(figure)
    rows = ''.join('<tr>'+''.join(f'<td>{value}</td>' for value in [r['name'], r['radius_m'],
        f"{r['one_db_rate_deg_per_s']:.5f}", f"{r['one_db_total_angle_deg']:.3f}",
        f"{r['phase_only_loss_at_001_deg_per_s_db']:.2f}"] )+'</tr>' for r in records)
    (args.output/'rotation-coherence-budget.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Rotation coherence budget</title>
<style>body{{font:17px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}</style>
<h1>How slow would heading change need to be for a fixed-orientation filter?</h1>
<p><a href="analysis.html">Index</a> · <a href="rotation-coherence-budget.json">Calculation and input identities</a> · {report['updated_utc']}</p>
<p class="note">This is a sensitivity calculation, not a fitted ship rotation or a proposed correction. The recorded-data motion and videos remain exactly zero-yaw. A fixed north/east display orientation is compatible with a physically rotating target model; the two operations are distinct.</p>
<p>For a constant-amplitude point with white noise, the SNR fraction of a mismatched phase template is |⟨exp(−ik ΔL(t))⟩|². This is the squared normalized inner product of the true and assumed templates, obtained directly from matched-filter output signal and noise powers. Centre translation is assumed perfectly compensated. The example uses the saved 150.772–180.772 s translation and nominal sites, plus one hypothetical horizontal point direction shared by every band. It is chosen for maximum first-order Ten sensitivity, not inferred from the ship image.</p>
<p>For small constant yaw rate ω, ΔL ≈ [(uTX + uRX) · (up × q)] ωt. A rectangular aperture therefore gives sinc²(f AωT/c), using the normalized sinc. The numerical curves use exact rotating/fixed bistatic distance differences and Gaussian quadrature. They omit the small associated range-envelope displacement and other scene/noise effects; they quantify phase sensitivity, not measured image SNR or actual yaw.</p>
<img src="rotation-coherence-budget.png" alt="Conditional matched-filter phase loss versus hypothetical constant yaw rate for 50 and 150 metre lever arms">
<table><tr><th>Band</th><th>Lever arm (m)</th><th>Rate giving 1 dB loss (deg/s)</th><th>Angle change over {duration:g} s (deg)</th><th>Loss at 0.01 deg/s (dB)</th></tr>{rows}</table>
<p>The first-order sinc approximation differs from the exact-path calculation by at most {max(r['linear_exact_discrepancy_before_one_db'] for r in records):.3g} dB before the first 1 dB crossing. No comparison here establishes that this vessel actually rotated, that a rotation parameter is identifiable jointly with translation/geometry, or that including it improves withheld prediction.</p></html>''')
    print(json.dumps({'report': str(args.output/'rotation-coherence-budget.html'),
        'one_db_total_angle_deg': [r['one_db_total_angle_deg'] for r in records]}))


if __name__ == '__main__':
    main()
