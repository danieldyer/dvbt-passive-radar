"""Publish delayed-return transfer, matched motion controls and their limits."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint

NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('acquisition', 'validation', 'gradient-audit', 'search-equivalence', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--videos', type=Path)
    p.add_argument('--test-log', type=Path, required=True)
    args = p.parse_args()
    acquisition, validation = [json.loads(path.read_text()) for path in (args.acquisition, args.validation)]
    gradient = json.loads(args.gradient_audit.read_text())
    equivalence = json.loads(args.search_equivalence.read_text())
    durations = json.loads((args.output/'fixed-ne-duration-statistics.json').read_text())
    yaw = [{k: r[k] for k in ('window', 'duration_s', 'paired_convergence',
        'heldout_error_change_percent', 'heldout_matched_energy_change_db')}
        for r in durations['records'] if r['duration_s'] == 30.]
    if not acquisition['complete'] or not validation['complete']:
        raise ValueError('Finish the matched controls before publishing')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in [None, *range(5)]:
        trials = [t for t in acquisition['trials'] if c is None or t['channel'] == c]
        before = sum(t['primary_before']['prediction_error'] for t in trials)
        after = sum(t['primary_after']['prediction_error'] for t in trials)
        rows.append({'channel': c, 'trials': len(trials), 'converged': sum(t['fit']['converged'] for t in trials),
            'error_change_percent': 100*(after/before-1),
            'failed_companion_predictions': sum(t['companion_validation']['explained_fraction'] < 0 for t in trials)})
    record = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'transfer': rows,
        'acquisition': str(args.acquisition), 'acquisition_sha256': fingerprint(args.acquisition),
        'validation': validation, 'validation_sha256': fingerprint(args.validation),
        'gradient_audit': gradient, 'gradient_audit_sha256': fingerprint(args.gradient_audit),
        'search_equivalence': equivalence, 'search_equivalence_sha256': fingerprint(args.search_equivalence),
        'earlier_30s_yaw_controls': yaw,
        'test_log': str(args.test_log), 'test_result': args.test_log.read_text().strip().splitlines()[-3:],
        'calibration_adopted': False, 'nuisance_adopted_in_production': False, 'yaw_fitted': False,
        'original_profiles_modified': False}
    if args.videos:
        record['video'] = json.loads((args.videos/'videos.json').read_text())
        record['video_validation'] = json.loads((args.videos/'validation.json').read_text())
    (args.output/'companion-validation.json').write_text(json.dumps(record, indent=2)+'\n')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), layout='constrained')
    for c, name in enumerate(NAMES):
        t = [r for r in acquisition['trials'] if r['channel'] == c]
        x = [r['centre_s'] for r in t]
        axes[0, 0].plot(x, [r['parameters'][0] for r in t], '.-', label=name, lw=1.)
        axes[0, 1].plot(x, [r['parameters'][1] for r in t], '.-', label=name, lw=1.)
        axes[1, 0].plot(x, [r['primary_error_change_percent'] for r in t], '.-', label=name, lw=1.)
    axes[0, 0].set(ylabel='Extra path (m)', title='Companion path acquired independently')
    axes[0, 1].set(ylabel='Relative path rate (m/s)', title='Range migration and carrier phase agree by construction')
    axes[1, 0].set(ylabel='Primary heldout error change (%)', title='Same complex amplitude transferred into ship crop')
    axes[1, 0].axhline(0, color='#777', lw=.6)
    axes[1, 1].bar(NAMES, [r['error_change_percent'] for r in rows[1:]], color='#2876a3')
    axes[1, 1].axhline(0, color='#777', lw=.6)
    axes[1, 1].set(ylabel='Aggregate heldout error change (%)', title='All fifteen 3-second blocks; none discarded')
    for ax in axes.ravel()[:3]:
        ax.set(xlabel='Capture time (s)')
    axes[0, 0].legend(ncol=3, fontsize=9)
    fig.suptitle('Delayed-return transfer at fixed yaw-free motion\nTraining estimates predict withheld samples; this is not a ship-SNR measurement')
    fig.savefig(args.output/'companion-transfer.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    leak = validation['range_crop_leakage']
    for name, channel in zip(NAMES, leak['channels']):
        axes[0].plot(leak['extra_path_m'], channel['raw_energy_ratio_db'], label=name, lw=1.)
        axes[1].plot(leak['extra_path_m'], channel['whitened_energy_ratio_db'], label=name, lw=1.)
    for ax, title in zip(axes, ('Raw range crop', 'After declared thermal range whitening')):
        ax.set(xlabel='Extra path from crop centre (m)', ylabel='Relative template energy (dB)', title=title)
        ax.axvline(500, color='#777', lw=.6)
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle('Out-of-crop points still contribute to the ship likelihood\nEqual echo amplitude, 17 samples; normalized to a centred point in each metric')
    fig.savefig(args.output/'companion-leakage.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained')
    curves = validation['motion_curves']
    for key, label, color in [('source', 'Initial no-yaw fit', '#888'),
            ('without_companion', 'Matched control', '#c1792e'), ('with_companion', 'With companion nuisance', '#2466a2')]:
        axes[0, 0].plot(curves['time_s'], curves['path_m'][key], label=label, color=color)
        axes[0, 1].plot(curves['time_s'], curves['path_rate_mps'][key], color=color)
    axes[0, 0].set(xlabel='Capture time (s)', ylabel='Path correction (m)', title='Yaw remains exactly zero')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel='Capture time (s)', ylabel='Path-rate correction (m/s)', title='Physical translation and velocity')
    for i, family in enumerate(('stationary', 'stationary_and_companion')):
        values = validation['families'][family]['candidates']
        a, b = [values[k]['channels'] for k in ('without_companion', 'with_companion')]
        changes = [100*(y['prediction_error']/x['prediction_error']-1) for x, y in zip(a, b)]
        axes[1, i].bar(NAMES, changes, color='#2466a2')
        axes[1, i].axhline(0, color='#777', lw=.6)
        axes[1, i].set(ylabel='Heldout error change (%)',
            title='Same '+('stationary' if i == 0 else 'stationary + companion')+' projection')
    fig.suptitle('Does modelling the companion change the estimated ship motion usefully?\nSame Gaussian scene model and original rows in both fits')
    fig.savefig(args.output/'companion-motion.png', dpi=150)
    plt.close(fig)

    transfer_rows = ''.join(f'<tr><td>{"All" if r["channel"] is None else NAMES[r["channel"]]}</td>'
        f'<td>{r["error_change_percent"]:+.3f}%</td><td>{r["converged"]}/{r["trials"]}</td>'
        f'<td>{r["failed_companion_predictions"]}</td></tr>' for r in rows)
    motion_rows = ''.join(f'<tr><td>{family.replace("_", " ")}</td>'
        f'<td>{v["companion_vs_control_error_change_percent"]:+.4f}%</td>'
        f'<td>{v["companion_vs_control_matched_change_db"]:+.4f} dB</td></tr>'
        for family, v in validation['families'].items())
    cost_rows = ''.join(f'<tr><td>{name}</td><td>{max(d["point_information_loss_db"]):.4f} dB</td>'
        f'<td>{max(d["scene_mode_information_loss_db"]):.4f} dB</td></tr>'
        for name, d in zip(NAMES, validation['information_cost']))
    matched = validation['families']['stationary_and_companion']['candidates']
    phase_rows = ''.join(f'<tr><td>{name}</td><td>{before["prediction_phase_rms_degrees"]:.2f}°</td>'
        f'<td>{after["prediction_phase_rms_degrees"]:.2f}°</td></tr>' for name, before, after in
        zip(NAMES, matched['without_companion']['channels'], matched['with_companion']['channels']))
    yaw_rows = ''.join(f'<tr><td>{r["window"]}</td><td>{r["heldout_matched_energy_change_db"]:+.3f} dB</td>'
        f'<td>{r["heldout_error_change_percent"]:+.2f}%</td></tr>' for r in yaw)
    video_html = ''
    if args.videos:
        video_html = '<h2>Original-data preview videos</h2><ul>'+''.join(
            f'<li><a href="../../../{html.escape(v["path"])}">{Path(v["path"]).name}</a></li>'
            for v in record['video']['files'])+f'</ul><p>These are {record["video"]["frames"]/record["video"]["fps"]:.1f}-second diagnostic previews covering capture centres {record["video"]["frame_centres_s"][0]:.3f}–{record["video"]["frame_centres_s"][1]:.3f} s, not replacements for the full-capture videos. Fixed north/east axes, 30-second coherent windows, 30 fps, cyan AIS track and fixed colour scale. The companion remains in the backprojection input. This is a conditional image, not an accepted multiband phase calibration.</p>'
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Yaw-free autofocus: delayed-return controls</title>
<style>body{{font:17px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#243040}}img{{max-width:100%}}td,th{{text-align:left;padding:7px 13px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}.note{{background:#fff0ce;padding:17px}}</style>
<h1>Yaw-free autofocus: accounting for the delayed return</h1>
<p><a href="fixed-ne-followup.html">No-yaw comparison and earlier videos</a> · <a href="secondary-covariance.html">Covariance controls</a> · <a href="companion-validation.json">Measurements and identities</a></p>
<p class="note">The matched test finds no meaningful autofocus improvement: heldout matched energy changes by only {validation['families']['stationary_and_companion']['companion_vs_control_matched_change_db']:+.6f} dB. The separately measured delayed component does explain additional signal in the ship crop, reducing prediction error by {abs(rows[0]['error_change_percent']):.2f}% in that transfer test. These are different results. The extra nuisance remains a support diagnostic and is not added to production. All new motion models keep yaw exactly zero.</p>
<h2>Does removing yaw improve the result?</h2>
<p>The earlier paired 30-second controls both converged, but removing yaw reduced heldout matched energy and increased prediction error:</p>
<table><tr><th>Test aperture</th><th>No-yaw matched-energy change</th><th>No-yaw prediction-error change</th></tr>{yaw_rows}</table>
<p>The <a href="fixed-ne-followup.html">subsequent numerical follow-up</a> found −0.164 dB in the converged regularized 3-second comparison, and −3.779 dB for the selected 45-second no-yaw model. Another short-window rotating control remains numerically unresolved and is not counted as evidence. These are conditional comparisons using the declared point scene and nuisance model; a statistical advantage for a flexible yaw model does not validate the earlier implausible physical heading estimates. The new north/east previews retain broad ridges rather than a resolved hull.</p>
<h2>Independent range-crop transfer</h2>
<p>The companion crop is shifted by 13 range samples (487.16 m), retaining the same original epochs, centre deramp and recorded response. Fifteen nonoverlapping 3-second blocks span the fitted 45-second interval. Each fit estimates an extra path, path rate and one complex amplitude from training half-second blocks; stationary clutter and the main ship point responses are explicit nuisance terms. Three acquisition starts are selected using training likelihood only.</p>
<p>The measured companion amplitude is transferred unchanged into the primary crop. Main-target strengths are re-estimated from primary training data, and both predictions are frozen on heldout rows. Range, phase and within-symbol Doppler all follow the same path history. Twelve companion-window predictions are worse than zero prediction; they are retained in the statistics.</p>
<table><tr><th>Band</th><th>Primary error change</th><th>Stationary fits</th><th>Failed companion predictions</th></tr>{transfer_rows}</table>
<img src="companion-transfer.png" alt="Delay, rate, heldout transfer errors and aggregate results">
<h2>Why an apparently distant return matters</h2>
<p>The 17-sample ship crop extends approximately 300 m to either side of its centre. A 500 m delayed point lies outside that crop, but its sidelobes are retained. The range covariance weights those sidelobes differently from a raw-power plot. The calculation below uses the same pulse and whitening operator as the likelihood and compares equal-amplitude points. It does not assume the recorded companion has unit amplitude or that the residual covariance is perfectly thermal.</p>
<img src="companion-leakage.png" alt="Delayed point leakage before and after conditional whitening">
<h2>Motion control with the same statistical model</h2>
<p>One yaw-free motion fit uses stationary nuisance alone. The other additionally profiles one unknown complex companion amplitude per 3-second block; companion path hypotheses remain fixed from the separate training crop. Both use the same physical motion bounds, fixed point support and training-estimated Gaussian scene variance. A faster midpoint forward model proposes search steps for the companion-aware fit; original-row objective values, gradients and feasibility accept its final corrections. The source and both refits pass the original-row stationarity check.</p>
<p>Every motion candidate is then evaluated under each of two identical nuisance projections. Comparisons are within a row below: projected energies from different rows are not compared. Main-target coefficients are learned from training data; heldout samples eliminate only nuisance amplitudes.</p>
<table><tr><th>Common evaluation projection</th><th>Companion-fit error change</th><th>Matched-energy change</th></tr>{motion_rows}</table>
<img src="companion-motion.png" alt="Yaw-free path curves and matched heldout motion comparisons">
<table><tr><th>Band</th><th>Control prediction phase RMS</th><th>Companion-aware motion phase RMS</th></tr>{phase_rows}</table>
<p>These are weighted one-second heldout prediction-phase residuals under the same augmented nuisance projection. They include noise, interference, unresolved scattering and model mismatch. They are not isolated clock errors or direct measurements of physical motion-estimation accuracy.</p>
<h2>Information cost and theoretical scope</h2>
<p>For whitened data y = A(θ)a + Bβ + n, eliminating unknown β applies P = I − BB⁺ to both observations and target templates. This is the nuisance-subspace construction used by <a href="https://api.mountainscholar.org/server/api/core/bitstreams/0bd25191-6792-419a-b526-2b4914839e17/content">Scharf and Friedlander, Matched Subspace Detectors (1994)</a>. Its conditional optimality requires the stated covariance and fixed signal/interference subspaces. Estimated companion paths, real-scene mismatch and correlated interference prevent upgrading that conditional result to a measured global CRLB claim here.</p>
<p>The table reports loss of template information compared with stationary nuisance alone. The second column considers each declared ship point; the last considers the worst complex combination of all eight point templates through the eigenvalues of UᴴPU, with U an orthonormal basis for the original projected scene. Unknown companion phase and gain are thus charged an explicit information cost.</p>
<table><tr><th>Band</th><th>Worst individual point</th><th>Worst scene mode</th></tr>{cost_rows}</table>
<p>The original profiles are unchanged. Nuisance elimination occurs only in the autofocus likelihood. There is no zero-Doppler notch, image-dependent normalization, or subtraction of the companion from rendered data.</p>
<p>If the second component belongs to the same vessel, a validated common physical model could contain more motion information than this nuisance treatment. Its physical identity has not been established; this experiment does not claim optimal use of every return.</p>
{video_html}
<h2>Numerical checks and remaining limits</h2>
<p>The tests check the delayed path's range/phase/Doppler derivatives, transfer of a known complex amplitude, exact nuisance elimination without modifying observations, motion derivatives through the augmented projection, and rejection of missing or overlapping block support. Recorded-data directional gradients in all five bands agree with finite differences to a maximum absolute error of {max(r['absolute_error'] for c in gradient['channels'] for r in c):.3g} across the tested step sizes. A separate analytic regression fixes cancellation error in Gaussian inverse-covariance evaluation when near-coincident point columns require large opposing coefficients. Another regression verifies that approximate search curvature cannot replace the exact objective or its stationarity check. <a href="../../../{args.test_log}">Complete test log</a>.</p>
<p>The independent original-row search also completed. Its final objective agrees with the accelerated route to {abs(equivalence['objective_difference']):.3g}; their fitted path curves differ by at most {equivalence['maximum_curve_difference_by_component'][0]*1e6:.2f} μm. This measures numerical agreement between solvers, not physical positioning accuracy. Both routes pass the same first-order stationarity criterion.</p>
<p>Attempted continuous point-coordinate refinement at fixed yaw did not pass numerical stationarity and is not adopted. The comparison remains conditional on the same eight points per band. The heldout half-second blocks are not an untouched recording; initial support and motion were selected earlier. Physical heading, hull origin, transmitter locations and cross-band phase calibration remain unvalidated. The independent-band likelihood can improve motion prediction without establishing coherent calibration across different scattering responses.</p>
</html>'''
    (args.output/'companion-validation.html').write_text(page)
    print(str(args.output/'companion-validation.html'), flush=True)


if __name__ == '__main__':
    main()
