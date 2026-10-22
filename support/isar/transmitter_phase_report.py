"""Publish the ship range-rate, licensed-site and spatial-phase controls."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('build/isar-20261015'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    paths = {'rates': args.work/'profiled-rate-64.json',
        'independent_sites': args.work/'licensed-site-screen-64/fit.json',
        'coherent_sites': args.work/'coherent-licensed-sites-64.json',
        'known_truth_sites': args.work/'coherent-licensed-sites-known-truth-64.json',
        'spatial_geometry': args.work/'spatial-phase-geometry.json',
        'site_signatures': args.work/'ship-site-signatures.json'}
    evidence = {key: json.loads(path.read_text()) for key, path in paths.items()}
    if not all(evidence[key]['complete'] for key in ('rates', 'independent_sites', 'coherent_sites', 'known_truth_sites')):
        raise ValueError('Finish all declared channel and site controls before publishing')
    test_log = Path('build/isar-tests-20261015-rate-sites.log')
    log = test_log.read_text()
    passed = re.search(r'Ran (\d+) tests', log)
    if not passed or not log.rstrip().endswith('OK'):
        raise ValueError('Numerical tests must pass before publication')
    args.output.mkdir(parents=True, exist_ok=True)
    names, folds = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC'], ['all', 'even to odd', 'odd to even']
    independent, coherent, truth = [evidence[key] for key in ('independent_sites', 'coherent_sites', 'known_truth_sites')]
    selected = coherent['trials'][coherent['training_selected_index']]
    baseline = independent['combinations'][0]['training_selected']
    penalty = 100*(selected['validation']['coupled']['prediction_error']/baseline['test_error']-1)
    best_truth = truth['trials'][truth['training_selected_index']]
    truth_phase_error = float(np.max(abs(np.degrees(np.angle(np.exp(1j*(
        np.array(best_truth['fit']['channel_phase_radians'])-truth['known_truth']['channel_phase_radians'])))))))
    if best_truth['sites'] != truth['known_truth']['sites']:
        raise ValueError('Known-truth site search failed')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    all_independent = independent['combinations'][0]['all']
    axes[0].scatter([t['training_error']/all_independent[0]['training_error'] for t in all_independent],
        [t['test_error']/baseline['test_error'] for t in all_independent], s=30, label='Independent band responses')
    scale = sum(c['options'][0]['trials'][0]['training_energy'] for c in independent['channels'])
    axes[0].scatter([t['fit']['objective']*scale/all_independent[0]['training_error'] for t in coherent['trials']],
        [t['validation']['coupled']['prediction_error']/baseline['test_error'] for t in coherent['trials']],
        marker='x', s=35, label='Common point phases + band phases')
    axes[0].scatter(selected['fit']['objective']*scale/all_independent[0]['training_error'],
        selected['validation']['coupled']['prediction_error']/baseline['test_error'], marker='*', s=150,
        color='#b3382d', label='Training-selected coherent sites')
    axes[0].axhline(1, color='black', lw=.7)
    axes[0].set(xlabel='Training error / best independent training error',
        ylabel='Withheld error / main-site independent error', title='All 48 licensed combinations; fixed motion/support')
    axes[0].legend(fontsize=8)
    errors = sorted(t['fit']['objective'] for t in truth['trials'])
    axes[1].semilogy(np.arange(1, 49), errors, 'o', ms=4)
    axes[1].set(xlabel='Combination rank by training error', ylabel='Training residual / synthetic signal energy',
        title='Known-truth mixed-site control\nCorrect sites recovered; no measured ship values used')
    fig.savefig(args.output/'transmitter-site-controls.png', dpi=160); plt.close(fig)

    rates = evidence['rates']['channels']
    if len(rates) != 5 or any(len(c['trials']) != 3 or not all(t['fit']['converged'] for t in c['trials']) for c in rates):
        raise ValueError('All five channels and three converged range/rate folds are required')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout='constrained')
    offsets = [-.15, 0., .15]
    rate_summary = []
    for c, channel in enumerate(rates):
        values = []
        for j, trial in enumerate(channel['trials']):
            fit = trial['fit']
            for axis, value in zip(axes, [1000*fit['path_rate_mps'], fit['range_origin_m'], trial['prediction_error_change_percent']]):
                axis.plot(c+offsets[j], value, ['o', 's', '^'][j], color=['#176a91', '#b05b25', '#4e8149'][j],
                    label=folds[j] if c == 0 else None)
            values.append({'fold': trial['fold'], 'range_origin_m': fit['range_origin_m'],
                'path_rate_mps': fit['path_rate_mps'], 'matched_gain_db': trial['matched_gain_db'],
                'prediction_error_change_percent': trial['prediction_error_change_percent'],
                'maximum_projected_gradient': fit['maximum_projected_gradient'], 'bound_hit': fit['bound_hit']})
        rate_summary.append({'channel': names[c], 'trials': values})
    for axis in axes:
        axis.set_xticks(range(5), names)
        axis.legend(fontsize=8)
    axes[0].axhline(0, color='black', lw=.7)
    axes[0].set(ylabel='Effective excess-path rate (mm/s)', title='Physical rate changes envelope, phase and Doppler')
    axes[1].set(ylabel='Envelope origin (m)', title='Independent fitted envelope origins')
    axes[2].axhline(0, color='black', lw=.7)
    axes[2].set(ylabel='Withheld prediction-error change (%)', title='Negative is better; complementary range checks')
    fig.savefig(args.output/'transmitter-range-rate.png', dpi=160); plt.close(fig)

    spatial = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for row, assignment in enumerate(('main', 'selected')):
        spatial[assignment] = []
        for label, marker in [('full', 'o'), ('first', 's'), ('second', '^')]:
            path = args.work/f'spatial-phase-{assignment}-{label}-64.json'
            fit = json.loads(path.read_text())['trials'][0]
            slopes = np.degrees(fit['fit']['spatial_phase_gradient_radians_per_m'])
            for axis in range(2):
                axes[row, axis].plot(names, slopes[:, axis], marker+'-', label=label)
                axes[row, axis].axhline(0, color='black', lw=.5)
                axes[row, axis].set(ylabel='Phase slope (degrees / m)',
                    title=f'{assignment.capitalize()} sites / '+['forward', 'port'][axis]+' direction')
                axes[row, axis].legend()
            spatial[assignment].append({'interval': label, 'source': str(path),
                'converged': fit['fit']['converged'], 'channel_phase_radians': fit['fit']['channel_phase_radians'],
                'phase_gradient_radians_per_m': fit['fit']['spatial_phase_gradient_radians_per_m'],
                'withheld_error_penalty_percent': 100*(fit['validation']['coupled']['prediction_error']/
                    fit['validation']['independent']['prediction_error']-1)})
    fig.savefig(args.output/'transmitter-spatial-phase.png', dpi=160); plt.close(fig)

    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'numerical_tests_passed': int(passed[1]),
        'licensed_combinations': len(coherent['trials']), 'roll_pitch_fitted': False, 'correction_adopted': False,
        'independent_training_selected_sites': baseline['sites'], 'coherent_training_selected_sites': selected['sites'],
        'coherent_selected_heldout_penalty_percent': penalty,
        'known_truth_selected_sites': best_truth['sites'], 'known_truth_maximum_phase_error_degrees': truth_phase_error,
        'rates': rate_summary, 'spatial': spatial, 'spatial_geometry': evidence['spatial_geometry']['trials'],
        'remaining': 'No accepted physical phase-centre correction, hull-heading calibration or replacement coherent video from these controls.'}
    projected_image = json.loads((args.output/'stationary-profiled-image.json').read_text())
    result['stationary_projection_control'] = projected_image
    (args.output/'transmitter-phase-controls.json').write_text(json.dumps(result, indent=2)+'\n')
    rate_rows = []
    for c in rates:
        trial = c['trials'][0]; fit = trial['fit']
        rate_rows.append(f'<tr><td>{names[c["channel"]]}</td><td>{fit["range_origin_m"]:.3f}</td>'
            f'<td>{1000*fit["path_rate_mps"]:+.4f}</td><td>{trial["matched_gain_db"]:+.3f}</td>'
            f'<td>{trial["prediction_error_change_percent"]:+.3f}%</td></tr>')
    geometry_rows = ''.join(f'<tr><td>{Path(t["source"]).stem.removeprefix("spatial-phase-").removesuffix("-64")}</td>'
        f'<td>{100*t["bounded_exact_residual_fraction"]:.1f}%</td>'
        f'<td>{t["linear_exact_difference_degrees_per_m"]:.4f}°/m</td></tr>' for t in evidence['spatial_geometry']['trials'])
    def link(path, label):
        return f'<a href="{escape(os.path.relpath(path.resolve(), args.output.resolve()))}">{escape(label)}</a>'
    links = '<ul>'+''.join(f'<li>{link(path, key.replace("_", " "))}</li>' for key, path in paths.items())+f'<li>{link(test_log, "Numerical test log")}</li></ul>'
    links += '<ul>'+''.join(f'<li>{link(Path(t["source"]), "Spatial phase fit: "+assignment+" / "+t["interval"])}</li>'
        for assignment, trials in spatial.items() for t in trials)+'</ul>'
    maximum_gradient = max(t['fit']['maximum_projected_gradient'] for c in rates for t in c['trials'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship autofocus: transmitter range and phase controls</title>
<style>body{{font:16px/1.55 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4d9;padding:8px;text-align:left}}th{{background:#eef2f5}}.note{{background:#fff1cf;padding:16px}}code{{background:#eff2f4;padding:2px 4px}}</style>
<h1>Ship autofocus: transmitter range and phase controls</h1>
<p>Updated {result['updated_utc']}. Capture <code>live-vhf-20261013-193334</code>, 142–172 s.</p>
<p class="note"><strong>Roll and pitch remain fixed.</strong> The tests below examine physical channel path rates, all 48 licensed site combinations, and phase variation across the ship. The fits converge numerically, but no transmitter correction passes the combined prediction and physical-plausibility checks. Existing images remain conditional; no new calibrated video or coordinates are released from these tests.</p>
<p>This follows the <a href="physical-validation.html">physical motion and channel-phase audit</a>. The ship trajectory and 64 point coordinates are held fixed at the earlier control for these comparisons. The motion and support were discovered using this aperture, so withheld tests are conditional validation, not an independent experiment in target discovery.</p>
<h2>1. Channel range origins and path rates</h2>
<p>Each band has two new fitted quantities: a compressed-envelope origin and an excess-path rate. A rate changes range migration, RF phase and within-symbol Doppler together at the original sample-row epochs. Every candidate re-solves the complex reflector coefficients and stationary nuisance. Constant carrier phase is absorbed into those coefficients; it is not counted as independently measured transmitter phase.</p>
<p>Five deterministic rate starts are evaluated for each fit. Training uses alternating one-second blocks plus the final block; untouched interior blocks test prediction. Two additional controls train on alternating whitened range components and predict the complementary components. These are range-vector coordinates, not isolated physical range gates. All 15 fits meet the projected-gradient gate; the largest residual is {maximum_gradient:.3g}.</p>
<img src="transmitter-range-rate.png" alt="Physical path rate, envelope origin and withheld error across three range folds">
<table><tr><th>Channel</th><th>Envelope origin (m)</th><th>Full-range rate (mm/s)</th><th>Withheld matched-energy change (dB)</th><th>Withheld error change</th></tr>{''.join(rate_rows)}</table>
<p>The folds do not provide one stable set of channel corrections. Matched energy and prediction error are both shown: a larger matched output alone can accompany a worse prediction. These effective rates include residual motion and scattering-model error; they must not be reported as measured transmitter clock drift.</p>
<h2>2. All licensed site combinations</h2>
<p>There are 2 × 2 × 2 × 3 × 2 = 48 combinations. For independent band responses, the likelihood factors by band, so all combinations are scored exactly from 11 individual dictionaries. The full-range training selection is the current main-site assignment: Ornata for Seven/SBS/Nine/Ten and Ridge for ABC. Nine, Ten and ABC keep that preference in both training and prediction across the range-split checks; Seven and SBS are less decisive.</p>
<p>The coherent test instead shares each reflector phase across bands, with independent nonnegative strengths and one free constant phase per band. Every combination receives the same five initialization rules and the same support. Mean envelope shifts are aligned for each site, while the full point-dependent carrier phase and Doppler follow the alternate geometry. The coherent training selection is <code>{', '.join(selected['sites'])}</code>, ordered Seven, SBS, Nine, Ten, ABC. Its withheld error is {penalty:.2f}% above the main-site independent-response control. Selecting a different site alone has not repaired the model.</p>
<img src="transmitter-site-controls.png" alt="Training versus withheld errors for all site combinations and known-truth recovery">
<p>The independent and coherent selections differ because their scattering assumptions differ. This is a conditional screening experiment, not joint re-optimization of ship motion under every site assignment. It does not establish surveyed antenna positions.</p>
<p><strong>Known-truth control:</strong> a noiseless synthetic signal uses a deliberately mixed main/alternate assignment, the real epochs, geometry and observation operators, and declared shared reflector phases. It uses no measured ship values. The same search recovers the correct five sites and channel phases to {truth_phase_error:.2g} degrees. Its training residual fraction is {best_truth['fit']['objective']:.3g}; the next site combination has a nonzero residual. This verifies search plumbing and phase conventions under the stated model, not the adequacy of that model for the real ship.</p>
<h2>3. Point-dependent phase and physical antenna errors</h2>
<p>A further diagnostic adds a static affine phase across the horizontal ship coordinates for each non-reference band. This is more flexible than one band phase but has only two extra slope parameters per band. Ten fixes the shared phase-plane gauge. This diagnostic does not move the range envelope or reproduce time-varying geometry, so it is not applied as an imaging correction.</p>
<img src="transmitter-spatial-phase.png" alt="Spatial phase gradients for full, first-half and second-half apertures">
<p>The gradients differ substantially between halves and still leave worse withheld predictions than independent band responses. To check whether they even resemble plausible antenna-location errors, their midpoint phase slopes are compared with the existing correlated location prior: site uncertainty 30/30/75 m, independent per-band antenna uncertainty 5/5/10 m, and receiver uncertainty 15/15/7.5 m in east/north/up. The comparison permits ±3 on each whitened prior coordinate. This is a declared feasibility region, not a statistical confidence region inferred from the ship.</p>
<table><tr><th>Site/interval control</th><th>Phase-gradient norm left unexplained</th><th>Linear versus exact finite-offset slope difference</th></tr>{geometry_rows}</table>
<p>The constrained antenna model leaves most of the fitted phase-gradient norm unexplained. Exact finite-offset evaluation agrees closely with the local derivative at those bounded offsets. This rejects interpreting the entire empirical phase plane as a plausible location correction within the declared prior. It does not rule out a smaller useful geometric correction, or prove which scattering effect dominates the mismatch.</p>
<h2>4. Stationary clutter in the preview images</h2>
<p>The fit profiles stationary nuisance, while earlier previews retain it. A new comparison uses the same cached complex observations and physical templates on both sides. With <code>P = I − QQᴴ</code> for the whitened stationary basis, the profiled image uses <code>z = aᴴPy</code>, <code>I = aᴴPa</code> and power <code>|z|²/I</code>. This is the improvement in the complex least-squares likelihood when adding one target to an unknown stationary background. A numerical test compares it directly with the joint nuisance-plus-target solve, including a nearly stationary target.</p>
<p>There is no fixed-width Doppler notch and no fitted image brightness normalization. The projection retains at least 75.6–82.0% of point-template information in the tested crop, depending on channel. Across these images the main pattern changes little: the coherent peak changes by about 0.004 dB and individual-band peaks by less than 0.18 dB. This control does not explain the large calibration-transfer mismatch. The fixed colour scale is shared by every panel; blank areas were outside the declared dictionary.</p>
<p><a href="stationary-profiled-image.png">Open the full-size comparison</a> · <a href="stationary-profiled-image.json">operator and measurements</a> · <a href="stationary-profiled-image.npz">complex matched statistics</a></p>
<img src="stationary-profiled-image.png" style="max-height:1300px" alt="Five independent channel images and conditional coherent sum, before and after exact stationary-nuisance projection">
<h2>Numerical checks and status</h2>
<p>{passed[1]} support tests pass. New checks verify the exact profiled range/rate residual Jacobian and known-rate recovery, independent synthesis with displaced transmitter and receiver, spatial-phase gradients and known-phase recovery, and antenna-position derivatives of spatial phase. All computations are offline support analysis; production capture, recorded data and live display are unchanged.</p>
<p>The absolute hull heading and centroid remain unvalidated. The supplied AIS contains course over ground, not true heading or antenna-to-hull offsets. These controls have narrowed the proposed calibration explanations; they have not produced a physically validated five-band autofocus solution. The goal remains active.</p>
<p>A public <a href="https://taic.org.nz/inquiry/mo-2020-204">TAIC vessel report</a> supplies a longitudinal and deck drawing for the ship under its former name, Rio De La Plata; the name/IMO history is cross-checked against <a href="https://www.vesselfinder.com/vessels/details/9357951">the vessel record</a>. The <a href="rio-delta-general-arrangement-taic.png">unmodified drawing</a> and <a href="rio-delta-general-arrangement-source.json">source record</a> are retained as a possible hull-shape reference. They do not identify the AIS antenna reference or heading during this capture, and no hull-origin correction has been inferred from them.</p>
<h2>Evidence</h2>{links}
</html>'''
    (args.output/'transmitter-phase-controls.html').write_text(page)
    index = args.output/'analysis.html'
    text = re.sub(r'<p><strong>Latest work:</strong>.*?</p>',
        '<p><strong>Latest work:</strong> <a href="transmitter-phase-controls.html">Transmitter range/rate, all 48 licensed sites and spatial-phase controls</a>. '
        'Roll and pitch remain fixed. Numerical and known-truth checks pass; no new real-data calibration is accepted.</p>', index.read_text(), count=1)
    index.write_text(text)
    session_path = args.output/'session.json'
    session = json.loads(session_path.read_text())
    session.update(updated_utc=result['updated_utc'], pending=[], numerical_support_tests=int(passed[1]),
        transmitter_phase_report=str(args.output/'transmitter-phase-controls.html'),
        transmitter_phase_results=str(args.output/'transmitter-phase-controls.json'),
        status='Active. Channel path-rate, all licensed site combinations and spatial-phase tests are complete. Numerical controls pass; no physical calibration accepted.')
    session['stationary_projected_image'] = str(args.output/'stationary-profiled-image.png')
    session['vessel_drawing_source'] = str(args.output/'rio-delta-general-arrangement-source.json')
    session_path.write_text(json.dumps(session, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
