"""Report phase-only autofocus after continuous physical point refinement."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('limited', 'expanded', 'support', 'test-log', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    paths = (args.limited, args.expanded)
    controls = [json.loads(path.read_text()) for path in paths]
    support = json.loads(args.support.read_text())
    log = args.test_log.read_text()
    tests = re.search(r'Ran (\d+) tests', log)
    if not tests or not log.rstrip().endswith('OK'):
        raise ValueError('Numerical tests must pass before publication')
    if not all(c['complete'] and c['observe_aperture_edges'] for c in controls):
        raise ValueError('Complete both interpolation controls')
    if controls[0]['source_dictionary_sha256'] != controls[1]['source_dictionary_sha256']:
        raise ValueError('Both phase bounds must use the same physical data and dictionary')
    dictionary_path = Path(controls[0]['source_dictionary'])/'dictionary.json'
    if fingerprint(dictionary_path) != controls[0]['source_dictionary_sha256']:
        raise ValueError('The physical response cache has changed')
    dictionary = json.loads(dictionary_path.read_text())
    if (Path(dictionary['source_support']).resolve() != args.support.resolve()
            or dictionary['identity']['source_sha256'][str(args.support.resolve())] != fingerprint(args.support)):
        raise ValueError('Physical point source does not match its cache')
    baseline = support['after']['independent']['prediction_error']
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    rows, comparisons = [], []
    for control, label in zip(controls, ['0.8 rad limit', 'π rad limit']):
        trials = [t for c in control['channels'] for t in c['trials']]
        if len(trials) != 15 or not all(t['refinement']['converged'] for t in trials):
            raise ValueError('Complete all five channels and three range folds')
        reproduced = sum(c['trials'][0]['before']['prediction_error'] for c in control['channels'])
        if not np.isclose(reproduced, baseline, rtol=1e-9):
            raise ValueError('Uncorrected phase audit changes the established physical prediction')
        comparisons.append({'label': label, 'baseline_error': reproduced,
            'corrected_error': sum(c['trials'][0]['after']['prediction_error'] for c in control['channels']),
            'maximum_kkt_residual': max(t['refinement']['maximum_kkt_residual'] for t in trials),
            'phase_bound_hits': sum(t['refinement']['phase_bound_hit'] for t in trials)})
        for channel in control['channels']:
            full = channel['trials'][0]
            phases = np.array([t['phase_radians'] for t in channel['trials']])
            rows.append({'channel': names[channel['channel']], 'channel_index': channel['channel'],
                'limit': label, 'matched_gain_db': full['matched_gain_db'],
                'prediction_error_change_percent': full['prediction_error_change_percent'],
                'phase_rms_before_degrees': full['before']['one_second_phase_metrics']['phase_rms_degrees_by_point'][0],
                'phase_rms_after_degrees': full['after']['one_second_phase_metrics']['phase_rms_degrees_by_point'][0],
                'range_fold_phase_difference_rms_degrees': float(np.degrees(np.sqrt(np.mean((phases[1]-phases[2])**2)))),
                'full_range_phase_bound_hit': full['refinement']['phase_bound_hit'],
                'fold_error_changes_percent': [t['prediction_error_change_percent'] for t in channel['trials']]})
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'sources_sha256': {str(p): fingerprint(p) for p in (*paths, args.support, args.test_log)},
        'tests_passed': int(tests[1]), 'controls': comparisons, 'rows': rows,
        'maximum_cache_operator_discrepancy': max(c['forward_probe_relative_error'] for c in dictionary['channels']),
        'correction_adopted': False, 'absolute_heading_validated': False,
        'hull_centroid_validated': False, 'cross_band_calibration_validated': False}
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    for ax, control, label in zip(axes, controls, ['0.8 rad phase limit', 'π rad phase limit']):
        for index, legend in enumerate(['All components', 'Even → odd', 'Odd → even']):
            values = [c['trials'][index]['prediction_error_change_percent'] for c in control['channels']]
            ax.bar(np.arange(5)+(index-1)*.25, values, width=.25, label=legend)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xticks(range(5), names)
        ax.set(title=label, ylabel='Withheld prediction-error change (%)')
        ax.legend(fontsize=9)
    fig.savefig(args.output/'refined-phase-prediction.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True, layout='constrained')
    for ax, channel, name in zip(axes, controls[1]['channels'], names):
        for trial, label in zip(channel['trials'], ['All components', 'Train even', 'Train odd']):
            ax.plot(channel['time_s'], np.degrees(trial['phase_radians']), label=label)
        ax.axhline(0, color='black', linewidth=.5)
        ax.set(ylabel=name+'\nDegrees', ylim=(-185, 185))
    axes[0].legend(ncol=3)
    axes[0].set_title('Estimated residual phase with the expanded limit; no correction applied')
    axes[-1].set_xlabel('Seconds from capture start')
    fig.savefig(args.output/'refined-phase-histories.png', dpi=160); plt.close(fig)
    table = ''.join(f'<tr><td>{r["channel"]}</td><td>{r["phase_rms_before_degrees"]:.2f}°</td>'
        f'<td>{r["phase_rms_after_degrees"]:.2f}°</td><td>{r["matched_gain_db"]:+.3f}</td>'
        f'<td>{r["prediction_error_change_percent"]:+.2f}%</td>'
        f'<td>{r["range_fold_phase_difference_rms_degrees"]:.1f}°</td></tr>' for r in rows if r['limit'] == 'π rad limit')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship ISAR: refined-support phase autofocus</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #ccd4d9;text-align:left}}th{{background:#eef2f5}}.note{{background:#fff2cf;padding:16px}}</style>
<h1>Phase autofocus after physical reflector refinement</h1>
<p class="note">All 30 phase fits pass numerical stationarity. Widening the phase search removes clipping in all five full-range fits, but four then predict withheld data worse. ABC improves on full-range prediction but fails complementary-range transfer. No new phase correction is adopted. Absolute heading, hull centroid and coherent five-band calibration remain unresolved.</p>
<h2>What changed in this control</h2>
<p>The earlier test used fixed two-metre grid locations and every full-range phase fit hit its 0.8-radian limit. This test uses the 64 continuously refined physical locations from <a href="scattering-validation.html">the independent-band support fit</a>. The physical path/yaw trajectory, range origins, antenna sites and roll/pitch remain fixed. Both 0.8-radian and π-radian search bounds are declared and reported.</p>
<p>Each channel has a smooth residual phase with three-second spline knots. Constant and linear phase are removed using the same training-time convention in all range folds. Complex reflector coefficients and stationary clutter are re-solved at every phase trial. Independent complex responses in each band avoid assuming a common scattering phase across different carrier frequencies.</p>
<p>Odd interior one-second blocks are withheld; both aperture edges are observed. Additional folds train on alternate whitened range components and test the others. Data and prediction receive the identical stationary-nuisance projection. These are conditional tests: original support and motion discovery used this aperture. No phase or target coefficient is refitted on validation samples.</p>
<h2>Measured residual phase and prediction</h2>
<table><tr><th>Channel</th><th>Before: phase RMS</th><th>After: phase RMS</th><th>Matched-energy change (dB)</th><th>Prediction-error change</th><th>Complementary-range phase disagreement RMS</th></tr>{table}</table>
<p>The table uses the expanded limit. Phase RMS is measured from one-second projections of withheld data onto the frozen multi-reflector prediction. It includes glint, noise and scene mismatch; it is not a pure motion-error measurement. Matched energy can increase while squared prediction error worsens, as SBS demonstrates. It is therefore insufficient to select a correction using image sharpness or matched energy alone.</p>
<img src="refined-phase-prediction.png" alt="Prediction error changes for each channel, phase bound and range fold">
<p>Negative error change is better. The three folds are reported together; the favourable ABC full-range result is not adopted while its even-to-odd component test worsens.</p>
<img src="refined-phase-histories.png" alt="Estimated residual phase histories differ between complementary range components">
<p>These curves are model residual phases; a compensating operation would use their negative. They are evaluated on retained 16-symbol field bins. No phase screen has been applied to archived profiles, original sample rows, production data or the videos. Original-row delay/Doppler consistency would require a separate check before adoption.</p>
<h2>Numerical evidence</h2>
<p>{int(tests[1])} support tests pass, including exact profiled gradients and recovery of a known 1.6-radian phase history with unchanged complex scene coefficients and strong stationary nuisance. The regenerated physical response cache agrees with the established original-row operator to {result['maximum_cache_operator_discrepancy']:.3g} relative error. Its uncorrected withheld error reproduces the prior physical-point result ({baseline:.6f}) within 10⁻⁹ relative tolerance. The largest phase-fit KKT residual is {max(r['maximum_kkt_residual'] for r in comparisons):.3g}. These establish local numerical accuracy, not a global-optimum or physical-calibration certificate.</p>
<h2>Production phase-reference trace</h2>
<p>The matched-profile shader forms the noise-weighted product <code>(Y − H X (1+B)) H* X*</code>, with finite-reference prefix and time-moment corrections. Here H is the fitted response, X is the aligned reconstructed waveform and B is the frozen stationary transfer. A common acquisition rotation of Y and H cancels algebraically. The decoder equalizer copies the scientific measured carriers unchanged. This source-code trace does not establish a known physical scattering phase across bands: direction-dependent antenna response, the direct-path reference and target scattering still enter that interpretation.</p>
<p>Relevant code: <a href="../../../shaders/matched_pack.comp">reference/error packing</a>, <a href="../../../shaders/matched_product.comp">matched product</a>, <a href="../../../shaders/frame_equalize.comp">measurement preservation</a>, <a href="../../../src/live_archive.cpp">saved profile coordinates and response</a>. This check is algebraic and source-based; the production pipeline was not changed in this experiment.</p>
<h2>Evidence and remaining requirements</h2>
<p><a href="refined-phase-validation.json">Summary JSON</a> · <a href="../../../{args.limited}">Limited-bound fits</a> · <a href="../../../{args.expanded}">Expanded-bound fits</a> · <a href="../../../{args.test_log}">Test log</a> · <a href="joint-point-validation.html">Joint channel range/phase and anchor controls</a></p>
<p>The supplied AIS has position, speed and course, but no true heading or antenna-to-hull offsets. Those missing observations would help anchor the body frame; they do not by themselves resolve the separate cross-band scattering/calibration ambiguity. Existing videos remain conditional illustrations, not validated hull-centred coherent reconstructions. The full imaging goal is still open.</p></html>'''
    (args.output/'refined-phase-validation.html').write_text(page)
    (args.output/'refined-phase-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'tests_passed': int(tests[1]), 'controls': comparisons}))


if __name__ == '__main__':
    main()
