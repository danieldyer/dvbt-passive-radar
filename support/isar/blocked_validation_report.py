"""Publish the stronger temporal and scene-profiled autofocus controls."""
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
    parser.add_argument('--work', type=Path, default=Path('build/isar-20261015/extended-scene-142-172'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    core = read(args.work/'core-blocks-1s/fit.json')
    nested = read(args.work/'core-nested-blocks-1s/fit.json')
    sparse = read(args.work/'core-support-blocks-1s.json')
    extended = read(args.work/'extended-support-blocks-1s.json')
    regions = read(args.work/'prediction-regions.json')
    phases = {n: read(args.work/f'core-support-{n}-phase/fit.json') for n in (64, 128)}
    previous = read(Path('build/isar-20261015/channel-alignment-142-172/phase-audit-full-scene-2m/audit.json'))
    dictionary = read(args.work/'dictionary/dictionary.json')
    if not all(d['complete'] for d in [core, nested, sparse, extended, dictionary, *phases.values()]):
        raise ValueError('Finish all declared experiments before publishing')
    log = Path('build/isar-tests-20261015-blocked-final.log').read_text()
    tests = re.search(r'Ran (\d+) tests', log)
    if not tests or not log.rstrip().endswith('OK'):
        raise ValueError('Finish the numerical regression suite')
    for a, b in zip(sparse['trials'], extended['trials']):
        np.testing.assert_array_equal(a['points_body_m'], b['points_body_m'])
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    def link(path, label):
        return f'<a href="{escape(os.path.relpath(path.resolve(), args.output.resolve()))}">{escape(label)}</a>'
    def sparse_trial(count):
        return next(t for t in sparse['trials'] if t['count'] == count)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for key, label in [('gcv', 'GCV dense: interior blocks only'), ('nested_ridge', 'Nested ridge: all test blocks'),
                       ('sparse_64', '64 shared locations: all test blocks'), ('sparse_128', '128 shared locations: all test blocks')]:
        region = 'interior' if key == 'gcv' else 'all'
        values = [100*c['models'][key][region]['explained_fraction'] for c in regions['channels']]
        axes[0].plot(names, values, 'o-', label=label)
    axes[0].axhline(0, color='black', linewidth=.6)
    axes[0].set(ylabel='Explained held-out complex energy (%)', title='Fixed motion; no phase correction')
    axes[0].legend(fontsize=8)
    for key, label in [('gcv', 'GCV dense'), ('nested_ridge', 'Nested ridge'), ('sparse_128', '128 locations')]:
        values = [1-c['models'][key]['all']['explained_fraction'] for c in regions['channels']]
        axes[1].semilogy(names, values, 'o-', label=label)
    axes[1].axhline(1, color='black', linewidth=.6)
    axes[1].set(ylabel='Prediction error / stationary-only error', title='All test blocks, including terminal 1 s\nBelow 1 improves prediction; above 1 makes it worse')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'blocked-scene-prediction.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    before = previous['variants']['independent']['channels']
    retained = [c['trials'][0]['before']['target_information_retained'] for c in before]
    axes[0].bar(names, 100*np.array(retained))
    axes[0].set(ylabel='Template energy outside stationary subspace (%)',
        title='Earlier frozen dense template: only 0.05–0.21% survives\nLarge stationary components can artificially anchor phase')
    axes[1].plot(names, [c['trials'][0]['before']['one_second_phase_metrics']['phase_rms_degrees_by_point'][0] for c in before],
        'o--', label='Earlier alternating 17 ms bins')
    axes[1].plot(names, [c['models']['gcv']['interior']['phase_rms_degrees'] for c in regions['channels']],
        'o-', label='GCV dense / interior 1 s blocks')
    axes[1].plot(names, [c['models']['nested_ridge']['interior']['phase_rms_degrees'] for c in regions['channels']],
        'o-', label='Nested ridge / interior 1 s blocks')
    axes[1].set(ylabel='One-second prediction-phase RMS (degrees)', title='Different validation splits; not an isolated motion error')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'blocked-phase-validation.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout='constrained')
    phase_rows, phase_summary = [], []
    for row, (count, model) in enumerate(phases.items()):
        for fold, label in enumerate(['All ranges', 'Even → odd range components', 'Odd → even range components']):
            trials = [c['trials'][fold] for c in model['channels']]
            axes[row, 0].plot(names, [t['matched_gain_db'] for t in trials], 'o-', label=label)
            axes[row, 1].plot(names, [t['prediction_error_change_percent'] for t in trials], 'o-', label=label)
            for c, t in enumerate(trials):
                d = t['refinement']
                phase_summary.append({'points': count, 'channel': c, 'fold': label, 'matched_gain_db': t['matched_gain_db'],
                    'prediction_error_change_percent': t['prediction_error_change_percent'], 'converged': d['converged'],
                    'kkt_residual': d['maximum_kkt_residual'], 'phase_bound_hit': d['phase_bound_hit']})
                phase_rows.append(f'<tr><td>{count}</td><td>{names[c]}</td><td>{label}</td><td>{t["matched_gain_db"]:+.4f}</td>'
                    f'<td>{t["prediction_error_change_percent"]:+.3f}%</td><td>{d["maximum_kkt_residual"]:.2g}</td><td>{d["phase_bound_hit"]}</td></tr>')
        for col in (0, 1):
            axes[row, col].axhline(0, color='black', linewidth=.7)
        axes[row, 0].set(ylabel='Matched-energy change (dB)', title=f'{count} points, amplitudes refitted at every phase candidate')
        axes[row, 1].set(ylabel='Prediction-error change (%)', title='Negative improves prediction; positive degrades it')
        axes[row, 0].legend(fontsize=8)
    fig.savefig(args.output/'profiled-scene-phase-transfer.png', dpi=160)
    plt.close(fig)
    if not all(t['converged'] for t in phase_summary):
        raise ValueError('A phase refinement failed the numerical convergence check')
    comparison_rows = []
    for c in range(5):
        m = regions['channels'][c]['models']
        t = nested['channels'][c]['tuning']
        comparison_rows.append(f'<tr><td>{names[c]}</td><td>{100*m["gcv"]["interior"]["explained_fraction"]:.2f}%</td>'
            f'<td>{100*m["nested_ridge"]["all"]["explained_fraction"]:.2f}%</td><td>{t["relative_strengths"][t["chosen_index"]]:.4g}</td>'
            f'<td>{t["grid_edge"]}</td><td>{100*m["sparse_128"]["all"]["explained_fraction"]:.2f}%</td></tr>')
    methods = [(args.work/'core-blocks-1s/fit.json', 'GCV whole-block result'),
        (args.work/'core-nested-blocks-1s/fit.json', 'Nested penalty selection'),
        (args.work/'core-support-blocks-1s.json', 'Compact shared-location models'),
        (args.work/'extended-support-blocks-1s.json', 'Expanded-support control'),
        (args.work/'prediction-regions.json', 'Interior / terminal / all-region statistics'),
        *[(args.work/f'core-support-{n}-phase/fit.json', f'{n}-point scene-profiled phase tests') for n in (64, 128)]]
    verification = {'numerical_tests_passed': int(tests[1]), 'phase_fits': len(phase_summary),
        'maximum_phase_kkt_residual': max(t['kkt_residual'] for t in phase_summary),
        'exact_dictionary_probe_relative_error': max(c['forward_probe_relative_error'] for c in dictionary['channels']),
        'maximum_compact_score_change_when_extending_support': max(abs(a['heldout_explained_fraction']-b['heldout_explained_fraction']) for a, b in zip(sparse['trials'], extended['trials']))}
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'verification': verification,
        'inputs': [str(p) for p, _ in methods], 'phase_tests': phase_summary,
        'correction_adopted': False, 'roll_pitch_fitted': False,
        'status': 'The earlier alternating-sample phase interpretation is superseded. Whole-block prediction and scene-profiled phase controls do not yet validate a transferable correction, accurate physical heading, hull centroid, or coherent five-band calibration.',
        'next': 'Use observed aperture edges when testing offline interpolation, explicitly validate the stationary/target decomposition, and constrain physical motion before repeating channel range/phase calibration. Keep forecasting controls separate. Do not select a correction from the largest training focus score.'}
    (args.output/'blocked-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    links = '<ul>'+''.join(f'<li>{link(p, label)}</li>' for p, label in methods)+'</ul>'
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship autofocus: blocked prediction and phase validation</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;line-height:1.55;color:#18232d}} table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{padding:8px;border:1px solid #ccd4d9;text-align:left}} th{{background:#eef2f5}} img{{max-width:100%}} .note{{background:#fff1cf;padding:16px}} code{{background:#eff2f4;padding:2px 4px}}</style>
<h1>Ship autofocus: stronger validation exposes an overconfident phase result</h1>
<p class="note"><strong>The previous 0.6–2.0° phase residuals are not evidence of accurate autofocus.</strong> They came from a flexible scene fitted on alternating closely spaced samples. Withholding entire one-second blocks exposes instability, and most frozen-template energy was indistinguishable from stationary background. No new phase screen, transmitter position, roll or pitch correction has been adopted. All work here is offline analysis.</p>
<p>Capture: <code>live-vhf-20261013-193334</code>, ship aperture 142–172 s. Physical translation/yaw, nominal sites, envelope origins and existing channel gains are unchanged. The original-row forward operator, symbol moments, whitening and coherent averaging are retained exactly. This report corrects the interpretation of the <a href="distributed-scene-audit.html">earlier alternating-sample audit</a>; its measured numbers remain available.</p>
<h2>1. Whole-block prediction</h2>
<p>Even one-second blocks fit the scene; odd blocks test it. The final odd block is a one-second extrapolation beyond the last training sample. Interior blocks instead test interpolation across missing observations. Those are different tasks, so both are shown. The extreme GCV failure is dominated by the terminal extrapolation: pooled prediction error is about {1-core['trials'][0]['heldout_explained_fraction']:.0f} times the stationary-only baseline. Interior explained fractions are approximately 35%, 37%, 64%, 78% and 66%; these also fall below the earlier alternating-sample results.</p>
<img src="blocked-scene-prediction.png" alt="Blocked scene prediction errors and explained energy">
<p>Nested ridge selection uses blocks 0,4,8,… for an inner fit and 2,6,10,… for penalty validation. It then refits all even blocks using the selected absolute penalty. The outer odd blocks never choose coefficients or regularization. This removes catastrophic extrapolation but is <strong>not a successful universal model</strong>: ABC's inner validation selects the maximum tested penalty, effectively rejecting the target prediction. Its small phase statistic is therefore not evidence of useful recovered signal.</p>
<table><tr><th>Channel</th><th>GCV interior explained</th><th>Nested all-block explained</th><th>Inner relative penalty</th><th>Penalty at grid edge</th><th>128-point all-block explained</th></tr>{''.join(comparison_rows)}</table>
<p>The compact model selects physical locations shared by all channels while allowing independent complex scattering coefficients in each band. Each greedy step maximizes the exact reduction in summed whitened least-squares error; it does not equalize channel power. The declared 8, 16, 32, 64 and 128-point models are all retained. The 128-point model explains {100*sparse_trial(128)['heldout_explained_fraction']:.2f}% of the outer test energy. These are conditional prediction scores, not image-SNR gains or proof that the chosen points are physical reflectors.</p>
<p>Expanding the search from 4,961 to 6,536 locations includes the weaker negative-forward response outside the earlier rectangle. Every selected location remains unchanged through 128 steps and the score changes by less than {verification['maximum_compact_score_change_when_extending_support']:.2g}. This particular model is therefore not rescued by that omitted region; it does not establish the identity of the weaker return.</p>
<h2>2. Why the earlier phase test was overconfident</h2>
<img src="blocked-phase-validation.png" alt="Stationary overlap and phase statistics under stronger validation">
<p>Only 0.05–0.21% of the earlier independent-band template energy survives the stationary-nuisance projection. A flexible scene can contain large nearly stationary components that cancel against fitted background. Freezing that scene and varying only its phase can make tiny phase changes look very costly because they rotate those large components out of the stationary subspace. Thus a tiny fitted phase update need not mean the physical motion is already accurate.</p>
<p>All prediction-phase RMS values include noise, unresolved scattering, interference and model error. They are neither isolated motion-error measurements nor a calibrated CRLB. The data establish the failure of the earlier validation claim; they do not, by themselves, identify which remaining physical error dominates.</p>
<h2>3. Refit the scene at every phase candidate</h2>
<p>The new variable-projection control minimizes <code>min(a,b) ||y − exp(iφ) A a − B b||²</code> at every phase candidate, with complex reflector amplitudes <code>a</code> and stationary nuisance <code>b</code>. It uses three-second spline knots, removes constant/linear alignment terms, and bounds the diagnostic phase to ±0.8 rad. The 64- and 128-point supports were selected from training data only. Separate range folds test transfer to complementary whitened range components.</p>
<img src="profiled-scene-phase-transfer.png" alt="Matched-energy and prediction-error changes for profiled scene phase tests">
<p>All {verification['phase_fits']} numerical solves meet the stationarity gate, with maximum KKT residual {verification['maximum_phase_kkt_residual']:.3g}. Nevertheless the improvements are inconsistent across channels and range folds. All-range matched-energy changes are only about −0.47 to +0.22 dB, and every all-range fit reaches the phase bound somewhere. Large gains on a weak complementary-range predictor are not persuasive when its absolute prediction error worsens. These controls do not justify applying a new correction to the recorded data or images.</p>
<details><summary>Complete phase-transfer table</summary><table><tr><th>Points</th><th>Channel</th><th>Range fold</th><th>Matched-energy change (dB)</th><th>Prediction-error change</th><th>KKT residual</th><th>Phase bound hit</th></tr>{''.join(phase_rows)}</table></details>
<h2>4. Numerical checks and interpretation limits</h2>
<p>{verification['numerical_tests_passed']} support tests pass. New checks cover exhaustive verification of every greedy selection step, different band dictionaries, column-unit and phase invariance, exact QR equivalence, complex ridge solutions versus augmented least squares, outer-data isolation, analytic phase gradients and recovery of an injected phase curve across time gaps. Exact cached forward responses agree with the established operator to {verification['exact_dictionary_probe_relative_error']:.3g} relative error. A roundoff issue in zero circular variance was also fixed.</p>
<p>Shared-location models with independent complex band responses cannot determine arbitrary constant channel phases: replacing a channel gain by <code>g·exp(iδ)</code> and all that band's reflector coefficients by <code>a·exp(−iδ)</code> leaves its predicted data unchanged. A shared complex scene breaks this ambiguity only through an additional scattering assumption. This is separate from any phase reference preserved by the capture pipeline and from geometric, time-varying antenna effects. No arbitrary band phase from this diagnostic is a transmitter calibration.</p>
<p>Likewise, constant hull heading and a counter-rotation of unknown body-fixed scattering locations form an exact coordinate ambiguity without a hull-shape or independent heading reference. Numerical convergence alone does not resolve it. The supplied AIS track contains course over ground, not true heading or antenna-to-hull offsets.</p>
<p><strong>Next validation:</strong> distinguish offline interpolation with observed aperture edges from forecasting, validate the target/stationary decomposition, and revisit the physical motion fit before accepting channel range/phase calibration. The current trial videos remain conditional comparisons; accurate hull-centred, calibrated five-band autofocus is unfinished.</p>
<h2>Reproduction and sources</h2>{links}
<p>Method references: <a href="https://www.mins.ee.ethz.ch/pubs/p/allertonhb2012">Heckel &amp; Bölcskei: joint sparsity with different measurement matrices</a>; <a href="https://arxiv.org/abs/1910.10976">Kim et al.: orthogonal least squares</a>; <a href="https://www.cs.umd.edu/~oleary/software/varpro.pdf">O'Leary &amp; Rust: variable projection</a>. Their assumptions are not asserted to hold for this dense radar dictionary; the real-data checks above provide the relevant empirical evidence.</p>
</html>'''
    (args.output/'blocked-validation.html').write_text(page)
    index = args.output/'analysis.html'
    page = index.read_text()
    page = re.sub(r'<p><strong>Latest work:</strong>.*?</p>', '<p><strong>Latest work:</strong> <a href="blocked-validation.html">Whole-block prediction and scene-profiled phase checks</a>. These supersede the earlier small-residual interpretation. No new phase calibration is adopted. The <a href="amplitude-autofocus.html">earlier report</a> contains conditional trial videos.</p>', page, count=1)
    index.write_text(page)
    state_path = args.output/'session.json'
    state = read(state_path)
    state.update(updated_utc=result['updated_utc'], status='Active. Stronger validation exposed unstable dense predictions and a frozen-template phase ambiguity; no new correction is accepted.',
        blocked_validation_report=str(args.output/'blocked-validation.html'), blocked_validation_results=str(args.output/'blocked-validation.json'),
        pending=[], remaining=[
            'Keep roll and pitch fixed. Current absolute heading, hull centroid and transmitter calibration remain unvalidated.',
            'The previous 0.6–2 degree dense prediction-phase residuals do not establish autofocus accuracy; whole-block prediction and frozen-template stationary overlap invalidate that inference.',
            'Thirty scene-profiled phase solves converged, but transfer gains are inconsistent and phase bounds are reached. No new phase correction is adopted.',
            'The nested dense penalty test avoids catastrophic prediction but rejects ABC; compact independent-band models still leave substantial error. Validate interior interpolation and target/background separation before more physical motion fitting.',
            'Existing conditional videos use the earlier physical motion and gains. The calibrated hull-centred, genuinely coherent five-channel imaging objective is unfinished.',
            'True AIS heading and antenna-to-hull offsets remain an unanswered optional data request.'
        ])
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
