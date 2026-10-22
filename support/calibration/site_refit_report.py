"""Report calibration comparisons made with the audited common-source inputs.

Historical objective values are deliberately not mixed into these plots. A
small aggregate energy loss is not an acceptance test for a five-band model.
"""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from geometry_evidence import render as render_geometry


NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return '<table><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+\
        '</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+
                       '</tr>' for row in rows)+'</table>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('build/calibration-20261013'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/calibration_20261013'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    geometry_section=render_geometry(args.output)
    screen = read(args.work/'shared-seed-sites/progress.json')
    ranking = screen['results']
    ranking.sort(key=lambda result: result['cost'])
    minimum = ranking[0]['cost']
    figure, axis = plt.subplots(figsize=(13, 4.8), layout='constrained')
    axis.bar(range(len(ranking)), [r['cost']-minimum for r in ranking],
             color=['#278361' if r['converged'] else '#3876a8' for r in ranking])
    axis.set_xticks(range(len(ranking)), [r['code'] for r in ranking], rotation=90, fontsize=8)
    axis.set(ylabel='Objective above best screening result', xlabel='Site assignment in channel order',
             title='Same ten targets and revised observations in all 48 licensed assignments\n'
                   'Blue: evaluation budget reached; green: solver convergence criterion met')
    figure.savefig(args.output/'shared-seed-site-ranking.png', dpi=150)
    plt.close(figure)
    refined = []
    for code in ('00012', '00010', '00000', '00002'):
        path = args.work/'shared-seed-fit'/f'{code}-joint-refined.json'
        if path.exists():
            result = read(path)
            refined.append({'code': code, 'source': str(path), 'cost': result['cost'],
                            'converged': result['converged'], 'evaluations': result['evaluations'],
                            'zero_bands': result['zero_amplitude_channels_by_target']})
    branches = read(args.work/'shared-seed-fit/00012-phase-branches.json')
    validations = []
    for title, filename in (
            ('Main-site control', 'shared-seed-withheld.json'),
            ('Ten at Eyre', 'shared-seed-00012-withheld.json'),
            ('Five centres, main sites', 'shared-seed-five-centres-withheld.json'),
            ('Receiver uncertainty', 'shared-seed-receiver-withheld.json'),
            ('Five centres, Ten at Eyre', 'shared-seed-00012-five-centres-withheld.json'),
            ('Shared range origins', 'shared-seed-range-origin-withheld.json'),
            ('Band receive centres, 1 m prior', 'shared-seed-receiver-bands-withheld.json'),
            ('Band receive centres, 0.5 m prior', 'shared-seed-receiver-bands-half-withheld.json')):
        path = args.output/filename
        if path.exists():
            records = [r for r in read(path)['targets'] if r['label'] in ('A', 'K', 'R')]
            validations.append((title, filename, records))
    rows = [(f'{title} · {r["label"]}', r) for title, _, records in validations for r in records]
    figure, axes = plt.subplots(1, 3, figsize=(14, max(7,.5*len(rows)+1)), layout='constrained')
    labels = [name for name, _ in rows]
    phase = np.array([r['phase_contrast_degrees'] for _, r in rows])
    ratios = np.array([r['shared_to_independent_amplitude_ratio'] for _, r in rows])
    y = np.arange(len(rows))
    axes[0].barh(y, [r['heldout_loss_db'] for _, r in rows], color='#3876a8')
    axes[0].set(yticks=y, yticklabels=labels, xlabel='Loss versus independent bands (dB)',
                title='Combined energy\ncan hide a missing band', ylim=(len(rows)-.5,-.5))
    p = axes[1].imshow(phase, cmap='RdBu_r', vmin=-180, vmax=180, aspect='auto')
    axes[1].set(xticks=range(5), xticklabels=NAMES, yticks=y, yticklabels=[],
                title='Held-out phase contrast\nrelative to SBS')
    figure.colorbar(p, ax=axes[1], label='Degrees')
    p = axes[2].imshow(ratios, cmap='viridis', vmin=0, vmax=1.5, aspect='auto')
    axes[2].set(xticks=range(5), xticklabels=NAMES, yticks=y, yticklabels=[],
                title='Shared / independent amplitude')
    figure.colorbar(p, ax=axes[2], label='Amplitude ratio', extend='max')
    figure.suptitle('Whole-target and alternating-time-block checks; all five bands shown\n'
                   'Road discovery remains conditional on the full recording')
    figure.savefig(args.output/'shared-seed-model-validation.png', dpi=150)
    plt.close(figure)
    validation_rows = []
    for title, _, records in validations:
        for r in records:
            converged = all(r[k] for k in ('calibration_converged', 'target_converged', 'independent_converged'))
            validation_rows.append([title, r['label'], f'{r["heldout_loss_db"]:.3f}',
                ', '.join(f'{v:+.1f}' for v in r['phase_contrast_degrees']),
                ', '.join(f'{v:.2f}' for v in r['shared_to_independent_amplitude_ratio']),
                'yes' if converged else 'no'])
    receiver = read(args.work/'shared-seed-fit/00012-receiver.json')
    five = read(args.work/'shared-seed-fit/00002-five-centres.json')
    five_sigma = five['information']['channel_position_standard_deviation_m']
    alternative_five = read(args.work/'shared-seed-fit/00012-five-centres.json')
    alternative_sigma = alternative_five['information']['channel_position_standard_deviation_m']
    covariance = np.asarray(receiver['information']['data_motion_receiver_prior_posterior_covariance_normalized'])
    receiver_sigma = np.sqrt(np.diag(covariance)[-3:])*receiver['receiver_prior_sigma_m']
    extended = read(args.output/'shared-seed-extended-response.json')
    extended_rows = []
    for target in extended['targets']:
        baseline = target['models'][0]['variants'][0]['shared_prediction_error']
        for model in target['models']:
            variants = model['variants']
            errors = np.array([v['shared_prediction_error'] for v in variants])
            weakest = variants[0]['phase_information']['singular_values_per_radian'][-1]
            extended_rows.append([target['label'], model['grid'],
                f'{100*(baseline-errors[0])/baseline:+.2f}%',
                f'{100*(errors.max()-errors.min())/baseline:.2f}%', f'{weakest:.3f}'])
    slow = read(args.output/'shared-seed-slow-paths.json')
    slow_rows = []
    for target in slow['targets']:
        changes = [100*m['heldout_error_reduction_fraction'] for m in target['common_metrics']]
        slow_rows.append([target['label'], f'{target["median_common_path_rate_mps"]:+.3f}',
                          ', '.join(f'{c:+.2f}%' for c in changes)])
    association = read(args.work/'shared-seed-fit/00012-associated.json')
    carrier = read(args.output/'carrier-frequency-audit.json')
    range_origin = read(args.work/'shared-seed-fit/00012-range-origin.json')
    range_checks = read(args.output/'shared-seed-range-origin-withheld.json')['targets']
    receive_trials = []
    for name in ('00012-receiver-bands', '00012-receiver-bands-half-refined'):
        path = args.work/'shared-seed-fit'/f'{name}.json'
        if path.exists():
            result = read(path)
            receive_trials.append({'source': str(path), 'cost': result['cost'],
                'converged': result['converged'],
                'prior_sigma_m': result['receiver_channel_prior_sigma_m'],
                'common_correction_enu_m': result['receiver_correction_enu_m'],
                'offsets_enu_m': result['receiver_channel_offsets_enu_m'],
                'zero_bands': result['zero_amplitude_channels_by_target']})
    receive_rows = []
    for trial in receive_trials:
        for name, offset in zip(NAMES, trial['offsets_enu_m']):
            receive_rows.append([trial['prior_sigma_m'][0], name,
                ', '.join(f'{v:+.3f}' for v in offset), trial['converged']])
    road_rows = []
    for trial in association['association_history'][-1]['targets']:
        road_rows.append([trial['label'], trial['new_road'], len(trial['trials']),
                         ', '.join(sorted({r['name'] for r in trial['trials']}))])
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'accepted_phase_centres': False,
        'screening': {'count': len(ranking), 'converged': sum(r['converged'] for r in ranking),
                      'failures': screen['failures'], 'inputs': screen['inputs'],
                      'scope': screen['scope'], 'ranked_results': ranking},
        'continued_refinements': refined,
        'phase_branch_refinement': {'assignment': branches['assignment'], 'cost': branches['cost'],
                                    'converged': branches['converged']},
        'withheld_validations': [{'title': title, 'source': filename,
            'complete_targets': [r['label'] for r in records],
            'all_solvers_converged': all(r[k] for r in records for k in
                ('calibration_converged', 'target_converged', 'independent_converged'))}
            for title, filename, records in validations],
        'receiver': {'correction_enu_m': receiver['receiver_correction_enu_m'],
                     'prior_sigma_m': receiver['receiver_prior_sigma_m'],
                     'conditional_posterior_sigma_enu_m': receiver_sigma.tolist(),
                     'converged': receiver['converged']},
        'leading_assignment_five_centres': {
            'source': str(args.work/'shared-seed-fit/00012-five-centres.json'),
            'cost': alternative_five['cost'], 'converged': alternative_five['converged'],
            'corrections_enu_m': alternative_five['channel_position_corrections_enu_m'],
            'conditional_posterior_sigma_enu_m': alternative_sigma,
            'zero_amplitude_channels_by_target': alternative_five['zero_amplitude_channels_by_target']},
        'extended_response': str(args.output/'shared-seed-extended-response.json'),
        'slow_returns': str(args.output/'shared-seed-slow-paths.json'),
        'carrier_frequency_audit': str(args.output/'carrier-frequency-audit.json'),
        'shared_range_origins': {'source': str(args.work/'shared-seed-fit/00012-range-origin.json'),
            'cost': range_origin['cost'], 'converged': range_origin['converged'],
            'validation': str(args.output/'shared-seed-range-origin-withheld.json'),
            'scope': 'Four fixed envelope origins relative to SBS, shared by all targets. Diagnostic only; not identified as hardware delays.'},
        'frequency_dependent_receiver_trials': receive_trials,
        'finite_geometry_evidence': str(args.output/'shared-seed-geometry-discrimination.json'),
        'road_reassociation': {'source': str(args.work/'shared-seed-fit/00012-associated.json'),
                              'history': association['association_history']},
        'scope': 'Revised-input local models and conditional validation. No global-optimality certificate, road ground truth, calibrated transmitter coordinates, or accepted coherent geographic/ship image.'}
    (args.output/'shared-seed-site-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    refinements_table = table(['Code', 'Objective', 'Converged', 'Evaluations'],
        [[r['code'], f'{r["cost"]:.3f}', r['converged'], r['evaluations']] for r in refined])
    screen_table = table(['Code', 'Objective', 'Above screening best', 'Converged'],
        [[r['code'], f'{r["cost"]:.3f}', f'{r["cost"]-minimum:.3f}', r['converged']] for r in ranking])
    html_text = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Revised-input phase-centre validation</title>
<style>body{{font:16px/1.55 system-ui;max-width:1320px;margin:32px auto;padding:0 24px;color:#202b35}}
img{{max-width:100%}}table{{border-collapse:collapse;margin:16px 0;font-size:14px}}
th,td{{padding:7px 10px;border-bottom:1px solid #d4dce3;text-align:left}}
.status{{border-left:5px solid #ab6b21;padding:12px 18px;background:#fff7ea}}
code{{background:#eef1f4}}a{{color:#16659e}}</style>
<h1>Phase-centre refinement: revised-input validation</h1>
<p class="status"><b>No coordinates adopted.</b> All 48 licensed site assignments have
been evaluated against the same revised inputs. The leading local branch improves
the fitting objective, but individual-band phase transfer and antenna-height stability
still do not support a calibrated five-channel image.</p>
<p><a href="vehicle-analysis.html">Full analysis and historical experiments</a> ·
<a href="shared-seed-site-summary.json">Machine-readable summary</a> ·
<a href="../isar_20261013/analysis.html">Ship imaging and companion-return analysis</a></p>
<h2>What these comparisons use</h2>
<p>Ten road-associated target intervals, one source-track seed shared across bands,
17 complex range samples (about ±300 m), original carrier-phase origins and actual
reconstructed-waveform range responses. Source detections are counted once.
The common-source extraction replaces independently reselected band peaks; old
and new absolute objective values cannot be compared. All work here is offline
support analysis and does not change production data.</p>
<h2>Licensed main/alternative transmitter sites</h2>
<p>Codes list Seven, SBS, Nine, Ten, ABC. 0 = Ornata, 1 = Eyre, 2 = Ridge.
Every assignment received 160 evaluations with the same ten targets and likelihood.
{sum(r['converged'] for r in ranking)} of {len(ranking)} screening fits met the solver's
convergence criterion. These are local fits initialized with historical road/phase
hypotheses, not a fresh global road search or posterior site probabilities.</p>
<img src="shared-seed-site-ranking.png" alt="All 48 revised-input site screening objectives">
<p>The leading screened assignment, <b>00012</b>, places Seven/SBS/Nine at Ornata,
Ten at Eyre and ABC at Ridge. Continued refinements are listed separately below.
Its further phase-branch search reaches {branches['cost']:.3f} with solver convergence.
That does not establish the physical site assignment.</p>{refinements_table}
<details><summary>All 48 screening measurements</summary>{screen_table}</details>
<p>For the leading site assignment, the roads of A, K and R were also reconsidered
using their previously discovered alternatives as starting hypotheses. Every
candidate was rescored on the revised complex observations, with all ten targets
retained in the subsequent joint fit. No association changed; the joint objective
then reached {association['cost']:.3f}. One alternative reached its evaluation budget.
These finite comparisons do not establish observed vehicle positions.</p>
{table(['Target','Retained conditional road','Alternatives tested','Alternative road names'],road_rows)}
<h2>Prediction on withheld targets</h2>
<p>The full target is excluded while refitting calibration. Its even one-second
blocks fit motion and amplitudes; odd blocks test the result. Possible overlapping
fragments are excluded together. All five bands remain visible in the diagnostics.
The energy statistic profiles amplitudes on test blocks; the separate prediction-error
fields in each JSON use amplitudes learned on training blocks.</p>
<img src="shared-seed-model-validation.png" alt="Combined losses, individual-band phase differences and amplitude retention">
{table(['Model','Target','Combined loss (dB)','Phase contrast: Seven/SBS/Nine/Ten/ABC (degrees)',
        'Shared/independent amplitude','All fits converged'], validation_rows)}
<p>For example, the Ten-at-Eyre model loses only 0.164 dB in aggregate on A, yet
ABC differs by 89.4° and its fitted common-phase amplitude vanishes. A smaller
combined loss can therefore conceal loss of an entire weaker band. This is why
the aggregate metric alone cannot qualify coherent five-channel formation.</p>
<p>Five independent electrical centres are also tested; the plots distinguish the
main-site and Ten-at-Eyre hypotheses. Their held-out fits refit
all fifteen coordinates with priors centred on the original licensed offsets;
full-data fitted offsets are only initial guesses. In the main-site version, two
calibration folds reach their evaluation budget. All three folds of the leading
Ten-at-Eyre five-centre version converge, but still leave ABC phase contrasts of
about 77° on A and −121° on K. The completed trials are evidence about these
local models, not a proof against every possible calibration.</p>
<p>Source validation files: '''+' · '.join(f'<a href="{filename}">{html.escape(title)}</a>'
    for title, filename, _ in validations)+f'''</p>
<h2>Receiver uncertainty and vertical conditioning</h2>
<p>The receiver trial moves the antenna physically in both echo and direct-path
reference distances; it adds no independent per-band delay. Its original prior
is 15 m east/north and 7.5 m vertical. The fitted correction is
{np.array(receiver['receiver_correction_enu_m']).round(3).tolist()} m ENU, with conditional
posterior standard deviations {receiver_sigma.round(3).tolist()} m. The height
correction lies close to its ±22.5 m search bound. It is not a measured roof height.
Receiver position and calibration are refitted without each withheld target.</p>
<p>The separate five-centre trial reaches vertical bounds too. Its conditional
posterior standard deviations, including the stated coordinate priors, are:</p>
{table(['Channel','East σ (m)','North σ (m)','Up σ (m)'],
       [[name, *[f'{v:.2f}' for v in row]] for name, row in zip(NAMES, five_sigma)])}
<p>On the leading Ten-at-Eyre assignment, the five-centre fit reaches
{alternative_five['cost']:.3f} with solver convergence. Four of five antenna heights
hit their ±42.426 m limits. The fitted corrections below are diagnostic hypotheses,
not coordinates suitable for the capture metadata. Several targets still receive
zero fitted gain in Ten or ABC.</p>
{table(['Channel','Correction E / N / U (m)','Posterior σ E / N / U (m)'],
       [[name, ', '.join(f'{v:+.2f}' for v in delta), ', '.join(f'{v:.2f}' for v in sigma)]
        for name,delta,sigma in zip(NAMES,alternative_five['channel_position_corrections_enu_m'],alternative_sigma)])}
<h2>Shared envelope origins</h2>
<p>A separate control fits four constant range-envelope offsets relative to SBS,
shared across all ten targets. It preserves the carrier phase formula and permits
no target-specific channel offset. The full fit converges at objective
{range_origin['cost']:.3f}; the table shows how the envelope offsets change when a
whole target is excluded. Their variation and the remaining phase disagreement
do not support adopting these values as hardware delays.</p>
{table(['Calibration set','Seven origin (m)','Nine origin (m)','Ten origin (m)','ABC origin (m)'],
       [['All ten', *[f'{v:+.2f}' for v in range_origin['state'][-4:]]]]+
       [['Without '+r['label'], *[f'{v:+.2f}' for v in r['range_bias_m']]] for r in range_checks])}
<p>A negative aggregate held-out loss in the comparison plot is not a gain over
an oracle: the shared and independent controls can settle on different local
trajectories. Their prediction errors and the same-trajectory phase comparison
are retained in <a href="shared-seed-range-origin-withheld.json">the measurements</a>.</p>
<h2>Frequency-dependent receive phase centres</h2>
<p>The roof antenna was previously represented by one receive point across all
five bands. A broadband antenna can have a frequency-dependent phase centre;
for example, this is measured for log-periodic antennas by
<a href="https://www.ieice.org/publications/proceedings/summary.php?expandable=10&amp;iconf=ISAP&amp;number=WP-P-12&amp;session_num=WP-P&amp;year=2013">Song et al. (2013)</a>.
The installed antenna type and pattern are unknown, so this is a physical
sensitivity test, not an identified explanation of the recording.</p>
<p>Each band gets one receive-centre offset, fixed across every target; SBS fixes
the spatial reference. The exact bistatic path uses that point in both the echo
and the direct reference. Independent per-axis Gaussian widths of 1 m and 0.5 m
are tested, with limits at three times each width. These are declared sensitivity
priors, not measured antenna dimensions. The common roof-position prior stays
15 / 15 / 7.5 m. No samples or production settings are altered.</p>
{table(['Relative-coordinate prior σ (m)','Channel','Offset E / N / U relative to SBS (m)',
        'Full fit converged'],receive_rows)}
<p>The corresponding A/K/R validation rows above refit the common receiver and
all band offsets without the target. The target's samples never estimate those
calibration coordinates. Lower training error alone does not qualify this model;
coordinate-bound dependence and phase transfer must also be checked.</p>
<p>All three held-out targets converge under both prior widths. Candidate A improves:
its ABC contrast becomes −26.7° with the 1 m prior and −12.0° with the 0.5 m prior.
However, K still has about −120° ABC contrast and zero fitted common-phase ABC
gain. R retains about 41–43° Ten contrast and only 52–53% of its independent
Ten amplitude. Seven and Ten vertical receive offsets reach the imposed bounds
in both full fits. This model therefore remains a diagnostic hypothesis;
the improved training fit does not establish transferable receive centres.</p>
<h2>Extended vehicle response</h2>
<p>A rigid point, five-point line, fifteen-point plane and forty-five-point volume
were evaluated on the revised observations. The body spans 8 × 3 × 3 m. Complex
reflectivities use training-only generalized cross-validation regularisation;
one-second held-out blocks measure prediction. Geometry and motion remain fixed
conditional hypotheses. This does not independently validate those hypotheses.</p>
{table(['Target','Body model','Held-out error reduction vs point',
        'Error spread for Ten phase 0°/90°/180°','Weakest phase sensitivity / radian'],extended_rows)}
<p>Some extended models predict better, while others overfit badly. They can also
absorb a changed channel phase into their reflectivities. On R, the weakest local
phase sensitivity drops from about 8.0 to 0.16 per radian between point and volume
models, even before allowing motion/position uncertainty. Thus better image
sharpness alone cannot establish transferable transmitter phase centres.</p>
<p><a href="shared-seed-extended-response.json">Complete extended-model measurements</a>.</p>
<h2>Additional slow returns</h2>
<p>Six additional intervals were re-extracted with the same common-source rule.
The test fits moving response and stationary nuisance jointly, retaining the
original samples and phase origins. It assumes the four nominally co-sited bands
share a path and is consequently a conditional diagnostic, not site validation.</p>
{table(['Candidate','Median excess-path rate (m/s)','Held-out error reduction, four bands'],slow_rows)}
<p>No additional moving calibrator is accepted from this test. This does not imply
that every weak return is unusable. <a href="shared-seed-slow-paths.json">Measurements</a>.</p>
<h2>Configured RF versus measured carrier offset</h2>
<p>Approximately fifty profile records per channel were sampled across the recording.
Their archived frequency correction includes integer-carrier acquisition, not only
the fractional-subcarrier remainder. The largest sampled absolute correction is
{max(c['maximum_absolute_sampled_cfo_hz'] for c in carrier['channels']):.1f} Hz.
No 125 kHz channel-centre error is visible in these measurements.</p>
<p>For scale, treating the entire sampled correction as an RF modelling error would
change phase by at most {max(c['phase_difference_degrees_at_30km_if_full_cfo_is_carrier_error'] for c in carrier['channels']):.2f}°
over 30 km excess path, proportionally less for the nearer vehicles. This cannot
explain the tens-of-degrees discrepancies through that mechanism. The measured
CFO also contains receiver LO error; this is not an independent absolute-frequency
calibration. <a href="carrier-frequency-audit.json">Sampled metadata and assumptions</a>.</p>
{geometry_section}
<h2>Validation and remaining requirements</h2>
<p>The calibration suite passes 82 tests, including independent complex-waveform
phase checks, shared-source extraction, profiled nonnegative gains, physical
receiver derivatives, frequency-dependent receive centres, and original-prior handling for five-centre holdout.
Those establish numerical consistency, not real-scene identifiability.</p>
<p>The next acceptance gate remains a physical model that predicts the phase of
every supported band on withheld targets and yields stable coordinates across
target selections and plausible receiver/terrain uncertainty. Neither a local
optimizer success flag nor a sharper training image substitutes for that evidence.</p>
<p>Generated {summary['created_utc']}.</p></html>'''
    (args.output/'site-refits.html').write_text(html_text)


if __name__ == '__main__':
    main()
