"""Plots and report fragments for conditional calibration diagnostics."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_revised(directory):
    """Keep revised-input validation distinct from the historical trials."""
    path = directory/'shared-seed-withheld.json'
    if not path.exists():
        return ''
    result = json.loads(path.read_text())
    targets = result['targets']
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    labels = [target['label'] for target in targets]
    loss = np.array([target['heldout_loss_db'] for target in targets])
    phase = np.array([target['phase_contrast_degrees'] for target in targets])
    ratio = np.array([target['shared_to_independent_amplitude_ratio'] for target in targets])
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    axes[0].bar(labels, loss, color='#2373bb')
    axes[0].axhline(0, color='#888888', linewidth=.7)
    axes[0].set(title='Combined held-out comparison', ylabel='Loss versus independent bands (dB)')
    p = axes[1].imshow(phase, cmap='RdBu_r', vmin=-180, vmax=180, aspect='auto')
    axes[1].set(title='Same-path phase contrast', xticks=range(5), xticklabels=names,
                yticks=range(len(labels)), yticklabels=labels)
    figure.colorbar(p, ax=axes[1], label='Degrees relative to SBS')
    p = axes[2].imshow(ratio, cmap='viridis', vmin=0, vmax=2, aspect='auto')
    axes[2].set(title='Shared / independent amplitude', xticks=range(5), xticklabels=names,
                yticks=range(len(labels)), yticklabels=labels)
    figure.colorbar(p, ax=axes[2], label='Amplitude ratio (values above 2 clipped)', extend='max')
    figure.suptitle('Revised common-source extraction: whole-target and alternating-time holdout\n'
                   'One road/site branch; original phase origins; all five bands retained')
    figure.savefig(directory/'shared-seed-withheld.png', dpi=150)
    plt.close(figure)
    rows = ''.join(f'<tr><td>{target["label"]}</td><td>{target["heldout_loss_db"]:+.3f}</td>'
                   f'<td>{target["phase_only_loss_db_same_trajectory"]:.3f}</td><td>'+
                   ', '.join(names[c] for c, value in enumerate(target['shared_to_independent_amplitude_ratio'])
                             if value < .01)+'</td></tr>' for target in targets)
    complete = all(target['calibration_converged'] and target['target_converged']
                   and target['independent_converged'] for target in targets)
    corrections = np.array([target['transmitter_corrections_enu_m'] for target in targets])
    summary = {'source': str(path), 'targets': len(targets), 'all_fits_converged': complete,
               'maximum_combined_heldout_loss_db': float(loss.max()),
               'minimum_combined_heldout_loss_db': float(loss.min()),
               'site_correction_min_enu_m': corrections.min(axis=0).tolist(),
               'site_correction_max_enu_m': corrections.max(axis=0).tolist(),
               'accepted': False,
               'scope': 'Revised common-source seeds and 17-column range patches. Fixed main-site/road hypotheses, initialized from earlier fits. Source observations are not counted as independent measurements in other bands. No independent global association or site-permutation validation.',
               'reason': 'Small aggregate losses coexist with inconsistent individual-band phases, negligible fitted amplitudes and unstable antenna heights. No phase-centre metadata was updated.'}
    (directory/'shared-seed-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return f'''<h2 id="revised-input">Revised-input validation</h2>
<p>The twelve intervals have been re-extracted using the selected source track for
every channel, with 17 retained range samples (about ±300 m). Each source detection
is counted once. Ten previously associated road candidates have been jointly refitted
and each withheld from calibration in turn; possible overlapping fragments E/O are
excluded together. The withheld target's even one-second blocks fit its trajectory
and amplitudes, and odd blocks test prediction. All numerical fits converged.</p>
<p>The combined loss relative to independent-band fits lies between {loss.min():+.3f}
and {loss.max():+.3f} dB. Negative values favour the shared prediction on these samples;
they are not gains over an oracle. These figures do not compare old and new production
SNR: the extraction and observation dimensions changed.</p>
<img src="shared-seed-withheld.png" alt="Revised-input held-out loss, individual-band phases and amplitude retention">
<table><tr><th>Candidate</th><th>Combined held-out loss (dB)</th><th>Same-path phase loss (dB)</th><th>Negligible fitted bands</th></tr>{rows}</table>
<p>The combined metric is dominated by stronger bands. In A's withheld test, ABC's
phase differs by about 122° and its fitted shared amplitude vanishes. Other candidates
show similar failures in Ten or ABC. The leave-target-out antenna heights also range
to the ±30 m search bounds. The corrected extraction is necessary, but these results
still do not validate transferable five-band phase centres. The site and road branch
was initialized from historical searches, so the new result is conditional on that
branch; the old 48-way search is not a fresh likelihood comparison.</p>
<p>The calibration regression suite passes 80 tests, including common-source seed
provenance, conditional gate retention, and independent five-band complex waveform
checks through the widened extraction. These verify the operator, not physical
identifiability. <a href="shared-seed-summary.json">Summary</a> ·
<a href="shared-seed-withheld.json">All held-out measurements</a> ·
<a href="shared-seed-phase-diagnostics.json">Training-path block jackknife</a>.</p>'''


def render_slow(directory):
    result_path = directory/'slow-stationary-common-paths.json'
    if not result_path.exists():
        return ''
    result = json.loads(result_path.read_text())
    control = json.loads((directory/'slow-common-paths.json').read_text())
    controls = {target['label']: target for target in control['targets']}
    root = Path(__file__).resolve().parents[2]
    work = root/'build/calibration-20261013'
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    table = ''
    summaries = []
    for axis, target in zip(axes.flat, result['targets']):
        label = target['label']
        description = json.loads((work/'slow-histories'/f'{label}.json').read_text())
        seed = description['channels'][1]
        coarse = np.asarray(seed['coarse_observations'])
        time = np.asarray(target['time_s'])
        seed_path = np.polynomial.polynomial.polyval(
            (time-seed['epoch'])/seed['scale'], seed['path'])-seed['path'][0]
        axis.scatter(coarse[:, 0]-seed['epoch'], coarse[:, 1]-seed['path'][0],
                     s=13, color='#888888', label='SBS envelope peaks')
        axis.plot(time-seed['epoch'], seed_path, '--', color='#888888',
                  label='Doppler-integrated seed')
        previous = controls[label]
        axis.plot(np.asarray(previous['time_s'])-seed['epoch'],
                  previous['common_path_variation_m'], color='#b25a23',
                  label='Single-component phase fit')
        axis.plot(time-seed['epoch'], target['common_path_variation_m'],
                  color='#2373bb', label='Joint stationary + moving fit')
        axis.set(title=f'{label} · {description["start"]:g}–{description["end"]:g} s',
                 xlabel='Time from interval centre (s)', ylabel='Relative excess path (m)')
        seed_rate = float(seed['path'][1]/seed['scale'])
        changes = [100*r['heldout_error_reduction_fraction'] for r in target['common_metrics']]
        losses = [r['training_target_information_loss_db'] for r in target['common_metrics']]
        table += (f'<tr><td>{label}</td><td>{seed_rate:+.2f}</td>'
                  f'<td>{target["median_common_path_rate_mps"]:+.3f}</td>'
                  f'<td>{min(changes):+.1f}% to {max(changes):+.1f}%</td>'
                  f'<td>{min(losses):.2f}–{max(losses):.2f}</td></tr>')
        summaries.append({'label': label, 'interval_s': [description['start'], description['end']],
                          'sbs_seed_rate_mps_at_epoch': seed_rate,
                          'median_fitted_common_rate_mps': target['median_common_path_rate_mps'],
                          'heldout_error_reduction_percent_by_band': changes,
                          'conditional_target_information_loss_db_by_band': losses,
                          'accepted_as_moving_calibrator': False})
    axes.flat[0].legend(fontsize=7)
    figure.suptitle('Slow-return follow-up: selected envelopes versus fitted phase histories\n'
                   'Original complex profiles; fixed phase origins; no antenna coordinates fitted')
    figure.savefig(directory/'slow-return-validation.png', dpi=150)
    plt.close(figure)
    scans = []
    for name in ('candidates-slow-6s.jsonl', 'candidates-slow-8s-projected.jsonl'):
        counts = [0]*5
        last = 0.
        with (work/name).open() as source:
            header = json.loads(next(source))
            for line in source:
                frame = json.loads(line)
                if frame['type'] == 'map':
                    counts[frame['channel']] += 1
                    last = max(last, frame['time'])
        scans.append({'source': str((work/name).relative_to(root)),
                      'coherent_interval_s': header['cpi'], 'update_interval_s': header['update'],
                      'remove_aperture_dc_for_detection': header['remove_dc'],
                      'doppler_half_width_hz': header['doppler_hz'],
                      'minimum_abs_doppler_hz': header['minimum_doppler_hz'],
                      'selection_threshold_db': header['threshold_db'],
                      'maps_per_channel': counts, 'last_map_centre_s': last})
    summary = {'scans': scans, 'targets': summaries, 'accepted_count': 0,
               'road_search_coverage': '8072 connected road hypotheses, extending about 28.8 km east of the receiver; earlier screening was not restricted to near-receiver roads.',
               'selection': 'Track duration at least 20 s, at least 15 overlapping map detections, excess range below 6.5 km. Follow-up selection required absolute median path rate 0.35–4 m/s, range-slope/rate disagreement below 0.3 m/s, straight-range residual RMS below 15 m, nominal displacement over 20 m and median excess range above 800 m. These are screening criteria, not significance tests or added calibration observations.',
               'scope': 'Six additional candidate intervals, no accepted calibration or geographic position. The model fits constant-delay complex nuisance coefficients on alternating one-second blocks and predicts other blocks. Prediction errors and information losses are conditional diagnostics, not production SNR losses. These finite searches do not prove that the recording contains no weaker useful echo.'}
    (directory/'slow-search-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return f'''<h2 id="slow-search">Additional slow-return search</h2>
<p>The road-search catalogue already contains 8,072 connected hypotheses and extends
about 28.8 km east of the receiver. It was not limited to nearby streets. A separate
Doppler-coverage check found that the earlier detection apertures excluded much of
the slow-motion region. Two new Vulkan scans cover the full recording at ±12 Hz:
six-second integration with aperture DC retained, and eight-second integration with
a constant background projected out for detection. The latter uses a 0.25 Hz minimum
Doppler cutoff.
Neither scan changes the archived samples used by calibration.</p>
<p>Six additional intervals were selected from the eight-second maps. Their peaks
suggested migration at about 1–2 m/s, but the original complex samples contain much
stronger nearly stationary components. The follow-up therefore fits constant-delay
range responses jointly with a moving point. Both templates use the same measured
range response, covariance and temporal operator; fitted coefficients on alternating
one-second blocks predict the others.</p>
<img src="slow-return-validation.png" alt="Six additional candidate envelopes compared with their complex phase histories">
<table><tr><th>Candidate</th><th>SBS seed rate at centre (m/s)</th><th>Median fitted rate (m/s)</th><th>Held-out error reduction across four bands</th><th>Conditional nuisance information loss (dB)</th></tr>{table}</table>
<p>Positive error reductions favour the joint model over a stationary-only prediction.
The fitted moving components do not consistently recover the motion indicated by
the selected peaks; several also worsen withheld predictions. All numerical fits
converged, but none of these intervals is accepted as a moving phase calibrator.
The information-loss column quantifies overlap with the fitted stationary nuisance
subspace for these trial paths. It is neither a production-pipeline loss nor a reason
to discard zero Doppler in the ship data.</p>
<p>New synthetic tests inject a slow point among much stronger stationary responses,
recover its complex amplitude and phase, predict withheld samples, and verify the
path derivatives. The full calibration suite passes 77 tests.
These checks validate the observation/projection implementation, not a physical
calibration. The search does not prove that no weaker useful echo exists.
<a href="slow-search-summary.json">Search scope and numerical summary</a> ·
<a href="slow-stationary-common-paths.json">Full joint-nuisance results</a> ·
<a href="slow-common-paths.json">Single-component control</a>.</p>'''


def render(directory):
    extended_path = directory/'vehicle-extended-response.json'
    finite_path = directory/'vehicle-finite-site-sensitivity.json'
    if not extended_path.exists() or not finite_path.exists():
        return ''
    extended = json.loads(extended_path.read_text())
    finite = json.loads(finite_path.read_text())
    targets = extended['targets']
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    grid_names = ['1 point', '5 points', '15 points', '45 points']
    colours = ['#2373bb', '#da8a15', '#228d59', '#9a58b5']
    rows = ''
    for target, colour in zip(targets, colours):
        errors = np.array([[v['shared_prediction_error'] for v in m['variants']]
                           for m in target['models']])
        relative = 100*(errors[:, 0]/errors[0, 0]-1)
        axes[0].plot(range(4), relative, 'o-', color=colour, label=target['label'])
        phase_effect = 100*(errors[:, 2]/errors[:, 0]-1)
        axes[1].plot(range(4), phase_effect, 'o-', color=colour, label=target['label'])
        rows += '<tr><td>'+target['label']+'</td>'+''.join(
            f'<td>{v:+.1f}%</td>' for v in relative)+'</tr>'
    for axis in axes:
        axis.set_xticks(range(4), grid_names)
        axis.axhline(0, color='#999999', linewidth=.7)
        axis.set_xlabel('Fixed vehicle scattering grid')
    axes[0].set(title='Held-out error relative to the point model',
                ylabel='Prediction-error change (%)')
    axes[1].set(title='Effect of a 180° Ten phase change',
                ylabel='Prediction-error change from unperturbed phase (%)')
    axes[0].legend(title='Candidate', ncol=2)
    figure.suptitle('Extended vehicles: better fits do not necessarily identify channel phase\n'
                   'Fixed trial geometry; training-only ridge selection; alternating one-second holdout')
    figure.savefig(directory/'vehicle-extended-response.png', dpi=150)
    plt.close(figure)

    sites = finite['used_sites']
    directions = ['towards_receiver', 'east', 'north', 'up']
    names = {0: 'Ornata', 1: 'Eyre', 2: 'Ridge'}
    figure, axes = plt.subplots(len(sites), 4, figsize=(14, 6),
                               layout='constrained', squeeze=False, sharex=True)
    distances = [('unknown_common_target_reflectivity_squared_distance',
                  'Known band phases', '#2373bb'),
                 ('unknown_common_target_and_channel_phase_squared_distance',
                  'Fitted shared band phases', '#228d59'),
                 ('unknown_per_band_target_reflectivity_squared_distance',
                  'Separate target/band coefficients', '#b25a23')]
    radial_rows = ''
    for row, site in enumerate(sites):
        for column, direction in enumerate(directions):
            axis = axes[row, column]
            selected = [r for r in finite['records'] if r['site'] == site
                        and r['direction'] == direction and r['displacement_m'] > 0]
            for key, label, colour in distances:
                axis.loglog([r['displacement_m'] for r in selected],
                            [r[key] for r in selected], 'o-', label=label, color=colour)
            axis.axhline(1, color='#999999', linewidth=.7, linestyle='--')
            axis.set_title(names[site]+' · '+direction.replace('_', ' '))
            axis.set_xticks([10, 30, 100], ['10', '30', '100'])
            if row == len(sites)-1:
                axis.set_xlabel('Site displacement (m)')
            if column == 0:
                axis.set_ylabel('Squared whitened mean difference')
                for record in selected:
                    if record['displacement_m'] not in (10, 100):
                        continue
                    radial_rows += (f'<tr><td>{names[site]}</td><td>{record["displacement_m"]:g}</td>'
                                    +''.join(f'<td>{record[key]:.5g}</td>' for key, _, _ in distances)
                                    +'</tr>')
    axes[0, 0].legend(fontsize=7)
    figure.suptitle('Finite transmitter displacements: exact trajectories and receiver held known\n'
                   'Ten conditional road candidates; all measured band powers; no coordinate prior')
    figure.savefig(directory/'vehicle-finite-site-sensitivity.png', dpi=150)
    plt.close(figure)

    return f'''<h2 id="identifiability">Scattering freedom and geometric information</h2>
<p>A larger deterministic vehicle model uses 5, 15 or 45 rigidly attached points within an
8 × 3 × 3 m body. Every point has a common complex coefficient across the five bands;
positive per-band illumination magnitudes are estimated on training samples.
Ridge regularisation is selected using training-only
<a href="https://pages.stat.wisc.edu/~wahba/stat860/pdf1/golub.heath.wahba.pdf">generalised cross-validation</a>.
Geometry and road trajectories remain fixed to the unaccepted main-site trial and were
discovered using the complete recording. The held-out blocks therefore test conditional
prediction, not an independently validated calibration.</p>
<img src="vehicle-extended-response.png" alt="Extended vehicle prediction errors and sensitivity to Ten channel phase">
<table><tr><th>Candidate</th><th>1 point</th><th>5 points</th><th>15 points</th><th>45 points</th></tr>{rows}</table>
<p>The table gives held-out prediction-error changes relative to the single-point model;
negative is better. These are not radar SNR gains. Some targets improve, while A becomes
much worse as the model grows. Several responses also become less discriminating about
Ten's phase. The richer model is not accepted merely because its training fit improves.
<a href="vehicle-extended-response.json">Complete errors, regularisation choices and conditional phase information</a>.</p>
<p>There is an exact ambiguity if every target is instead given an arbitrary complex
reflectivity in every band: in the model
<code>μ[j,c] = exp(i β[c]) α[j,c] u[j,c]</code>, replacing
<code>β[c]</code> by <code>β[c] + δ[c]</code> and
<code>α[j,c]</code> by <code>α[j,c] exp(−i δ[c])</code> leaves every predicted sample
unchanged. Adding more targets under that model cannot identify the constant channel
phases. A transferable physical scattering constraint or an independent phase reference
is required. An analytic information-matrix test verifies this invariance.</p>
<h3>Exact finite geometry check</h3>
<p>This separate calculation holds all ten proposed road trajectories and the receiver
exactly known, uses an ideal point echo with each band's independently measured power,
and moves one active transmitter site at a time. The exact bistatic paths, RF phase and
measured range-response operator are reevaluated; no location prior contributes. Eyre is
unused in this particular site assignment and is excluded from the calculation.</p>
<img src="vehicle-finite-site-sensitivity.png" alt="Finite site displacement sensitivity with shared or independent target reflectivities">
<table><tr><th>Site</th><th>Move toward receiver (m)</th><th>Known band phases</th><th>Fitted shared band phases</th><th>Separate target/band coefficients</th></tr>{radial_rows}</table>
<p>Numbers are squared whitened mean differences after refitting the indicated constant
complex target coefficients. The first two cases retain one common complex coefficient
per target; the second also fits four phase corrections, each shared by every target,
relative to SBS. This small periodic fit uses deterministic multiple starts and is not
a certified global optimum; further optimisation could only reduce its distance.
With proper complex Gaussian unit covariance these distances equal
KL divergence in nats; the dashed line at one is a scale reference, not a detection or
calibration acceptance threshold. The empirical covariance and ideal point assumption
make these conditional sensitivity results, not a certified experimental CRLB.
The trajectories and receiver are treated more favourably than in the real fit.
They demonstrate weak radial sensitivity, including at finite displacements, independently
of an optimiser's convergence. Freeing unknown motion and response parameters cannot
increase the expected information under the same observation model.</p>
<p>The two tests identify distinct limitations: shared-phase target models have not yet
passed validation, and the available road hypotheses weakly constrain some antenna
directions even with exactly known paths. A tighter coordinate prior would narrow the
answer by assumption; it would not constitute a measurement of the phase centre.
<a href="vehicle-finite-site-sensitivity.json">Finite-displacement measurements and scope</a>.</p>'''+render_slow(directory)
