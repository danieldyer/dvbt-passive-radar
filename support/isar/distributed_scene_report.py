"""Publish the complete-scene channel-registration checks without adopting them."""
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
from matplotlib.patches import Rectangle
import numpy as np

from combine_images import combine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('build/isar-20261015/channel-alignment-142-172'))
    parser.add_argument('--images', type=Path, default=Path('build/isar-20261015/band-strength-images/geometry'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    read = lambda p:json.loads(p.read_text())
    def link(path, label):
        return f'<a href="{escape(os.path.relpath(path.resolve(), args.output.resolve()))}">{escape(label)}</a>'
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    scenes = []
    for filename in ['dense-constant-scene-200-registered.json', 'dense-constant-scene-648-registered.json',
        'dense-constant-scene-968-registered.json', 'full-scene-5m-qr.json', 'full-scene-2m.json',
        'full-scene-2m-shifted.json', 'full-scene-2m-blocked.json']:
        path = args.work/filename
        if path.exists():
            scene = read(path)
            if len(scene['trials']) == 3:
                scenes.append((path, scene))
    sensitivity = [(p, read(p)) for p in sorted(args.work.glob('full-scene-*-phase-sensitivity.json'))]
    sampling, qr = read(args.work/'spatial-sampling.json'), read(args.work/'qr-validation.json')
    rows, summaries = [], []
    for path, s in scenes:
        phase = s['trials'][2]['phase_registration']
        title = ('Complete rectangle' if s.get('support_bounds_m') else 'Eight local patches')+f", {s['spacing_m']:g} m, {len(s['points_body_m'])} points"
        if s.get('holdout_seconds', 0.):
            title += f", {s['holdout_seconds']:g} s blocked holdout"
        if 'shifted' in path.name:
            title += ', shifted grid'
        values = [t['heldout_explained_fraction'] for t in s['trials']]
        p = phase['relative_phase_degrees']
        rows.append(f"<tr><td>{link(path,title)}</td>"+''.join(f'<td>{100*v:.3f}%</td>' for v in values)+
            f"<td>{', '.join(f'{v:+.2f}' for v in p)}</td><td>{phase['maximum_gradient']:.2g}</td></tr>")
        summaries.append({'source':str(path), 'label':title, 'heldout_explained_fractions':values,
                          'phase_degrees':p, 'converged':phase['converged'],
                          'shared_gcv_grid_edge':s['trials'][0]['channels'][0]['regularization']['strength_grid_edge'],
                          'shared_relative_regularization':s['trials'][0]['channels'][0]['regularization']['regularization_relative_to_max_gram']})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for i, label in enumerate(['Shared complex scene', 'Independent band scenes', 'Shared scene + channel phases']):
        axes[0].plot(np.arange(len(summaries)), [100*s['heldout_explained_fractions'][i] for s in summaries], 'o-', label=label)
    axes[0].set(xticks=np.arange(len(summaries)), xticklabels=[f"{len(s['points_body_m'])}\n{s['spacing_m']:g} m" for _,s in scenes],
        xlabel='Points / grid interval (support differs)', ylabel='Held-out explained energy (%)', title='Prediction of complex data; this is not image SNR')
    axes[0].legend(fontsize=8)
    for s in summaries:
        axes[1].plot(names, s['phase_degrees'], 'o-', label=s['label'])
    axes[1].set(ylabel='Fitted phase update relative to current gains (degrees)', title='Same motion; phase calibration depends on the scene')
    axes[1].legend(fontsize=6)
    fig.savefig(args.output/'complete-scene-comparison.png', dpi=160)
    plt.close(fig)
    sensitivity_rows, information_rows = [], []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for path, s in sensitivity:
        title = Path(s['source_scene']).stem
        trial = sorted(s['trials'], key=lambda t:t['strength_multiplier'])
        x = [t['strength_multiplier'] for t in trial]
        axes[0].semilogx(x, [100*t['heldout_explained_fraction'] for t in trial], 'o-', label=title)
        for c, name in enumerate(names):
            if c == 3:
                continue
            axes[1].semilogx(x, np.degrees(np.unwrap(np.radians([t['relative_phase_degrees'][c] for t in trial]))),
                'o-', label=title+' / '+name)
        for t in trial:
            sensitivity_rows.append(f"<tr><td>{link(path,title)}</td><td>{t['strength_multiplier']:g}</td>"
                f"<td>{100*t['heldout_explained_fraction']:.3f}%</td><td>{', '.join(f'{v:+.2f}' for v in t['relative_phase_degrees'])}</td></tr>")
            if 'data_information' in t:
                for info in t['data_information']['profiled_scene']:
                    e = info['fraction_of_fixed_scene_information_eigenvalues']
                    pattern = info.get('weakest_phase_direction_by_channel')
                    mode = '—' if pattern is None else ', '.join(f'{v:+.3f}' for v in pattern)
                    information_rows.append(f"<tr><td>{title}</td><td>{info['relative_rank_tolerance']:.0e}</td>"
                        f"<td>{info['scene_rank']}</td><td>{min(e):.3g}</td><td>{max(e):.3g}</td><td>{mode}</td></tr>")
    axes[0].set(xlabel='Scene ridge strength / original training GCV choice', ylabel='Held-out explained energy (%)', title='Prior sensitivity; held-out scores do not select a model')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Scene ridge strength / original training GCV choice', ylabel='Relative phase update (degrees; unwrapped)', title='Ten is the fixed phase reference')
    axes[1].legend(fontsize=6)
    fig.savefig(args.output/'complete-scene-phase-sensitivity.png', dpi=160)
    plt.close(fig)
    # A constant band phase can be applied to existing original-row complex BP
    # images exactly. No motion, range origin, magnitude or pixel normalization
    # changes here. These are explicitly calibration-sensitivity diagnostics.
    description = read(args.images/'image-description.json')
    for path, scene in scenes:
        if not path.name.startswith('full-'):
            continue
        source_motion = read(Path(scene['source_motion']))
        for key in ('parameters', 'range_bias_m', 'channel_gains_real_imag', 'interval_s'):
            if source_motion[key] != description['motion_model'][key]:
                raise ValueError('Phase trial and complex images use different motion or registration')
    gains = np.array([complex(*g) for g in description['channel_gains_real_imag']])
    amplitudes, information = [], []
    for c in range(5):
        with np.load(args.images/f'channel-{c}-image.npz') as f:
            x, y = f['along_m'], f['port_m']
            amplitudes.append(f['complex_image'])
            information.append(f['information'])
    phase_options = [('Current conditional gains', np.zeros(5))]+[(s['label'],s['phase_degrees']) for s in summaries if 'Complete' in s['label']]
    powers, image_statistics = [], []
    for title, phase in phase_options:
        image, weight, _ = combine(amplitudes, information, gains*np.exp(1j*np.radians(phase)))
        db = 10*np.log10(np.maximum(8192*weight*abs(image)**2, 1e-30))
        powers.append(db)
        image_statistics.append({'label':title, 'maximum_matched_power_db':float(db.max()), 'phase_degrees':list(phase)})
    upper = max(p.max() for p in powers)
    fig, axes = plt.subplots(len(powers), 1, figsize=(12, 3.6*len(powers)+.6), sharex=True, sharey=True, squeeze=False, layout='compressed')
    for ax, p, (title, _) in zip(axes.ravel(), powers, phase_options):
        handle = ax.imshow(p, origin='lower', extent=[x[0],x[-1],y[0],y[-1]], vmin=upper-40, vmax=upper, cmap='viridis', aspect='equal')
        ax.set(title=title, xlabel='Forward from AIS antenna prior (m)', ylabel='Port (m)')
        ax.add_patch(Rectangle((-80,-20),240,80,fill=False,edgecolor='lightgrey',linewidth=1,linestyle='--'))
    fig.colorbar(handle, ax=axes.ravel().tolist(), label='Conditional coherent matched power (dB)')
    fig.suptitle('Actual five-channel complex BP, same 30 s data and colour scale\nPhase-sensitivity diagnostic; these calibration updates are not adopted')
    fig.savefig(args.output/'complete-scene-phase-images.png', dpi=150)
    plt.close(fig)
    audit_path = args.work/'phase-audit-full-scene-2m/audit.json'
    phase_html, phase_statistics = '', []
    if audit_path.exists():
        audit = read(audit_path)
        previous = read(args.work/'phase-audit-band-strengths-profiled/audit.json')
        phase_rows = []
        fig, axes = plt.subplots(1,2,figsize=(14,5),layout='constrained')
        axes[0].plot(names,[c['trials'][0]['before']['one_second_phase_metrics']['phase_rms_degrees_by_point'][0] for c in previous['channels']],
            'o--',label='Earlier eight-point prediction')
        for variant, detail in audit['variants'].items():
            channels = detail['channels']
            before, after = [], []
            for channel in channels:
                c = channel['channel']
                trials = channel['trials']
                rms = [trials[0][stage]['one_second_phase_metrics']['phase_rms_degrees_by_point'][0] for stage in ('before','after')]
                gains_db = [t['matched_gain_db'] for t in trials]
                gradient = max(t['refinement']['maximum_gradient'] for t in trials)
                converged = all(t['refinement']['converged'] for t in trials)
                phase_rows.append(f'<tr><td>{names[c]}</td><td>{variant}</td><td>{rms[0]:.3f}</td><td>{rms[1]:.3f}</td>'+
                    ''.join(f'<td>{g:+.5f}</td>' for g in gains_db)+f'<td>{converged}</td><td>{gradient:.2g}</td></tr>')
                phase_statistics.append({'channel':c, 'model':variant, 'phase_rms_degrees_before_after':rms,
                    'matched_gain_db_all_even_odd':gains_db, 'converged':converged, 'maximum_gradient':gradient,
                    'regenerated_template_prediction_error_relative_check':channel['prediction_error_relative_check']})
                before.append(rms[0]);after.append(rms[1])
            index = [c['channel'] for c in channels]
            axes[0].plot(index,before,'o-',label=variant+' scene / before')
            axes[0].plot(index,after,'x:',label=variant+' scene / after GPGA')
            for i,label in enumerate(['all ranges','even → odd','odd → even']):
                axes[1].plot(index,[c['trials'][i]['matched_gain_db'] for c in channels],'o-',label=variant+' / '+label)
        axes[0].set(xticks=np.arange(5),xticklabels=names,ylabel='One-second held-out prediction-phase RMS (degrees)',title='Same physical motion, a better spatial template')
        axes[0].legend(fontsize=7)
        axes[1].axhline(0,color='black',linewidth=.7)
        axes[1].set(xticks=np.arange(5),xticklabels=names,ylabel='Held-out matched-energy change (dB)',title='Common phase correction must transfer across ranges')
        axes[1].legend(fontsize=7)
        fig.savefig(args.output/'complete-scene-residual-phase.png',dpi=160)
        plt.close(fig)
        phase_html = f'''<h2>Residual phase after modelling the extended return</h2>
<p>These checks retain the same fitted physical motion and use the frozen distributed-scene coefficients to predict each range/time sample. A new one-second-knot phase screen is fitted on training times, with stationary background eliminated at every trial. Complementary whitened range components test whether the phase correction transfers. The original-row template regeneration is checked against the QR likelihood before interpreting the phase statistics.</p>
<img src="complete-scene-residual-phase.png" alt="Residual phase and independent range-transfer tests">
<table><tr><th>Channel</th><th>Scene</th><th>Before RMS (°)</th><th>After RMS (°)</th><th>All-range gain (dB)</th><th>Even → odd gain (dB)</th><th>Odd → even gain (dB)</th><th>Final phase solves converged</th><th>Maximum gradient</th></tr>{''.join(phase_rows)}</table>
<p>Independent band scenes do not establish coherent inter-band calibration. These residuals also include noise and model error; they are not a standalone motion CRLB. This remains same-aperture conditional validation. The phase screens have not been applied to original data or images. Line-search stalls are resolved by a Newton stationarity check in the identifiable spline subspace, with the two exact constant/linear gauges removed; the convergence threshold is unchanged.</p>
<p>{link(audit_path,'Full residual-phase audit and numerical convergence diagnostics')}</p>'''
    test_log = Path('build/isar-tests-20261015-distributed.log').read_text()
    test_match = re.search(r'Ran (\d+) tests',test_log)
    if not test_match or not test_log.rstrip().endswith('OK') or len(phase_statistics) != 10:
        raise ValueError('Finish numerical validation and all five-channel phase audits before publishing')
    if not all(s['converged'] for s in phase_statistics):
        raise ValueError('Final phase refinements have not all converged')
    polish_check = read(audit_path.parent/'polish-validation.json')
    verification = {'numerical_tests_passed':int(test_match[1]), 'phase_refinements':30,
        'maximum_gradient':max(s['maximum_gradient'] for s in phase_statistics),
        'maximum_regenerated_error_relative_difference':max(s['regenerated_template_prediction_error_relative_check'] for s in phase_statistics),
        'stationarity_polish':polish_check}
    result = {'updated_utc':datetime.now(timezone.utc).isoformat(), 'scene_models':summaries,
              'phase_sensitivity':[str(p) for p,_ in sensitivity], 'qr_validation':qr, 'spatial_sampling':sampling,
              'coherent_image_comparisons':image_statistics, 'phase_update_adopted':False,
              'residual_phase_statistics':phase_statistics,
              'verification':verification,
              'status':'Conditional scene and phase-identifiability audit. Physical motion, roll/pitch, sites and production data are unchanged.',
              'interpretation_status':'Superseded by whole-block validation; small alternating-sample phase residuals and negligible frozen-template phase updates do not establish accurate physical motion or exhausted autofocus gain.'}
    peak_table = ''.join(f"<tr><td>{escape(s['label'])}</td><td>{s['maximum_matched_power_db']:.3f}</td>"
        f"<td>{s['maximum_matched_power_db']-image_statistics[0]['maximum_matched_power_db']:+.3f}</td></tr>" for s in image_statistics)
    (args.output/'distributed-scene-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship ISAR: complete-scene and channel-phase audit</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;line-height:1.55;color:#18232d}} table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{padding:8px;border:1px solid #ccd4d9;text-align:left}} th{{background:#eef2f5}} img{{max-width:100%}} code{{background:#eff2f4;padding:2px 4px}} .note{{background:#fff2cf;padding:16px}}</style>
<h1>Ship ISAR: complete spatial model before further channel calibration</h1>
<p class="note"><strong>Historical alternating-sample diagnostic — interpretation superseded.</strong> The subsequent <a href="blocked-validation.html">whole-block validation</a> found unstable dense-scene predictions and large stationary components in the frozen templates. The small phase residuals and negligible phase updates below do not establish accurate physical motion or exhausted autofocus gain. Numerical values are retained for reproducibility; no correction from this analysis was adopted.</p>
<p>Roll and pitch remain fixed. This audit tests the higher-value channel-phase residuals while keeping the current physical translation/yaw, range origins and antenna locations fixed. It fills the gaps between the earlier eight reflector patches, tests spatial sampling, and separates numerical convergence from physical identifiability.</p>
<p class="note">No new phase update is adopted. Better prediction of held-out complex samples is evidence about the scene model; it is not automatically an image SNR gain, a true-heading measurement or a validated transmitter calibration.</p>
<h2>Same data, different scene support</h2>
<p>The common-scene model has one static complex reflectivity image for all five frequencies. The independent control allows a different complex image in each band. The phase trial adds four constant channel phases, with Ten fixed as reference, and profiles the shared image at every step. All use original-row near-field responses, the same stationary nuisance projection and training-only generalized cross validation (GCV). No Doppler notch or fitted phase screen is introduced.</p>
<table><tr><th>Spatial support</th><th>Shared scene</th><th>Independent bands</th><th>Shared + phases</th><th>Phase updates Seven/SBS/Nine/Ten/ABC (°)</th><th>Maximum phase gradient</th></tr>{''.join(rows)}</table>
<p>The score is <code>1 − ||y − prediction||² / ||y||²</code> after the declared stationary projection. The image coefficients are learned on training data and frozen on test data. The inherited motion/support decisions used this aperture, so this is conditional validation rather than independent end-to-end validation. Adjacent retained bins are 17.4 ms apart; blocked tests, when listed, are a stronger temporal-transfer check. The 2 m common-scene GCV choice reaches the lower boundary of its initial regularization grid (10⁻¹² of the maximum squared singular value). It is not established as an interior optimum; the sensitivity sweep includes weaker and stronger penalties.</p>
<img src="complete-scene-comparison.png" alt="Complex prediction and fitted channel phases across spatial supports">
<h2>Spatial sampling and numerical checks</h2>
<p>The observed forward/port wavevector spans are {sampling['wavevector_span_rad_per_m'][0]:.3f} and {sampling['wavevector_span_rad_per_m'][1]:.3f} rad/m. A conservative near-field gradient margin over a 220 m support radius gives maximum grid spacings about {sampling['maximum_axis_grid_spacing_m'][0]:.2f} and {sampling['maximum_axis_grid_spacing_m'][1]:.2f} m. A 5 m full-band grid fails this check; a 2 m grid passes it. This gradient bound is not proof that a finite support or a discretized inverse image has converged. The support rectangle is an experimental scene region, not a measured hull outline.</p>
<p>QR compression of each augmented observation <code>[A y]</code> preserves <code>||A a − y||² = ||R a − z||² + E⊥</code>, including the energy outside the scene and the original observation count for GCV. It does not form normal equations or discard small singular directions. On the 833-point recording control, maximum relative error-score difference is {max(t['prediction_error_relative_difference'] for t in qr['trials']):.2g}; the phase solutions differ by {qr['trials'][-1]['max_phase_difference_degrees']:.2g}°. Separate tests cover rank deficiency, wide matrices, phase gradients, nuisance elimination and the known unidentifiable case of disjoint band-specific scene modes.</p>
<p>The numerical suite passes {verification['numerical_tests_passed']} tests. All {verification['phase_refinements']} final dense-scene phase refinements converge with maximum normalized gradient {verification['maximum_gradient']:.3g}. Regenerated waveform predictions reproduce their QR error scores within {verification['maximum_regenerated_error_relative_difference']:.3g} relative error. The stationarity polish changes the phase solutions by at most {polish_check['maximum_phase_change_after_polishing_degrees']:.3g}° and the measured gains by at most {polish_check['maximum_gain_change_after_polishing_db']:.3g} dB; it resolves a numerical stopping issue without changing the scientific conclusion.</p>
<h2>Does the data determine the phase, or does the image prior?</h2>
<p>Numerical convergence of a regularized phase fit is necessary but insufficient. These trials vary the scene ridge strength without choosing the answer from held-out scores. The Gaussian phase-information calculation then removes the arbitrary complex scene coefficients and excludes the ridge penalty: <code>F = 2 Re[Jᴴ (I − AA†) J]</code>. The eigenvalue fractions compare the remaining information with the overly optimistic case where the scene were known exactly. Small fractions mean a reflector model can absorb most of the apparent channel-phase constraint.</p>
<img src="complete-scene-phase-sensitivity.png" alt="Sensitivity to the scene regularizer">
<table><tr><th>Scene</th><th>Ridge multiplier</th><th>Held-out energy explained</th><th>Phase updates (°)</th></tr>{''.join(sensitivity_rows)}</table>
<table><tr><th>Scene</th><th>Numerical singular-value cutoff</th><th>Scene rank</th><th>Smallest information fraction</th><th>Largest information fraction</th><th>Weakest direction, Seven/SBS/Nine/Ten/ABC</th></tr>{''.join(information_rows)}</table>
<p>These are conditional information diagnostics, not calibrated uncertainty intervals: residual interference, scene-model error and the approximate thermal covariance still matter. The numerical rank sensitivity is shown explicitly rather than interpreted as a noise-dependent threshold.</p>
<p>The fine-grid weakest direction moves the lower three channels almost together relative to Ten/ABC. Its persistence across rank cutoffs indicates weak registration between the two separated frequency groups in this model. This is an inference from the measured Jacobian, not a proof that a different calibrated target or scene prior could never resolve the phase.</p>
{phase_html}
<h2>Effect on the actual coherent image</h2><img src="complete-scene-phase-images.png" alt="Actual coherent backprojection with alternative fitted constant channel phases">
<table><tr><th>Phase choice</th><th>Peak matched power (dB)</th><th>Change from current gains (dB)</th></tr>{peak_table}</table>
<p>These panels re-use the original-row complex GPU backprojections and change only the declared constant channel gains. This is exactly equivalent to applying the same constant phases during the coherent sum. No magnitudes are combined, no brightest-pixel phase is fitted, and every panel uses the same colour scale. A larger peak alone cannot select a physical calibration.</p>
<p>The outlined rectangle is the spatial region used in these complete-grid fits. The weaker return farther to negative forward coordinates lies outside it; whether it belongs to the same rigid vessel response remains a scene-model question. This is another reason not to treat the fitted phases as established antenna measurements.</p>
<h2>Why the scene assumption needs testing</h2>
<p>Published multiband ISAR calibration methods explicitly rely on a shared point-scattering model and discuss off-grid bias; see <a href="https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.12030">Zhu et al., 2021</a>. Joint antenna autofocus is also an identifiability problem, especially outside the far-field phase-error approximation; see <a href="https://arxiv.org/abs/1805.03269">Mansour et al., 2018</a>. These motivate the controls above; they do not prove that this vessel has frequency-independent reflectivity.</p>
<p>The current trial videos remain available in the <a href="amplitude-autofocus.html">preceding report</a>. They include the five individual channels and their actual complex combination, at 1/3/10/15 s apertures. Their coordinate origin is the AIS antenna prior. Absolute hull heading/centroid and a transferable phase calibration remain open; the supplied AIS track contains course over ground but no true heading or antenna-to-hull offsets.</p>
<p><a href="distributed-scene-audit.json">Machine-readable audit</a> · {link(args.work/'spatial-sampling.json','Spatial sampling calculation')} · {link(args.work/'qr-validation.json','Recording-level QR equivalence')} · {link(Path('build/isar-tests-20261015-distributed.log'),'Numerical test log')}</p></html>'''
    (args.output/'distributed-scene-audit.html').write_text(page)
    session_path = args.output/'session.json'
    session = read(session_path)
    session.update(updated_utc=result['updated_utc'], distributed_scene_report=str(args.output/'distributed-scene-audit.html'),
        complete_scene_tests=[str(p) for p,_ in scenes if p.name.startswith('full-')],
        phase_identifiability_tests=[str(p) for p,_ in sensitivity],
        distributed_scene_phase_audit=str(audit_path),
        status='Active. Extended-scene prediction resolves most apparent phase residual, but physical heading, hull centroid and transferable five-channel phase calibration remain unvalidated.',
        remaining=[
            'Keep roll and pitch fixed. Earlier physical antenna fitting added about 0.08 dB; channel path rates about 0.02 dB. Neither was adopted.',
            'The 4961-point, 2 m complete rectangle explains 73.34% of held-out energy with shared reflectivity and fitted phases, versus 85.84% with independent band reflectivity. This is same-aperture conditional validation. The original GCV grid minimum is at its lower boundary.',
            'The weakest phase-information mode moves Seven/SBS/Nine together relative to Ten/ABC. Phase estimates change by more than 100 degrees under the tested scene penalties. No new phase update or antenna location was accepted.',
            'The dense-scene phase audit has 0.6-2.0 degree prediction-phase RMS for independent bands, unchanged physical motion, all 30 final refinements converged, and no positive complementary-range transfer. Do not apply these phase screens to original data or images.',
            'Next validate broader spatial support, including the weaker negative-forward return outside the current rectangle, and blocked temporal transfer. Current motion was learned using the earlier sparse amplitude model; a flexible dense prediction alone does not validate absolute heading or a physical coherent-aperture limit.',
            'An asynchronous question remains pending for true AIS heading and antenna-to-hull offsets. The supplied track contains position, speed and course over ground only.',
            'Existing 1/3/10/15-second videos remain conditional comparisons using the same motion, with their origin at the AIS antenna prior. The full calibrated hull-centred image/video goal remains unfinished.'
        ])
    session_path.write_text(json.dumps(session,indent=2)+'\n')
    index_path = args.output/'analysis.html'
    index = index_path.read_text()
    start = index.index('<p><strong>Latest work:</strong>')
    end = index.index('</p>',start)+4
    index = index[:start]+('<p><strong>Latest work:</strong> <a href="distributed-scene-audit.html">Complete-scene and channel-phase audit: numerical checks, phase identifiability, residual-phase tests and coherent image comparisons</a>. '
        'The <a href="amplitude-autofocus.html">preceding report</a> contains the trial videos. Roll and pitch remain fixed; no additional phase calibration has been adopted.</p>')+index[end:]
    index_path.write_text(index)
    print(json.dumps({'report':str(args.output/'distributed-scene-audit.html'), 'scenes':len(scenes)},indent=2))


if __name__ == '__main__':
    main()
