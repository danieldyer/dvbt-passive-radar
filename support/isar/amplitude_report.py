"""Publish the amplitude-aware autofocus experiments and their limits."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ais import read_csv, utc_seconds
from imaging import physical_state, sites
from motion import enu_basis
from spline_motion import SplineMotion
from validate_motion import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('build/isar-20261015/channel-alignment-142-172'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    parser.add_argument('--ais', type=Path, default=Path('docs/data/isar_20261013/rio-delta-ais.csv'))
    parser.add_argument('--geometry-directory', type=Path, default=Path('docs/data/calibration_20261013'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    read = lambda p: json.loads(p.read_text())
    def link(path, label):
        relative = os.path.relpath(path.resolve(), args.output.resolve())
        return f'<a href="{escape(relative)}">{escape(label)}</a>'
    candidates = [('Constant strengths', 'coherent-motion-range-control.json'),
                  ('Changing strengths; fixed motion', 'amplitude-fixed-motion-3s.json'),
                  ('Changing strengths; refitted motion', 'amplitude-motion-polished-3s.json'),
                  ('Refined shared points; fixed motion', 'amplitude-points-fixed-motion-3s.json'),
                  ('Frequency-dependent strengths; fixed motion', 'amplitude-band-strengths-fixed-motion-3s.json'),
                  ('Frequency-dependent strengths; refitted motion', 'amplitude-band-strengths-motion-3s.json'),
                  ('Relative channel path rates; fixed motion', 'amplitude-channel-rates-fixed-motion-3s.json')]
    models = [(title, args.work/name, read(args.work/name)) for title, name in candidates if (args.work/name).exists()]
    for _, _, model in models:
        c = model['coherent_motion']
        if 'after_summary' not in c:
            c['after_summary'] = summarize(c['after'])
    temporal_path = args.work/'temporal-scene-amplitude-polished.json'
    temporal = read(temporal_path)
    audit_path = args.work/'phase-audit-band-strengths-profiled/audit.json'
    if not audit_path.exists():
        audit_path = args.work/'phase-audit-amplitude-polished/audit.json'
    audit = read(audit_path)
    image = read(args.output/'amplitude-coherent-comparison.json')
    change = image['results'][1]['maximum_conditional_matched_power_db']-image['results'][0]['maximum_conditional_matched_power_db']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    modes = ['shared complex', 'positive amplitudes, fixed phases', 'independent positive amplitudes, fixed shared phases', 'independent complex by channel']
    labels = ['Shared complex coefficients', 'Shared positive strengths; fixed phases', 'Band-dependent positive strengths; fixed phases', 'Independent complex band coefficients']
    for mode, label in zip(modes, labels):
        trials = sorted([t for t in temporal['trials'] if t['coefficient_model'] == mode], key=lambda t:t['duration_s'])
        axes[0].plot([t['duration_s'] for t in trials], [100*t['heldout_explained_fraction'] for t in trials], 'o-', label=label)
    axes[0].set(xlabel='Coefficient block duration (s)', ylabel='Held-out explained energy (%)', title='Same motion, samples and stationary nuisance space')
    axes[0].legend(fontsize=8)
    for title, _, model in models:
        axes[1].plot(names, model['coherent_motion']['after_summary']['phase_rms_degrees_by_channel'], 'o-', label=title)
    axes[1].set(ylabel='One-second prediction-phase RMS (degrees)', title='Phase of full-range model residual, including mismatch')
    axes[1].legend(fontsize=7)
    fig.savefig(args.output/'amplitude-model-evidence.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained')
    x = np.arange(5)
    for i, label in enumerate(['All ranges', 'Fit even / test odd ranges', 'Fit odd / test even ranges']):
        axes[0].bar(x+(i-1)*.25, [c['trials'][i]['matched_gain_db'] for c in audit['channels']], width=.25, label=label)
    axes[0].axhline(0, color='black', linewidth=.8)
    axes[0].set(xticks=x, xticklabels=names, ylabel='Held-out matched-energy change (dB)', title='A common phase screen must transfer across range components')
    axes[0].legend(fontsize=9)
    for key, label in [('before', 'Before GPGA'), ('after', 'After GPGA, all ranges')]:
        axes[1].plot(names, [c['trials'][0][key]['one_second_phase_metrics']['phase_rms_degrees_by_point'][0] for c in audit['channels']], 'o-', label=label)
    axes[1].set(ylabel='One-second prediction-phase RMS (degrees)', title='Low full-range residual alone does not establish a physical correction')
    axes[1].legend()
    fig.savefig(args.output/'amplitude-gpga-validation.png', dpi=160)
    plt.close(fig)
    track = read_csv(args.ais, reference_height_ellipsoid_m=models[0][2]['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    motion_statistics = []
    for title, path, model in models:
        if path.name not in ('coherent-motion-range-control.json', 'amplitude-motion-polished-3s.json', 'amplitude-band-strengths-motion-3s.json'):
            continue
        t = np.linspace(*model['interval_s'], 301)
        curve = SplineMotion(model)
        prior = physical_state(track, utc_seconds(model['nominal_start_utc'])+t, model['attitude_prior'])
        state = curve.apply(t, prior, tx[3], rx)
        forward = np.einsum('nji,nj->ni', enu_basis(prior['latitude'], prior['longitude']), state['body_to_ecef'][:, :, 0])
        heading = np.degrees(np.unwrap(np.arctan2(forward[:, 0], forward[:, 1])))+360
        axes[0].plot(t, curve.curves(t)[0], label=title)
        axes[1].plot(t, heading)
        motion_statistics.append({'model': str(path), 'conditional_heading_endpoints_deg': heading[[0,-1]].tolist(),
            'conditional_heading_rate_range_deg_ps': [float(np.gradient(heading,t).min()), float(np.gradient(heading,t).max())]})
    axes[0].set(ylabel='Reference path correction (m)', title='Physical motion fit; body-frame orientation remains conditional on the scene')
    axes[0].legend(fontsize=9)
    axes[1].set(xlabel='Capture time (s)', ylabel='Conditional heading (degrees true)')
    fig.savefig(args.output/'amplitude-motion-curves.png', dpi=160)
    plt.close(fig)
    table, summaries = [], []
    for title, path, model in models:
        c = model['coherent_motion']
        s = c['after_summary']
        explained = 100*(1-s['prediction_error']/s['stationary_only_error'])
        table.append(f'<tr><td>{link(path,title)}</td><td>{c["converged"]}</td><td>{c["maximum_kkt_residual"]:.3g}</td><td>{explained:.2f}%</td><td>'+', '.join(f'{v:.1f}' for v in s['phase_rms_degrees_by_channel'])+'</td></tr>')
        summaries.append({'title': title, 'path': str(path), 'converged': c['converged'], 'kkt_residual': c['maximum_kkt_residual'], 'heldout_explained_percent': explained, 'phase_rms_degrees': s['phase_rms_degrees_by_channel']})
    gpga_rows = []
    for name, c in zip(names, audit['channels']):
        full = c['trials'][0]
        before = full['before']['one_second_phase_metrics']['phase_rms_degrees_by_point'][0]
        after = full['after']['one_second_phase_metrics']['phase_rms_degrees_by_point'][0]
        gpga_rows.append(f'<tr><td>{name}</td><td>{before:.2f} → {after:.2f}°</td><td>{full["matched_gain_db"]:+.3f}</td><td>{c["trials"][1]["matched_gain_db"]:+.3f}</td><td>{c["trials"][2]["matched_gain_db"]:+.3f}</td></tr>')
    motion = read(args.work/'amplitude-motion-polished-3s.json')
    polish = motion['coherent_motion']['gradient_polishing']
    curvature = polish['curvature_eigenvalues']
    registration_rows = ''.join(f'<tr><td>{name}</td><td>{r:.3f}</td><td>{abs(complex(*g)):.4f}</td><td>{np.degrees(np.angle(complex(*g))):.2f}</td></tr>'
        for name, r, g in zip(names, motion['range_bias_m'], motion['channel_gains_real_imag']))
    movie_root = Path('captures/live-vhf-20261013-193334/isar-band-strength-20261015')
    videos = read(movie_root/'videos.json') if (movie_root/'videos.json').exists() else []
    video_links = ''.join(f'<li>{link(Path(v["path"]), str(v["requested_aperture_s"])+" s aperture video")}: {v["fps"]} fps, peak {v["maximum_power_db"]:.2f} dB, clipping {v["display_saturated"]}.</li>' for v in videos)
    dense_rows, dense_results, phase_comparisons = [], [], []
    for filename in ('dense-constant-scene-200-registered.json', 'dense-constant-scene-648-registered.json', 'dense-constant-scene-968-registered.json'):
        path = args.work/filename
        if not path.exists():
            continue
        dense = read(path)
        for trial in dense['trials']:
            registration = trial.get('phase_registration')
            label = 'Independent band responses' if trial['independent_complex_band_responses'] else ('Shared scene, refitted channel phases' if registration else 'Shared scene, previous channel phases')
            phases = ', '.join(f'{x:+.1f}' for x in registration['relative_phase_degrees']) if registration else '—'
            if registration:
                phase_comparisons.append({'points': len(dense['points_body_m']), **registration})
            dictionary_link = link(path, str(len(dense['points_body_m']))+' points')
            dense_rows.append(f'<tr><td>{dictionary_link}</td><td>{label}</td><td>{100*trial["heldout_explained_fraction"]:.2f}%</td><td>{phases}</td></tr>')
        dense_results.append(str(path))
    if phase_comparisons:
        fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
        for entry in phase_comparisons:
            ax.plot(names, entry['relative_phase_degrees'], 'o-', label=f'{entry["points"]} points; converged {entry["converged"]}')
        ax.set(ylabel='Phase update relative to previous gains (degrees)', title='Channel phase estimates depend on the distributed-scene representation')
        ax.legend()
        fig.savefig(args.output/'distributed-scene-phase-registration.png', dpi=160)
        plt.close(fig)
    dense_text = ('<h2>Distributed static-scene control</h2><p>A finer collection of constant complex reflectors can generate apparent strength/phase changes through interference as the ship turns. This is a different explanation from intrinsic target decorrelation. Deterministic grids around the original points test it without adding temporal motion states. Regularization is selected by training-only generalized cross validation; the full stationary projection is retained on held-out data.</p>'
        '<table><tr><th>Spatial dictionary</th><th>Response model</th><th>Held-out explained energy</th><th>Relative phase updates, Seven/SBS/Nine/Ten/ABC (°)</th></tr>'+''.join(dense_rows)+
        '</table><figure><img src="distributed-scene-phase-registration.png"></figure><p>Independent per-band reflectivity does not establish coherent frequency registration. The phase-refitted common-scene trial has only four nonlinear parameters and profiles the distributed scene at every step, with nine deterministic starting points. Agreement across spatial resolutions is needed before adopting those phase updates. These remain conditional target-model experiments.</p><p>The phase solve uses the SVD of the full design, rather than forming normal equations. An independently tested matrix with condition number 10⁹ verifies the gradient and phase recovery. The earlier normal-equation diagnostic failed its convergence/curvature check at 648 points; its results were not adopted.</p>') if dense_rows else ''
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'models': summaries,
        'temporal_experiment': str(temporal_path), 'phase_audit': str(audit_path),
        'actual_uniform_bp_peak_change_db': change, 'videos': [v['path'] for v in videos],
        'distributed_scene_tests': dense_results,
        'conditional_motion': motion_statistics,
        'distributed_scene_phase_updates': phase_comparisons,
        'numerical_tests': {'passed': 78, 'log': 'build/isar-tests-20261015-final.log'},
        'phase_registration_adopted_from_dense_scene': False,
        'accepted_roll_pitch': False, 'absolute_heading_validated': False, 'antenna_locations_adopted': False,
        'scope': 'Conditional autofocus research on 142–172 seconds. Numerical convergence is established for named models; global motion uniqueness and an absolute hull frame are not.'}
    (args.output/'amplitude-autofocus.json').write_text(json.dumps(result, indent=2)+'\n')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship autofocus: separating strength and phase</title>
<style>body{{font:16px/1.55 system-ui;max-width:1220px;margin:36px auto;padding:0 24px;color:#20252c}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.status{{padding:18px;background:#fff1cd;border-left:5px solid #b98620}}code{{background:#eee;padding:2px 5px}}figure{{margin:26px 0}}</style>
<h1>Ship autofocus: separating changing strength from phase error</h1>
<p>Updated {escape(result['updated_utc'])}. MAERSK RIO DELTA hypothesis, capture 142–172 s. <a href="analysis.html">Earlier analysis</a> · <a href="numerical-audit.html">Numerical validation</a> · <a href="geometry-refinement.html">Antenna-location experiment</a>.</p>
<p class="status">The amplitude-aware translation/yaw, range-origin and channel-gain fit has passed strict local numerical convergence. Roll and pitch remain fixed. A lower prediction-phase residual is not, by itself, evidence of an accurate physical trajectory: reflector mixtures change with time and frequency. No new surveyed antenna coordinates or independently verified hull heading are claimed.</p>
<h2>What changed, and what did not</h2>
<p>The earlier model assigned each point one complex response over the full 30 seconds. Allowing its nonnegative magnitude to change every three seconds, while retaining one constant point phase, changes held-out explained energy from 12.25% to 43.89% with motion held fixed. Subsequent joint refitting reaches 45.50%. This is evidence against a constant-strength scene model, not an equivalent image-SNR improvement. The actual uniform-aperture, five-channel coherent image peak changes by only <strong>{change:.3f} dB</strong>.</p>
<p>The controlled comparison uses the same samples, whitened range operator and stationary nuisance spaces. The score is <code>2 Re(mᴴ y) − ||m||²</code> for a prediction learned on even time bins and evaluated on odd bins; target phase is not refitted on test data. Support selection and physical motion used the training portion of this aperture. This remains conditional validation, not a new independent vessel observation or a global sensitivity bound.</p>
<figure><img src="amplitude-model-evidence.png"><figcaption>Left: temporal/frequency response models with physical motion fixed. Right: residual phase includes noise, glint and incomplete scattering support as well as possible motion error.</figcaption></figure>
<table><tr><th>Model</th><th>Numerically converged</th><th>Maximum KKT residual</th><th>Held-out explained energy</th><th>Phase RMS, Seven/SBS/Nine/Ten/ABC (°)</th></tr>{''.join(table)}</table>
<h2>Numerical convergence and registration</h2>
<p>The polished motion fit has KKT residual {motion['coherent_motion']['maximum_kkt_residual']:.3g} and maximum constraint violation {motion['coherent_motion']['maximum_constraint_violation']:.3g}. The information-scaled constrained tangent Hessian is positive definite: its {len(curvature)} eigenvalues span {min(curvature):.3f} to {max(curvature):.3f}. These certify a local constrained minimum in this model. They do not prove a unique global solution or include scene-model bias. Kinematic priors are explicit in the linked JSON.</p>
<p>Each range origin shifts the compressed envelope only. The fitted complex gain supplies carrier registration independently; adding the envelope offset to carrier phase would double-count it. Physical motion updates range, carrier phase and symbol Doppler together. The independently synthesized displaced-antenna GPU check measured 0.00026° combined phase error.</p>
<table><tr><th>Channel</th><th>Envelope origin (m)</th><th>Gain magnitude, Ten = 1</th><th>Gain phase (°)</th></tr>{registration_rows}</table>
<figure><img src="amplitude-motion-curves.png"><figcaption>Changing the motion fit is distinct from changing the assumed reflector strengths. The absolute heading remains conditional on the chosen body-coordinate support.</figcaption></figure>
<p>Frequency-dependent nonnegative point strengths are also tested without per-frequency point phases. Their band-gain magnitudes are held fixed because band gain and reflector strength have an exact scale ambiguity. These models preserve a coherent phase hypothesis, but do not justify arbitrary frequency-flat scattering.</p>
<p>The relative channel path-rate trial changes envelope delay, RF phase and symbol Doppler together. It converges, but adds only about 0.021 dB of held-out matched energy. Its effective rates are not transmitter-clock measurements and have not been added to imaging. The earlier joint antenna-location refinement adds about 0.08 dB; those coordinates remain unadopted.</p>
<h2>GPGA: improvement and failed transfer tests</h2>
<p>A common residual phase is estimated from the matched sufficient statistic of the full whitened range vector. All phase screens are trained on even time bins and evaluated on odd bins. Two additional tests train on complementary whitened range components. Their constant/linear phase gauges are identical. In the latest audit, stationary nuisance is re-solved at every phase trial using the full complex range residual, rather than holding its initial estimate fixed. Phase-screen reductions that fail those transfer tests are not applied to the images.</p>
<figure><img src="amplitude-gpga-validation.png"></figure>
<table><tr><th>Channel</th><th>Full-range phase RMS</th><th>Full-range gain (dB)</th><th>Even → odd ranges (dB)</th><th>Odd → even ranges (dB)</th></tr>{''.join(gpga_rows)}</table>
<p>The frequency-dependent-strength model still fails range-transfer tests in four channels despite smaller full-range residuals. That is evidence that a single phase screen is absorbing some range-dependent scene mismatch. It does not establish a transmitter phase error. Screens currently operate on retained 17.4 ms fields and have not been promoted into the original-row image operator.</p>
<h2>Actual coherent images and aperture comparison</h2>
<figure><img src="amplitude-coherent-comparison.png"><figcaption>Original C0/C1 data, identical fixed colour scale, nominal antennas, fitted envelope origins and complex channel gains. No zero-Doppler notch and no image-dependent normalization.</figcaption></figure>
<figure><img src="amplitude-coherent-comparison-channels.png"></figure>
<figure><img src="band-strength-coherent-comparison.png"><figcaption>Additional frequency-dependent-strength motion refit: the uniform-aperture image peak changes by only about −0.027 dB. Prediction quality and image peak gain are distinct quantities.</figcaption></figure>
<p>The videos contain all five independent channels and the actual information-weighted complex combination. They use complete fixed-length windows, 30 fps and all-intra H.264 for scrubbing. Their origin is still the AIS antenna prior. They are comparisons of a numerically converged conditional model, not an independently calibrated hull-centred deliverable.</p><ul>{video_links}</ul>
<p>Three seconds remains a useful short-aperture working comparison. The coarse point-model coefficients vary substantially beyond about 3–5 seconds; this does not establish intrinsic loss of scene coherence. The distributed-scene control below tests unresolved spatial interference. A universal optimal CPI has not been established. More parameters improving a predictive score must not be mistaken for information gained by uniform-aperture BP.</p>
{dense_text}
<p>The 648- and 968-point phase fits both converge, yet differ by up to about 63° in their channel updates while those updates add only about 0.02 dB. This is model sensitivity, not solver nonconvergence. No phase update from the dense-scene trials has been applied to the videos. Finer static structure substantially improves prediction, so the coarse model's changing coefficients cannot establish a short physical coherence limit.</p>
<h2>Remaining identifiability and references</h2>
<p>The supplied AIS contains positions, speed and course over ground, but no true heading or antenna-to-hull offsets. A constant rotation of the body points can be exchanged with body heading, leaving their physical trajectories unchanged. Reflectivity-based orientation therefore needs a hull-shape assumption or an independent heading reference. The first fitted point anchors the local scene; it does not survey the hull centroid.</p>
<p>The separable constrained least-squares approach follows the variable-projection framework of <a href="https://www.cs.umd.edu/~oleary/software/varpro.pdf">O’Leary and Rust</a>; our particular positive-strength model and acceptance tests are explicit modeling choices. <a href="https://scholar.afit.edu/etd/2240/">Evers’ generalized phase-gradient autofocus</a> motivates estimating phase from complex target histories; it does not make a changing extended target a rigid point reflector.</p>
<p>{link(args.output/'amplitude-autofocus.json','Machine-readable summary')} · {link(temporal_path,'Temporal model results')} · {link(audit_path,'Full phase audit')}</p></html>'''
    (args.output/'amplitude-autofocus.html').write_text(html)
    state_path = args.output/'session.json'
    if state_path.exists():
        state = read(state_path)
        state.update(updated_utc=result['updated_utc'], amplitude_report=str(args.output/'amplitude-autofocus.html'), amplitude_results=str(args.output/'amplitude-autofocus.json'))
        state.update(status='Active autofocus investigation. Conditional motion fits converge; extended-scene uncertainty still prevents validated absolute heading, hull centroid and transmitter phase calibration.',
            current_conditional_motion=str(args.work/'amplitude-band-strengths-motion-3s.json'),
            current_trial_videos=str(movie_root), distributed_scene_tests=dense_results,
            remaining=[
                'Keep roll and pitch fixed. Antenna-position changes add about 0.08 dB; relative channel path rates add about 0.02 dB. Neither was promoted.',
                'Distinguish extended spatial interference from changing point strengths. Static 968-point scenes explain about 67.6% of held-out energy with a shared response and 77.7% with independent band responses; sparse temporal-coefficient changes do not prove an intrinsic coherence limit.',
                'Dense channel-phase fits converge numerically but change by up to 63 degrees between the 648- and 968-point grids. Test support/grid and physical-scene assumptions before adopting further calibration. Current dense grids are local patches around the original eight points, not a complete hull model.',
                'GPGA with full stationary-background profiling still fails complementary-range transfer in four channels. Do not apply those phase screens to original rows or images.',
                'An asynchronous question asks for true AIS heading and antenna-to-hull offsets. The supplied track has only position, speed and course over ground. Absolute heading and hull centroid remain unresolved.',
                'Select a defensible aperture using the improved distributed-scene model. Current 1/3/10/15-second videos are conditional comparisons over the 142–172-second working interval, with original complex BP and a fixed colour scale.'])
        state_path.write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
