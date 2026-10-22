"""Publish sub-grid, motion and cross-channel phase controls for ship autofocus."""
import argparse
from datetime import datetime, timezone
from html import escape
from itertools import combinations
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def wrapped_degrees(value):
    return np.degrees(np.angle(np.exp(1j*np.asarray(value))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('build/isar-20261015'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    paths = {name: args.work/path for name, path in {
        'subgrid': 'subgrid-profile-64.json', 'motion': 'subgrid-motion-64.json',
        'common_phases': 'subgrid-phase-expanded-starts-64.json',
        'pairs': 'pair-phase-expanded-starts-64.json', 'original_pairs': 'pair-phase-64.json',
        'partial_phases': 'partial-phase-64.json'}.items()}
    source = {key: json.loads(path.read_text()) for key, path in paths.items()}
    for key in ('subgrid', 'common_phases', 'pairs', 'original_pairs', 'partial_phases'):
        if not source[key]['complete']:
            raise ValueError(f'Finish {key} before publishing')
    point, motion = source['subgrid'], source['motion']['physical_profile']
    test_log = Path('build/isar-tests-20261015-subgrid.log')
    log = test_log.read_text()
    match = re.search(r'Ran (\d+) tests', log)
    if not match or not log.rstrip().endswith('OK'):
        raise ValueError('Numerical tests must pass before publication')
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    before = point['before']['independent']['prediction_error']
    after = point['after']['independent']['prediction_error']
    if not np.isclose(after, motion['before_summary']['prediction_error'], rtol=1e-9):
        raise ValueError('The motion comparison changed its observations or support')
    point_change = 100*(after/before-1)
    motion_change = 100*(motion['after_summary']['prediction_error']/motion['before_summary']['prediction_error']-1)
    total_change = 100*(motion['after_summary']['prediction_error']/before-1)
    args.output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
    p = np.array(point['initial_points_body_m'])
    q = np.array(point['points_body_m'])
    bound = np.any(abs(q[:, :2]-p[:, :2]) >= 1-1e-6, axis=1)
    axes[0].scatter(p[:, 0], p[:, 1], s=15, c=np.where(bound, '#b84133', '#1b7194'))
    axes[0].quiver(p[:, 0], p[:, 1], 5*(q[:, 0]-p[:, 0]), 5*(q[:, 1]-p[:, 1]),
        angles='xy', scale_units='xy', scale=1, width=.004)
    displayed = np.r_[p[:, :2], p[:, :2]+5*(q[:, :2]-p[:, :2])]
    axes[0].set_xlim(displayed[:, 0].min()-5, displayed[:, 0].max()+5)
    axes[0].set_ylim(displayed[:, 1].min()-5, displayed[:, 1].max()+5)
    axes[0].set(xlabel='Body-frame forward (m)', ylabel='Port (m)', aspect='equal',
        title='64 basis points; arrows enlarged 5×\nRed: at least one ±1 m bound active')
    x = np.arange(5)
    point_errors = [100*(a['prediction_error']/b['prediction_error']-1)
        for a, b in zip(point['after']['independent']['channels'], point['before']['independent']['channels'])]
    motion_errors = [100*(a['prediction_error']/b['prediction_error']-1)
        for a, b in zip(motion['after'], motion['before'])]
    axes[1].bar(x-.18, point_errors, width=.36, label='Refine support; motion fixed')
    axes[1].bar(x+.18, motion_errors, width=.36, label='Then refit motion + origins')
    axes[1].axhline(0, color='black', lw=.6)
    axes[1].set_xticks(x, names)
    axes[1].set(ylabel='Withheld prediction-error change (%)', title='Successive comparisons; negative is better')
    axes[1].legend(fontsize=8)
    for stage, marker in [('before', 'o'), ('after', 's')]:
        axes[2].plot(names, [c['prediction_phase_rms_degrees'] for c in motion[stage]], marker+'-', label=stage)
    axes[2].set(ylabel='One-second prediction-phase RMS (degrees)', title='Physical motion refit at fixed refined support\nIncludes scattering mismatch, noise and glint')
    axes[2].legend()
    fig.savefig(args.output/'scattering-support-controls.png', dpi=160); plt.close(fig)

    phase_trials = source['common_phases']['trials']
    phase_penalties = [100*(t['validation']['coupled']['summary']['prediction_error']/
        t['validation']['independent']['summary']['prediction_error']-1) for t in phase_trials]
    half_difference = wrapped_degrees(np.array(phase_trials[2]['fit']['channel_phase_radians'])-
        phase_trials[1]['fit']['channel_phase_radians'])
    phase_starts = [len(t['fit']['trials']) for t in phase_trials]
    if phase_starts != [10, 10, 10]:
        raise ValueError('Use both plain and coordinate-escape phase initializations')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    for t, label, marker in zip(phase_trials, ['30 s', 'First 15 s', 'Second 15 s'], ['o', 's', '^']):
        axes[0].plot(names, wrapped_degrees(t['fit']['channel_phase_radians']), marker+'-', label=label)
    axes[0].set(ylabel='Band phase relative to Ten (degrees)', ylim=(-190, 190), title='Refined physical point coordinates; constant band phases')
    axes[0].legend()
    axes[1].bar(['30 s', 'First 15 s', 'Second 15 s'], phase_penalties)
    axes[1].axhline(0, color='black', lw=.6)
    axes[1].set(ylabel='Withheld error penalty vs independent bands (%)', title='Common point phases still predict worse')
    fig.savefig(args.output/'scattering-common-phase.png', dpi=160); plt.close(fig)

    pair_entries = []
    for i, j in combinations(range(5), 2):
        trials = {t['interval']: t for t in source['pairs']['trials'] if t['channels'] == [i, j]}
        delta = wrapped_degrees(trials['second']['fit']['channel_phase_radians'][1]-
            trials['first']['fit']['channel_phase_radians'][1])
        pair_entries.append({'channels': [i, j], 'name': names[i]+' / '+names[j],
            'prediction_error_penalty_percent': trials['full']['prediction_error_penalty_percent'],
            'half_phase_difference_degrees': float(delta)})
    closure = {label: wrapped_degrees([t['phase_radians'] for t in values])
        for label, values in source['pairs']['closures'].items()}
    closure_rms = {key: float(np.sqrt(np.mean(value**2))) for key, value in closure.items()}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), layout='constrained')
    axes[0].barh([p['name'] for p in pair_entries], [p['half_phase_difference_degrees'] for p in pair_entries])
    axes[0].axvline(0, color='black', lw=.6)
    axes[0].set(xlabel='Second-half minus first-half phase (degrees)', xlim=(-185, 185),
        title='Independently fitted channel pairs; original 64-point grid')
    labels = ['/'.join(str(i+1) for i in t['channels']) for t in source['pairs']['closures']['full']]
    for key, marker in [('full', 'o'), ('first', 's'), ('second', '^')]:
        axes[1].plot(range(10), closure[key], marker+'-', label=key)
    axes[1].axhline(0, color='black', lw=.6)
    axes[1].set_xticks(range(10), labels, rotation=45)
    axes[1].set(ylabel='Wrapped phase around a three-channel loop (degrees)',
        title='A consistent set of band offsets closes at zero\nIndices: 1 Seven, 2 SBS, 3 Nine, 4 Ten, 5 ABC', ylim=(-185, 185))
    axes[1].legend()
    fig.savefig(args.output/'scattering-pair-closure.png', dpi=160); plt.close(fig)

    partial_entries = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
    for label, marker in [('full', 'o'), ('first', 's'), ('second', '^')]:
        trials = [t for t in source['partial_phases']['trials'] if t['interval'] == label]
        axes[0].plot([t['count'] for t in trials], [t['prediction_error_penalty_percent'] for t in trials], marker+'-', label=label)
    for count in (2, 4, 8, 16, 32):
        trials = {t['interval']: t for t in source['partial_phases']['trials'] if t['count'] == count}
        difference = wrapped_degrees(np.array(trials['second']['fit']['channel_phase_radians'])-
            trials['first']['fit']['channel_phase_radians'])
        partial_entries.append({'anchors': count, 'half_phase_difference_degrees': difference.tolist(),
            'full_prediction_error_penalty_percent': trials['full']['prediction_error_penalty_percent']})
        axes[1].plot(names, difference, 'o-', label=str(count)+' anchors')
    axes[0].set(xlabel='Number of points constrained to common phases', ylabel='Withheld error penalty (%)',
        title='Other points retain independent complex band responses')
    axes[0].axhline(0, color='black', lw=.6); axes[0].legend()
    axes[1].set(ylabel='Half-aperture phase difference (degrees)', ylim=(-185, 185), title='Weak constraints do not establish stable calibration')
    axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(args.output/'scattering-partial-phase.png', dpi=160); plt.close(fig)

    point_phase_rms = {stage: [c['one_second_phase_metrics']['phase_rms_degrees_by_point'][0]
        for c in point[stage]['independent']['channels']] for stage in ('before', 'after')}
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'numerical_tests_passed': int(match[1]),
        'scope': 'Conditional controls on 142–172 s of live-vhf-20261013-193334. Roll/pitch fixed; no accepted antenna correction.',
        'subgrid': {'converged': point['fit']['converged'], 'kkt_residual': point['fit']['maximum_kkt_residual'],
            'coordinate_bound_hits': point['bound_hits'], 'points_with_bound_hit': int(bound.sum()),
            'prediction_error_change_percent': point_change, 'prediction_error_change_by_channel_percent': point_errors,
            'conditional_phase_rms_degrees': point_phase_rms},
        'motion': {'converged': motion['converged'], 'kkt_residual': motion['maximum_kkt_residual'],
            'prediction_error_change_percent': motion_change, 'prediction_error_change_from_original_grid_percent': total_change,
            'prediction_error_change_by_channel_percent': motion_errors, 'correction_adopted': False,
            'scaled_tangent_curvature_eigenvalues': motion['stationarity_polish']['curvature_eigenvalues']},
        'common_phase_penalties_percent': phase_penalties, 'common_phase_half_difference_degrees': half_difference.tolist(),
        'pair_phase_closure_rms_degrees': closure_rms, 'pairs': pair_entries, 'partial_phases': partial_entries,
        'antenna_correction_adopted': False, 'absolute_heading_validated': False, 'hull_centroid_validated': False,
        'inputs': {key: str(path) for key, path in paths.items()}}
    (args.output/'scattering-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    def link(path, label):
        return f'<a href="{escape(os.path.relpath(path.resolve(), args.output.resolve()))}">{escape(label)}</a>'
    evidence = ('<ul>'+''.join(f'<li>{link(path, key.replace("_", " "))}</li>' for key, path in paths.items())+
        f'<li>{link(test_log, "Numerical test log")}</li></ul>')
    pair_rows = ''.join(f'<tr><td>{p["name"]}</td><td>{p["prediction_error_penalty_percent"]:+.2f}%</td>'
        f'<td>{p["half_phase_difference_degrees"]:+.1f}°</td></tr>' for p in pair_entries)
    motion_rows = ''.join(f'<tr><td>{name}</td><td>{a["prediction_phase_rms_degrees"]:.2f}°</td>'
        f'<td>{b["prediction_phase_rms_degrees"]:.2f}°</td><td>{change:+.2f}%</td></tr>'
        for name, a, b, change in zip(names, motion['before'], motion['after'], motion_errors))
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship autofocus: scattering support and phase consistency</title>
<style>body{{font:16px/1.55 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4d9;padding:8px;text-align:left}}th{{background:#eef2f5}}.note{{background:#fff1cf;padding:16px}}code,pre{{background:#eff2f4;padding:2px 4px}}</style>
<h1>Ship autofocus: scattering support and phase consistency</h1>
<p>Updated {result['updated_utc']}. This follows the <a href="transmitter-phase-controls.html">transmitter range/rate and licensed-site controls</a>.</p>
<p class="note"><strong>Refining the scattering support improves prediction, but has not established transmitter phase calibration.</strong> With motion fixed, sub-grid horizontal coordinates reduce withheld squared prediction error by {-point_change:.2f}%. The stronger common-phase search still predicts {phase_penalties[0]:.2f}% worse than independent band responses. Roll and pitch remain fixed. No antenna coordinates, absolute hull heading or calibrated replacement video are adopted from these tests.</p>
<h2>1. Exact sub-grid support</h2>
<p>The previous 64-point basis used a two-metre grid. Each shared horizontal coordinate now moves by at most one metre, with height, motion, antenna sites and envelope origins fixed. Every trial moves physical range, RF phase and within-symbol Doppler together before the recorded row averaging. Independent complex band responses and the declared stationary nuisance are re-solved at every trial. The analytic residual Jacobian includes that coefficient refit.</p>
<p>Training uses alternating one-second blocks and the final block; odd interior blocks test prediction. The seed trajectory and original support were discovered using this aperture, so these are conditional interpolation checks, not independent target discovery or absolute location validation. No held-out phase or amplitude is used to align the frozen target prediction. Stationary nuisance is projected identically from the validation data and prediction.</p>
<p>The constrained fit meets the first-order convergence gate: KKT residual {point['fit']['maximum_kkt_residual']:.3g}; maximum constraint violation {point['fit']['maximum_constraint_violation']:.3g} m. However, {point['bound_hits']} of 128 coordinates, on {bound.sum()} points, meet a one-metre bound. This is a better representation of the observed scattering, not evidence that all fitted points are literal hull reflectors. The independent-band withheld error changes from {before:.1f} to {after:.1f} in the fixed whitened observation units.</p>
<img src="scattering-support-controls.png" alt="Sub-grid point movement, channel prediction errors, and motion-refit phase statistics">
<h2>2. Physical motion after support refinement</h2>
<p>The next fit holds those refined points fixed and re-estimates the shared physical path/yaw splines and five envelope origins, with the existing kinematic constraints. Antenna sites and roll/pitch remain fixed. Numerical convergence is <strong>{motion['converged']}</strong>, with KKT residual {motion['maximum_kkt_residual']:.3g}. Its aggregate withheld error changes by {motion_change:+.2f}% relative to the same refined support before the motion refit; the change relative to the original grid/control is {total_change:+.2f}%.</p>
<p>The local curvature check is positive in all {motion['stationarity_polish']['tangent_dimensions']} free tangent directions: eigenvalues {min(motion['stationarity_polish']['curvature_eigenvalues']):.3f}–{max(motion['stationarity_polish']['curvature_eigenvalues']):.3f} in the solver's information-scaled coordinates. This establishes local constrained curvature, not physical parameter accuracy or global uniqueness. The extra motion correction makes the withheld prediction worse and is not adopted. The original conditional motion remains the working control; the improved support is retained for further analysis, not interpreted as a measured hull geometry.</p>
<table><tr><th>Channel</th><th>Phase RMS before motion refit</th><th>After motion refit</th><th>Prediction-error change</th></tr>{motion_rows}</table>
<p>These phases are one-second projections of withheld data onto the frozen multi-reflector prediction. They contain scattering mismatch, noise and glint. They are not pure motion-error RMS, an absolute phase calibration measurement or a CRLB. A stationary numerical optimum alone does not validate heading or the physical scatterer interpretation.</p>
<h2>3. A local phase-optimisation failure mode, tested explicitly</h2>
<p>The common-phase model uses one reflector phase shared by bands, independent nonnegative band/point strengths, and constant band phase offsets. A local phase solver can get stuck when a reflector has zero fitted strength in every band: its local phase gradient is then zero even when another phase would activate it. A regression test constructs that exact case and confirms the previous local stationary point is not the optimum.</p>
<p>The added coordinate step solves a point's common phase and all its nonnegative strengths exactly while holding the remaining points fixed. With energy-normalised residual correlations b<sub>c</sub>, it maximises <code>Σ max(0, Re(exp(−iφ)b_c))²</code>. The active set changes at only two quadrature crossings per band; between crossings the objective is a constant plus a second harmonic. Evaluating those endpoints and stationary phases contains the global maximum of this one-point subproblem. Exact band-phase updates and a joint NNLS step follow. Tests check this solution against a dense angular grid, recover a deliberately invisible known reflector, and verify monotone updates with correlated dictionaries and spatial phases.</p>
<p>For the real comparisons, both the original starts and the new coordinate-initialised starts are retained. Training likelihood chooses among them; validation does not select the phase solution. This prevents a new initialization from discarding a better original basin. The full five-band control has ten starts, and each pair has four. This improves the search but does not certify global optimality of the complete nonlinear scene.</p>
<h2>4. Shared phases with refined physical points</h2>
<p>The stronger search improves the training objective, but does not repair phase transfer. The full-aperture prediction penalty is {phase_penalties[0]:.2f}%; the first/second-half penalties are {phase_penalties[1]:.2f}% and {phase_penalties[2]:.2f}%. The four non-reference phase differences between halves are {', '.join(f'{half_difference[c]:+.1f}°' for c in [0, 1, 2, 4])}. The halves reuse the discovered support and motion, so their role is a conditional stability check.</p>
<img src="scattering-common-phase.png" alt="Common-phase fits on the full aperture and its halves, with prediction penalties">
<h2>5. Independent pair fits and phase closure</h2>
<p>To avoid forcing the five-band model onto nearby frequencies, all ten channel pairs are fitted independently on the original fixed grid. If each pair estimates the same physical band offsets θ, then δ<sub>ij</sub> = θ<sub>j</sub> − θ<sub>i</sub> and every triangle satisfies <code>δ_ij + δ_jk − δ_ik = 0 mod 2π</code>. Noise and model error perturb those estimates; closure is a necessary consistency check, not a proof of calibration.</p>
<p>After retaining both initialization families, the wrapped closure RMS is {closure_rms['full']:.1f}° for the full aperture, {closure_rms['first']:.1f}° for the first half and {closure_rms['second']:.1f}° for the second half. The current point-phase fits do not yield one transferable set of offsets. This result does not imply that the recorded channels lost acquisition coherence; it concerns phase inferred through this scattering model.</p>
<img src="scattering-pair-closure.png" alt="Pairwise phase transfer and three-channel closure errors">
<table><tr><th>Pair</th><th>Full-aperture prediction penalty</th><th>Phase change between halves</th></tr>{pair_rows}</table>
<h2>6. Only some common reflectors</h2>
<p>A separate control constrains only 2, 4, 8, 16 or 32 candidate reflectors to common phases and lets all other coefficients be independent complex band responses. Candidates are ranked using full-aperture training data only, by the weakest band's leave-one-column fitted energy. That statistic is not called target SNR. With very few constrained reflectors, the total prediction penalty can be small, but inferred phases still change markedly between halves.</p>
<img src="scattering-partial-phase.png" alt="Mixed common and independent scattering: prediction penalties and phase transfer">
<p>This mixed-scattering control used the original local phase initializations and is retained as exploratory evidence. The full-scene and pair consistency conclusions above use the expanded search. It would be incorrect to treat a nearly unchanged total prediction error in the weakly constrained model as a measured phase calibration.</p>
<h2>Status and numerical checks</h2>
<p>{match[1]} numerical support tests pass. The new tests cover exact profiled coordinate derivatives, known sub-grid recovery from independently synthesised echoes, pair-phase closure, the closed-form positive-amplitude phase maximum and the inactive-reflector failure case. A body-frame invariance test also verifies that origin/heading relabelling preserves physical point positions, velocities and complex observations. Production capture and imaging code are unchanged.</p>
<p>For unrestricted complex reflectivity in each band, a constant band phase can be absorbed exactly into that band's reflector coefficients; it is unidentifiable without a valid cross-band scattering or external calibration constraint. Similarly, rotating an unknown body-coordinate scene while counter-rotating its attitude preserves the physical point trajectories. Hull shape or independent heading information is needed to establish the absolute body frame. These are model gauges, not failures that a smaller optimisation tolerance can fix.</p>
<p>The useful positive result is improved independent-band prediction from continuous support. The remaining work is to validate physical motion and a cross-band scattering/calibration constraint before presenting a hull-centred coherent five-band image as calibrated. The goal remains active.</p>
<h2>Next physical hypothesis: unresolved sea-reflected paths</h2>
<p>A small extra delay can be unresolved in a seven-megahertz range profile while still changing VHF carrier phase substantially. In a simple scalar four-path model, direct and reflected illumination/reception give the factor <code>(1 + Γ_T exp(−ikΔL_T))(1 + Γ_R exp(−ikΔL_R))</code>. Even for an ideal reflection coefficient of −1, <code>1 − exp(−ikΔL) = 2i exp(−ikΔL/2) sin(kΔL/2)</code>: the real factor can change sign across frequencies or reflector heights. Positive band strengths with one shared reflector phase need not represent that situation.</p>
<p>This motivates a physical hypothesis, not an explanation established by these data. Four-path and quasi-image ship/sea models are developed by <a href="https://www.jpier.org/issues/volume.html?paper=11101108">Chen, Zhang and Yin (2012)</a>. Their microwave approximations do not validate an exact VHF model for this capture. The earlier companion-return delay-scale test concerned a separate return hundreds of metres away; it cannot exclude unresolved sea paths affecting phase within the main ship response.</p>
<p>A small offline reference operator is now tested in <a href="../../../support/isar/sea_paths.py">sea_paths.py</a>. Each path uses the physical transmitter–receiver baseline, its own exact delay and path rate, spherical spreading and the unchanged range/moment/averaging operator. Independent Fourier/quadrature synthesis checks the combined complex response; finite time differences check Doppler; direct-only and ideal zero-height cancellation check limiting cases. The test model is explicitly a fixed plane with supplied scalar reflection coefficients, not a VHF sea-state model. No real-data correction has been applied. Curved-water geometry, reflector heights and reflection assumptions must be stated and tested before a withheld-data comparison.</p>
<h2>Evidence</h2>{evidence}
</html>'''
    (args.output/'scattering-validation.html').write_text(page)
    index = args.output/'analysis.html'
    text = re.sub(r'<p><strong>Latest work:</strong>.*?</p>',
        '<p><strong>Latest work:</strong> <a href="scattering-validation.html">Sub-grid support, physical motion and phase-consistency checks</a>. '
        'Continuous support improves prediction; no new antenna calibration or absolute hull heading is accepted.</p>', index.read_text(), count=1)
    index.write_text(text)
    session_path = args.output/'session.json'
    session = json.loads(session_path.read_text())
    session.update(updated_utc=result['updated_utc'], numerical_support_tests=int(match[1]), pending=[],
        scattering_validation_report=str(args.output/'scattering-validation.html'),
        scattering_validation_results=str(args.output/'scattering-validation.json'),
        status='Active. Sub-grid, physical motion and expanded phase-search controls completed. No accepted antenna calibration.')
    session_path.write_text(json.dumps(session, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
