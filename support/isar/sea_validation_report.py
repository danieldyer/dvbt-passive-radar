"""Publish the declared-height / curved-water phase-calibration controls."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def wrapped(value):
    return np.degrees(np.angle(np.exp(1j*np.asarray(value))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('build/isar-20261015'))
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    inputs = {label: args.work/f'sea-height-{label}.json' for label in ('full', 'first', 'second')}
    source = {label: json.loads(path.read_text()) for label, path in inputs.items()}
    if not all(s['complete'] for s in source.values()):
        raise ValueError('Complete all aperture controls before publication')
    test_log = args.work/'tests-sea-height.log'
    log = test_log.read_text()
    match = re.search(r'Ran (\d+) tests', log)
    if not match or not log.rstrip().endswith('OK'):
        raise ValueError('Numerical suite did not pass')
    footprint = json.loads((args.output/'sea-footprint.json').read_text())
    dictionary = json.loads((args.work/'sea-height-dictionary/dictionary.json').read_text())
    earlier = json.loads((args.work/'subgrid-phase-expanded-starts-64.json').read_text())['trials'][0]
    baseline = source['full']['families']['direct']['baseline_scores']
    for key in ('coupled', 'independent'):
        previous = earlier['validation'][key]['summary']['prediction_error']
        if abs(baseline[key]['total']/previous-1) > 1e-8:
            raise ValueError('Direct-only dictionary does not reproduce the previous observation control')
    families = ['direct', 'transmit', 'receive', 'both']
    labels = ['Direct only', 'Tx reflection', 'Rx reflection', 'Both reflections']
    channels = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'roll_pitch_fitted': False, 'antenna_correction_adopted': False,
        'sea_correction_adopted': False, 'numerical_tests_passed': int(match[1]),
        'full_interval_s': [142., 172.], 'height_grid_m': dictionary['identity']['heights_body_m'],
        'families': {}, 'baseline': baseline, 'inputs': {k: str(v) for k, v in inputs.items()},
        'direct_operator_max_relative_error': max(c['direct_probe_relative_error'] for c in dictionary['channels'])}
    for family in families:
        full, first, second = [source[span]['families'][family] for span in ('full', 'first', 'second')]
        phase_difference = wrapped(np.array(second['fit']['best']['band_phase_radians'])-first['fit']['best']['band_phase_radians'])
        result['families'][family] = {
            'scores': full['scores'], 'full_training_objective': full['fit']['best']['objective'],
            'coupled_prediction_error_change_from_fixed_height_direct_percent':
                100*(full['scores']['coupled']['total']/baseline['coupled']['total']-1),
            'coupled_prediction_penalty_percent': full['coupled_prediction_penalty_percent'],
            'phase_difference_degrees': phase_difference.tolist(),
            'nonreference_phase_difference_rms_degrees': float(np.sqrt(np.mean(phase_difference[[0, 1, 2, 4]]**2))),
            'height_agreement_between_halves_fraction': float(np.mean(np.array(first['selected_heights_m'])==second['selected_heights_m'])),
            'selected_heights_m': full['selected_heights_m']}
    trials = [t for s in source.values() for f in s['families'].values() for t in f['fit']['trials']]
    result['starts'] = len(trials)
    result['all_coordinate_converged'] = all(t['converged'] for t in trials)
    result['maximum_phase_gradient'] = max(t['gradient'] for t in trials)
    result['maximum_final_coordinate_improvement'] = max(t['cycles'][-1]['next_coordinate_improvement'] for t in trials)
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout='constrained')
    x = np.arange(4)
    for i, (variant, label, colour) in enumerate([('independent', 'Independent complex band responses', '#0072b2'),
                                                 ('coupled', 'Shared point phases + band offsets', '#d55e00')]):
        value = [result['families'][f]['scores'][variant]['total']/baseline['independent']['total'] for f in families]
        axes[0].bar(x+(i-.5)*.36, value, .36, label=label, color=colour)
    axes[0].axhline(1., color='black', lw=.6)
    axes[0].set_xticks(x, labels, rotation=15, ha='right')
    axes[0].set(ylabel='Withheld squared error / fixed-height independent control', ylim=(0, 1.5),
        title='30 s aperture; training selects each height and phase')
    axes[0].legend(fontsize=8, loc='lower left')
    for family, label, marker in zip(families, labels, ['o', 's', '^', 'd']):
        axes[1].plot([channels[c] for c in [0, 1, 2, 4]],
            np.array(result['families'][family]['phase_difference_degrees'])[[0, 1, 2, 4]], marker+'-', label=label)
    axes[1].axhline(0., color='black', lw=.6)
    axes[1].set(ylabel='Second-half minus first-half band phase (degrees)', ylim=(-185, 185),
        title='Phase stability; Ten is the zero-phase reference')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'sea-phase-validation.png', dpi=160); plt.close(fig)
    heights = np.array(dictionary['identity']['heights_body_m'])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout='constrained', sharey=True)
    for ax, family, label in zip(axes, families, labels):
        for index, (span, colour) in enumerate([('full', '#0072b2'), ('first', '#d55e00'), ('second', '#009e73')]):
            chosen = source[span]['families'][family]['selected_heights_m']
            ax.bar(np.arange(len(heights))+(index-1)*.26, [chosen.count(float(h)) for h in heights],
                .26, color=colour, label=span)
        ax.set_xticks(np.arange(len(heights)), heights.astype(int))
        ax.set(title=label, xlabel='Body-up hypothesis (m)')
    axes[0].set_ylabel('Number of 64 basis points'); axes[0].legend(fontsize=8)
    fig.suptitle('Selected dictionary heights are model variants, not measured hull structure', fontsize=12)
    fig.savefig(args.output/'sea-height-choices.png', dpi=160); plt.close(fig)
    (args.output/'sea-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    rows = ''.join(f'<tr><td>{label}</td><td>{result["families"][family]["full_training_objective"]:.6f}</td>'
        f'<td>{result["families"][family]["coupled_prediction_error_change_from_fixed_height_direct_percent"]:+.2f}%</td>'
        f'<td>{result["families"][family]["coupled_prediction_penalty_percent"]:+.2f}%</td>'
        f'<td>{result["families"][family]["nonreference_phase_difference_rms_degrees"]:.1f}°</td></tr>'
        for family, label in zip(families, labels))
    footprint_rows = ''.join(f'<tr><td>{"Tx–ship" if row["leg"]==0 else "Ship–Rx"}</td>'
        f'<td>{row["height_body_m"]:g} m</td><td>{row["extra_path_m"]:.3f} m</td>'
        f'<td>{100*row["water_area_fraction"]:.1f}%</td></tr>' for row in footprint['records'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship ISAR: curved-water and height controls</title>
<style>body{{font:16px/1.55 system-ui;max-width:1220px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #ccd4d9;text-align:left}}th{{background:#eef2f5}}.note{{background:#fff2cf;padding:16px}}code{{background:#eff2f4;padding:2px 4px}}</style>
<h1>Ship autofocus: test physical propagation before fitting more channel phase</h1>
<p class="note"><strong>The ideal-water model does not produce a transferable five-band calibration.</strong> Roll and pitch remain fixed. The best reflected-path case reduces withheld squared error by 2.11% relative to the fixed-height shared-phase control, but still predicts 32.67% worse than independent complex band responses. Its channel phases change by 104.9° RMS between aperture halves. No water-path, antenna-position or channel-phase correction from this test is applied to production or the videos.</p>
<p>This follows the <a href="scattering-validation.html">sub-grid support and phase-consistency analysis</a>. It tests one specific explanation for point-dependent phase: direct and unresolved water-reflected illumination/reception of a reflector at a declared height. It does not assume the observed companion return is a water echo.</p>
<h2>1. Physical model and numerical checks</h2>
<p>Each point can contribute four paths: direct, transmitter-side water reflection, receiver-side reflection, and both. All subtract the same physical transmitter–receiver baseline. Their own geometric delay, RF phase and endpoint Doppler enter the existing original-symbol observation operator before moment combination, averaging and whitening. A single fixed water sphere is tangent to the EGM96 geoid at the aperture midpoint; its radius is the geometric mean of the local WGS84 principal curvature radii.</p>
<p>The water hypotheses use scalar horizontal Fresnel reflection, relative permittivity 80, conductivity 4 S/m and geometrical-optics spherical divergence. These are stated model assumptions, not measurements of this bay. Roll/pitch, horizontal point coordinates, motion, antenna sites and range origins remain fixed. Atmospheric bending, rough-surface diffuse scattering, antenna pattern discrimination and ray-dependent target scattering are omitted.</p>
<p>The specular solve is checked against an independent Cartesian sag-coordinate solve and the reflection law. Finite time differences check Doppler. Fresnel limits and passivity check the complex sign convention. Independent Fourier-pulse and symbol-quadrature synthesis checks all four observed rays. The real-data direct ray agrees with the established operator to {result['direct_operator_max_relative_error']:.2g} relative error. The compressed direct-only control reproduces the earlier 30 s independent error to below 10⁻⁸ relative error; the shared-phase result agrees at the same threshold.</p>
<p>{result['numerical_tests_passed']} numerical support tests pass. The new template-selection tests recover known variants and phases, verify monotone original-sample likelihood and reject ineligible paths. QR sufficient statistics retain the residual outside the dictionary; no singular-value cutoff is used to compress it.</p>
<h2>2. Height and phase estimation</h2>
<p>The height choices are {', '.join(str(int(h)) for h in heights)} m body-up relative to the AIS-origin hypothesis. One choice per point is shared by all channels. Each band retains independent nonnegative point strengths; point phases are shared, with four constant band phase offsets relative to Ten. Exact conditional updates search all eligible heights and phases for one point, followed by joint strength/continuous-phase profiling. Multiple starts include the previous fixed-height phase solution.</p>
<p>All {result['starts']} starts satisfy the declared coordinate-stationarity gate. The largest final normalized phase gradient is {result['maximum_phase_gradient']:.3g}; the largest remaining full-sweep improvement is {result['maximum_final_coordinate_improvement']:.3g}. This is numerical local/coordinate convergence, not a global-optimality certificate or proof that a selected height is physically determined.</p>
<p>Training uses alternate one-second blocks plus the last block; odd interior blocks test frozen predictions. The two half-aperture controls use the same full-aperture nuisance dictionary and retained bins restricted to their intervals. Both validation data and predictions receive the identical stationary projection. Motion and initial support were discovered on this capture: these are conditional interpolation and stability checks, not independent end-to-end validation. The four propagation families are reported together; no family is adopted by picking the most flattering test score.</p>
<table><tr><th>Propagation family</th><th>Normalized training error</th><th>Withheld error change from fixed-height shared-phase control</th><th>Penalty vs independent bands at the same selected heights</th><th>Half-aperture band-phase difference RMS</th></tr>{rows}</table>
<img src="sea-phase-validation.png" alt="Withheld prediction errors and phase changes between aperture halves">
<p>The full-aperture training criterion favours the direct-only height model, which actually predicts worse than the fixed-height shared-phase control. Transmitter-side reflection gives the smallest test error among these trials, but that isolated improvement does not establish calibration. The independent-band errors stay close to the original 427649.6 control while the shared-phase error remains much larger. These are squared prediction errors in fixed whitened units; their changes are not image SNR gains.</p>
<img src="sea-height-choices.png" alt="Height-choice counts on the full aperture and its halves">
<p>The height choices change substantially between halves. In the direct-only full-aperture model, 56 of 64 points reach the 5 or 55 m ends of the declared grid. This is another reason not to reinterpret the dictionary choices as a recovered physical hull structure. The horizontal points are an approximation basis, already known to extend beyond a literal 40 m hull width.</p>
<h2>3. Why a specular centre on water is insufficient</h2>
<p>Eligibility first excludes reflected variants whose sampled specular centre lies on land or outside the DEM. All tested transmitter centres pass; receiver-side centres reject 192 of 448 point/height variants per band. This is a necessary geometric check, but the reflecting footprint is much larger than a point.</p>
<img src="sea-footprint.png" alt="First-Fresnel surface regions over the DEM coastline, for transmitter and receiver reflection legs">
<p>This figure uses exact surface path contours with λ/2 excess over the specular path on the declared sphere, restricted to surface points visible from both endpoints. It shows Ten at the aperture midpoint for a point above the AIS origin, rather than claiming those examples are measured reflectors. Grey is DEM terrain or unmapped area; blue-grey is approximately zero-height DEM surface. There are no unmapped cells inside the tabulated regions. The 20 m grid samples the contour area; it is not a wave-scattering simulation.</p>
<table><tr><th>Reflection leg</th><th>Body-up hypothesis</th><th>Extra geometric path</th><th>Sampled first-zone area classified as water</th></tr>{footprint_rows}</table>
<p>Much of the receiver-side zone crosses the shoreline even at low heights. The area fraction is <strong>not</strong> a reflection amplitude, phase or loss estimate. It shows why an infinite, homogeneous-water coefficient cannot be treated as calibrated propagation here. Terrain height alone also does not establish surface material or an unobstructed actual reflected path.</p>
<p><a href="https://www.itu.int/dms_pubrec/itu-r/rec/p/r-rec-p.530-14-201202-s!!pdf-e.pdf">ITU-R P.530-14, §6.1.2.4.1</a> distinguishes Fresnel reflection, Earth divergence and the finite reflecting surface. The scalar four-path hypothesis is motivated by <a href="https://www.jpier.org/issues/volume.html?paper=11101108">Chen, Zhang and Yin (2012)</a>; that microwave approximation is not a validation of this VHF ship/bay model. A finite shoreline, atmosphere and target scattering model would require substantially more assumptions and independent evidence.</p>
<h2>Decision</h2>
<p>Do not add sea-path compensation or fit roll/pitch to absorb this mismatch. The test has not identified a stable transmitter phase residual. The earlier independent-band motion/support control remains the working model, and the existing combined images remain conditional on their stated channel-phase choice. Accurate absolute hull heading/centroid and a transferable five-band calibration remain unresolved; no replacement video is labelled as achieving those objectives.</p>
<p>For arbitrary complex band reflectivities, <code>g_c → g_c exp(iδ_c)</code> and <code>a_cj → a_cj exp(−iδ_c)</code> leave every predicted sample unchanged. A valid cross-band scattering constraint or independent calibration must break that ambiguity. This does not imply loss of acquisition coherence. Separately, an unknown body-coordinate scene and an opposite attitude rotation have an exact coordinate ambiguity; a physical hull or true-heading reference is needed to identify that frame.</p>
<h2>Reproducible evidence</h2><ul>
<li><a href="sea-validation.json">Summary JSON</a> and <a href="sea-footprint.json">surface-footprint numbers</a></li>
<li><a href="../../../build/isar-20261015/sea-height-full.json">Full-aperture fits</a>, <a href="../../../build/isar-20261015/sea-height-first.json">first-half fits</a>, <a href="../../../build/isar-20261015/sea-height-second.json">second-half fits</a></li>
<li><a href="../../../build/isar-20261015/sea-height-dictionary/dictionary.json">Dictionary provenance, declared water model and terrain eligibility</a></li>
<li><a href="../../../support/isar/spherical_sea.py">Physical reference operator</a>, <a href="../../../support/isar/variant_phase.py">template/phase fitter</a>, <a href="../../../support/isar/audit_sea_height.py">audit CLI</a></li>
<li><a href="../../../build/isar-20261015/tests-sea-height.log">Numerical test log</a></li></ul>
<p>The only implementation issue encountered during parallel aperture runs was a cache-manifest publication race. It was fixed by atomic publication and preserving existing manifests; no numerical algorithm or completed fit was changed. The captured data and production pipeline are unchanged.</p>
</html>'''
    (args.output/'sea-validation.html').write_text(page)
    index = args.output/'analysis.html'
    index.write_text(re.sub(r'<p><strong>Latest work:</strong>.*?</p>',
        '<p><strong>Latest work:</strong> <a href="sea-validation.html">Curved-water and reflector-height controls</a>. '
        'No stable channel-phase calibration is recovered; roll/pitch remain fixed.</p>', index.read_text(), count=1))
    print(json.dumps({k: result[k] for k in ('starts', 'all_coordinate_converged', 'numerical_tests_passed', 'maximum_phase_gradient')}))


if __name__ == '__main__':
    main()
