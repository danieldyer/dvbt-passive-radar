"""Publish the no-yaw numerical controls, point response and preview videos."""
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


def load(path):
    return json.loads(Path(path).read_text())


def entry(path):
    m = load(path)
    return {'path': str(path), 'sha256': fingerprint(path), 'fit': {k: m['fit'][k] for k in
        ('converged', 'objective', 'maximum_kkt_residual', 'maximum_constraint_violation')},
        'validation': m['after_summary'], 'per_band_validation': m['original_row_validation_after'],
        'fixed_ne_frame': bool(m.get('fixed_ne_frame')), 'scene_variances': m.get('scene_variances'),
        'effective_scene_ranks': m.get('effective_scene_ranks')}


def selected(candidates):
    minimum = min(r['fit']['objective'] for r in candidates)
    tied = [r for r in candidates if r['fit']['converged'] and r['fit']['objective'] <= minimum+1e-10]
    return min(tied or candidates, key=lambda r:r['fit']['objective'])


def comparison(free, fixed):
    return {'free': free, 'fixed': fixed,
        'paired_convergence': free['fit']['converged'] and fixed['fit']['converged'],
        'heldout_error_change_percent': 100*(fixed['validation']['prediction_error']/free['validation']['prediction_error']-1),
        'heldout_matched_energy_change_db': 10*np.log10(fixed['validation']['matched_energy']/free['validation']['matched_energy'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('build/isar-20261016'))
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    short = []
    for w in (6, 7):
        fixed = [entry(p) for p in (args.work/'marginal-local-fixed-ne-3s').glob(f'window-{w:03d}-fixed-ne*.json')]
        free = [entry(p) for folder in ('marginal-local-fixed-ne-3s', 'marginal-nested-control-3s', 'marginal-fisher-control-3s')
            for p in (args.work/folder).glob(f'window-{w:03d}-free*.json')]
        short.append({'window': w, **comparison(selected(free), selected(fixed)),
            'fixed_starts': fixed, 'free_starts': free})
    long_fixed = [entry(p) for p in (args.work/'fixed-ne-motion-45s').glob('window-007-fixed-ne*.json')]
    long = comparison(entry(args.work/'fixed-ne-motion-45s/window-007-free.json'), selected(long_fixed))
    audit = load(args.work/'local-geometry-audit.json')
    point_info = load(args.work/'fixed-ne-point-information.json')
    spread = load(args.work/'fixed-ne-point-spread/point-spread.json')
    videos = Path('captures/live-vhf-20261013-193334/isar-fixed-ne-30s-20261016')
    video = load(videos/'videos.json')
    record = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'short_aperture_gaussian_scene_controls': short, 'long_aperture_control': long,
        'long_fixed_starts': long_fixed, 'local_geometry_audit': audit, 'point_information': point_info,
        'point_response': spread, 'video': video, 'video_validation': load(videos/'validation.json'),
        'support_tests': 200, 'test_log': 'build/isar-20261016-tests-fixed-ne-final.log',
        'selection': 'Minimum training likelihood; numerical ties within 1e-10 prefer a converged fit. Withheld statistics never select starts.',
        'holdout_scope': 'Alternating half-second blocks conditional on previously selected point support and initial motion; not an untouched recording or a population performance estimate.',
        'calibration_accepted': False}
    (args.output/'fixed-ne-followup.json').write_text(json.dumps(record, indent=2)+'\n')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for result, color, title in zip(spread['records'], ('#d47924', '#2466a2'), ('Fitted yaw', 'Zero yaw')):
        with np.load(result['cuts']) as a:
            x = a['distance_m']
            for row in range(2):
                for column in range(2):
                    for key, style, label in (('coherent_power', '-', 'known relative phases'),
                                              ('independent_power', '--', 'independent band amplitudes')):
                        axes[row, column].plot(x, 10*np.log10(np.maximum(a[key][row], 1e-12)),
                            color=color, ls=style, lw=1.2, label=title+', '+label)
    for ax, extent, title in zip(axes.ravel(), (200, 2000, 30, 200),
            ('Weak direction: central response', 'Weak direction: wider ambiguity',
             'Strong direction: central response', 'Strong direction: wider ambiguity')):
        ax.set(xlim=(-extent, extent), ylim=(-40, 1), xlabel='Displacement from known point (m)',
            ylabel='Point-response power (dB)', title=title)
        ax.axhline(-3.0103, color='#888', lw=.6)
    axes[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle('30-second known-point response on the original GPU operator\nUnit power at the injected point; this is a conditional ambiguity test, not measured ship SNR')
    fig.savefig(args.output/'fixed-ne-point-response.png', dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    x = np.arange(len(short))
    for i, r in enumerate(short):
        axes[0].bar(i, r['heldout_matched_energy_change_db'], color='#2466a2' if r['paired_convergence'] else '#ddd',
            hatch=None if r['paired_convergence'] else '///', edgecolor='#777')
    axes[0].set(xticks=x, xticklabels=[f'3 s, window {r["window"]}' for r in short],
        ylabel='No-yaw matched-energy change (dB)', title='Training-selected matched models')
    axes[0].axhline(0, color='#555', lw=.7)
    for i, r in enumerate(point_info['results']):
        values = [r['without_stationary_nuisance'][k]['principal_axis_standard_deviation_m'][0]
            for k in ('independent_complex_band_amplitudes', 'known_relative_band_calibration')]
        bars = axes[1].bar(np.arange(2)+(.17 if i else -.17), values, width=.34,
            label='Zero yaw' if i else 'Fitted yaw', color='#2466a2' if i else '#d47924')
        axes[1].bar_label(bars, fmt='%.2f m', padding=3, fontsize=9)
    axes[1].set(xticks=[0, 1], xticklabels=['Independent band amplitudes', 'Known relative phases'],
        yscale='log', ylim=(.1, 150), ylabel='Conditional weak-axis position SD (m)',
        title='Known-point local CRLB at 30 dB integrated SNR')
    axes[1].legend(fontsize=9)
    fig.suptitle('Yaw-free follow-up: hatched result is unresolved, and local CRLBs assume known motion/geometry')
    fig.savefig(args.output/'fixed-ne-followup-statistics.png', dpi=155)
    plt.close(fig)

    rows = ''.join(f'<tr><td>{r["window"]}</td><td>{r["heldout_error_change_percent"]:+.3f}%</td>'
        f'<td>{r["heldout_matched_energy_change_db"]:+.3f} dB</td>'
        f'<td>{"Both stationary" if r["paired_convergence"] else "Unresolved: best rotating trial is not stationary"}</td></tr>' for r in short)
    response_rows = ''.join(f'<tr><td>{"Zero yaw" if r["fixed_ne_frame"] else "Fitted yaw"}</td>'
        f'<td>{r["coherent_central_fwhm_m"][0]:.1f} m</td><td>{r["coherent_central_fwhm_m"][1]:.2f} m</td>'
        f'<td>{r["independent_central_fwhm_m"][0]:.1f} m</td></tr>' for r in spread['records'])
    links = ''.join(f'<li><a href="../../../{html.escape(v["path"])}">'
        f'{"Ship-following" if "following" in v["path"] else "Fixed geographic"} video</a></li>' for v in video['files'])
    maximum_change = max(audit['per_band_relative_template_change'])
    numerical = audit['results']
    local_error = numerical['local']['probes'][-1]['maximum_absolute_derivative_error']
    old_error = numerical['ecef_round_trip']['probes'][-1]['maximum_absolute_derivative_error']
    maximum_kkt = max(r['fit']['maximum_kkt_residual'] for s in short for r in s['fixed_starts'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Fixed N/E autofocus follow-up</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#243040}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}code{{font-size:.9em}}.preview{{max-height:650px}}</style>
<h1>Fixed north/east autofocus: no fitted yaw</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="fixed-ne-duration-validation.html">Earlier 3/10/30-second controls</a> · <a href="fixed-ne-followup.json">Measurements and input identities</a></p>
<p class="note">Yaw is exactly zero in the new fixed-frame forward model. The revised calculation resolves its short-window numerical failures, but removing yaw has not established a consistent sensitivity or focusing improvement. The new videos show broad range ridges; they are a diagnostic, not a calibrated ship reconstruction.</p>
<h2>Video result</h2>
<p>Two {video['frames']/video['fps']:.1f}-second previews at {video['fps']} fps, with a fixed 30-second coherent window, cover capture centres {video['frame_centres_s'][0]:.3f}–{video['frame_centres_s'][1]:.3f} s. The entire viewport is backprojected with translation only in fixed east/north axes. Cyan is the interpolated AIS antenna track. The colour scale stays at {video['scale_db'][0]:.3f} to {video['scale_db'][1]:.1f} dB. No DC notch is used.</p>
<ul>{links}</ul>
<p>Both videos passed full decoding and frame/packet checks: 447 frames each, 30 fps, no B frames, every frame independently coded, and faststart. Actual coherent support is {video['actual_aperture_extrema_s'][0]:.6f}–{video['actual_aperture_extrema_s'][1]:.6f} s. These are complete previews of the fitted 45-second interval, not replacements for the full-capture videos.</p>
<a href="../../../{videos}/ship-geographic-north-up-30s-30fps.mp4"><img class="preview" src="../../../{videos}/geographic-frame-00223.png" alt="No-yaw geographic backprojection with cyan AIS track"></a>
<h2>Does it improve the statistics?</h2>
<p>The same 45-second observations and fixed point support were fitted with and without yaw, trying five translation starts. All six fits pass the independent stationarity test. The training-selected no-yaw model has {long['heldout_error_change_percent']:.2f}% higher withheld prediction error and {long['heldout_matched_energy_change_db']:.3f} dB less matched energy than the rotating control. The preview uses that selected no-yaw model. This tests removing rotation from the declared scene model; it does not prove that fitted yaw is the ship's physical heading.</p>
<p>The short-aperture follow-up uses a Gaussian prior on complex point strengths to control poorly distinguishable scene terms. One variance per band is fitted using training data. The following scores compare the same likelihood family, not a regularized no-yaw fit against an unregularized rotating fit:</p>
<table><tr><th>3-second window</th><th>Withheld error change</th><th>Matched-energy change</th><th>Numerical status</th></tr>{rows}</table>
<p>The rotating family contains the fixed-frame solution. I explicitly embedded each selected fixed-frame fit into the rotating family as another initialization, preserving every physical point history. Window 7 finds a lower training objective but fails stationarity even with covariance-Fisher step scaling. Its apparent heldout gain or loss therefore remains unresolved. Window 6's selected paired fits are stationary. Starts are chosen using training likelihood only.</p>
<p>Withheld observations are alternating half-second blocks, conditional on reflector support and initial motion selected earlier. They are useful checks on the new fit, not a claim of untouched independent data or a universal performance guarantee. The earlier converged 10- and 30-second controls also lost matched energy with yaw removed.</p>
<img src="fixed-ne-followup-statistics.png" alt="Conditional withheld statistics and point information with and without yaw">
<h2>Why the no-yaw image is a ridge</h2>
<p>I injected a known coherent point using the original row epochs, information weights, archived mid-aperture spectral response and exact C0/C1 model, then ran the actual GPU backprojector. A 30-second aperture gives these central half-power widths:</p>
<table><tr><th>Model</th><th>Weak direction, known band phases</th><th>Strong direction, known band phases</th><th>Weak direction, independent bands</th></tr>{response_rows}</table>
<img src="fixed-ne-point-response.png" alt="Known-point central response and wider ambiguity cuts">
<p>Removing yaw substantially reduces angular diversity in this aperture. Even with perfect relative calibration, the known point is about 128 m wide in the weak direction. With independent unknown band amplitudes, the corresponding power-combination response along the same cut is about 1.2 km wide. The narrow and repeated features in the wider cuts also matter: a narrow central response or local CRLB alone does not certify a unique geographic solution. Each model's cuts follow its known-relative-phase information axes. The independent-band curves use those same cuts; their own principal directions differ slightly.</p>
<p>The information plot assumes 30 dB total integrated point SNR, known geometry and known motion. Its standard deviations are local conditional bounds, not observed ship-position uncertainties. The synthetic point uses mean symbol energy and the archived filter spectrum; it is not a complete finite-waveform ambiguity test or a statement that five-frequency ship scattering is calibrated.</p>
<h2>Numerical fix and statistical model</h2>
<p>Small fitted translations previously passed through Earth-sized ECEF coordinates before being subtracted again. The likelihood now carries those corrections as relative vectors. The physical transformation is unchanged: the largest relative change of a real-data point template is {maximum_change:.3g}. In the ill-conditioned recorded-data derivative probe, the maximum absolute error at a 10⁻⁶ parameter step falls from {old_error:.3g} to {local_error:.3g}, about {old_error/local_error:.0f} times smaller. This is a numerical fix, not an SNR claim.</p>
<p>All ten regularized fixed-frame starts now pass stationarity; maximum KKT residual {maximum_kkt:.3g}. A synthetic derivative test resolves 10⁻⁸ parameter steps without rounding through ECEF positions. All 200 support tests pass, including physical phase/range/Doppler derivatives, fixed N/E conversion, identical nested point histories, fixed-duration window support, and the complex Gaussian likelihood/information identities.</p>
<p>The diagnostic scene model is <code>y = A(θ)a + n</code>, with <code>a ~ CN(0,vI)</code> and unit whitened noise. Motion minimizes <code>yᴴC⁻¹y + log det C</code>, where <code>C = I + vAAᴴ</code>; point strengths are integrated out and the posterior mean predicts withheld samples. Training-only variance fitting retains the determinant penalty. This follows the Gaussian marginal-likelihood construction in <a href="https://papers.nips.cc/paper/1719-the-relevance-vector-machine.pdf">Tipping's Bayesian linear-model work</a>; this implementation has one variance per band and does not perform ARD sparsity selection.</p>
<p>The step-scaling matrix uses the proper-complex covariance information <code>Jᵢⱼ = Re tr(C⁻¹CᵢC⁻¹Cⱼ)</code>, eliminating unknown scene variance. It is evaluated as a positive square-root Gram matrix without subtracting nearly equal information matrices. <a href="https://arxiv.org/html/1510.02551">He, Hu, Blum and Wu</a> treat Gaussian reflection coefficients and correlated clutter in distributed-radar bounds; applying such bounds requires the stated covariance model. Our direct-covariance tests verify the implementation, but the recording's residual covariance and scattering prior remain physical assumptions.</p>
<p>A 6,000-realisation known-variance Gaussian test agrees with posterior-MMSE predictions within its declared 4% test tolerance. This does not certify empirical-Bayes motion estimates or real ship imaging as CRLB-tight. No transmitter or receiver calibration has been promoted from these experiments.</p>
</html>'''
    (args.output/'fixed-ne-followup.html').write_text(page)
    print(json.dumps({'short': [{k:r[k] for k in ('window', 'paired_convergence', 'heldout_error_change_percent', 'heldout_matched_energy_change_db')} for r in short],
        'long': {k:long[k] for k in ('paired_convergence', 'heldout_error_change_percent', 'heldout_matched_energy_change_db')}, 'report': str(args.output/'fixed-ne-followup.html')}), flush=True)


if __name__ == '__main__':
    main()
