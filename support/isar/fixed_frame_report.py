"""Publish the paired fixed-N/E motion experiment, including negative results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('docs/data/isar_20261015')


def main():
    result = json.loads((ROOT/'fixed-ne-comparison.json').read_text())
    records = result['records']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    labels = [f'{r["interval_s"][0]:.2f}–{r["interval_s"][1]:.2f} s' for r in records]
    axes[0].bar(labels, [r['heldout_error_change_percent'] for r in records], color='#bd573e')
    axes[0].set(ylabel='Change in withheld squared prediction error (%)', title='Fixed N/E versus free yaw: higher is worse')
    axes[1].bar(labels, [r['heldout_matched_energy_change_db'] for r in records], color='#bd573e')
    axes[1].set(ylabel='Change in matched energy (dB)', title='Prediction matched to withheld data: lower is worse')
    for ax in axes:
        ax.axhline(0, color='black', lw=.7)
        ax.set_xlabel('Capture aperture')
    fig.suptitle('Same 30-second observations; translation is refitted with yaw disabled')
    fig.savefig(ROOT/'fixed-ne-statistics.png', dpi=160)
    plt.close(fig)
    rows, channel_rows, start_rows = [], [], []
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    max_kkt = 0.
    for r in records:
        free, fixed = r['variants']
        max_kkt = max(max_kkt, free['fit']['maximum_kkt_residual'], fixed['fit']['maximum_kkt_residual'])
        phase = np.asarray(r['frozen_attitude_point_phase_excursions_deg'])
        rows.append(f'<tr><td>{r["interval_s"][0]:.2f}–{r["interval_s"][1]:.2f}</td>'
            f'<td>{free["parameters"]} → {fixed["parameters"]}</td>'
            f'<td>{free["fit"]["objective"]:.6f} → {fixed["fit"]["objective"]:.6f}</td>'
            f'<td>{r["heldout_error_change_percent"]:+.2f}%</td>'
            f'<td>{r["heldout_matched_energy_change_db"]:+.3f} dB</td>'
            f'<td>{r["free_yaw_excursion_deg"]:.3f}°</td><td>{np.median(phase):.1f}° / {phase.max():.1f}°</td></tr>')
        for name, a, b in zip(names, free['per_channel_validation'], fixed['per_channel_validation']):
            error = 100*(b['prediction_error']/a['prediction_error']-1)
            gain = 10*np.log10(b['heldout_matched_energy']/a['heldout_matched_energy'])
            channel_rows.append(f'<tr><td>{r["window"]}</td><td>{name}</td><td>{error:+.3f}%</td><td>{gain:+.3f} dB</td>'
                f'<td>{a["prediction_phase_rms_degrees"]:.2f}° → {b["prediction_phase_rms_degrees"]:.2f}°</td></tr>')
        for trial in r['starts']:
            start_rows.append(f'<tr><td>{r["window"]}</td><td>{Path(trial["path"]).stem}</td>'
                f'<td>{trial["training"]:.9f}</td><td>{trial["heldout_error"]:.3f}</td>'
                f'<td>{trial["path"] == r["selected_fixed_start"]}</td></tr>')
    aperture_lengths = [v['actual_aperture_s'][1]-v['actual_aperture_s'][0] for r in records for v in r['variants']]
    low, high = result['colour_scale_db']
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Fixed N/E translation-only ISAR control</title>
<style>body{{font:17px/1.55 system-ui;max-width:1220px;margin:35px auto;padding:0 20px;color:#243040}}img{{max-width:100%;height:auto}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse;font-size:15px}}.note{{background:#fff0ce;padding:18px}}pre{{white-space:pre-wrap;background:#f3f4f6;padding:15px}}</style>
<h1>Removing yaw: translation-only imaging in fixed east/north axes</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="fixed-ne-comparison.json">Numerical results and exact input identities</a> · <a href="radar-acquisition.html">Earlier motion and video audit</a></p>
<p class="note">Yaw has been removed from the forward model and optimizer, and translation refitted. On these two overlapping 30-second apertures, it does not improve predictive statistics: error rises by {records[0]['heldout_error_change_percent']:.2f}% and {records[1]['heldout_error_change_percent']:.2f}%, while matched energy falls by {abs(records[0]['heldout_matched_energy_change_db']):.3f} and {abs(records[1]['heldout_matched_energy_change_db']):.3f} dB. These are conditional comparisons of the current point model, not a proof that the fitted free-yaw trajectory is the real ship heading.</p>
<h2>What changed</h2>
<p>The geographic model uses x(t,q) = c(t) + E q<sub>E</sub> + N q<sub>N</sub> + U q<sub>U</sub>, with E/N/U constant across the aperture. Only the horizontal centre trajectory c(t) is fitted. Angular velocity is exactly zero; roll and pitch remain fixed. Selected reflectors are transformed into N/E coordinates so their physical midpoint positions are unchanged. The initial centre trajectory is also unchanged before the refit. There is no rotation of a formed complex image and no new phase-only correction.</p>
<p>The paired models use the same original observations, eight previously selected reflectors per band, alternating 0.5-second training/validation blocks, conditional range whitening, stationary-nuisance subspace and physical translation bounds. Complex reflector amplitudes are profiled separately in each band. The renderer retains the same provisional channel gains and range origins. AIS supplies the existing coarse midpoint pose/velocity and the cyan display reference; no interpolated AIS dynamics enter this fit.</p>
<p>Five translation-only starts were tested: the original centre start and four starts following the original rotating support's corners. Starts are chosen solely by training likelihood. All fits converge; the selected paired fits have maximum KKT residual {max_kkt:.3g}. The centre start wins both comparisons. Numerical convergence does not establish a global optimum or validate the physical reflector support.</p>
<table><tr><th>Aperture (s)</th><th>Motion parameters<br>free → fixed</th><th>Training objective<br>free → fixed</th><th>Withheld error change</th><th>Matched energy change</th><th>Free-yaw excursion</th><th>Freezing-only phase excursion<br>median / maximum</th></tr>{''.join(rows)}</table>
<img src="fixed-ne-statistics.png" alt="Withheld error and matched-energy changes when yaw is removed">
<h2>Images on a common geographic grid</h2>
<p>Each pair uses exactly the same original rows and physical viewport. Actual integration lengths are {min(aperture_lengths):.6f}–{max(aperture_lengths):.6f} seconds. Pixel spacing before display interpolation is {result['scientific_grid_spacing_m']:g} m. All four panels share one fixed {low:g}–{high:g} dB scale, with its upper limit above the largest displayed value. Complex channel sums precede power detection. Cyan denotes the interpolated AIS antenna track and midpoint; it is not a hull-centre constraint.</p>
<a href="fixed-ne-comparison.png"><img src="fixed-ne-comparison.png" alt="Paired original-row coherent images with free yaw and fixed north/east translation"></a>
<p>A larger isolated image peak is not sufficient evidence of improved focusing: a broad ambiguity ridge can be bright while predicting less of the withheld signal. The independent prediction scores above do not use these image pixels, their grid spacing or their display scale.</p>
<h2>What the statistics mean</h2>
<p>Withheld squared error compares the fixed training prediction with data after applying the same declared stationary-nuisance projection. Matched energy is |ŷᴴz|²/(ŷᴴŷ), where ŷ is the training-fitted prediction and z the projected withheld observation. The change in this quantity measures how much withheld signal the fitted template captures. It is not an absolute SNR loss relative to an unknown ideal ship reconstruction.</p>
<p>The older conditional dictionary-efficiency metric can decrease even when prediction gets worse: its denominator is the energy explained by a separately fitted test dictionary, and that dictionary changes with the motion model. It must not be used alone to select a model. The raw absolute prediction and matched-energy comparisons are retained here instead.</p>
<p>The reflector support and coarse seeds were obtained in earlier work using broader data. This is therefore a paired conditional validation of the refit, not fully independent validation of discovery or point selection. The two apertures overlap by ten seconds. Their results do not establish the best model throughout the capture, unique geographic location, physical heading, or a validated inter-band scattering phase.</p>
<details><summary>Per-channel prediction statistics</summary><table><tr><th>Window</th><th>Channel</th><th>Error change</th><th>Matched-energy change</th><th>Prediction phase RMS<br>free → fixed</th></tr>{''.join(channel_rows)}</table><p>Phase RMS includes noise, glint, scene mismatch and motion error. It is not a direct measurement of oscillator or trajectory phase accuracy.</p></details>
<details><summary>Every translation-only starting point</summary><table><tr><th>Window</th><th>Start</th><th>Training objective</th><th>Withheld squared error</th><th>Selected by training</th></tr>{''.join(start_rows)}</table></details>
<h2>Why a slow turn can still matter coherently</h2>
<p>For a point displaced by δx, the first-order phase change is δφ = (2πf/c)(u<sub>T</sub> + u<sub>R</sub>)·δx. A small yaw change gives δx ≈ δψ(U × q). Thus the relevant small quantity is the resulting path change relative to a wavelength, not simply degrees turned over the aperture. A 0.5° change moves a point 150 m from the reference by about 1.31 m before bistatic projection.</p>
<p>The table's freezing-only phase excursions use exact path differences at the fitted points, after removing constant phase, and before refitting translation. They illustrate sensitivity under the free-yaw model; they do not measure actual vessel rotation. Translation refitting can absorb the common component but cannot generally reproduce the different motion of every extended-scene point.</p>
<p>The prior implausible heading oscillations remain rejected as a physical interpretation. The fixed-N/E control is now available and tested; its poorer score here is a reason to continue examining scattering support and shared range/phase geometry, not to label arbitrary yaw fluctuations as measured heading. The broader autofocus and calibration goal remains open.</p>
<h2>Verification and reproduction</h2>
<p>182 support tests passed. The new regressions check preservation of midpoint physical reflectors, centre position/velocity, exactly zero frame velocity, spline restriction and translation-start projection accuracy. Original-row validation checks the fitting approximation; the selected fits' maximum conditional template loss from midpoint evaluation is below 4.3×10<sup>−7</sup> dB. This small numerical discrepancy does not address physical model mismatch.</p>
<pre>env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 build/validation-venv/bin/python \
  support/isar/fit_fixed_frame_motion.py \
  --fields build/isar-20261015/full-fields \
  --seed build/isar-20261015/full-capture-radar-motion.json \
  --source build/isar-20261015/radar-rate-refined-motion \
  --output build/isar-20261015/fixed-ne-motion-30s \
  --windows 6 7 --duration 30 --offset-starts --resume</pre>
<p>Implementation: <a href="../../../support/isar/fixed_frame_motion.py">coordinate conversion and translation starts</a>, <a href="../../../support/isar/fit_fixed_frame_motion.py">paired fitter</a>, <a href="../../../support/isar/render_fixed_frame_comparison.py">original-row image comparison</a>. Production capture and range-compression code are unchanged.</p></html>'''
    (ROOT/'fixed-ne-validation.html').write_text(text)
    print(ROOT/'fixed-ne-validation.html')


if __name__ == '__main__':
    main()
