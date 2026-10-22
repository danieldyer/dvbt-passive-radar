"""Publish the best retained ship image and evidence for remaining focus limits."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import struct

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    a = p.parse_args(); r = a.results
    audit = json.loads((r/'analysis.json').read_text())
    sensitivity = json.loads((r/'motion-sensitivity.json').read_text())
    root = Path('docs/data/isar_20261018')
    synthetic = json.loads((root/'known-hull-joint-phase-image-strong-defocus/analysis.json').read_text())
    fading = json.loads((root/'fading-acquisition-recovery-review/analysis.json').read_text())
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    source = Path(sensitivity['source']); validity = []
    for c in range(5):
        with (source/f'channel-{c}.bin').open('rb') as f:
            _, rows, _, _, _ = struct.unpack('<8sIIIf', f.read(24))
            geometry = np.fromfile(f, '<f4', count=rows*40).reshape(rows, 10, 4)
        validity.append(dict(channel=c, rows=rows, positive_information_rows=int(np.count_nonzero(geometry[:, 9, 3]>0))))
    clock_root = Path('docs/data/clock_audit_20261018')
    old = np.genfromtxt(clock_root/'controller-incumbent.csv', delimiter=',', names=True)
    new = np.genfromtxt(clock_root/'controller-weighted-common-horizon.csv', delimiter=',', names=True)
    np.testing.assert_array_equal(old['time_s'], new['time_s'])
    keep = (old['time_s']>=142) & (old['time_s']<172)
    difference_ns = (old['offset_samples']-new['offset_samples'])[keep]/64e6*1e9
    clock = dict(interval_s=[142,172], counterfactual_map_difference_span_ns=float(np.ptp(difference_ns)),
                 old_rate_std_ppm=float(old['rate_ppm'][keep].std()), new_rate_std_ppm=float(new['rate_ppm'][keep].std()),
                 actual_clock_error_measured=False)
    train, test = audit['temporal_calibration_diagnostics']; anchor = audit['calibration_anchor_point']
    row_table = ''.join(f'<tr><td>{names[v["channel"]]}</td><td>{v["blocks_used"]}</td>'
                        f'<td>{v["circular_phase_spread_degrees"]:.1f}°</td>'
                        f'<td>{v["maximum_conditional_thermal_phase_sigma_degrees"]:.2f}°</td></tr>'
                        for v in audit['differential_phase_checks'])
    sensitivity_rows = []
    for rate in [.001,.003,.01]:
        values = [v for v in sensitivity['records'] if v['perturbation']==f'heading_rate_{rate:g}_degps' and v['duration_s']==30]
        loss = max(z['coherence_loss_db'] for v in values for z in v['point_metrics'])
        spread = max(v['differential_point_phase']['phase_rms_degrees'] for v in values)
        sensitivity_rows.append(f'<tr><td>{rate:g}°/s</td><td>{spread:.1f}°</td><td>{loss:.3f} dB</td></tr>')
    synthetic_rows = ''.join(f'<tr><td>{html.escape(v["label"])}</td><td>{v["case"]["phase_only_five_band_loss_db"]:.6f} dB</td>'
                             f'<td>{-10*np.log10(min(v["case"]["phase_only_five_band_coherent_fraction_by_point"])):.6f} dB</td></tr>'
                             for v in fading['records'])
    maximum_synthetic_loss = max(v['coherence_loss_db'] for v in synthetic['known_point_coherence'])
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship image and remaining focus contributors</title>
<style>body{{font:17px/1.55 system-ui;max-width:1300px;margin:35px auto;padding:0 24px;color:#222}}img{{width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:9px 15px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff1d6;padding:16px}}code{{font-size:.9em}}</style>
<h1>Best retained ship image and remaining focus contributors</h1>
<p class="note"><strong>The strongest ship-like appearance remains the original 142–172 s aperture.</strong> It is reproducible but is not an accepted resolved hull or calibrated five-band reconstruction. New checks below identify limits beyond the clock controller. No capture, production DSP, scientific complex image or motion estimate was changed by this review.</p>
<h2>The image requested</h2>
<img src="best-ship-six-panels.png" alt="Five independent channel images and the nominal complex sum from the strongest retained aperture">
<p>The panels replot the saved complex images, with their original information, nominal unit band gains and one fixed 5–50 dB scale. These are the original forward/port coordinates, not geographic north/east. No peak alignment, magnitude fusion, image rotation, sharpening or fitted phase enters this replot. The original long comparison is <a href="../isar_20261015/physical-motion-comparison.png">here</a>; its retained numerical replay is <a href="../isar_20261017/ship-baseline-replay/analysis.json">here</a>. “Best” here means the retained ship-like appearance, not the best independently established physical reconstruction.</p>
<p>A crucial qualification: this hypothesis inherits an orientation excursion as fast as <strong>{abs(audit['original_heading_rate_extrema_degps'][0]):.2f}°/s</strong>. That is not measured vessel heading. The newer physically constrained models have not surpassed this appearance. A sharper image obtained by changing an unverified angular history cannot itself certify that history.</p>
<h2>1. New original-observation phase and calibration checks</h2>
<p>Eight common cells are selected from the complete independent-band image. The existing GPU operator then re-extracts their complex histories from all original observations, without refitting motion. Their sums reproduce the saved complex images to at most <strong>{max(audit['history_image_relative_errors'])*1e6:.2f} ppm RMS</strong>. Different channel epochs are integrated separately; RF phase is never interpolated between them. The selection used the full aperture, so the temporal split is conditional rather than a blinded discovery experiment.</p>
<img src="selected-cells.png">
<p>For the brightest one percent of cells in the declared central corridor, the nominal complex sum retains {100*audit['nominal_fraction_bright_corridor']:.1f}% of the sum of independent matched energies. At the independent peak it retains {100*audit['nominal_fraction_at_independent_peak']:.1f}%. The independent statistic permits more complex nuisance responses than the coherent model; these ratios are <strong>not target-SNR efficiencies or achievable-gain bounds</strong>. They do establish that the old nominal sum has appreciable destructive combination.</p>
<p>One constant complex response per band is estimated at cell {anchor+1} using 142–157 s, then frozen. At that cell, coherent/independent energy is exactly 100% on training by construction, but only <strong>{100*test['frozen_complex_response_fraction_by_point'][anchor]:.1f}%</strong> on 157–172 s. Summed over the eight selected cells it changes from <strong>{100*train['frozen_complex_response_sum_fraction']:.1f}% to {100*test['frozen_complex_response_sum_fraction']:.1f}%</strong>. This includes both phase and relative magnitude changes. It rejects that simple fixed-response transfer model for these histories; it does not prove that the channels lost acquisition coherence, that different physical reflectors dominate each frequency, or that an instrumental calibration alone can recover the missing energy.</p>
<img src="phase-and-calibration-transfer.png">
<p>The same-channel phase difference between cells {anchor+1} and {audit['comparison_point']+1} is particularly useful: <code>(z₁ exp(iφ)) conjugate(z₂ exp(iφ)) = z₁ conjugate(z₂)</code>. A multiplicative channel phase common to both histories cancels exactly. During the bright 158–166 s period, the variation below remains even when both two-second cells exceed 20 dB conditional matched power.</p>
<table><tr><th>Band</th><th>Usable 2 s blocks</th><th>Circular phase spread</th><th>Largest conditional thermal σ</th></tr>{row_table}</table>
<p>There are only two high-power SBS blocks, so its spread is a particularly weak summary. The thermal column assumes independent point errors; these image histories actually share correlated unresolved contributions. It is a scale comparison, not a calibrated hypothesis-test p-value. Large relative phases nevertheless show that one common multiplicative phase correction is not a complete explanation. Wrong differential target motion, unresolved scene mixtures and range-dependent instrumental effects remain distinguishable candidates. No phase screen is applied.</p>
<p>The two-second matched-power participation durations are {', '.join(f'{n} {v:.1f} s' for n,v in zip(names,audit['bright_cell_two_second_power_participation_duration_s']))}. These use <code>2 (ΣP)² / ΣP²</code> across fifteen bins and describe concentration of measured power. They are not effective coherent apertures or point-information bounds. In particular the ABC response is strongly dominated by the brief flash; a nominal thirty-second window does not mean that one stable scatterer supplied thirty seconds of useful angular coverage.</p>
<h2>2. New controlled differential-motion sensitivity</h2>
<p>Exact range differences are calculated using the saved rays, frames, epochs and information at two declared material points 300 m apart. One constant phase is profiled per point. The differential-point statistic additionally cancels any common phase screen. These are perturbation experiments around the legacy hypothesis, not fits or measured heading errors.</p>
<img src="motion-versus-site-sensitivity.png">
<table><tr><th>Controlled heading-rate error</th><th>Worst-band differential RMS, 30 s</th><th>Worst individual-point temporal loss, 30 s</th></tr>{''.join(sensitivity_rows)}</table>
<p>Even a very slow heading-rate error can therefore matter on a hull-scale lever arm. This does not justify unconstrained yaw, roll or pitch fitting: a valid model must explain different spatial cells with one physical motion and transfer to excluded epochs. The constant-angle/body-coordinate ambiguity also remains. The height/TX/RX comparison concerns differential <em>temporal</em> phase only, with all other parameters fixed. It does not dismiss antenna uncertainty: antenna offsets can still produce large constant relative band phases and affect geographic registration.</p>
<h2>3. What the synthetic success actually establishes</h2>
<p>The current GPU self-test agrees with its independent double-precision operator and gives 0.000618° maximum analytic point-phase disagreement. The six-coordinate conditional motion-recovery, forward-model and derivative tests pass again. This is strong numerical evidence for those operators, not a proof of the real scene model.</p>
<p>The eleven-source, off-grid unknown-image/common-path control has worst material-point motion-coherence loss <strong>{maximum_synthetic_loss:.4f} dB</strong>. It supplies coarse geometry, angular motion and constant band calibration, uses 30 dB integrated SNR per point, and estimates a restricted nonlinear common path with its linear gauge fixed. It does not jointly discover the actual ship pose, transmitter sites and band responses. <a href="../isar_20261018/joint-phase-acquisition-review/index.html">Original control and matched images</a>.</p>
<p>The separate six-point fading control supplies support and attitude. Its performance depends strongly on acquisition:</p>
<table><tr><th>Initializer</th><th>Mean phase-only loss</th><th>Worst material-point loss</th></tr>{synthetic_rows}</table>
<p>Thus “works on synthetic data” needs those qualifications. Some plausible initializers fail even the declared synthetic case. The successful short joint acquisition is real progress, but its controlled interfering return is independently synthesized and scaled to the recording's crop power; it is not arbitrary measured interference. Neither control is the fully blind unknown-attitude/unknown-calibration problem. <a href="../isar_20261018/fading-acquisition-recovery-review/index.html">Acquisition cases and scope</a>.</p>
<h2>4. Other explanations checked or still open</h2>
<p><strong>Dropped samples and FEC erasures:</strong> all five original 142–172 s packets retain positive information on every one of their 27,573–27,574 rows. The capture summary reports no missing samples, host discards, timestamp gaps or receiver resets. Those known discontinuity/erasure mechanisms do not explain this aperture. This does not certify against undetected waveform or capture-time processing errors.</p>
<p><strong>Clock controller:</strong> over this best aperture, the old and fixed controller-only replay maps differ by {clock['counterfactual_map_difference_span_ns']:.2f} ns peak to peak; old/fixed applied-rate standard deviations are {clock['old_rate_std_ppm']:.4f}/{clock['new_rate_std_ppm']:.4f} ppm. These are counterfactual controller outputs from the same timing observations, not measurements of true SDR-clock error. The defect is much larger late in the recording. Timing remains plausible, but it is not established as the explanation for the much larger differential phase variations above. <a href="../clock_audit_20261018/index.html">Clock audit and physical-waveform tests</a>.</p>
<p><strong>Finite-aperture point spread:</strong> the gapped RF coverage and limited effective angular coverage leave sidelobes and sometimes a long ridge even for an exactly matched point. Backprojection is an adjoint, not an inverse deconvolution. Existing <a href="../isar_20261017/exact-point-and-hull-controls/window-0333.32/comparison.png">known point/hull controls</a> separate this from motion failure. Sparse pictures are not accepted as evidence unless they predict excluded complex observations.</p>
<p><strong>Scene/likelihood mismatch and acquisition:</strong> the existing physically calibrated flash models improve short-fold prediction but transfer poorly across two-second omissions. Correlated clutter, unresolved moving contributions, aspect-dependent complex scattering and imperfect target association remain candidates. The new same-channel and temporal-transfer results make these higher-priority checks than simply adding a more flexible common phase spline. They do not yet select a unique cause.</p>
<h2>Literature and next discriminating test</h2>
<p><a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5233">Garry and Smith's passive-ISAR framework</a> treats translational and apparent rotational compensation separately and ties physical image scaling to viewing geometry. This supports the distinction tested above; it does not supply our unknown vessel heading. <a href="https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.12030">Zhu et al.'s multiband compensation method</a> explicitly assumes a common point-scattering model. Its successful fusion demonstrations do not establish that our unresolved histories meet that assumption.</p>
<p>The most discriminating extension is a blind injected extended target in the measured residual background, with angular motion and constant band responses hidden as well as the nonlinear path. It should use the same association/search as the recording, then transfer a frozen physical fit across contiguous omitted intervals and spatial cells. That harder end-to-end recovery is <strong>not yet established</strong>. A fresh clock-fixed capture will test the upstream fix, but cannot substitute for this motion/model validation.</p>
<h2>Reproduction and evidence</h2>
<p><a href="analysis.json">New history/calibration results and source hashes</a> · <a href="motion-sensitivity.json">Exact perturbation results</a> · <a href="diagnostic-tests.log">Three new diagnostic checks</a> · <a href="recovery-tests.log">Three motion/forward-model checks rerun</a> · <a href="backprojection-tests.log">GPU/double/analytic and history-partition checks</a>. Tools are under <code>support/isar</code>; no production code changed in this investigation.</p>
</html>'''
    (r/'index.html').write_text(text)
    sources = [r/'analysis.json', r/'motion-sensitivity.json', Path(__file__),
               Path(__file__).with_name('audit_ship_motion_sensitivity.py'), Path(__file__).with_name('test_focus_diagnostics.py')]
    provenance = {str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in sources}
    (r/'review-summary.json').write_text(json.dumps(dict(complete=True, original_packet_validity=validity,
        counterfactual_clock_comparison=clock, source_sha256=provenance,
        whole_capture_focus_accepted=False, production_changes=False), indent=2)+'\n')


if __name__ == '__main__': main()
