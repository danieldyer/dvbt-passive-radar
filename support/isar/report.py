"""Build the reviewable ship-association, propagation and imaging report."""
import json
from pathlib import Path

from multipath import symbol_psf_summary
from rigid_motion_report import render as render_rigid_motion


def main():
    root = Path(__file__).resolve().parents[2]
    destination = root/'docs/data/isar_20261013'
    work = root/'build/isar-20261013'
    delay = json.loads((destination/'companion-delay-tests.json').read_text())
    phase = json.loads((work/'autofocus-fine-90-210/shared-path-phase.json').read_text())
    coherence = json.loads((work/'coherence-90-210/coherence.json').read_text())
    point = json.loads((work/'point-response-90-210/point-test.json').read_text())
    full = json.loads((work/'full-scatterer-model-90-210/scatterer-validation.json').read_text())
    stationary = [json.loads((work/f'stationary-model-90-210/channel-{c}.json').read_text()) for c in range(5)]
    control = json.loads((destination/'stationary-phase-control.json').read_text())
    timing = json.loads((destination/'utc-alignment-test.json').read_text())
    dense = json.loads((work/'dense-scatterers-142-172/channel-3.json').read_text())
    symbols = symbol_psf_summary(root/'build/calibration-20261013/static-with-psf.gprf')
    (destination/'symbol-point-response.json').write_text(json.dumps(symbols, indent=2)+'\n')
    symbol_levels = [channel['maximum_response_350_to_750_m_db'] for channel in symbols['channels']]
    rigid_section = render_rigid_motion(work, destination)
    delay_rows = ''.join(f'<tr><td>{x["channel"]}</td><td>{x["paired_frames"]}</td>'
                         f'<td>{x["median_measured_companion_separation_m"]:.0f}</td>'
                         f'<td>{x["median_companion_relative_power_db"]:+.1f}</td>'
                         f'<td>{x["spectral_psf_max_350_to_750_m_db"]:.1f}</td></tr>' for x in delay['channels'])
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    phase_rows = ''.join(f'<tr><td>{name}</td><td>{x["held_out_coherent_power_gain_db"]:+.2f}</td></tr>'
                         for name, x in zip(names, phase['diagnostics']))
    coherence_rows = ''.join(f'<tr><td>{i+1}</td><td>{a:.2f}</td><td>{b:.2f}</td></tr>'
                             for i, (a, b) in enumerate(zip(coherence['interleaved_2s_held_out_efficiency_loss_db'],
                                                           coherence['first_half_to_second_half_efficiency_loss_db'])))
    peak = max(x['peak_amplitude'] for x in point['channels'])
    control_rows = ''.join(f'<tr><td>{ch["name"]}</td><td>{ch["regions"][0]["phase_p01_p50_p99_deg"][0]:+.2f} to '
                           f'{ch["regions"][0]["phase_p01_p50_p99_deg"][2]:+.2f}</td>'
                           f'<td>{ch["regions"][0]["adjacent_phase_step_absolute_p99_deg"]:.2f}</td></tr>' for ch in control['channels'])
    information_loss = max(max(ch['target_information_loss_db']) for ch in stationary)
    page = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RIO DELTA: association, companion return and ISAR validation</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;max-width:1120px;margin:36px auto;padding:0 24px;color:#20242a;background:#fafafa}}
h1{{line-height:1.2}} h2{{margin-top:2em}} img{{max-width:100%;background:white;border:1px solid #ddd}}
table{{border-collapse:collapse;width:100%;margin:18px 0}} th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #ddd}}
.status{{padding:16px;background:#fff4dc;border-left:4px solid #bf851a}} .caption{{color:#555;font-size:.94em}}
code{{background:#eee;padding:2px 5px}} a{{color:#075aa8}} figure{{margin:24px 0}} pre{{overflow:auto;background:#eee;padding:14px}}
</style>
<h1>RIO DELTA: companion-return checks and first ISAR images</h1>
<p>Capture: <code>live-vhf-20261013-193334</code>. Analysis updated 15 October 2026.</p>
<p class="status"><b>Formation and calibration remain in progress.</b> Exact GPU backprojection is implemented and passes
analytic phase tests. Independent-channel images are available. The real data have not yet passed the shared five-channel
phase-response test, and no refined transmitter coordinates or autofocus correction have been adopted.</p>
<p><a href="../calibration_20261013/site-refits.html">Latest phase-centre validation:</a>
all 48 site assignments have been evaluated with the corrected common-source extraction;
the report includes individual-band holdouts, receiver uncertainty and extended-vehicle controls.</p>

<h2>The companion is too delayed for a simple sea bounce</h2>
<p>A radar branch follows the unshifted AIS-predicted bistatic path. A second, often stronger branch lies about
480–520 m farther out during the 58–88 s diagnostic interval. This is the return previously selected as candidate D.
Its offset must not be absorbed into antenna calibration.</p>
<p>For a flat-water specular reflection on one leg, the extra path is
<code>sqrt(d²+(h₁+h₂)²) − sqrt(d²+(h₁−h₂)²) ≈ 2h₁h₂/d</code>.
Allowing reflections on both legs, nominal antenna heights of 750 m and 50 m, and ship scatterers as high as 60 m,
the extra path is at most about <b>3.1 m</b> in this geometry. This is a delay-scale calculation under the stated
two-ray model, not a comprehensive curved-Earth/rough-sea propagation simulation.</p>
<p>The 286.53 m × 40 m hull, aligned with the course prior, spans about 276–318 m of excess path in this interval.
True heading is unavailable, so this is conditional too. A longer route through a fixed reflector on land is plausible.
A different target or finite-waveform processing ambiguity is still possible. The data do not currently identify the reflector.</p>
<table><thead><tr><th>Channel</th><th>Paired frames</th><th>Separation (m)</th><th>Companion/direct power (dB)</th>
<th>Largest spectral PSF sidelobe, 350–750 m (dB)</th></tr></thead><tbody>{delay_rows}</tbody></table>
<p class="caption">Peak pairs are screening observations from unnotched 3 s maps, selected near the AIS rate.
Their reported relative powers are conditional on detection and should not be treated as unbiased RCS estimates.
The spectral PSF uses the archived |H|²/noise weights. It excludes finite-symbol ambiguity and distributed scattering;
it rules against a simple replica from this filter shape, not every possible processing artefact.</p>
<p>A separate replay check includes the actual reconstructed DVB-T symbol powers:
the zero-relative-Doppler point sidelobes over 350–750 m reach only
{min(symbol_levels):.1f} to {max(symbol_levels):.1f} dB across the five channels.
This agrees with the filter-spectrum check. It is still a point-response calculation,
not a complete distributed-target or finite-waveform ambiguity analysis.
<a href="symbol-point-response.json">Reconstructed-symbol check and scope</a>.</p>
<figure><a href="companion-delay-tests.png"><img src="companion-delay-tests.png" alt="Measured companion delay and spectral point response"></a></figure>

<h2>The aperture zero-Doppler notch is disabled</h2>
<p>Archived PROF records precede the sliding-map DC projection. The analysis now retains zero Doppler and no longer
excludes peaks within 2/CPI of DC. This recovers the ship through its near-zero path-rate turn. The archive still contains
the fixed stationary-background subtraction trained at capture start; that is a separate earlier transformation.</p>
<figure><a href="unnotched/ais-radar-association.png"><img src="unnotched/ais-radar-association.png" alt="Unnotched AIS and radar association"></a>
<figcaption class="caption">Red: AIS prediction using licensed site coordinates and file-creation UTC as a nominal origin.
No range bias, timing shift or antenna fit was used to force agreement. Grey intervals denote AIS gaps longer than 90 s.</figcaption></figure>

<h2>What the supplied AIS does and does not constrain</h2>
<p>The user-provided Data Docked export has 16 positions, median spacing 60 s, and a 392 s gap overlapping the latter
part of the recording. It contains speed and course, but no true heading or rate of turn. The current 90–210 s imaging
aperture avoids that long gap. Position interpolation uses reported speed/course tangents; it is a prior, not a
carrier-phase-accurate vessel trajectory. File creation is not a calibrated first-sample UTC timestamp.</p>
<p>Two explicit attitude hypotheses were compared: course derived from the interpolated position velocity, and a
shape-preserving interpolation of the reported course itself. Neither is relabelled as measured heading. Zero roll/pitch,
a 5 m ellipsoidal reference height and a 20 m body-plane offset are initial hypotheses.</p>

<h2>Independent images and the point-response control</h2>
<p>The operator evaluates <code>L = |x−T| + |x−R| − |T−R|</code> and its velocity derivative for every body point and
original profile epoch. It interpolates both matched-profile moments with the existing 64-tap symmetric sinc convention,
applies exp(+j2πfL/c), and sums with archived information weights. There is no slow-time resampling, local normalization,
image blending, or aperture DC projection. Large ECEF coordinates remain double precision on the host; the GPU uses
stable differences of centre-relative path lengths.</p>
<p>Each 120 s channel aperture contains about 110,294 valid rows. Forming 80,661 pixels took approximately 8.1–9.0 s
per channel in these runs, including preparation and input/output of the offline working arrays. This is a formation
measurement, not a full live-pipeline benchmark.</p>
<figure><a href="ais-prior-fine-images.png"><img src="ais-prior-fine-images.png" alt="Five independent AIS-prior images"></a>
<figcaption class="caption">One-metre grid, one fixed colour scale. No autofocus correction applied.
The conditional thermal normalization is not a calibrated false-alarm probability.</figcaption></figure>
<p><b>Do not interpret the stripes as hull detail.</b> A single known point passed through the same aperture and archived
spectral filter weights produces many of them. The point response has substantial ambiguity lobes; a narrow central fringe
alone does not establish unambiguous spatial resolution. The real return is broader than an isolated central fringe, but
its physical structure is not yet separated from motion error and these lobes.</p>
<figure><a href="point-response.png"><img src="point-response.png" alt="Independent and coherent five-channel point responses"></a></figure>
<p>The known-point peak is at the correct coordinates in all channels; the largest recovered amplitude is {peak:.7f}
for an injected unit echo. Coherent five-band addition also preserves the unit echo. The independent analytic GPU test
has maximum phase error 0.00062°. These validate numerical formation, not real transmitter locations or vessel motion.</p>
<p><a href="reported-course-images.png">Alternative reported-course attitude images</a> are retained for comparison.
They increase some peak powers, but do not improve the fixed-point multichannel validation. Sharpness alone is therefore
not used to select the attitude.</p>

<h2>Autofocus and coherent combination have explicit acceptance tests</h2>
<p>A smooth common phase model was fitted to fixed scatterer histories, profiling out each complex amplitude.
Constant and linear phase gauges were held fixed. Alternating 2 s time blocks were withheld from the phase fit.
Points were selected from the full prior image, so this holdout tests the phase fitting, not the initial feature selection.</p>
<table><thead><tr><th>Channel</th><th>Held-out power gain from shared phase correction (dB)</th></tr></thead>
<tbody>{phase_rows}</tbody></table>
<p>The shared correction yields no useful consistent gain; independent corrections can make validation worse.
They have <b>not</b> been applied to the displayed images. This is consistent with the documented difficulty of ordinary
PGA on narrowband passive ISAR: a strong image pixel need not isolate a stable point scatterer.
See <a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5236">Garry and Smith, Passive ISAR part II (2019)</a>.
Exact bistatic BP with motion information is also used in
<a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5622">Pisciottano et al., maritime DVB-T/DVB-S ISAR (2019)</a>.</p>
<p>A second test estimates a five-channel complex signature at a fixed body point and predicts withheld data.
The coherent score is compared with independent channel responses using the weighted Cauchy ratio. No complex-image
correlation is used: disjoint RF bands have different complex PSFs, making that simple correlation misleading.</p>
<table><thead><tr><th>Fixed body point</th><th>Interleaved 2 s holdout loss (dB)</th><th>First-half to second-half loss (dB)</th></tr></thead>
<tbody>{coherence_rows}</tbody></table>
<p>Transferring one point's signature to other points is also poor. This does not isolate a unique cause: unresolved
scattering, attitude/motion error, antenna geometry and frequency-dependent reflectivity remain coupled. It does show
that fitting one bright point's phases is insufficient evidence for a calibrated coherent image of the entire ship.</p>

<h2>Stationary controls pass; a more complete ship likelihood still needs work</h2>
<p>The stationary profile control was repeated at <b>32.64 ms cadence</b>, using a separate 30–90 s reference
and evaluating the 90–210 s ship aperture. The strong 0.6–5 km returns remain phase-stable across all five channels.
This argues against a common receive-chain phase reset as the explanation for the ship fit failures.
It does not validate absolute interband phase or every moving-target response.</p>
<table><thead><tr><th>Channel</th><th>Stationary phase, 1st–99th percentile (°)</th>
<th>Adjacent phase step, 99th percentile (°)</th></tr></thead><tbody>{control_rows}</tbody></table>
<figure><a href="stationary-phase-control.png"><img src="stationary-phase-control.png" alt="Stationary phase control at 32.64 millisecond cadence"></a></figure>
<p>The ship test now uses its complete complex range response and the corresponding range covariance, with
the original symbol epochs and the same 16-row averaging in both observation and prediction. A four-point model
still has a {full['independent_held_out']['efficiency_loss_db']:.2f} dB held-out signature mismatch with independent
channel amplitudes and {full['shared_held_out']['efficiency_loss_db']:.2f} dB with a shared five-band scattering response.
<b>These are model-rejection diagnostics, not measured DSP SNR losses or CRLB gaps.</b> They cannot be attributed
solely to transmitter phase-centre error.</p>

<h2>Preserving near-zero ship Doppler while treating stationary clutter</h2>
<p>The narrow ridge in the residual-rate spectrum follows the stationary-return prediction after the AIS translation
has been removed (green dashed line). This is distinct from the delayed moving companion discussed above. It overlaps
the ship during its turn, so a short-window zero-Doppler notch is particularly inappropriate here.</p>
<figure><a href="residual-motion-spectrum.png"><img src="residual-motion-spectrum.png" alt="Five-channel residual rate spectra with stationary and hull-motion predictions"></a>
<figcaption class="caption">Eight-second Hann windows are used only to diagnose spectral content. They do not change
the archived data or the rectangular-aperture backprojection. Cyan hull bounds use an explicit course-derived heading
hypothesis; the jumps at report boundaries expose a limitation of that hypothesis.</figcaption></figure>
<p>A support-only likelihood now fits constant-delay nuisance amplitudes jointly over the long aperture.
In whitened coordinates its projector is <code>P = I − S S†</code>; both measurements and target templates receive
the same projection. Target information is therefore <code>Aᴴ P A</code>, rather than assuming cancellation is free.
This is the standard matched-subspace construction described by
<a href="https://www.mssc.mu.edu/~daniel/mscs6960/spring2010/ScharfFriedlanderTSP94.pdf">Scharf and Friedlander (1994)</a>.
The slow-target losses caused by short-batch cancellation are also discussed by
<a href="https://iris.uniroma1.it/handle/11573/906920">Colone et al. (2016)</a>.</p>
<p>For the four tested AIS point trajectories across the five channels, the largest conditional information loss from
this long-aperture nuisance fit is <b>{information_loss:.3f} dB</b>. This is not a guarantee for an arbitrary ship trajectory,
nor a claim of optimal interference whitening. The nuisance coefficients and original observations are retained.
The stationary fit alone does not make the moving-scatterer model pass withheld-data validation.</p>

<h2>Additional checks before adopting a correction</h2>
<p>Continuous scatterer-position fitting, a 30 s Ten aperture with smoothly interpolated reported course as the heading
prior, and a 64-scatterer model have also been tested. The larger dictionary explains about
{100*dense['training_captured_fraction']:.0f}% of the projected training energy, but fails to predict held-out samples
({dense['held_out_efficiency_loss_db']:.2f} dB signature mismatch). Its regularization was selected on training data only.
A common smooth phase fit also worsens prediction. None of these fits has been adopted as autofocus or calibration.</p>
<p>The archive was created before receiver warm-up. Its original closure timestamp and continuous sample count bound
the capture-start offset to <b>0–{timing['offset_bounds_s'][1]:.2f} s</b> after the nominal creation time, assuming no filesystem
clock jump or subsequent rewrite. Closure includes drainage, so it is an upper bound, not a measured UTC epoch.
An exact motion/attitude shift within that range selects roughly
{timing['selected_by_training_only']['offset_s']:.1f} s on training data but still fails validation; it is not accepted UTC calibration.
This small permitted shift cannot explain the roughly 500 m companion separation.</p>
<p>The remaining task is to establish a motion and scattering model that predicts independent observations,
including uncertain heading and possibly nonconstant aspect-dependent scattering. Increasing image sharpness alone
continues to be insufficient evidence for refined antenna coordinates or coherent five-band ship formation.</p>

{rigid_section}
<h2>Reproducible artifacts</h2>
<p>Source tools: <code>support/isar/</code>. Full complex images and working profiles:
<code>build/isar-20261013/</code>. Source capture and AIS export are unchanged. Production DSP stages are unchanged;
the calibration probe has a trailing <code>--keep-dc</code> option.</p>
<p>Validation: 26 ISAR support tests, 77 calibration tests, GPU/double and independent moving/static point tests,
and the full-aperture five-channel point control. The active work is a physical motion/scattering/calibration model with
held-out validation, rather than accepting a phase fit solely because it concentrates an image.</p>
<p><a href="companion-delay-tests.json">Companion measurements</a> ·
<a href="ais-radar-association.json">AIS association metadata</a> ·
<a href="session.json">Session status and assumptions</a> ·
<a href="../calibration_20261013/vehicle-analysis.html#revised-input">Revised vehicle extraction and held-out calibration tests</a> ·
<a href="ais-options.html">AIS acquisition notes</a></p>
</html>'''
    (destination/'analysis.html').write_text(page)
    for source, name in ((work/'autofocus-fine-90-210/shared-path-phase.json', 'autofocus-validation.json'),
                         (work/'coherence-90-210/coherence.json', 'coherence-validation.json'),
                         (work/'full-scatterer-model-90-210/scatterer-validation.json', 'full-scatterer-validation.json'),
                         (work/'dense-scatterers-142-172/channel-3.json', 'dense-scatterer-validation.json'),
                         (work/'point-response-90-210/point-test.json', 'known-point-validation.json')):
        (destination/name).write_text(source.read_text())
    print(destination/'analysis.html')


if __name__ == '__main__':
    main()
