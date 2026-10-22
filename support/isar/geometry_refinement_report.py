"""Publish controlled coherent antenna-location fits and validation limitations."""
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

from spline_motion import SplineMotion
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    definitions = [('Fixed motion, gains + delays', 'geometry-range-control'),
        ('Fixed motion + antenna locations', 'coherent-geometry-fixed-motion'),
        ('Motion, gains + delays; nominal locations', 'coherent-motion-range-control'),
        ('Joint motion + antenna locations', 'coherent-geometry-joint-motion')]
    trials = []
    for label, name in definitions:
        model = json.loads((args.trial/(name+'.json')).read_text())
        trials.append((label, name, model, summarize(model['coherent_motion']['after'])))
    control, joint = trials[2], trials[3]
    gain = 10*np.log10(joint[3]['matched_energy']/control[3]['matched_energy'])
    improvement = 100*(1-joint[3]['prediction_error']/control[3]['prediction_error'])
    audit = json.loads((args.output/'numerical-audit.json').read_text())
    gpu = json.loads((args.output/'geometry-backprojection-validation.json').read_text())
    image = json.loads((args.output/'geometry-coherent-comparison.json').read_text())
    phase_names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    summaries, transfers = [], []
    for label, name in (('Nominal locations', 'coherent-motion-range-transfer-projected'),
                        ('Fitted locations', 'coherent-geometry-transfer-projected')):
        data = json.loads((args.trial/(name+'.json')).read_text())
        transfers.append((label, name, data))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for label, _, model, scores in trials:
        axes[0].plot(phase_names, scores['phase_rms_degrees_by_channel'], 'o-', label=label)
    axes[0].set(ylabel='One-second held-out prediction phase RMS (degrees)', title='Residual includes scattering changes and noise')
    axes[0].legend(fontsize=8)
    for key, label in [('antenna_information', 'Point coordinates held fixed'),
                       ('antenna_information_with_unknown_points', 'Point coordinates also unknown')]:
        e = np.array(joint[2][key]['antenna_information_eigenvalues_before_prior'])
        axes[1].semilogy(np.arange(1, 19), np.maximum(e, 1e-13), 'o-', label=label)
    axes[1].axhline(1., color='#6c737a', ls='--', label='Information in location prior')
    axes[1].set(xlabel='Location combination, weakest to strongest', ylabel='Conditional information / prior information', title='Most location combinations remain weak')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'geometry-residuals-and-conditioning.png', dpi=165)
    plt.close(fig)
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True, layout='constrained')
    for label, _, model, _ in (control, joint):
        for ax, channel in zip(axes, model['coherent_motion']['after']):
            blocks = channel['prediction_phase_by_one_second_block']
            ax.plot([b['time_s'] for b in blocks], np.degrees([b['phase_radians'] for b in blocks]), 'o-', ms=3, label=label)
    for name, ax in zip(phase_names, axes):
        ax.set(ylabel=name+' (degrees)', ylim=(-180, 180))
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel('Recording time (s)')
    fig.suptitle('Fixed-location control versus joint geometry · identical held-out bins\nPhase of the complete multi-point prediction; not an isolated point-target motion error')
    fig.savefig(args.output/'geometry-phase-residuals.png', dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout='constrained')
    for label, _, model, _ in (control, joint):
        curve = SplineMotion(model)
        t = np.linspace(*curve.interval, 601)
        path, yaw = curve.curves(t)
        axes[0].plot(t, path, label=label)
        axes[1].plot(t, np.degrees(yaw))
    axes[0].set_ylabel('Reference path correction (m)')
    axes[1].set(ylabel='Body-yaw correction (degrees)', xlabel='Recording time (s)')
    axes[0].legend()
    fig.suptitle('Physical path and yaw fits · roll/pitch fixed\nYaw is relative to the AIS course prior and the chosen scattering coordinates')
    fig.savefig(args.output/'geometry-motion-curves.png', dpi=155)
    plt.close(fig)
    def link(path, text):
        return f'<a href="{escape(os.path.relpath(Path(path).resolve(), args.output.resolve()))}">{escape(text)}</a>'
    rows = []
    for label, name, model, scores in trials:
        fit = model['coherent_motion']
        rows.append(f'<tr><td>{link(args.trial/(name+".json"), label)}</td><td>{fit["converged"]}</td>'
            f'<td>{fit["maximum_kkt_residual"]:.3g}</td><td>{scores["matched_energy"]:.1f}</td>'
            f'<td>{scores["conditional_coherent_loss_db"]:.3f} dB</td><td>{fit["antenna_bound_hit"]}</td></tr>')
        summaries.append({'name': label, 'input': str(args.trial/(name+'.json')), 'summary': scores,
                          'converged': fit['converged'], 'maximum_kkt_residual': fit['maximum_kkt_residual']})
    transfer_rows, maximum_extrapolation = [], 0.
    for label, _, data in transfers:
        for trial in data['trials']:
            channels = trial['channels']
            extrapolation = sum(c['extrapolated_stationary_energy'] for c in channels)/sum(c['raw_heldout_energy'] for c in channels)
            maximum_extrapolation = max(maximum_extrapolation, extrapolation)
            retention = min(c['test_target_information_retained_fraction'] for c in channels)
            transfer_rows.append(f'<tr><td>{label}</td><td>{"First → second" if trial["training_half"] == 0 else "Second → first"}</td>'
                f'<td>{"Half-aperture gains" if trial["refit_gains"] else "Frozen full-aperture gains"}</td>'
                f'<td>{trial["summary"]["conditional_coherent_loss_db"]:.2f} dB</td><td>{100*retention:.1f}%</td></tr>')
    antenna = joint[2]['antenna_fit']
    offsets = antenna['transmitter_offsets_enu_m']+[antenna['receiver_offset_enu_m']]
    prior = joint[2]['antenna_information_with_unknown_points']['prior_sigma_enu_m']
    posterior = joint[2]['antenna_information_with_unknown_points']['conditional_posterior_sigma_enu_m']
    location_rows = ''.join(f'<tr><td>{name}</td><td>{", ".join(f"{v:+.2f}" for v in offset)}</td>'
        f'<td>{", ".join(f"{v:.1f}" for v in before)}</td><td>{", ".join(f"{v:.1f}" for v in after)}</td></tr>'
        for name, offset, before, after in zip(phase_names+['Receiver'], offsets, prior, posterior))
    information = joint[2]['antenna_information_with_unknown_points']['antenna_information_eigenvalues_before_prior']
    weak = int(np.sum(np.array(information) < 1))
    numerical = audit['known_antenna_location_recovery']
    image_gain = image['results'][1]['maximum_conditional_matched_power_db']-image['results'][0]['maximum_conditional_matched_power_db']
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'roll_pitch_fixed': True,
        'location_gain_db': float(gain), 'heldout_error_improvement_percent': improvement,
        'location_directions_weaker_than_prior': weak, 'location_combinations': len(information),
        'maximum_stationary_extrapolated_to_observed_energy': maximum_extrapolation,
        'image_peak_gain_db': image_gain, 'trials': summaries,
        'phase_centres_accepted': False, 'absolute_heading_validated': False,
        'scope': 'Conditional common-scene fits and diagnostics; no production metadata changed.'}
    (args.output/'geometry-refinement.json').write_text(json.dumps(report, indent=2)+'\n')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship coherent geometry refinement</title>
<style>body{{font:17px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#233143}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #ccd3dd}}img{{max-width:100%;height:auto}}.status{{background:#e9eff7;border-left:4px solid #25659b;padding:18px}}.correction{{background:#fff1cf;padding:18px}}code{{font-size:14px}}a{{color:#175f9a}}</style>
<h1>Ship ISAR: transmitter/receiver residuals before additional attitude states</h1>
<p>Updated {report['updated_utc']}. Capture live-vhf-20261013-193334, interval 142–172 s.</p>
<p class="status">Roll and pitch remain fixed. Both the matched control and the joint motion/location fit converge numerically.
Freeing antenna locations adds <strong>{gain:.3f} dB</strong> of held-out coherent matched energy and reduces squared prediction error by {improvement:.3f}%.
The location solution reaches search bounds, and {weak}/18 location combinations remain weaker than their prior.
No phase-centre coordinates are accepted or written to production metadata.</p>
<h2>1. What is being fitted</h2>
<p>The common complex scene uses the eight shared subpixel points from the preceding experiment. Physical path and yaw corrections change the envelope delay, RF phase and symbol Doppler together. Five transmitter positions and one receiver position have the declared correlated site/antenna prior. Four relative complex channel gains are eliminated at every candidate, with Ten fixing the common gain/phase gauge. All five compressed-envelope origins are free: a delay-origin error must not masquerade as a surveyed antenna displacement. The scene and data receive the same stationary-nuisance projection.</p>
<p>In whitened coordinates the model is <code>y[c] = B[c] beta[c] + g[c] A[c](motion, locations, origin[c]) alpha + noise[c]</code>.
Linear scene/background coefficients are eliminated by least squares; gains are optimized to their checked stationary solution. This is the variable-projection construction described by
<a href="https://www.cs.umd.edu/~oleary/software/varpro.pdf">O’Leary and Rust, Variable Projection for Nonlinear Least Squares Problems</a>.
It does not guarantee a global solution, a frequency-independent ship response, or correctly calibrated real-data uncertainties.</p>
<h2>2. Numerical validation before fitting this recording</h2>
<p>All 61 support tests pass, including independent forward synthesis, all motion/location/origin derivatives, shared-gain profiling, the correct real-coordinate information projection, and a regression for background extrapolation. In a noiseless experiment with known motion/support and two identifiable antenna directions, the maximum recovered coordinate error is {numerical['maximum_antenna_coordinate_error_m']*1000:.4f} mm and the delay-origin error is {numerical['maximum_range_origin_error_m']*1000:.4f} mm. These are numerical checks, not claims of survey accuracy.</p>
<p>The independent five-channel Vulkan round trip now includes displaced physical transmitters/receiver as well as envelope biases and arbitrary channel gains. Combined phase error is {gpu['combined_phase_error_degrees']:.6f}°, complex error {gpu['combined_complex_error']:.3g}. The template never counts the carrier phase twice when shifting the envelope origin.</p>
<h2>3. Matched controls on identical held-out observations</h2>
<table><tr><th>Fit</th><th>Numerical convergence</th><th>KKT residual</th><th>Held-out matched energy</th><th>Conditional dictionary loss</th><th>Antenna bound hit</th></tr>{''.join(rows)}</table>
<p>Training uses alternate retained 17.4 ms bins; the others evaluate frozen target predictions. The dB loss compares this prediction with independent per-band responses in the same finite dictionary. It excludes unexplained scattering and temporal changes, so {joint[3]['conditional_coherent_loss_db']:.3f} dB is <strong>not</strong> a bound on total image SNR loss. The point support and earlier motion seeds also used broader data, so these are conditional comparisons.</p>
<img src="geometry-residuals-and-conditioning.png" alt="Residual phase and antenna-location information">
<img src="geometry-phase-residuals.png" alt="One-second prediction phase before and after joint location fitting">
<p>The remaining one-second phase residuals include unresolved reflectors, glint, noise and motion mismatch. They must not be reported as isolated physical path errors. Their size still prevents claiming a fully validated 30 s autofocus result.</p>
<img src="geometry-motion-curves.png" alt="Fitted path and yaw curves with and without antenna refinement">
<h2>4. A correction to the earlier time-transfer test</h2>
<p class="correction">The previous half-aperture test extrapolated stationary coefficients fitted in one half into the other half. Because the moving range crop exposes different portions of constant-delay responses, some coefficients are almost unobserved in training. Their extrapolated energy reached {maximum_extrapolation:.0f} times the actual observed energy. Those earlier transfer-error numbers cannot by themselves establish failure of motion or channel-phase calibration.</p>
<p>The corrected diagnostic projects the declared stationary response space from the validation observations <em>and</em> the frozen target prediction: <code>P = I − B B⁺</code>, score <code>||P y − P m||²</code>. Only background nuisance coefficients are profiled in validation; target motion, gains and point amplitudes are never fitted there. This discards the known nuisance subspace instead of pretending that poorly constrained background coefficients extrapolate accurately. The table reports target-template information retained. A regression demonstrates exact recovery with this extrapolation failure and confirms that an injected target phase error remains visible.</p>
<table><tr><th>Locations</th><th>Training → validation</th><th>Gain treatment</th><th>Corrected conditional loss</th><th>Minimum retained target information</th></tr>{''.join(transfer_rows)}</table>
<p>The corrected result still does not support one constant scattering scene across the entire 30 s interval. It diagnoses motion/scattering-model transfer jointly; it does not prove that the underlying captured data have lost coherence. Geometry, support and motion are fixed from full-aperture training bins, so this remains a conditional transfer diagnostic.</p>
<h2>5. What the data say about antenna locations</h2>
<p>The information calculation eliminates the common complex amplitudes, band gains, envelope origins and motion. The conservative curve additionally eliminates all shared point coordinates as real nuisance parameters. Antenna parameters use prior-standardized coordinates, so an eigenvalue below one indicates less information than the prior. The resulting optimistic location uncertainties barely improve in many directions. More numerical iterations cannot recover information absent from this scene/geometry.</p>
<table><tr><th>Channel</th><th>Trial east, north, up offset (m)</th><th>Prior sigma E/N/U (m)</th><th>Conditional sigma with unknown points (m)</th></tr>{location_rows}</table>
<p>These are local, thermal-model sensitivities, conditional on tower assignments, the discrete common-scene hypothesis and the AIS midpoint pose. They are not measured coordinate confidence intervals. Search bounds and model mismatch are not captured by this local Gaussian calculation.</p>
<h2>6. Actual coherent backprojection</h2>
<img src="geometry-coherent-comparison.png" alt="Exact GPU coherent image with nominal and fitted antenna locations">
<p>Both panels use the same original profile rows, fixed 30 s support and one colour scale. Each includes its fitted physical motion, channel gains and envelope origins; the second additionally changes the physical TX/RX rays and direct-path baseline. Peak matched power changes by {image_gain:+.3f} dB. This is a diagnostic image comparison, not a global SNR or hull-resolution claim.</p>
<img src="geometry-coherent-comparison-channels.png" alt="Five independent channel images and their complex coherent combination">
<p>The combined image is an information-weighted complex backprojection, not a sum of image magnitudes. No zero-Doppler notch or local image normalization is applied. The frame origin remains the AIS antenna prior; true heading and the hull centroid are not yet established.</p>
<h2>Remaining work</h2>
<p>Use these converged models as controls while separating temporal scattering changes from residual motion, then measure useful coherent aperture length with the corrected nuisance treatment. A defensible hull-centred video still requires that validation and a physical body-frame estimate. Additional roll/pitch states are not justified by this result.</p>
<p>{link(args.output/'geometry-refinement.json', 'Summary JSON')} · {link(args.output/'numerical-audit.json', 'Independent numerical audit')} · {link(args.output/'geometry-backprojection-validation.json', 'GPU validation')} · {link('build/isar-tests-20261015-geometry.log', '61-test log')} · {link(args.output/'numerical-audit.html', 'Earlier alignment stages')} · {link(args.output/'analysis.html', 'Full experiment history')}</p>
<p>{' · '.join(link(args.trial/(name+'.json'), label+' transfer details') for label, name, _ in transfers)}</p></html>'''
    (args.output/'geometry-refinement.html').write_text(page)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
