"""Publish the paired profile/IQ injection, including frozen held-out prediction."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample

from ais import read_csv
from field import observation, average
from flash_siting_report import EPOCH
from motion import C
from piecewise_ais import LocalAisReference
from quadratic_motion_kernel import wavenumber_focus
from recorded_injection_fields import read_profiles
from recovery_experiment import SCALE
from validate_recorded_injection import metrics

NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
COLOURS = ['#0072b2', '#009e73', '#e69f00', '#cc79a7', '#d55e00']


def dense_range(values, factor=16):
    """Display interpolation of the complex response with explicit zero guards."""
    n = len(values)
    padded = np.pad(values, (2*n, 2*n))
    return resample(padded, factor*len(padded))[2*n*factor:(3*n-1)*factor+1]


def wide_observations(path, plan, track):
    header, channels = read_profiles(path)
    result = []
    for c in range(5):
        ch = channels[c]
        n = len(ch['information'])//16*16
        meta = dict(channel=c, channel_configuration=header['applied_channels'][c],
                    first_range_sample=header['first_range_sample'])
        aperture = dict(time_s=ch['rows'][:n, 1], information=ch['information'][:n],
                        valid=ch['rows'][:n, 4] != 0, moments=ch['values'][:, :n], metadata_json=json.dumps(meta))
        result.append(observation(aperture, track, EPOCH, np.array(plan['transmitters_ecef_m'][c]),
                      np.array(plan['receiver_ecef_m']), 'course-prior', offsets=np.arange(-20, 21)))
    return result


def paths(time, state, plan, channel):
    t = np.asarray(time)-plan['midpoint_s']
    local = state[0]+t[:, None]*state[1]+.5*t[:, None]**2*state[2]
    rx = np.asarray(plan['receiver_ecef_m'])
    tx = np.asarray(plan['transmitters_ecef_m'][channel])
    position = rx+local@np.asarray(plan['ecef_to_enu'])
    return np.linalg.norm(position-rx, axis=1)+np.linalg.norm(position-tx, axis=1)-np.linalg.norm(tx-rx)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('docs/data/injection_20261018'))
    p.add_argument('--replay', type=Path, default=Path('build/injection-20261018'))
    args = p.parse_args();out = args.output
    plan = json.loads((out/'plan.json').read_text())
    cases = [json.loads((out/name/'analysis.json').read_text()) for name in ['exact-profile-fit', 'final-iq-fit']]
    if not all(case['complete'] and case['passed'] for case in cases):
        raise ValueError('Both matched-motion recovery stages must pass')
    comparison = json.loads((out/'exact-operator-comparison.json').read_text())
    controls = [json.loads((out/name/'analysis.json').read_text()) for name in
                ['profile-3s', 'profile-10s', 'profile-minus6', 'profile-plus6']]
    summaries = [json.loads((args.replay/name/'summary.json').read_text()) for name in
                 ['deterministic-baseline', 'exact-profile', 'deterministic-iq']]
    if not all(s['channels'] == summaries[0]['channels'] for s in summaries):
        raise ValueError('Paired replay row/decoder identities differ')
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout='constrained')
    arrays = [np.load(out/name/'observations.npz') for name in ['exact-profile-fit', 'final-iq-fit']]
    track = read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'), reference_height_ellipsoid_m=5., mmsi=563052200)
    baseline = wide_observations(args.replay/'deterministic-baseline/profiles.bin', plan, track)
    linear = wide_observations(args.replay/'exact-profile/profiles.bin', plan, track)
    iq = wide_observations(args.replay/'deterministic-iq/profiles.bin', plan, track)
    curve_comparisons = []
    for c, ax in enumerate(axes.flat[:5]):
        t = baseline[c]['time']
        predictions = []
        for d, colour, label in zip(arrays, ['#0072b2', '#d55e00'], ['Profile injection', 'IQ injection']):
            expected = d[f'injected_{c}'].reshape(len(t), -1)
            fitted = d[f'prediction_{c}'].reshape(len(t), -1)
            phase = np.degrees(np.angle(np.sum(expected.conj()*fitted, axis=1)))
            ax.plot(t, phase, color=colour, linewidth=1, label=label)
            predictions.append(fitted.ravel())
        curve_comparisons.append(dict(channel=c, **metrics(predictions[1], predictions[0])))
        ax.axvspan(plan['interval_s'][0], cases[0]['training_end_s'], color='#dddddd', alpha=.6)
        ax.set(title=NAMES[c], xlabel='Capture time (s)', ylabel='Predicted echo phase error (degrees)')
    axes[0, 0].legend(fontsize=9)
    axes[2, 1].bar(np.arange(5)-.18, [c['excluded']['coherence_loss_db'] for c in cases[0]['channels']], .36, label='Profile injection')
    axes[2, 1].bar(np.arange(5)+.18, [c['excluded']['coherence_loss_db'] for c in cases[1]['channels']], .36, label='IQ injection')
    axes[2, 1].set(xticks=np.arange(5), xticklabels=NAMES, ylabel='Excluded-interval mismatch loss (dB)',
                    title='Motion and band coefficients frozen after 161.5 s')
    axes[2, 1].legend(fontsize=9)
    fig.suptitle('Known moving point in the actual five-band recording\nGrey: fit; white: excluded prediction. No fitted time-varying phase screen.')
    fig.savefig(out/'phase-and-transfer.png', dpi=160);plt.close(fig)
    rx = np.asarray(plan['receiver_ecef_m']);rotation = np.asarray(plan['ecef_to_enu'])
    reference = LocalAisReference(track, EPOCH+plan['midpoint_s'], rotation, rx)
    coarse = np.array([reference(0., i) for i in range(3)])
    states = [np.asarray(plan['state_enu'])]
    gains = [np.array([complex(*v) for v in plan['gains_real_imag']])]
    for case in cases:
        state = coarse.copy();state[:, :2] += np.asarray(case['selected']['physical_correction']).reshape(3, 2)
        states.append(state)
        gains.append(np.array([complex(*v[0]) for v in case['selected']['coefficients_real_imag']]))
    offsets = baseline[0]['offset_samples']*C/8e6
    curves = [[], [], []];velocity = [[], [], []]
    rates = np.linspace(-.25, .25, 801)
    combined = [np.zeros(len(offsets), complex) for _ in range(3)]
    combined_rate = [np.zeros(len(rates), complex) for _ in range(3)]
    total = np.zeros(3)
    for c in range(5):
        data = baseline[c];time = data['time'];w = data['information']
        nominal = average(data['centre_path_m'], data['row_information'], 16)[0]
        signals = [linear[c]['value']-data['value'], linear[c]['value']-data['value'], iq[c]['value']-data['value']]
        for index in range(3):
            delta = paths(time, states[index], plan, c)-nominal
            focused = wavenumber_focus(signals[index], C/8e6, delta, plan['rf_hz'][c])
            amplitude = w@focused/np.sqrt(w.sum())
            curves[index].append(amplitude)
            tone = np.exp(2j*np.pi*plan['rf_hz'][c]/C*(time-plan['midpoint_s'])[:, None]*rates)
            score = (w*focused[:, len(offsets)//2])@tone
            velocity[index].append(abs(score)/np.sqrt(w.sum()))
            combined[index] += gains[index][c].conjugate()*(w@focused)
            combined_rate[index] += gains[index][c].conjugate()*score
            total[index] += w.sum()*abs(gains[index][c])**2
    for index in range(3):
        curves[index].append(combined[index]/np.sqrt(total[index]))
        velocity[index].append(abs(combined_rate[index])/np.sqrt(total[index]))
    curves = [[abs(dense_range(curve)) for curve in case] for case in curves]
    offsets = np.linspace(offsets[0], offsets[-1], len(curves[0][0]))
    scale = max(curves[0][-1])
    for name, x, values, label in [('range-response', offsets, curves, 'Excess path relative to the target (m)'),
                                    ('path-rate-response', rates, velocity, 'Residual excess-path rate (m/s)')]:
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout='constrained')
        for c, ax in enumerate(axes.flat):
            for index, (colour, text) in enumerate(zip(['#444444', '#0072b2', '#d55e00'],
                    ['Known true motion', 'Recovered from profiles', 'Recovered from IQ'])):
                ax.plot(x, 20*np.log10(np.maximum(values[index][c]/scale, 1e-8)), color=colour,
                        linewidth=1.2, label=text, linestyle='--' if index==0 else '-')
            ax.set(title=NAMES[c] if c<5 else 'Five channels coherently combined', xlabel=label,
                   ylabel='Conditional matched power (dB)', ylim=(-65, 2), xlim=(-150, 150) if name=='range-response' else (-.25, .25))
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Isolated injected-point response; one shared scale for every panel\n'
                     '0 dB = ideal combined range peak. Operational motion fits used background + target.')
        fig.savefig(out/(name+'.png'), dpi=160);plt.close(fig)
    peak_losses = [-20*np.log10(max(curves[i][-1])/max(curves[0][-1])) for i in [1, 2]]
    positional_errors = [np.linalg.norm(state[0, :2]-states[0][0, :2]) for state in states[1:]]
    original = Path('captures/live-vhf-20261013-193334/capture.dvblive').stat()
    index = json.loads(Path('build/calibration-20261013/histories/archive-index.json').read_text())
    if [original.st_size, original.st_mtime_ns] != index['identity'][1:]:
        raise ValueError('Original capture identity changed')
    summary = dict(passed=all(case['passed'] for case in cases) and all(c['passed'] for c in controls),
                   profile_combined_excluded_loss_db=cases[0]['combined']['excluded']['coherence_loss_db'],
                   iq_combined_excluded_loss_db=cases[1]['combined']['excluded']['coherence_loss_db'],
                   exact_operator_comparison=comparison, fitted_template_comparison=curve_comparisons,
                   focussed_combined_peak_losses_db=peak_losses, horizontal_position_errors_m=positional_errors,
                   controls=[dict(interval_s=c['interval_s'], additional_gain_db=c['additional_gain_db'],
                                  worst_excluded_loss_db=max(ch['excluded']['coherence_loss_db'] for ch in c['channels'])) for c in controls],
                   original_capture_unchanged=True, replay_channels=summaries[0]['channels'])
    summary['passed'] = bool(summary['passed'] and max(r['coherence_loss_db'] for r in comparison)<.05
                             and max(abs(20*np.log10(r['amplitude_ratio'])) for r in comparison)<.1)
    sources = ['support/validation/recorded_injection_probe.cpp', 'support/isar/recorded_injection_fields.py',
               'support/isar/validate_recorded_injection.py', 'support/isar/recovery_experiment.py',
               'support/isar/compare_recorded_injections.py', str(Path(__file__))]
    summary['source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    rows = ''.join(f"<tr><td>{NAMES[c]}</td><td>{r['coherence_loss_db']:.6f}</td><td>{20*np.log10(r['amplitude_ratio']):+.5f}</td><td>{r['phase_bias_deg']:+.5f}</td><td>{r['relative_complex_error']:.3%}</td></tr>" for c, r in enumerate(comparison))
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Paired recorded-background injection</title>
<style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:16px;background:#eef6fa}}pre{{overflow:auto;background:#eee;padding:16px}}</style>
<h1>Paired range-profile and channelised-IQ injection</h1>
<p class="note"><strong>Both levels recover the injected point's coherent response.</strong> On the excluded final 10.5 seconds, conditional five-band template mismatch is {summary['profile_combined_excluded_loss_db']:.5f} dB after profile injection and {summary['iq_combined_excluded_loss_db']:.5f} dB after IQ injection. This validates the declared point-motion case in measured interference; it does not establish a resolved ship image or a unique geographic motion fit.</p>
<h2>Data and protocol</h2><p>The original five-band capture supplies the background at 142–172 s. Its decoded waveform supplies a noiseless reference with recorded source timing, fitted response and explicit carrier/CFO coordinates. A single translating point follows independently evaluated Cartesian bistatic distances, including full physical RF phase. Its fixed target/direct amplitudes are approximately 0.067–0.216%. The estimator receives a coarse AIS-based position/velocity prior, a bounded correction search and a one-point, known-height model. The injected corrections, exact trajectory and constant complex band responses are not supplied as initial guesses or coefficients.</p>
<p>The same coarse range/rate/acceleration acquisition and six-parameter physical variable-projection fitter used in the existing ship analysis are used here. Motion and one constant complex coefficient per band are fitted using 142–161.5 s, selected by training cost, and frozen for 161.5–172 s. Every complete 16-symbol bin enters the final fits: 1,723 bins per channel, with native row times retained by the forward model. There is no fitted time-dependent phase screen or yaw adjustment.</p>
<h2>Two injection levels and the exact operator control</h2><p>The initial profile injector uses the measured waveform autocorrelation and analytic symbol moments. Its comparison with IQ injection has a small 2.4–6.8% complex-shape discrepancy, although the corresponding conditional mismatch loss is below 0.020 dB. That approximation was not silently treated as an exact finite CAF.</p>
<p>The final profile control transforms the known injected time waveform through the baseline's frozen finite range-compression operator, adding that response to the unchanged background profiles. The echo's window FFT is evaluated on the CPU with exported sample/phase/CFO coordinates; the production finite compressor supplies both C0/C1 moments and preceding-symbol context. The IQ case instead adds the same echo before the actual receiver, FEC, reconstruction, response and noise estimation, and range compression. Full point paths are evaluated independently of the estimator. The 64-sample synthesis path interpolation agrees with Cartesian distances to 2.0 nm.</p>
<p>An offline drain at each input-block boundary gives all three final replays identical decoder-feedback scheduling; it changes no production DSP formulas. Native rows and decoded-packet checksums agree exactly, with zero RS failures. The 2.4–6.8% complex-shape difference persists with the exact finite-operator control, so it cannot be attributed solely to the initial analytic pulse approximation. The IQ case re-estimates the response, noise and receiver coordinates, whereas the profile control freezes them; this experiment measures their net effect without isolating each contribution. Exact bitwise profile equality is not claimed. The corresponding conditional mismatch loss remains below 0.020 dB, gain difference below 0.005 dB and mean phase difference below 0.024 degrees. No output phase or gain is fitted in the operator comparison:</p>
<table><tr><th>Band</th><th>Mismatch loss (dB)</th><th>Gain difference (dB)</th><th>Mean phase difference (degrees)</th><th>Complex RMS difference</th></tr>{rows}</table>
<h2>Frozen prediction</h2><p>The reported template mismatch is −10 log<sub>10</sub>(|sᴴWŝ|² / [(sᴴWs)(ŝᴴWŝ)]), with the declared conditional range covariance and information weights. It measures the loss from using the recovered template in place of the known template, and is invariant to one overall complex scale. Gain and mean phase are therefore reported separately for the unaligned operator comparison.</p><img src="phase-and-transfer.png" alt="Individual phase errors and held-out loss">
<p>The plotted phase includes the constant coefficients learned on training data. The combined mismatch metric permits only one overall complex scaling; it penalizes disagreement between band phases. These losses are conditional on the declared range/noise weighting, not a claim of total sensitivity or a calibrated CRLB in arbitrary correlated clutter.</p>
<h2>Focused responses</h2><img src="range-response.png" alt="Individual and coherently combined range responses"><img src="path-rate-response.png" alt="Individual and coherently combined residual path-rate responses">
<p>For operator verification, these plots subtract the paired background to isolate the known injected signal. That subtraction is never supplied to motion acquisition or fitting. All six panels share one fixed scale. Range profiles are aligned by their own physical bistatic path; the rate cut holds that range trajectory fixed. Complex range responses are Fourier-interpolated 16-fold for plotting, with zero guards outside the saved crop; this adds no measured bandwidth or resolution. This is a point-response diagnostic, not a terrain-projected ship image.</p>
<h2>What is still unresolved</h2><p>The test fits a single point with planar quadratic translation, fixed height and fixed test antenna positions. Constant complex band responses remain nuisance parameters. The recovered horizontal positions are still about {min(positional_errors):.1f}–{max(positional_errors):.1f} m from the injected position: accurate phase prediction does not imply equally accurate geographic localization in this geometry. Multiple scatterer association, unknown support/attitude, scattering changes and antenna refinement are not validated by this experiment. No claim of convergence to a unique physical ship model follows.</p>
<p>The saved IQ was already retimed during capture. Neither injection level tests impairments introduced before that boundary, nor does a clean decoded-waveform injection reproduce every unmodelled impairment of a real reflected waveform. The original capture was not modified; its size and modification time still match the existing archive index. There are no numerical production-code changes in this work.</p>
<h2>Controls and reproduction</h2><p>Three-second and ten-second profile controls and injections 6 dB weaker/stronger all passed the declared 0.2 dB conditional mismatch gate. Their retained logs and individual results are linked below. Final paired operator acceptance requires less than 0.05 dB mismatch and 0.1 dB gain difference.</p>
<pre>build/release/support/validation/recorded_injection_probe prepare CAPTURE PLAN.json NEW_SOURCE_DIR NEW_REFERENCE_DIR
build/release/support/validation/recorded_injection_probe baseline CAPTURE PLAN.json SOURCE_DIR NEW_BASELINE_DIR
build/release/support/validation/recorded_injection_probe linear CAPTURE PLAN.json SOURCE_DIR NEW_PROFILE_INJECTION_DIR
build/release/support/validation/recorded_injection_probe inject CAPTURE PLAN.json SOURCE_DIR NEW_IQ_INJECTION_DIR
build/validation-venv/bin/python support/isar/validate_recorded_injection.py --profiles PROFILES.bin --plan PLAN.json --level linear --stride 1 --output NEW_FIT_DIR
build/validation-venv/bin/python support/isar/compare_recorded_injections.py
build/validation-venv/bin/python support/isar/report_recorded_injection.py</pre>
<p>Use <code>--level iq</code> for the IQ replay. Large decoded-source and native-profile working products remain under <code>build/injection-20261018</code>. All tooling is under <code>support</code>.</p>
<p><a href="summary.json">Summary and source hashes</a> · <a href="plan.json">Declared injection truth</a> · <a href="exact-profile-fit/analysis.json">Exact profile fit</a> · <a href="final-iq-fit/analysis.json">IQ fit</a> · <a href="exact-operator-comparison.json">Unaligned operator comparison</a> · <a href="profile-3s/analysis.json">3 s control</a> · <a href="profile-10s/analysis.json">10 s control</a> · <a href="profile-minus6/analysis.json">−6 dB control</a> · <a href="profile-plus6/analysis.json">+6 dB control</a></p></html>''')
    print(json.dumps({k: summary[k] for k in ['passed', 'profile_combined_excluded_loss_db', 'iq_combined_excluded_loss_db',
                                             'focussed_combined_peak_losses_db', 'horizontal_position_errors_m']}))
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
