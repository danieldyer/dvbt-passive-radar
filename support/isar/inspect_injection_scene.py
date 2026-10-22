"""Measure injected-point strength and proximity to recorded matched responses.

All background measurements use the same frozen thermal metric and stationary
projection as the localisation fit. These are offline diagnostic controls.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from motion import C
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment, SCALE, NAMES
from report_measured_scatterer_phases import serial


def publish(root):
    data = json.loads((root/'scene-strength.json').read_text())
    original = json.loads((root/'background-diagnosis.json').read_text())['cases']['profile']['original_horizontal_error_m']
    rows = ''.join(f"<tr><td>{c['name']}</td><td>{c['nominal_thermal_coherent_snr_db']:.1f}</td>"
        f"<td>{c['injected_to_matched_background_db']:.1f}</td>"
        f"<td>{c['strongest_background']['range_offset_m']:.0f}</td>"
        f"<td>{c['strongest_background']['doppler_offset_hz']:+.3f}</td>"
        f"<td>{c['strongest_background']['relative_to_injected_db']:.1f}</td></tr>" for c in data['channels'])
    powers = f'<tr><td>Original</td><td>{original:.2f} m</td></tr>'
    powers += ''.join(f"<tr><td>+{c['added_power_db']:.0f} dB</td><td>{c['horizontal_error_m']:.2f} m</td></tr>"
                      for c in data['strength_controls'])
    control = data['doppler_separation_control']
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Injected point strength and nearby returns</title>
<style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Was the point weak, or close to other returns?</h1>
<p>The injection is strong relative to the declared thermal covariance. Other recorded responses are present a few hundred metres away, and background overlap affects the position estimate. The fit's weak geometric direction amplifies relatively small complex perturbations. This experiment does not identify the physical source of every recorded residual.</p>
<h2>Strength in the 19.5 s fitting interval</h2>
<table><tr><th>Band</th><th>Nominal coherent thermal SNR (dB)</th><th>Injected / background at point-template output (dB)</th><th>Largest background peak offset (m)</th><th>Peak Doppler offset (Hz)</th><th>Peak / injected power (dB)</th></tr>{rows}</table>
<p>Nominal thermal SNR is ||s||² after the fixed stationary projection and thermal range whitening, with unit complex-noise covariance assumed. The measured target/background ratio is ||s||⁴/|sᴴb|², with the same projection. It is a coherent projection ratio for this recording, not a Gaussian-noise SINR. The −53 to −63 dB injection power relative to the direct path does not specify its SNR.</p>
<h2>Where the nearby responses are</h2><img src="scene-strength.png" alt="Paired recorded-background and injected-point matched responses">
<p>Both columns use the same model, stationary projection, range metric and fixed colour scale, referenced to each channel's injected-point response. Cyan + marks the injected point; cyan × marks the largest background response in the displayed bank. Delay and Doppler are relative to the known injected point. Original-row templates retain its physical motion. Residual Doppler in this diagnostic is a tone at each 16-symbol-bin epoch, without additional range migration or curvature search. Therefore these are local fixed-trajectory response slices, not fully focused images of every competing target. The injection is about 225 m beyond the AIS ship-centre delay; the prominent recorded responses lie roughly 175–270 m nearer.</p>
<h2>Change only injected power</h2><table><tr><th>Power increase</th><th>Position error</th></tr>{powers}</table>
<p>These profile-level controls retain the identical measured background, epochs, frozen operator and weights, scaling only the known finite-waveform injected component. They use the previous solution as a diagnostic local start. Higher power reduces the displacement approximately in proportion to voltage ratio. This establishes sensitivity to the background relative to the injected point, rather than a fixed numerical position offset.</p>
<h2>Change motion while retaining signal energy</h2><p>A physical velocity perturbation increases Nine's Doppler by 0.5 Hz, with corresponding frequency-dependent changes in all bands. Midpoint position, acceleration, background and operator are unchanged. Each injected band is scaled by one known constant to preserve its full-aperture energy. Both cases use analytic profile injection and start at their own truth for this diagnostic. Position error changes from {control['original']['horizontal_error_m']:.2f} m to {control['separated']['horizontal_error_m']:.2f} m. This shows sensitivity to the point's placement in the structured background; it does not isolate one particular interfering scatterer. The changed motion also changes the measurement derivatives, so this is not a proof that spectral proximity alone accounts for the entire change.</p>
<p>These new power/motion controls are profile-level diagnostics; they are not additional full IQ replays. The earlier paired experiment established the same original error at both levels. None of these controls establishes that the present background likelihood is statistically efficient or that the measured localisation error is unavoidable.</p>
<p><a href="scene-strength.json">Metrics and control fits</a> · <a href="geographic-localisation.html">Background-isolation diagnosis</a> · <a href="index.html">Original injection report</a></p></html>'''
    (root/'scene-strength.html').write_text(html)


def response_bank(ex, c, truth, background, injected, ranges, doppler):
    f = ex.fields[c]; keep = ex.training[c]
    use = np.repeat(keep, len(f.offsets)); q = ex.q[c]
    time = f.time[keep]-ex.midpoint
    tone = np.exp(2j*np.pi*time[:, None]*doppler)
    qb = background[use]-q@(q.conj().T@background[use])
    qs = injected[use]-q@(q.conj().T@injected[use])
    qb = qb.reshape(len(time), -1); qs = qs.reshape(qb.shape)
    q_time = q.reshape(len(time), len(f.offsets), -1)
    position, velocity, _, _ = ex.kinematics(truth, c, [0, 0, 0])
    result = np.empty((2, len(ranges), len(doppler)))
    for i, bias in enumerate(ranges):
        h = f.trajectory(position, velocity, range_bias_m=bias).ravel()[use]
        ht = h.reshape(qb.shape)
        projected = np.einsum('trq,tr->tq', q_time.conj(), ht).T@tone
        energy = np.vdot(h, h).real
        norm = energy-np.sum(abs(projected)**2, axis=0)
        valid = norm > 100*np.finfo(float).eps*energy
        for plane, data in enumerate([qb, qs]):
            cross = np.sum(ht.conj()*data, axis=1)@tone.conj()
            result[plane, i] = np.divide(abs(cross)**2, norm,
                out=np.full(len(doppler), np.nan), where=valid)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/data/injection_20261018'))
    parser.add_argument('--replay', type=Path, default=Path('build/injection-20261018'))
    args = parser.parse_args(); root = args.root
    previous = json.loads((root/'exact-profile-fit/analysis.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    fields, rotation, rx, tx, state, mid = load_fields(args.replay/'deterministic-baseline/profiles.bin')
    injected_fields = load_fields(args.replay/'exact-profile/profiles.bin')[0]
    masks = [f.time < previous['training_end_s'] for f in fields]
    ex = Experiment(fields, rotation, rx, tx, state, mid, training=masks)
    truth = np.asarray(previous['truth_parameters'])
    background = [f.y.ravel() for f in fields]
    injected = [f.apply(p.data['value']-f.data['value']).ravel() for f, p in zip(fields, injected_fields)]
    ranges = np.unique(np.r_[np.arange(-400, 51, C/8e6/4), 0.])
    doppler = np.linspace(-.6, .6, 145)
    records, banks = [], []
    for c, f in enumerate(fields):
        use = np.repeat(masks[c], len(f.offsets)); q = ex.q[c]
        s = injected[c][use]; b = background[c][use]
        s -= q@(q.conj().T@s); b -= q@(q.conj().T@b)
        energy = np.vdot(s, s).real
        background_at_point = abs(np.vdot(s, b))**2/energy
        bank = response_bank(ex, c, truth, background[c], injected[c], ranges, doppler)
        banks.append(bank)
        ir, idop = np.unravel_index(np.nanargmax(bank[0]), bank[0].shape)
        record = dict(channel=c, name=NAMES[c], training_bins=int(masks[c].sum()),
            nominal_thermal_coherent_snr_db=10*np.log10(energy),
            nominal_thermal_snr_per_bin_db=10*np.log10(energy/masks[c].sum()),
            injected_to_matched_background_db=10*np.log10(energy/background_at_point),
            strongest_background=dict(range_offset_m=ranges[ir], doppler_offset_hz=doppler[idop],
                                      relative_to_injected_db=10*np.log10(bank[0, ir, idop]/energy)))
        records.append(record)
        print(serial(record), flush=True)
    result = dict(training_interval_s=[previous['interval_s'][0], previous['training_end_s']],
        scope=__doc__, channels=records, strength_controls=[],
        bank_note='Known point motion removed. Original-row range templates and fixed stationary projection; residual Doppler is a tone at each retained 16-symbol-bin epoch. This display search does not add residual-Doppler range migration or higher-order motion.',
        conditional_snr_note='Nominal thermal-noise SNR under the declared covariance, not total SINR. Background response uses the injection-free recording and includes other echoes and processing/model errors.')
    # Freeze the measured operator/background, changing only injected power.
    seed = np.asarray(previous['selected']['parameters'])
    for db in [12., 20.]:
        scale = 10**(db/20)
        fit, _ = ex.fit([b+scale*s for b, s in zip(background, injected)], seed, 100)
        error = (np.asarray(fit['parameters'])-truth)*SCALE
        result['strength_controls'].append(dict(added_power_db=db,
            horizontal_error_m=np.linalg.norm(error[:2]), physical_error=error, fit=fit))
        print('power control', serial(result['strength_controls'][-1]), flush=True)
    # A physical velocity change separates the point by +0.5 Hz in Nine.
    pos, _, _, _ = ex.kinematics(truth, 2, [0, 0, 0]); x = pos[len(pos)//2]
    gradient = sum((x-a)/np.linalg.norm(x-a) for a in [tx[2], rx])@rotation.T
    delta_rate = -.5*C/plan['rf_hz'][2]
    shifted = truth.copy()
    shifted[2:4] += delta_rate*gradient[:2]/(gradient[:2]@gradient[:2])/SCALE[2:4]
    gains = np.array([complex(*v) for v in plan['gains_real_imag']])
    analytic = [ex.independent_signal(truth, c, [g]) for c, g in enumerate(gains)]
    moved = [ex.independent_signal(shifted, c, [g]) for c, g in enumerate(gains)]
    for c in range(5):
        moved[c] *= np.linalg.norm(analytic[c])/np.linalg.norm(moved[c])
    result['doppler_separation_control'] = {}
    for name, target, point in [('original', analytic, truth), ('separated', moved, shifted)]:
        fit, _ = ex.fit([b+s for b, s in zip(background, target)], point, 100)
        error = (np.asarray(fit['parameters'])-point)*SCALE
        result['doppler_separation_control'][name] = dict(horizontal_error_m=np.linalg.norm(error[:2]),
            physical_error=error, fit=fit)
        print('Doppler control', name, serial(result['doppler_separation_control'][name]), flush=True)
    result['doppler_separation_control']['nine_shift_hz'] = .5
    (root/'scene-strength.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n')
    np.savez(root/'scene-strength-banks.npz', ranges_m=ranges, doppler_hz=doppler, banks=np.asarray(banks))
    fig, axes = plt.subplots(5, 2, figsize=(13, 16), layout='constrained')
    all_db = [10*np.log10(np.maximum(bank/10**(record['nominal_thermal_coherent_snr_db']/10), 1e-12))
              for bank, record in zip(banks, records)]
    top = max(10, np.ceil(max(np.nanmax(b[0]) for b in all_db)/5)*5)
    for c, (bank, record) in enumerate(zip(all_db, records)):
        for plane, label in enumerate(['Recorded background only', 'Injected point only']):
            ax = axes[c, plane]
            im = ax.pcolormesh(ranges, doppler, bank[plane].T, shading='auto', vmin=-35, vmax=top, cmap='magma')
            ax.plot([0], [0], '+', color='#00d6cf', markersize=10)
            if plane == 0:
                peak = record['strongest_background']
                ax.plot([peak['range_offset_m']], [peak['doppler_offset_hz']], 'x', color='#00d6cf')
            ax.set(title=NAMES[c]+' — '+label, xlabel='Excess path relative to injected point (m)',
                   ylabel='Residual Doppler relative to point (Hz)')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Conditional matched power relative to injected-point peak (dB)')
    fig.suptitle('Same stationary projection and thermal range metric as the fit\n19.5 s training interval; point motion removed; cyan + is injection truth')
    fig.savefig(root/'scene-strength.png', dpi=150)
    publish(root)
    print('Scene strength diagnostics written', flush=True)


if __name__ == '__main__':
    main()
