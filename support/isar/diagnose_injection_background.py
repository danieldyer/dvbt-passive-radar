"""Isolate the common recorded background at each injection level.

For IQ, the paired difference also contains the receiver's adaptive response to
the injection. It is explicitly a net operator difference, not a pure echo.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recorded_injection_fields import load_fields
from recovery_experiment import Experiment, SCALE
from report_measured_scatterer_phases import serial


def publish(root):
    result = json.loads((root/'background-diagnosis.json').read_text())
    geometry = json.loads((root/'localisation/analysis.json').read_text())
    if set(result['cases']) != {'profile', 'iq'}:
        raise ValueError('Both background-isolation controls are required')
    rows = ''
    changes = ''
    for level, case in result['cases'].items():
        a, b = case['paired_change_fits']
        rows += (f"<tr><td>{level}</td><td>{case['original_horizontal_error_m']:.3f} m</td>"
                 f"<td>{a['horizontal_error_m']:.4f} m</td><td>{b['horizontal_error_m']:.4f} m</td></tr>")
        d = case['selected_minus_truth']
        changes += (f"<tr><td>{level}</td><td>{d['mixed_rss']:+.3f}</td>"
                    f"<td>{d['background_rss']:+.3f}</td><td>{d['paired_change_rss']:+.3f}</td>"
                    f"<td>{d['cross_term']:+.3f}</td></tr>")
    info = geometry['conditional_thermal_position_information']
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Injection localisation and recorded background</title>
<style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:16px;background:#eef6fa}}</style>
<h1>Why the injected point's geographic position moved</h1>
<p class="note">The controls isolate the <strong>non-injected recorded data</strong> as the term moving the fit. They do not identify that term's physical origin, and do not establish unavoidable localisation error or statistical optimality.</p>
<h2>What “background” means here</h2><p>It is the exact complex profile data from the injection-free replay, after the receiver, reconstruction and compression stages. It contains the recording's other returns, thermal noise, remaining direct-path/clutter response and any receiver/reconstruction/model error already present. This experiment does not partition those components. “Interference” was too imprecise if understood as newly introduced adjacent-channel RF leakage.</p>
<p>Both injection levels retain that same recording. Agreement between their original position errors alone is not evidence about the cause. The causal control is subtraction of the paired injection-free profiles, with the range weighting, epochs and motion model held fixed within each case.</p>
<h2>Background-isolation controls</h2>
<table><tr><th>Level</th><th>Background retained</th><th>Paired change alone, starting at wrong solution</th><th>Paired change alone, starting at truth</th></tr>{rows}</table>
<p>These are explicit truth-controlled diagnostic refits, not new blind acquisition benchmarks. A separate profile test starting at truth with the background retained returns the same {geometry['controls']['recorded_background_truth_start']['horizontal_error_m']:.3f} m error. The exact finite-waveform point alone recovers within {geometry['controls']['finite_point_only']['horizontal_error_m']:.4f} m; the analytic point-only control reaches the numerical floor.</p>
<p>For the IQ case, the paired difference also contains the receiver's adaptive response/noise/coordinate changes induced by injection. It is a net operator difference and must not be labelled a pure isolated physical echo. The profile-level difference is through a frozen operator.</p>
<h2>Exact objective decomposition</h2><p>At each fixed motion p, profiling the stationary and constant complex point coefficients gives a linear residual projector R<sub>p</sub>. If y=b+s, the training objective is ||R<sub>p</sub>b||² + ||R<sub>p</sub>s||² + 2 Re⟨R<sub>p</sub>b,R<sub>p</sub>s⟩. The numerical residual decomposition is checked to relative error below 10⁻¹⁰. The table gives each term at the displaced solution minus its value at truth. Negative total differences favour the displaced location:</p>
<table><tr><th>Level</th><th>Total change</th><th>Background-only change</th><th>Paired-change-only change</th><th>Cross-term change</th></tr>{changes}</table>
<h2>Why this changes position while preserving focus</h2><p>The test assigned four bands to one site and ABC to a second site, {geometry['transmitter_baseline_m']:.1f} m apart. Its range-gradient condition number is {geometry['geometry']['range_gradient_condition_number']:.1f}. The 76 m error has only {abs(geometry['geometry']['normal_error_m']):.3f} m normal to the local equal-range contour; the rest is tangent to it. Independent constant complex band coefficients absorb constant phase offsets, while the fitted velocity/acceleration preserve most time-dependent phase. This test did not impose a calibrated absolute cross-band phase relationship.</p>
<img src="localisation/diagnosis.png" alt="Position error along an equal-range contour and training-cost slices">
<p>The ideal conditional thermal-noise local standard deviations along the strong and weak position directions are {info['principal_sigma_m'][0]:.4f} m and {info['principal_sigma_m'][1]:.2f} m, after profiling velocity, acceleration and complex coefficients. The actual fitted training residual has about {geometry['training_objective']['selected']['rss_per_complex_cell']:.1f} times the declared unit thermal power per complex cell. That excess may be correlated or structured; it must not be converted into a confidence interval by simply scaling the thermal bound. The present background likelihood is not validated as statistically efficient.</p>
<p><a href="background-diagnosis.json">Paired controls and exact decomposition</a> · <a href="localisation/analysis.json">Geometry, information and control fits</a> · <a href="index.html">Original injection report</a></p></html>'''
    (root/'geographic-localisation.html').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/data/injection_20261018'))
    parser.add_argument('--replay', type=Path, default=Path('build/injection-20261018'))
    args = parser.parse_args()
    baseline = load_fields(args.replay/'deterministic-baseline/profiles.bin')[0]
    results = dict(scope=__doc__, cases={})
    for level, replay, analysis in [('profile', 'exact-profile', 'exact-profile-fit'),
                                     ('iq', 'deterministic-iq', 'final-iq-fit')]:
        previous = json.loads((args.root/analysis/'analysis.json').read_text())
        fields, rotation, rx, tx, state, mid = load_fields(args.replay/replay/'profiles.bin')
        masks = [f.time < previous['training_end_s'] for f in fields]
        ex = Experiment(fields, rotation, rx, tx, state, mid, training=masks)
        truth = np.asarray(previous['truth_parameters'])
        selected = np.asarray(previous['selected']['parameters'])
        background = [f.apply(b.data['value']).ravel() for f, b in zip(fields, baseline)]
        mixed = [f.y.ravel() for f in fields]
        change = [y-b for y, b in zip(mixed, background)]
        case = dict(original_horizontal_error_m=float(np.linalg.norm((selected-truth)[:2]*SCALE[:2])),
                    paired_change_fits=[], objectives={})
        # Different seeds test whether the isolated component has the same
        # apparent location independently of the previous solution.
        for label, seed in [('previous_wrong_location', selected), ('truth', truth)]:
            fit, _ = ex.fit(change, seed, 120)
            error = (np.asarray(fit['parameters'])-truth)*SCALE
            record = dict(start=label, fit=fit, physical_error=error,
                          horizontal_error_m=float(np.linalg.norm(error[:2])))
            case['paired_change_fits'].append(record)
            print(level, label, 'isolated horizontal error m', record['horizontal_error_m'], flush=True)
        for label, p in [('truth', truth), ('selected', selected)]:
            rb = ex.profile(p, background, derivative=False)[0]
            rs = ex.profile(p, change, derivative=False)[0]
            ry = ex.profile(p, mixed, derivative=False)[0]
            discrepancy = np.linalg.norm(ry-rb-rs)/np.linalg.norm(ry)
            if discrepancy > 1e-10:
                raise ValueError('Fixed-coordinate profiled residual is not linear')
            case['objectives'][label] = dict(mixed_rss=ry@ry, background_rss=rb@rb,
                paired_change_rss=rs@rs, cross_term=2*rb@rs,
                linear_decomposition_relative_error=discrepancy)
        first, last = case['objectives']['truth'], case['objectives']['selected']
        case['selected_minus_truth'] = {k: last[k]-first[k]
            for k in ['mixed_rss', 'background_rss', 'paired_change_rss', 'cross_term']}
        results['cases'][level] = case
        (args.root/'background-diagnosis.json').write_text(json.dumps(serial(results), indent=2, allow_nan=False)+'\n')
        print(level, 'objective changes', serial(case['selected_minus_truth']), flush=True)
    publish(args.root)
    print('Background diagnosis written', flush=True)


if __name__ == '__main__':
    main()
