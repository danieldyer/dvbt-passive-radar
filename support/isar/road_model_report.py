"""Publish the matched road averaging, covariance and reflecting-point checks."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

import numpy as np

from scene_dictionary import fingerprint


def verified(path):
    data = json.loads(path.read_text())
    if not data['complete']:
        raise ValueError('Incomplete experiment: '+str(path))
    for source, expected in data['source_sha256'].items():
        if fingerprint(Path(source)) != expected:
            raise ValueError('Scientific input changed: '+source)
    return data


def checked_fit(fit):
    best = fit['starts'][fit['selected_start']]
    result = best['fit']
    if (not result['converged'] or result['maximum_kkt_residual'] > 1e-7
            or result['maximum_constraint_violation'] > 1e-7):
        raise ValueError('Require stationary selected fits')
    np.testing.assert_allclose(result['objective'],
        min(s['fit']['objective'] for s in fit['starts']), atol=1e-12, rtol=0)
    return result['maximum_kkt_residual']


def table(headers, rows):
    cells = lambda row, tag: '<tr>'+''.join(
        '<'+tag+'>'+html.escape(str(value))+'</'+tag+'>' for value in row)+'</tr>'
    return '<table>'+cells(headers, 'th')+''.join(cells(row, 'td') for row in rows)+'</table>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binning', 'noise', 'reflectors', 'tests'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--joint-experiment', type=Path,
        help='Archive the current state of the matched common-origin geometry extension.')
    p.add_argument('--geometry-information', type=Path,
        help='Completed local information audit with unknown reflecting points.')
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    paths = [args.binning, args.noise, args.reflectors]
    binning, noise, reflectors = [verified(path) for path in paths]
    labels = [[t['label'] for t in data['targets']] for data in (binning, noise, reflectors)]
    if labels[0] != labels[1] or labels[0] != labels[2]:
        raise ValueError('Require identical target sets')
    log = args.tests.read_text()
    tests = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$', log)
    if not tests:
        raise ValueError('Require a completed passing support suite')
    rows, noise_rows, body_rows = [], [], []
    summary = []
    for b, n, s in zip(binning['targets'], noise['targets'], reflectors['targets']):
        label = b['label']
        information = {
            'maximum_averaging_snr_loss_db': max(c['matched_snr_loss_db'] for c in b['channels']),
            'maximum_local_information_loss_db_cutoff_1e_8': max(
                c['local_information'][0]['worst_information_loss_db'] for c in b['channels']),
            'maximum_fixed_empirical_metric_loss_db': max(
                c['fixed_empirical_metric_snr_loss_db'] for c in b['channels'])}
        rows.append([label, *[f'{v:.6f}' for v in information.values()]])
        for trial in n['trials']:
            checked_fit(trial['fit'])
        for metric, comparison in n['comparison'].items():
            noise_rows.append([label, metric,
                f"{comparison['moving_matched_energy_change_db']:+.5f}",
                f"{comparison['moving_prediction_error_change_percent']:+.5f}",
                f"{comparison['full_waveform_prediction_error_change_percent']:+.3f}"])
        body = []
        for trial in s['trials']:
            kkt = checked_fit(trial['fit'])
            before = [sum(c[key] for c in s['point_prediction'])
                for key in ('prediction_error', 'matched_energy')]
            after = [sum(c[key] for c in trial['fit']['prediction'])
                for key in ('prediction_error', 'matched_energy')]
            comparison = {'prediction_error_change_percent': 100*(after[0]/before[0]-1),
                'matched_energy_change_db': float(10*np.log10(after[1]/before[1]))}
            for key, value in comparison.items():
                np.testing.assert_allclose(value, trial['comparison'][key], atol=1e-12, rtol=1e-12)
            body.append({'model': trial['name'], **comparison,
                'bound_hits': trial['offset_bounds_hit'], 'maximum_kkt_residual': kkt,
                'original_row_template_loss_db': trial['maximum_original_row_moving_template_loss_db']})
            body_rows.append([label, trial['name'], f"{comparison['matched_energy_change_db']:+.5f}",
                f"{comparison['prediction_error_change_percent']:+.5f}", trial['offset_bounds_hit']])
        summary.append({'label': label, 'binning': information,
            'noise_comparison': n['comparison'], 'reflecting_points': body})
    degrees = [check['complex_degrees_minmax'] for t in noise['targets'] for check in t['pooled_noise_checks']]
    losses = [check['conditional_large_bin_count_weight_loss_db_max']
        for t in noise['targets'] for check in t['pooled_noise_checks']]
    source_paths = paths+[args.tests, Path(__file__)]
    report = {'complete': True, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'targets': summary, 'tests_passed': int(tests[1]),
        'source_sha256': {str(path): fingerprint(path) for path in source_paths},
        'accepted_calibration': False, 'production_changed': False,
        'decision': 'Averaging loss is small in the declared local Gaussian model. Pooled noise weights have mixed recorded-data prediction and are not adopted. Per-band reflecting points give small gains but frequent physical-bound hits; these are sensitivity results, not measured dimensions or antenna calibration.'}
    information_note = ''
    if args.geometry_information:
        geometry_information = verified(args.geometry_information)
        report['geometry_information'] = geometry_information
        report['source_sha256'][str(args.geometry_information)] = fingerprint(args.geometry_information)
        information_rows = []
        for case in geometry_information['cases']:
            value = case['combined']
            information_rows.append([
                'unknown per-band positions' if case['free_ship_reflector_positions'] else 'fixed positions',
                'unknown' if case['free_vehicle_reflectors'] else 'held known',
                f"{case['cutoff']:.0e}",
                str(value['directions_weaker_than_declared_prior'])+'/18',
                f"{max(value['information_eigenvalues_before_prior']):.5g}"])
        # Check the numerical property on the recorded data as well as in the
        # independent synthetic tests. Do not silently clip negative differences.
        for i in range(0, len(geometry_information['cases']), 2):
            fixed, free = geometry_information['cases'][i:i+2]
            if fixed['free_vehicle_reflectors'] or not free['free_vehicle_reflectors']:
                raise ValueError('Expected matched known/unknown vehicle-point information')
            for name in ('ship_only', 'vehicles_only', 'combined'):
                a, b = [np.asarray(case[name]['information_before_prior']) for case in (fixed, free)]
                if np.linalg.eigvalsh(a-b).min() < -1e-12*max(np.linalg.norm(a), np.finfo(float).tiny):
                    raise ValueError('Unknown body offsets spuriously added antenna information')
        information_note = '''<h2>How much precision comes from assuming reflector positions?</h2>
<p>This audit uses the completed fixed-antenna control's fitted mean, motion, amplitudes and physical vehicle offsets. It compares treating those offsets as known with treating them as unknown real nuisance parameters. Complex gains and local target motion are removed first, then origins shared across targets and the additional body offsets. Nested orthogonal projections ensure that adding unknowns cannot add information, including when small numerical singular values are truncated.</p>
<p>All likelihood weights and fitted means remain identical within each comparison. Neither motion penalties nor search bounds count as measurements. Information eigenvalues use the declared antenna prior as the coordinate scale: one means data information equal to that prior in that direction. These are conditional local sensitivities, not calibrated confidence intervals for the actual antennas.</p>'''+table(
            ['Ship reflector model', 'Vehicle offsets', 'Rank cutoff', 'Directions weaker than prior',
             'Largest information eigenvalue'], information_rows)+'''
<p>Allowing independent unknown three-dimensional ship points is a deliberately permissive comparison; it is not an assertion that a ship lacks shared physical structure. Conversely, fixed fitted points are not surveyed anchors. The bracket shows that physical scene correspondence matters to the calibration. It does not prove that a better supported shared scene, other road branches or additional views cannot resolve it.</p>
<p>Independent tests compare sequential elimination with a single dense Gaussian projection, recover the original common-point information at zero offsets, and verify non-increasing information when offsets become unknown over four numerical rank thresholds. The same monotonicity check also passes on every recorded-data case.</p>'''
    joint_note = ''
    if args.joint_experiment:
        manifest = args.joint_experiment/'experiment.json'
        raw = manifest.read_bytes(); state = json.loads(raw)
        for group in ('implementation_sha256', 'additional_sources_sha256'):
            for path, expected in state['identity'][group].items():
                if fingerprint(Path(path)) != expected:
                    raise ValueError('Joint geometry experiment input changed: '+path)
        report['joint_geometry_snapshot'] = {'manifest': str(manifest), 'state': state,
            'scope': 'Saved snapshot; the running experiment may later advance.'}
        checks = state.get('recorded_derivative_checks', [])
        status = ('complete; the transfer result is retained in the snapshot' if state['complete'] else
            'still running; no new geometry-transfer outcome is claimed')
        joint_note = f'''<h2>Joint origin and antenna extension</h2>
<p>The same reflecting-point model now has explicit derivatives for common per-band range origins shared with the ship. Zero offsets reproduce the previous joint objective and all original parameter derivatives. Known displaced reflectors recover their synthetic timing offsets, and the original model's bounds and single antenna prior remain intact.</p>
<p>The matched fixed/free antenna experiment is {status}. Its original-row derivative audit completed {len(checks)} probes, with maximum absolute normalized-objective derivative discrepancy {max((c['absolute_error'] for c in checks), default=0.):.3g}. O/K/Q and the ship fit calibration; A is completely excluded. This numerical check verifies the enlarged forward model, not physical accuracy of its fitted coordinates.</p>'''
    bin_table = table(['Target', 'Averaging SNR loss (dB)', 'Worst resolved local information loss (dB)',
        'Fixed empirical weighting loss (dB)'], rows)
    noise_table = table(['Target', 'Identical test metric', 'Moving matched energy (dB)',
        'Moving squared error (%)', 'Full waveform squared error (%)'], noise_rows)
    body_table = table(['Target', 'Extent hypothesis', 'Moving matched energy (dB)',
        'Moving squared error (%)', 'Offset bounds hit / 12'], body_rows)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Road calibration model checks</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}code{{font-size:.9em}}</style>
<h1>Vehicle data: information retention, noise weights and reflecting points</h1>
<p><a href="siting-assumptions.html">Siting assumptions</a> · <a href="road-model-validation.json">Values and source identities</a> · {report['updated_utc']}</p>
<p class="note">These controls hold the nominal antennas and common range origins fixed. No physical site correction or production processing change is adopted. The purpose is to check whether the vehicle likelihood is giving the joint antenna fit misleading information.</p>
<h2>What is lost by the saved 20 ms averages?</h2>
<p>The reference is the original cropped range-profile rows, with archived row information, the uniform-symbol first-moment noise factor and the existing mean range covariance. Independent proper-complex noise is stipulated. Exact bistatic signal and motion/antenna/range-origin derivatives pass through the recorded weighted averaging operator; an unknown complex target gain is eliminated. Fisher modes below a relative threshold of 10⁻⁸ are excluded and retained explicitly in the source results. Results at 10⁻¹⁰ are also saved.</p>
{bin_table}
<p>The small averaging loss is conditional on those target paths and covariance assumptions. It does not establish losslessness of earlier range extraction, correlated interference, finite within-symbol covariance or the final geometry estimator. In particular, the empirical-weight column holds the measured weights fixed against the stipulated covariance; it is not the unconditional risk of estimating weights from the same noisy data. Neither comparison includes a static or moving interference nuisance.</p>
<p>Independent tests compare against an explicit dense Gaussian averaging operator, demonstrate exact sufficiency for piecewise-constant means with inverse-variance weights, and check removal of the complex-gain phase gauge.</p>
<h2>Can less noisy variance estimates improve the recorded fit?</h2>
<p>The alternative estimates each bin's scalar noise scale from range-whitened original rows after projecting out two temporal mean modes: a constant moving response and a stationary response with the known seed deramp. Arbitrary range coefficients are allowed in both modes. With the stipulated separable Gaussian covariance and exact mean span, the orthogonal innovations are independent of the fitted mean and their variance estimate is unbiased.</p>
<p>This uses {min(d[0] for d in degrees)}–{max(d[1] for d in degrees)} independent complex innovation degrees of freedom. Under that model, the large-bin-count inverse-variance weighting efficiency is (ν−2)/(ν−1), corresponding here to at most {max(losses):.4f} dB loss. This theoretical value is not a measured pipeline SNR guarantee. Tests with 1,600 independent Gaussian draws verify variance, mean independence and unbiasedness; deterministic tests verify rejection of both mean components and the coincident-mode rank case.</p>
<p>Both recorded-data fits use all saved bins and the same fixed stationary dictionary. Motion and complex moving/static coefficients are fitted on training blocks only. Each fitted prediction is evaluated with both identical test operators; test-bin innovation estimates still use that bin's samples and their independence remains conditional on the declared Gaussian model. Positive squared-error changes mean worse prediction.</p>
{noise_table}
<p>The outcome is mixed, including regression on Q and larger full-waveform errors for A and Q. Pooled noise weights remain experimental; they are not substituted into the running joint geometry comparison. A lower estimated variance or a flatter display alone would not justify that change.</p>
<h2>Must all bands reflect from the same point on a vehicle?</h2>
<p>The baseline assumes one shared moving point, with an independent complex gain per band. The alternative uses fixed physical reflecting-point offsets for four bands, with SBS defining the trajectory gauge. All bands retain the same translating road trajectory. Offsets are in a fixed along-road/across-road/up frame, with no fitted yaw, pitch, roll or arbitrary phase trajectories. Exact two-leg-minus-direct propagation gives each band's phase and delay.</p>
<p>Two extent hypotheses are tested: ±(3,2,2) m and ±(10,3,3) m along/across/up. These are sensitivity bounds, not observed vehicle dimensions. They add no Gaussian offset prior; road and motion priors remain unchanged. Broader fits include the compact solution and the point solution as starts; training objective alone selects the numerical result.</p>
{body_table}
<p>All eight selected fits meet the unchanged stationarity threshold. The broad hypothesis improves this withheld matched-energy statistic by about 0.06–0.14 dB, but four to seven of its twelve offset coordinates reach bounds. This does not demonstrate correctly recovered reflecting points or justify assigning the residual to antenna location. It does show why a point-target assumption deserves a sensitivity check before treating vehicle phase histories as a physical calibration.</p>
<p><a href="https://www.iwaenc.org/proceedings/1997/nsip97/pdf/scan/ns970424.pdf">Gerry, Potter, Moses and Koets (1997)</a> develop localized/distributed scattering models with frequency and aspect dependence. Their high-frequency monostatic model motivates testing the simplification, but does not validate this one-point-per-band VHF vehicle model. The experiment uses exact bistatic paths.</p>
{joint_note}
{information_note}
<h2>Numerical evidence and remaining physical test</h2>
<p>The complete support suite passes {int(tests[1])} tests. Added reflecting-point checks independently calculate path lengths and derivatives, recover the original model exactly at zero offset, and check the reference-band gauge and profiled objective. Mean-epoch and original-row moving templates are compared at every fitted solution; the maximum mismatch is {max(t['maximum_original_row_moving_template_loss_db'] for s in reflectors['targets'] for t in s['trials']):.7f} dB.</p>
<p>The <a href="preferred-road-full-bin-validation.html">separate full-bin joint calibration and completely excluded vehicle test</a> is the relevant transfer check for site corrections. These within-target scene/weighting comparisons cannot replace it. Road identity, covariance shape, shared origins and the scattering assumptions remain explicit hypotheses; no global CRLB or geographic uniqueness claim follows from numerical convergence.</p></html>'''
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'road-model-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'road-model-validation.html').write_text(document+'\n')
    print(json.dumps({'report': str(args.output/'road-model-validation.html'), 'tests': int(tests[1]),
        'accepted_calibration': False}))


if __name__ == '__main__':
    main()
