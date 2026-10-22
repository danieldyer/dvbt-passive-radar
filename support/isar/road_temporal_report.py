"""Publish original-row vehicle response and covariance-aware gate audits."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

from finite_scene_report import digest, load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('moving', 'legacy', 'whitened', 'tests'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--implementation-archive', type=Path, nargs='*', default=[])
    p.add_argument('--prediction', type=Path)
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    paths = [args.moving, args.legacy, args.whitened]
    loaded = [load(path, args.implementation_archive) for path in paths]
    moving, legacy, whitened = [d for d, _ in loaded]
    tests = args.tests.read_text(); count = re.search(r'Ran (\d+) tests', tests)
    if not count or tests.strip().splitlines()[-1] != 'OK':
        raise ValueError('Require completed passing support tests')
    if [r['label'] for r in legacy['targets']] != [r['label'] for r in whitened['targets']]:
        raise ValueError('Gate comparisons must use the same target hypotheses')
    for path, value in legacy['source_sha256'].items():
        if not path.endswith('road_gate_information.py') and whitened['source_sha256'].get(path) != value:
            raise ValueError('Gate comparison changed a scientific input: '+path)
    summaries = []
    for a, b in zip(legacy['targets'], whitened['targets']):
        summaries.append({'label': a['label'], **{name: {
            'worst_matched_snr_loss_db': max(c['matched_snr_loss_db'] for c in item['channels']),
            'worst_local_parameter_information_loss_db': max(c['information'][1]['worst_information_loss_db'] for c in item['channels'])}
            for name, item in (('legacy', a), ('whiten_first', b))}})
    prediction = None
    if args.prediction:
        prediction, replaced = load(args.prediction, args.implementation_archive)
        if not prediction['all_selected_fits_stationary']:
            raise ValueError('Require stationary recorded-data refits')
        if [r['label'] for r in prediction['targets']] != [r['label'] for r in legacy['targets']]:
            raise ValueError('Prediction must use the same targets')
        for path in set(prediction['source_sha256']) & set(legacy['source_sha256']):
            if prediction['source_sha256'][path] != legacy['source_sha256'][path]:
                raise ValueError('Prediction changed a shared scientific input: '+path)
        paths.append(args.prediction)
    result = {'complete': True, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'scope': __doc__, 'moving': moving, 'gate_summary': summaries, 'gates': [legacy, whitened],
        'tests': int(count[1]), 'source_sha256': {str(path): digest(path) for path in [*paths, args.tests, Path(__file__)]},
        'verified_archived_sources': [v for _, v in loaded], 'accepted_calibration': False,
        'production_changed': False, 'recorded_prediction': prediction}
    row = lambda values: '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    moving_rows = ''.join(row([r['label'], f"{r['worst_template_loss_db']:.7f}",
        f"{r['prediction_error_change_percent']:+.6f}", f"{r['matched_energy_change_db']:+.6f}"])
        for r in moving['targets'])
    gate_rows = ''.join(row([r['label'], name,
        f"{r[name]['worst_matched_snr_loss_db']:.4f}",
        f"{r[name]['worst_local_parameter_information_loss_db']:.4f}"])
        for r in summaries for name in ('legacy', 'whiten_first'))
    prediction_text = ''
    if prediction:
        prediction_rows = ''.join(row([r['label'], metric,
            f"{v['prediction_error_change_percent']:+.5f}", f"{v['matched_energy_change_db']:+.5f}"])
            for r in prediction['targets'] for metric, v in r['comparison'].items())
        prediction_text = f'''<h2>Recorded-data motion refits</h2>
<p>Both gates fit each target's road-constrained motion and complex responses on training blocks, from identical starts and with nominal antennas and common origins fixed. All eight selected fits pass stationarity. Their frozen predictions are evaluated on both held-out observation operators, so each table comparison uses identical test samples and noise units. Training-derived complex coefficients are transferred without refitting them on test data.</p>
<table><tr><th>Target</th><th>Test operator</th><th>Whiten-first error change (%)</th><th>Matched-energy change (dB)</th></tr>{prediction_rows}</table>
<p>The measured effects are mixed: O/Q improve slightly, while A/K worsen. No universal recorded-data gain or site correction is established. This does not contradict the exact sufficient-statistic result for band-limited Gaussian means: nonlinear target motion, clutter and the empirical covariance still need validation. The mean-epoch moving model used in these refits has a separate small-error audit above.</p>'''
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Vehicle time and Doppler projection audit</title>
<style>body{{font:17px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 22px;color:#243040}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Vehicle calibration: time integration and Doppler projection</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="road-temporal-validation.json">Values and source identities</a> · {result['updated_utc']}</p>
<p>The mean-epoch moving response is accurate at the tested road fits. The narrow Fourier gate is a more significant source of conditional information loss. Whitening before constructing its signal subspace recovers some target energy, but still leaves substantial loss in some local parameter directions. No improved antenna coordinates or production change are claimed.</p>
<h2>Original-row response versus the bin-mean approximation</h2>
<p>The audit reconstructs each saved 20-ms bin from original row epochs and information weights. It evaluates the physical target path, carrier phase and range response at those epochs, then applies the same averaging, Doppler projection and range whitening as the data. The ten fixed motion fits, nominal sites, common origins, split and noise units stay unchanged. Only complex target amplitudes are refitted on training data for the frozen-motion prediction comparison.</p>
<table><tr><th>Target</th><th>Worst template SNR loss (dB)</th><th>Prediction-error change (%)</th><th>Matched-energy change (dB)</th></tr>{moving_rows}</table>
<p>This comparison tests the forward-model approximation; it does not measure information discarded by averaging or by the Doppler gate. Both models still condition on the interval-averaged range pulse and declared empirical covariance.</p>
<h2>Projection ordering and its exact linear-model result</h2>
<p>The legacy gate uses fixed Fourier rows F and whitens their covariance FDFᴴ afterward. This is a valid noise-normalized projection, but it is not generally sufficient when bin variances D differ. For a mean in a declared Fourier subspace Ba, form D⁻¹ᐟ²B = UR by QR and retain z = UᴴD⁻¹ᐟ²y. Then Cov(z) = I and the discarded whitened component has zero mean for every a. Its likelihood factor is independent of a, so this statistic preserves all information for that linear mean model.</p>
<p>This is the fixed-covariance linear-Gaussian special case of information-preserving compression discussed by <a href="https://academic.oup.com/mnrasl/article/476/1/L60/4909822">Alsing and Wandelt (2018)</a>. A nonlinear moving target and its derivatives need not lie exactly in the chosen Fourier space; the following measured retention calculation tests that separate condition. No target-driven covariance scale is introduced.</p>
<h2>Physical motion and geometry information</h2>
<p>Each original-row integrated template is compared with all selected saved bins, retaining the same range samples and empirical time/range covariance. Unknown complex target amplitude is eliminated in each operator. Mean derivatives include road motion, antenna coordinates and common origins. Their full-data columns are scaled to unit norm before comparing generalized information modes; the table uses relative power cutoff 1e−10. The JSON retains 1e−8 and 1e−12 as well.</p>
<table><tr><th>Target</th><th>Projection</th><th>Worst channel target-SNR loss (dB)</th><th>Worst local parameter-mode loss (dB)</th></tr>{gate_rows}</table>
<p>A large worst-mode loss is not a measured antenna-position error or a claim of equally large image-SNR loss. Motion and antenna directions remain coupled here; shared calibration must still eliminate local motion and combine targets. The reference conditions on the empirical covariance and one moving point response. Unknown stationary and moving interference must be treated before interpreting these as real-data estimator-efficiency bounds.</p>
{prediction_text}
<p>{count[1]} support tests pass. New tests independently verify irregular-time complex integration, path/origin derivatives, full-bin whitening, projection identities, and exact retention of arbitrary complex band-limited signals under strongly unequal variances. The next validation must test full-bin or enlarged sufficient-subspace likelihoods with explicit clutter nuisance and held-out prediction. Changing projection alone is not an accepted site calibration.</p></html>'''
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'road-temporal-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'road-temporal-validation.html').write_text(document+'\n')
    print(json.dumps({'report': str(args.output/'road-temporal-validation.html'), 'tests': int(count[1])}))


if __name__ == '__main__':
    main()
