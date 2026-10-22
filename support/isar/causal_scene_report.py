"""Publish conditional causal-scattering tests without adopting their phases."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint


def relevance_extension(path):
    """Retain failed prediction and convergence checks instead of filtering them."""
    result = json.loads(path.read_text())
    if not result['complete'] or result['identity']['fold'] != 'inner':
        raise ValueError('Require a completed inner relevance experiment')
    cache_path = Path(result['identity']['cache'])/'cache.json'
    if fingerprint(cache_path) != result['identity']['cache_sha256']:
        raise ValueError('Relevance template cache changed')
    for source, digest in result['identity']['implementation_sha256'].items():
        if fingerprint(Path(source)) != digest:
            raise ValueError('Relevance implementation changed: '+source)
    records = []
    for case in result['cases']:
        fit, values = case['fit']['selected'], case['summary']
        records.append({'shared_response': case['shared_response'], 'summary': values,
            'stationary': fit['stationary'], 'maximum_kkt_residual': fit['maximum_kkt_residual'],
            'variance_upper_hits': fit['variance_upper_hits'], 'gain_bound_hits': fit['gain_bound_hits'],
            'prediction_error_change_from_zero_percent': 100*(values['prediction_error']/values['data_energy']-1),
            'conditional_phase_certificate': case['fit']['phase_certificate']})
    shared = next(r for r in records if r['shared_response'])
    independent = next(r for r in records if not r['shared_response'])
    rows = ''.join('<tr>'+''.join(f'<td>{value}</td>' for value in [
        'Shared' if r['shared_response'] else 'Independent', r['stationary'],
        f"{r['prediction_error_change_from_zero_percent']:+.2f}%",
        f"{r['maximum_kkt_residual']:.3g}", r['variance_upper_hits'], r['gain_bound_hits']])+'</tr>'
        for r in records)
    details = {'source': str(path), 'records': records, 'accepted': False,
        'outer_fold_evaluated': False, 'scope': result['limitations']}
    section = f'''<h2 id="learned-variances">Allowing each physical point its own prior power</h2>
<p>A second experiment replaces the uniform point powers with individually estimated nonnegative variances. The common response retains one latent coefficient per point and constant relative band gains; the independent control has separate coefficients and variances in each band and is therefore more flexible. Motion, site coordinates, envelope origins, point positions, observation weights and inner folds are unchanged.</p>
<p>The method integrates out proper-complex Gaussian coefficients and optimises their marginal likelihood, log det C + yᴴC⁻¹y, with C = I + DΓDᴴ. It follows the Type-II evidence principle described by <a href="https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf">Tipping, Sparse Bayesian Learning and the Relevance Vector Machine (2001)</a>; the complex physical dictionary and band-gain parameters here are our implementation. This is not a performance guarantee for ISAR or a claim to reproduce the variational multiband algorithm cited in the literature review.</p>
<table><tr><th>Response</th><th>Stationary</th><th>Error change vs stationary-only</th><th>Raw KKT residual</th><th>Point-variance upper bounds</th><th>Gain bounds</th></tr>{rows}</table>
<p>The shared fit reaches the original gradient criterion without upper-power or gain-bound hits, but predicts {shared['prediction_error_change_from_zero_percent']:.2f}% more squared error than the stationary-only control. The independent fit misses the declared 1e-5 gradient gate and hits {independent['variance_upper_hits']} variance bounds, so it is retained as an unfinished control rather than called an evidence optimum. Neither case is promoted to the outer fold. No phases, physical coordinates or videos are changed.</p>
<p>Four additional tests compare the complete covariance, posterior and every analytic gradient against a dense calculation; check the exact conditional phase quadratic; recover the analytic orthogonal-dictionary evidence solution; and recover a known common phase with unknown sparse point powers. All singular directions are retained. The learned-power model remains a local, conditional hypothesis; numerical stationarity and a conditional phase bound do not establish physical coherence or uncertainty after hyperparameter fitting.</p>'''
    return details, [path, cache_path], section


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--relevance', type=Path, help='Also report the learned point-variance inner control')
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    test_text = args.tests.read_text()
    count = re.search(r'Ran (\d+) tests', test_text)
    if not count or test_text.strip().splitlines()[-1] != 'OK':
        raise ValueError('Require a completed passing support-test log')
    rows, sources, checks = [], [args.tests], []
    for width in (0, 6, 12, 24):
        path = args.experiment/f'inner-{width}.json'
        cache_path = args.experiment/f'width-{width}/cache.json'
        result, cache = [json.loads(p.read_text()) for p in (path, cache_path)]
        if not result['complete'] or not cache['complete'] or result['identity']['fold'] != 'inner':
            raise ValueError('Complete each inner comparison and original-template check first')
        if result['identity']['cache_sha256'] != fingerprint(cache_path):
            raise ValueError('The fitted template cache changed')
        for value in (result, cache):
            for file, digest in value['identity']['implementation_sha256'].items():
                if fingerprint(Path(file)) != digest:
                    raise ValueError(f'Implementation changed after the reported run: {file}')
        sources.extend([path, cache_path])
        checks.append({'width_m': width,
            'analytic_kernel_error': cache['analytic_covariance_maximum_absolute_error'],
            'original_operator_covariance_relative_error': cache['original_operator_covariance_probe_relative_error'],
            'maximum_qr_likelihood_relative_error': max(z['relative_likelihood_discrepancy']
                for c in cache['channels'] for z in c['checks'].values())})
        for case in result['cases']:
            fit, summary = case['fit'], case['summary']
            selected, certificate = fit['selected'], fit['phase_certificate']
            if not selected['stationary']:
                raise ValueError('Do not interpret a nonstationary experimental fit')
            rows.append({'width_m': width, 'shared_response': case['shared_response'],
                'summary': summary,
                'prediction_error_change_from_zero_percent': 100*(summary['prediction_error']/summary['data_energy']-1),
                'maximum_kkt_residual': selected['maximum_kkt_residual'],
                'power_bound_hits': selected['amplitude_bound_hits'],
                'conditional_phase_certified': certificate['retained_solution_globally_certified'] if certificate else None,
                'phase_bound_gap': certificate['retained_solution_gap'] if certificate else None,
                'comparison': result['comparison']})
    eligible = [r for r in rows if r['shared_response'] and r['power_bound_hits'] == 0
                and r['prediction_error_change_from_zero_percent'] < 0]
    selected = min(eligible, key=lambda r: r['summary']['prediction_error']) if eligible else None
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'rows': rows, 'numerical_checks': checks,
        'sources': {str(p): fingerprint(p) for p in sources},
        'implementation_sha256': {str(Path(__file__)): fingerprint(Path(__file__))},
        'tests': int(count[1]), 'selected_support_m': selected['width_m'] if selected else None,
        'selection_gate': 'Must predict better than the stationary-only control on inner withheld blocks and have no power-bound hits before any outer test or calibration proposal.',
        'outer_fold_evaluated': False, 'accepted_calibration': False,
        'scope': 'Conditional test with frozen motion, licensed sites, envelope origins and union of original spatial supports. Source support and motion used this recording. Does not establish independent discovery, global physical optimality or actual-data CRLB attainment.',
        'physical_model': 'Independent proper-complex Gaussian scattering density at each of 35 spatial points, uniform internal excess-path power over [0,W]. The same latent scene is seen by all bands; matched control has independent latent scenes with identical per-band marginal covariance.',
        'numerical_change': 'A full singular-basis rotation and parameter-dependent diagonal equilibration replace ill-conditioned normal equations. All singular directions and the original Gaussian prior are retained. Original Gram runs are preserved under causal-uniform-30s and causal-gram-legacy.',
        'videos_changed': False, 'production_changed': False}
    relevance_section = ''
    if args.relevance:
        extra, extra_sources, relevance_section = relevance_extension(args.relevance)
        report['learned_point_variances'] = extra
        report['sources'].update({str(p): fingerprint(p) for p in extra_sources})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'causal-scattering-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout='constrained')
    widths = np.array([0, 6, 12, 24])
    for shared, color, label in [(False, '#777777', 'Independent band responses'),
                                  (True, '#277da8', 'Shared causal response')]:
        values = [r['prediction_error_change_from_zero_percent'] for r in rows if r['shared_response'] == shared]
        axes[0].plot(widths, values, 'o-', color=color, label=label)
    axes[0].axhline(0, color='black', lw=.8)
    axes[0].set(xlabel='Maximum internal excess path W (m)', ylabel='Prediction error change (%)',
                title='Inner withheld blocks; lower is better', ylim=(-2, 38), xticks=widths)
    axes[0].legend(fontsize=9)
    change = [r['comparison']['shared_matched_energy_change_db'] for r in rows if r['shared_response']]
    axes[1].plot(widths, change, 'o-', color='#a3443b')
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].set(xlabel='Maximum internal excess path W (m)', ylabel='Matched-energy change (dB)',
                title='Shared versus matched independent response', xticks=widths)
    fig.suptitle('Frozen physical motion and antenna coordinates; all original observation rows\n'
                 'These hypotheses fail inner prediction; no phases adopted')
    fig.savefig(args.output/'causal-scattering-comparison.png', dpi=160)
    plt.close(fig)
    table = ''.join('<tr>'+''.join(f'<td>{value}</td>' for value in [r['width_m'],
        'Shared' if r['shared_response'] else 'Independent',
        f"{r['prediction_error_change_from_zero_percent']:+.2f}%", f"{r['summary']['matched_energy']:.1f}",
        f"{r['maximum_kkt_residual']:.2g}", r['power_bound_hits']])+'</tr>' for r in rows)
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Causal scattering validation</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}code{{font-size:.9em}}</style>
<h1>Can frequency-dependent ship scattering supply transferable band phases?</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="siting-assumptions.html">Siting assumptions</a> · <a href="causal-scattering-validation.json">Results and source identities</a> · {report['updated_utc']}</p>
<p class="note">This specific shared-scattering family fails the inner predictive check. No new band phases, antenna coordinates or images are adopted. This does not rule out causal scattering models generally or establish an irreducible coherence limit.</p>
<p>The experiment holds the 150.772–180.772 s ship motion, licensed main-site geometry, envelope origins and fixed N/E axes unchanged. It asks whether a physically delayed response can explain band differences that a location fit might otherwise absorb. The 35 spatial points are the union of the previous five eight-point supports, deduplicated. They were discovered from this recording: this is conditional model validation.</p>
<h2>Physical and statistical model</h2>
<p>At each point, internal excess path d shifts both the range envelope and carrier phase: the original template is evaluated at envelope origin + d and multiplied by exp(−i 2π f d/c). Constant internal delay does not change path rate. Full original symbol rows are evaluated before the saved averaging and whitening. Observations are not rephased or renormalised.</p>
<p>A uniform proper-complex Gaussian scattering density on [0,W] produces cross-frequency covariance exp(−iπ Δf W/c) sinc(Δf W/c), with unit integrated prior power per point. Gauss quadrature approximates the continuous delay integral; it is not an arbitrary discrete tap model. W = 0 is the instantaneous common response. The uniform density and Gaussian prior are modelling choices being tested, not facts about this ship.</p>
<p>Band amplitudes and four relative phases are estimated from marginal Gaussian likelihood, integrating out the common scene coefficients. The independent control has separate latent coefficients per band and exactly the same marginal covariance family. Gaussian posterior means have their conditional MMSE interpretation only under this specified prior and noise covariance.</p>
<p>The physical motivation is the causal frequency-dependent reflectivity formulation in <a href="https://www.math.colostate.edu/~cheney/papers/FerraraHomanCheneyHSAR.pdf">Ferrara, Homan and Cheney, Hyperspectral SAR (2017)</a>. That paper also discusses space–delay ambiguity. Its monostatic scalar-wave model and resolution results are not a performance guarantee for our multistatic VHF experiment.</p>
<h2>Prediction before calibration</h2>
<p>Outer training consists of even half-second blocks. Within those samples, blocks 0 modulo 4 fit the model and blocks 2 modulo 4 test it. Each fold independently profiles its stationary nuisance. All eight fitted cases reach the original unnormalised gradient criterion, but every shared case hits at least one power bound and predicts worse than the stationary-only (zero moving-template) control. The outer test has not been evaluated for this family. No selection or reported gain is taken from that outer fold.</p>
<img src="causal-scattering-comparison.png" alt="Prediction errors exceed the stationary-only control and shared responses lose withheld matched energy relative to independent responses">
<table><tr><th>W (m)</th><th>Response</th><th>Error change vs stationary-only</th><th>Withheld matched energy</th><th>Raw KKT residual</th><th>Power bounds</th></tr>{table}</table>
<p>Matched energy is the squared correlation of the concatenated five-band prediction and test observations divided by prediction energy. It is not measured image SNR. These inner-fold scores cannot be compared numerically with the earlier mixed model's outer-fold score. The power search bounds were ±12 in global design-scaled log-amplitude units; reaching them is explicitly reported rather than treated as an unconstrained evidence optimum.</p>
<h2>Numerical checks and limits of the bounds</h2>
<p>The original QR sufficient statistics retain the parameter-independent outside energy, with relative likelihood discrepancy below {max(c['maximum_qr_likelihood_relative_error'] for c in checks):.2g}. Analytic cross-band covariance errors are below {max(c['analytic_kernel_error'] for c in checks):.2g}; increasing delay quadrature by eight nodes changes a cross-band covariance probe through the original operator by at most {max(c['original_operator_covariance_relative_error'] for c in checks):.2g} relative Frobenius norm.</p>
<p>The initial Gram solve lost numerical precision at large response powers. Its results and code are retained. The replacement rotates the isotropic latent prior by a full SVD, retains every singular direction, and equilibrates posterior precision before Cholesky. It changes no likelihood, weighting, basis rank or prior. A high-power regression with a 10¹² design condition number agrees with a direct observation-space singular-value calculation; finite differences independently verify its score. The full support suite passes {int(count[1])} tests.</p>
<p>Other tests verify exact envelope/carrier delay bookkeeping, dense covariance equivalence, 10,000 Gaussian posterior-risk realizations, and 4,000 simple common-phase trials against their known-model CRLB. The marginal phase quadratic has a global upper-bound certificate at the fitted powers in all four recorded shared cases. That conditional phase certificate does not prove a global joint power optimum, physical scattering correctness, surveyed antenna locations or CRLB-tight performance on this recording.</p>
{relevance_section}
<p>The next antenna-location experiment remains the combined road/ship search over all licensed assignments. These scattering results supply no justification for changing its locations, its data weights or the production waveform processing.</p></html>'''
    (args.output/'causal-scattering-validation.html').write_text(text)
    print(json.dumps({'tests': int(count[1]), 'selected_support_m': report['selected_support_m'],
        'accepted_calibration': False, 'report': str(args.output/'causal-scattering-validation.html')}))


if __name__ == '__main__':
    main()
