"""Publish the complete joint site/motion sweep, including conditioning limits."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'control', 'information', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    experiment, control, information = [json.loads(path.read_text()) for path in
        (args.experiment, args.control, args.information)]
    if not experiment['complete'] or len(experiment['trials']) != 48:
        raise ValueError('Publish all 48 licensed combinations together')
    trials = sorted(experiment['trials'], key=lambda t: t['index'])
    selected = next(t for t in trials if t['index'] == experiment['training_selected_index'])
    fitted = json.loads(Path(selected['path']).read_text())
    base = control['after_summary']
    nominal = trials[0]
    gain = 10*np.log10(selected['validation']['matched_energy']/base['matched_energy'])
    reduction = 100*(1-selected['validation']['prediction_error']/base['prediction_error'])
    models = [json.loads(Path(t['path']).read_text()) for t in trials]
    approximation = max(v['matched_snr_loss_db'] for m in models for channel in m['midpoint_approximation'] for v in channel)
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    x = np.array([t['index'] for t in trials])
    axes[0].plot(x, [t['objective'] for t in trials], '.-', label='Penalized training objective')
    axes[0].scatter([selected['index']], [selected['objective']], marker='*', s=180, color='#bd573e', label='Training-selected assignment')
    axes[0].axhline(control['fit']['objective'], color='#777', ls='--', label='Nominal sites fixed; motion + origins refitted')
    axes[0].set(ylabel='Training residual / fixed data energy', title='Yaw fixed at zero; motion, sites and origins jointly refitted')
    axes[0].legend(fontsize=9)
    axes[1].plot(x, [100*(1-t['validation']['prediction_error']/base['prediction_error']) for t in trials], '.-')
    axes[1].set(xlabel='Licensed-site assignment index', ylabel='Withheld error reduction versus fixed locations (%)', title='Heldout observations do not select the winning assignment')
    fig.savefig(args.output/'joint-sites-scores.png', dpi=170)
    plt.close(fig)
    chosen_info = next(t for t in information['trials'] if t['index'] == selected['index'])
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for unknown, colour, label in ((False, '#25659b', 'Point coordinates fixed'), (True, '#bd573e', 'Independent band point coordinates unknown')):
        records = [r for r in chosen_info['information'] if r['point_coordinates_profiled'] == unknown]
        middle = next(r for r in records if r['motion_and_range_elimination']['cutoff'] == 1e-10)
        values = np.array([r['antenna_information_eigenvalues_before_prior'] for r in records])
        ax.fill_between(np.arange(1, 19), np.maximum(values.min(axis=0), 1e-14),
            np.maximum(values.max(axis=0), 1e-14), color=colour, alpha=.15)
        ax.semilogy(np.arange(1, 19), np.maximum(middle['antenna_information_eigenvalues_before_prior'], 1e-14),
            'o-', color=colour, label=label)
    ax.axhline(1, color='#555', ls='--', label='Location prior information')
    ax.set(xlabel='Antenna-coordinate combination, weakest to strongest', ylabel='Conditional data information / prior information',
        title='Training-selected site assignment; motion, origins and amplitudes eliminated')
    ax.legend(fontsize=9)
    fig.savefig(args.output/'joint-sites-information.png', dpi=170)
    plt.close(fig)
    def link(path, label):
        target = os.path.relpath(Path(path).resolve(), args.output.resolve())
        return f'<a href="{escape(target)}">{escape(label)}</a>'
    rows = []
    for t in trials:
        v = t['validation']
        rows.append(f'<tr><td>{link(t["path"], str(t["index"]))}</td><td>{" / ".join(t["sites"])}</td>'
            f'<td>{t["objective"]:.9f}</td><td>{v["prediction_error"]:.2f}</td><td>{v["matched_energy"]:.2f}</td>'
            f'<td>{t["converged"]}</td><td>{t["maximum_kkt_residual"]:.2g}</td><td>{t["antenna_bound_hits"]} / {t["range_bound_hits"]}</td></tr>')
    controls = [('Original yaw-free fit; fixed envelope origins', json.loads(Path(fitted['source_motion']).read_text())['after_summary']),
        ('Nominal sites fixed; motion and origins refitted', base), ('Nominal assignment; locations also fitted', nominal['validation']),
        ('Training-selected licensed assignment; joint fit', selected['validation'])]
    control_rows = ''.join(f'<tr><td>{label}</td><td>{v["prediction_error"]:.3f}</td><td>{v["matched_energy"]:.3f}</td></tr>' for label, v in controls)
    site_names = {'12013': 'Ornata Road', '12014': 'Ridge Road', '12015': 'Eyre Road'}
    local_offsets = fitted['site_local_fit']['transmitter_offsets_enu_m']+[fitted['site_local_fit']['receiver_offset_enu_m']]
    location_rows = ''.join(f'<tr><td>{name}</td><td>{site_names.get(site, site)}</td><td>{", ".join(f"{v:+.2f}" for v in offset)}</td></tr>'
        for name, site, offset in zip(['Seven', 'SBS', 'Nine', 'Ten', 'ABC', 'Receiver'], selected['sites']+['East Melbourne'], local_offsets))
    rank_rows = ''.join(f'<tr><td>{r["point_coordinates_profiled"]}</td><td>{r["motion_and_range_elimination"]["cutoff"]:g}</td>'
        f'<td>{r["directions_weaker_than_prior"]} / 18</td><td>{r["motion_and_range_elimination"]["nuisance_rank"]}</td></tr>' for r in chosen_info['information'])
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'experiment': str(args.experiment),
        'complete_assignments': len(trials), 'converged_assignments': sum(t['converged'] for t in trials),
        'training_selected_index': selected['index'], 'selected_sites': selected['sites'],
        'selected_gain_db_vs_fixed_locations': float(gain), 'selected_heldout_error_reduction_percent': float(reduction),
        'maximum_kkt_residual': max(t['maximum_kkt_residual'] for t in trials),
        'maximum_midpoint_matched_snr_loss_db': approximation, 'controls': controls,
        'information': str(args.information), 'accepted_calibration': False,
        'scope': experiment['scope'], 'ranking_scope': experiment['ranking_scope']}
    (args.output/'joint-sites-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Joint licensed-site and yaw-free motion fits</title>
<style>body{{font:17px/1.55 system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#243040}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}code{{background:#f3f4f6}}</style>
<h1>All 48 licensed-site combinations: joint motion, TX/RX and range-origin fits</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="fixed-ne-validation.html">Paired yaw-free images</a> · <a href="joint-sites-validation.json">Summary JSON</a> · {link(args.experiment, 'Every fit and input identity')} · {link(args.information, 'Information audit')}</p>
<p class="note">Yaw remains exactly zero in geographic N/E axes. {report['converged_assignments']}/48 joint fits satisfy the numerical stationarity check. The training-selected assignment improves withheld matched energy by {gain:.4f} dB and reduces prediction error by {reduction:.3f}% versus a matched fixed-location control. It still reaches {selected['antenna_bound_hits']} location-coordinate and {selected['range_bound_hits']} envelope-origin bounds. This does not establish a trustworthy antenna calibration or unique ship geolocation.</p>
<h2>The controlled experiment</h2>
<p>The original-row interval is {fitted['interval_s'][0]:.6f}–{fitted['interval_s'][1]:.6f} s. Every assignment uses the same 17 translation parameters, 18 prior-standardized antenna coordinates, five envelope origins and eight fixed reflector positions per band. Unknown complex amplitudes are independently profiled per band. Training and validation alternate in 0.5-second blocks; both use the same declared stationary projection. Physical propagation and the direct-path baseline change with the chosen sites, while the original motion-reference coordinates remain fixed.</p>
<p>Unlike the earlier fixed-motion site screen, every combination receives a joint motion/location/origin refit. Common data factorizations are reused exactly, without changing the objective. The winning assignment is selected using penalized training likelihood alone. This is a local constrained comparison with a fixed scattering support and coarse transverse anchor, not an exhaustive search over all trajectories or reflector configurations. Scores are not marginal Bayesian evidence or site probabilities.</p>
<table><tr><th>Matched control</th><th>Withheld squared prediction error</th><th>Withheld matched energy</th></tr>{control_rows}</table>
<p>Matched energy is |ŷᴴz|²/(ŷᴴŷ), using the training-fitted prediction and projected validation data. Its change is not an absolute loss from an unknown ideal ship image. The original support and motion seeds used broader data; these are conditional refit checks, not independent target discovery.</p>
<img src="joint-sites-scores.png" alt="Training and withheld scores for all 48 joint site fits">
<h2>Numerical validation</h2>
<p>Derivative tests cover physical motion, all transmitter/receiver coordinates, envelope origins and the amplitude-refitting term. A separate finite-difference test checks expected information from mean derivatives. Synthetic tests verify motion recovery with antenna nuisance, and that unknown real reflector coordinates cannot add geometry information. A solver regression covers reaching a new inequality boundary before rebuilding the feasible Newton correction. Feasible iterates can receive this checked correction even after the optimizer iteration limit; neither the objective nor the convergence threshold is relaxed.</p>
<p>The largest final KKT residual across the sweep is {report['maximum_kkt_residual']:.3g}. Final validation uses original profile rows. The maximum matched-template loss of the midpoint approximation used during optimization is {approximation:.3g} dB. Stationarity is not a global-optimum or physical-model certificate.</p>
<h2>How much location information is actually present?</h2>
<p>The audit constructs expected proper-complex Gaussian information <code>J = 2 Re(DᴴD)</code> from derivatives of the whitened mean, after eliminating complex scene and stationary amplitudes. It then eliminates the real translation and envelope-origin variables. The conservative variant also eliminates every band's three-dimensional reflector coordinates as real nuisance variables. It does not use the noisy residual Hessian as Fisher information or count hard search bounds as measured information.</p>
<p>Nuisance elimination uses direct projection of the whitened real/imaginary derivative vectors, then forms their Gram matrix. An earlier subtraction of large information matrices produced small negative eigenvalues through cancellation; it is superseded by this square-root calculation. A synthetic regression retains weak information next to nuisance derivatives eight orders of magnitude larger.</p>
<img src="joint-sites-information.png" alt="Location information with fixed versus unknown reflector coordinates">
<p>The shaded range repeats the nuisance elimination with rank cutoffs from 10⁻⁸ to 10⁻¹². Values below one mean less conditional information than the declared location prior. These are plug-in sensitivities under the thermal covariance and current scene model; residual interference and model error prevent treating them as calibrated coordinate uncertainties. Reflector positions in the conservative variant are independent between bands, matching this fit's selected support; it does not claim the same result for an independently verified common physical reflector. Fixed midpoint location, fixed global altitude and a discrete scene remain assumptions.</p>
<table><tr><th>Unknown point coordinates</th><th>Rank cutoff</th><th>Location combinations weaker than prior</th><th>Motion/origin nuisance rank</th></tr>{rank_rows}</table>
<h2>Trial locations, for audit only</h2>
<p>Offsets below are relative to the chosen licensed-site hypotheses. They are not accepted survey updates. Common-site and per-frequency location uncertainty use the existing correlated Gaussian prior; receiver uncertainty is carried explicitly.</p>
<table><tr><th>Channel</th><th>Selected licensed site</th><th>Local east / north / up offset (m)</th></tr>{location_rows}</table>
<h2>Literature check and implications</h2>
<p><a href="https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.70217">Rattan, Andre and Finnis (2025)</a> describe multistatic backprojection autofocus using a localized sharpness metric and a six-degree-of-freedom motion search. Their experiment uses an anechoic ship model and a much wider fractional bandwidth and angular diversity than this recording. The paper also discusses how intensity-squared sharpness can reward smeared bright images. Full methods were reviewed. For this pipeline, that supports checking physical prediction as well as image concentration; it does not justify adding unobserved roll/pitch states or claiming CRLB efficiency from brightness.</p>
<p><a href="https://iris.uniroma1.it/handle/11573/1689231">Testa, Pastina and Santi (2023)</a> estimate per-sensor Doppler and rate before multistatic velocity/acceleration fusion, choosing whether constant velocity is adequate. The institutional abstract was verified; the full PDF could not be retrieved. This is relevant to radar-derived translational initialization and model-order checks, but does not validate our unknown cross-band scattering phases.</p>
<p><a href="https://doi.org/10.3390/s26154954">Weiss et al. (2026)</a> formulate joint radar self-calibration with geometry, timing and target variables, distinguishing gauge removal and prior information from measurement information. The abstract and publisher-indexed excerpts were available; a complete full-text review was unavailable. Their identifiability claims are not applied as a theorem for this clustered DVB-T geometry. Our measured rank sensitivity and heldout comparisons remain the evidence.</p>
<h2>What remains</h2>
<p>The next useful constraint is shared geometry across independent apertures and targets, with reflector support treated as unknown. Constant inter-band scattering/calibration phase also needs a validated model before interpreting combined-image fringes as hull detail. This experiment neither proves unique geographic location nor proves that it cannot be obtained. No new antenna locations are written into production metadata.</p>
<details><summary>All 48 assignments (Seven / SBS / Nine / Ten / ABC)</summary>
<p>12013: Ornata Road; 12014: Ridge Road; 12015: Eyre Road. Bounds column lists antenna / envelope-origin coordinates.</p>
<table><tr><th>Index</th><th>Sites</th><th>Training objective</th><th>Withheld error</th><th>Matched energy</th><th>Converged</th><th>KKT</th><th>Bounds</th></tr>{''.join(rows)}</table></details></html>'''
    (args.output/'joint-sites-validation.html').write_text(page)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
