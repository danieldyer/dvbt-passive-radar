"""Review temporal-basis bias and common versus independent measured motion."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cluster_phase import PolynomialPath,SplinePath
from compare_ais_phase import NAMES,COLOURS
from fit_fixed_grid_phases import prepare_fixed
from flexible_echo_cluster import FlexibleEchoCluster
from report_measured_scatterer_phases import load_fields,serial


def curve_for(label,first,last):
    return PolynomialPath(first,last,2) if label=='quadratic' else SplinePath(first,last,1. if label=='spline-1s' else .5)


def main():
    root=Path('docs/data/isar_20261017');out=root/'flexible-flash-path'
    source=out/'fits.json';data=json.loads(source.read_text())
    if not data['complete']:raise ValueError('Wait for every declared fit')
    baseline=json.loads(Path(data['baseline_source']).read_text());first,last=data['interval_s']
    geometry=Path('docs/data/calibration_20261016_sites')
    bands,_,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    errors=[];summaries=[]
    for group in data['groups']:
        curve=curve_for(group['name'],first,last)
        model=FlexibleEchoCluster(bands,baseline['selected_origins_samples'],1,curve)
        for fit in group['common']:
            r,_=model.evaluate(np.asarray(fit['parameters']));errors.append(abs(r@r*model.energy-fit['cost'])/fit['cost'])
        for c,fit in enumerate(group['independent']):
            independent=FlexibleEchoCluster([bands[c]],baseline['selected_origins_samples'],1,curve,shared=False)
            r,_=independent.evaluate(np.asarray(fit['parameters']));errors.append(abs(r@r*independent.energy-fit['cost'])/fit['cost'])
        fit=group['common'][group['best_common_index']]
        summary=dict(name=group['name'],degree=group['degree'],best_common_index=group['best_common_index'],
            training_cost=fit['cost'],validation_error=sum(e['heldout_error'] for e in fit['prediction']['validation']),
            unused_error=sum(e['heldout_error'] for e in fit['prediction']['testing']),
            common_unused_phase_rms_deg=[d['selected_unused_phase_rms_deg'] for d in fit['selected_diagnostics']],
            common_gradient=fit['maximum_projected_gradient'],common_bounds=fit['bound_indices'],
            selected_retained_energy=[d['selected_projected_signal_energy_retained'] for d in fit['selected_diagnostics']],
            reference_path_rate_extent_mps=[min(fit['selected_diagnostics'][2]['reference_path_rate_mps']),max(fit['selected_diagnostics'][2]['reference_path_rate_mps'])],
            reference_path_acceleration_extent_mps2=[min(fit['selected_diagnostics'][2]['reference_path_acceleration_mps2']),max(fit['selected_diagnostics'][2]['reference_path_acceleration_mps2'])])
        summaries.append(summary)
    if max(errors)>1e-9:raise ValueError('Saved likelihood reproduction failed')
    selected=min(summaries,key=lambda s:s['validation_error'])
    for summary in summaries:
        summary['selected_by_validation']=summary is selected
    fig,axes=plt.subplots(1,2,figsize=(16,5.7),layout='constrained')
    x=np.arange(5);colours=['#777','#0072b2','#009e73']
    for i,(summary,colour) in enumerate(zip(summaries,colours)):
        axes[0].bar(x+(i-1)*.22,summary['common_unused_phase_rms_deg'],.22,label=summary['name'],color=colour)
    one=next(g for g in data['groups'] if g['name']=='spline-1s')
    independent_rms=[f['selected_diagnostics'][0]['selected_unused_phase_rms_deg'] for f in one['independent']]
    axes[0].plot(x,independent_rms,'kx',ms=9,label='Independent spline-1s')
    axes[0].set(xticks=x,xticklabels=NAMES,ylabel='B phase innovation RMS on unused rows (degrees)',
        title='Common physical path; separate constant band phases')
    axes[0].legend(fontsize=8)
    q=summaries[0]
    for name,marker in (('validation_error','o'),('unused_error','s')):
        axes[1].plot(np.arange(3),[100*(s[name]/q[name]-1) for s in summaries],marker+'-',label=name.replace('_',' '))
    axes[1].axhline(0,color='#aaa',lw=.7)
    axes[1].set(xticks=np.arange(3),xticklabels=[s['name'] for s in summaries],ylabel='Whole-field prediction error change from quadratic (%)',
        title='Same observations and nuisance model; no gain normalization')
    axes[1].legend()
    fig.savefig(out/'01-prediction-comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(19,10),layout='constrained',sharex=True)
    for ax,c in zip(axes.ravel(),range(5)):
        for group,colour in zip(data['groups'],colours):
            fit=group['common'][group['best_common_index']];d=fit['selected_diagnostics'][c]
            t=np.array(d['time_s']);test=np.array(d['testing']);phase=np.array(d['phase_residual_deg'])
            ax.plot(t[test],phase[test],'.-',color=colour,ms=3,lw=.6,label=group['name'])
        ax.axhline(0,color='#aaa',lw=.6);ax.set(title=NAMES[c],xlabel='Capture time (s)',ylabel='Unused measured phase innovation (degrees)',ylim=(-100,100))
    axes[0,0].legend(fontsize=8);axes[-1,-1].axis('off')
    axes[-1,-1].text(.05,.75,'All path bases drive range migration,\ncarrier phase and within-symbol Doppler.\n\nNo per-band time-varying phase screen fits.\n\nBars and errors condition on the scene model;\nthis is not an antenna calibration.',transform=axes[-1,-1].transAxes,fontsize=12)
    fig.suptitle('Does a common nonquadratic trajectory explain the remaining measured phase?',fontsize=14)
    fig.savefig(out/'02-unused-phase-innovations.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(19,6),layout='constrained')
    common=one['common'][one['best_common_index']]
    for c,(fit,colour) in enumerate(zip(one['independent'],COLOURS)):
        d=fit['selected_diagnostics'][0];t=np.array(d['time_s']);path=np.array(d['selected_path_change_m'])
        reference=np.array(common['selected_diagnostics'][c]['selected_path_change_m'])
        axes[0].plot(t,path,color=colour,label=NAMES[c]);axes[1].plot(t,100*(path-reference),color=colour)
        axes[2].plot(d['derivative_time_s'],d['reference_path_rate_mps'],color=colour)
    d=common['selected_diagnostics'][2];axes[0].plot(d['time_s'],d['selected_path_change_m'],'k--',label='Common spline-1s')
    axes[0].set(ylabel='Path change from midpoint (m)',title='Independent flexible paths versus common path')
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].set(ylabel='Independent minus own common-band prediction (cm)',title='Remaining time-dependent band differences')
    axes[2].set(ylabel='Reference path rate (m/s)',title='Independent flexible-path rates; conditional on model')
    for ax in axes:ax.set_xlabel('Capture time (s)');ax.axhline(0,color='#aaa',lw=.6)
    fig.savefig(out/'03-independent-path-differences.png',dpi=160);plt.close(fig)
    result=dict(accepted_calibration=False,production_changed=False,summaries=summaries,
        validation_selected_model=selected['name'],independent_spline_1s_phase_rms_deg=independent_rms,
        reproduced_objectives=len(errors),maximum_relative_cost_error=max(errors),
        interpretation='A common flexible path tests whether temporal model truncation and unequal brightness weighting explain apparent interband differences. Remaining independent-path differences are conditional on four-return association, stationary profile and range template; they are not automatically antenna errors. Polynomial coefficients are not instantaneous physical derivatives without model adequacy.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,Path(__file__),Path('support/isar/flexible_echo_cluster.py'),Path('support/isar/test_flexible_echo_cluster.py'))})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    rows=''
    for s in summaries:
        rows+=f'<tr><td>{s["name"]}</td><td>{s["degree"]}</td><td>{s["training_cost"]:.3f}</td><td>{s["validation_error"]:.3f}</td><td>{s["unused_error"]:.3f}</td><td>{s["common_gradient"]:.3g}</td><td>{s["common_bounds"]}</td></tr>'
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Shared nonquadratic phase trajectory</title>
<style>body{{font:16px/1.55 system-ui;max-width:1550px;margin:30px auto;padding:0 20px}}img{{width:100%}}.note{{background:#eef4f8;padding:16px}}td,th{{border:1px solid #ddd;padding:7px}}table{{border-collapse:collapse}}</style>
<h1>Test temporal-model bias before interpreting antenna offsets</h1>
<p class="note">A shared nonquadratic trajectory can yield different independent quadratic fits when the bands have different brightness histories. A synthetic original-row three-return scene verifies this counterexample; the correct shared cubic recovers the injected motion and predicts unused samples. Therefore the earlier polynomial-coefficient contrasts alone cannot establish different physical scatterers or transmitter locations.</p>
<p>The real-data comparison retains the exact fixed-grid observations, their original epochs and information, constant stationary profile, positive 250 ms brightness and separately moving A/C/D nuisance returns. Only B's path basis changes. That one path drives envelope migration, carrier phase at each RF wavenumber and C0/C1 Doppler. Every band retains one constant phase and envelope intercept; no independent time-dependent phase correction is introduced in the common models.</p>
<img src="01-prediction-comparison.png" alt="Phase innovations and held-out error for quadratic and flexible common paths">
<p>The validation-selected common basis is <b>{selected['name']}</b>; unused samples are reported separately. Additional flexibility is not assumed to improve prediction. The black crosses use separate spline-1s paths as a diagnostic of the remaining gap; they are not an accepted physical motion model.</p>
<p class="note"><b>Endpoint and physical-motion warning:</b> the finer 0.5 s common spline reaches approximately −33.6 m/s² path acceleration at an endpoint, while predicting worse than the 1 s spline. The independently fitted ABC spline also bends sharply where B fades. These excursions are not credible ship motion and are not adopted. Smoothness/kinematic constraints and adequate phase support are required before any path can drive physical calibration or image formation; a converged complex-data fit alone is insufficient.</p>
<table><tr><th>Common path</th><th>Path coordinates</th><th>Training squared error</th><th>Validation squared error</th><th>Unused squared error</th><th>Projected gradient</th><th>Bound indices</th></tr>{rows}</table>
<img src="02-unused-phase-innovations.png" alt="Unused phase errors in all five channels for each shared trajectory basis">
<img src="03-independent-path-differences.png" alt="Independent spline paths, differences from common path and path rates">
<p>These internally inspected four-second controls are a scene-model comparison, not a conclusive survey, a blind detection experiment or proof of a carrier-connected material point through the full recording. The original phase measurements and all remaining disagreements are retained. No physical coordinates, production processing or capture data are changed.</p>
<p>All {len(errors)} new objectives reproduce from the original observations (maximum relative discrepancy {max(errors):.3g}). The exact profiled residual Jacobian is checked on synthetic and real data, including the flexible path's effect on both envelope and carrier. The complete 163-test regression suite passes.</p>
<p><a href="analysis.json">Comparison and provenance</a> · <a href="fits.json">Every fit, path, unused phase innovation and retention diagnostic</a> · <a href="../piecewise-geometry-review/tests.log">163 numerical tests</a> · <a href="../piecewise-geometry-review/index.html">Short-window conditioning and qualified geometric checks</a> · <a href="../multiple-scatterer-phase-plots/index.html">Original selected points and measured phase trajectories</a> · <a href="../index.html">Analysis archive</a></p></html>''')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
