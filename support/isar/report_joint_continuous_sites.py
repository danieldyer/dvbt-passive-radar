"""Review location/motion fits without promoting a conditional calibration."""
import argparse,json,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuous_ship_motion import ContinuousShipMotion

NAMES=['Seven','SBS','Nine','Ten','ABC','Receiver'];SITE={'0':'Ornata','1':'Eyre','2':'Ridge'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extra-case',nargs=2,action='append',default=[],metavar=('LABEL','ANALYSIS_JSON'),help='Completed continuation or independent initialization, shown separately')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);experiment=json.loads((a.experiment/'experiment.json').read_text());records=[]
    extra=a.output/'extra-cases.json'
    if extra.exists():a.extra_case.extend(json.loads(extra.read_text()))
    reference=ContinuousShipMotion(a.reference);t=np.linspace(*reference.interval,1201);base=reference.state(t)
    fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    rows=[];pending=[];expected=['control-00002',*['joint-'+s for s in experiment['identity']['assignments']]]
    control_path=a.experiment/'control-00002'/'analysis.json'
    control=json.loads(control_path.read_text()) if control_path.exists() else None
    completed={d['case'] for d in experiment['cases']}
    for name in expected:
        if name in completed:continue
        path=a.experiment/name/'progress.json'
        if path.exists():
            try:d=json.loads(path.read_text())
            except json.JSONDecodeError:continue
            if d['history']:pending.append(dict(case=name,**d['history'][-1],best_recorded_objective=min(r['objective'] for r in d['history'])))
    paths=[(name,a.experiment/name/'analysis.json') for name in expected if name in completed]
    paths.extend((label,Path(path)) for label,path in a.extra_case)
    for index,(name,path) in enumerate(paths):
        d=json.loads(path.read_text());motion=ContinuousShipMotion(path.parent/'motion.json')
        style='--' if d.get('antenna_up_offsets_fixed',False) else '-';colour=f'C{index%10}'
        offset=np.r_[d['prior']['locations']['transmitter_offsets_enu_m'],[d['prior']['locations']['receiver_offset_enu_m']]]
        for i,label in enumerate(['East','North','Up']):ax[0,i].plot(np.arange(6),offset[:,i],marker='o',linestyle=style,color=colour,label=name)
        state=motion.state(t);delta=(state['position_ecef_m']-base['position_ecef_m'])@reference.rotation.T
        ax[1,0].plot(t,delta[:,0],linestyle=style,color=colour,label=name);ax[1,1].plot(t,delta[:,1],linestyle=style,color=colour,label=name)
        ax[1,2].scatter(d['radar_rss_fraction'],d['objective']-d['radar_rss_fraction'],color=colour,marker='D' if style=='--' else 'o',label=name)
        summary=dict(case=name,analysis_source=str(path),assignment=[SITE[c] for c in d['assignment']],objective=d['objective'],radar_rss_fraction=d['radar_rss_fraction'],
            numerical_convergence=d['numerical_convergence'],feasible_linearized_improvement=d['maximum_linearized_feasible_improvement'],
            solver_success=d['solver_success'],solver_message=d['message'],iterations_this_pass=d['iterations'],seconds_this_pass=d['seconds'],
            previous_iterations=(d.get('continuation') or {}).get('previous_iterations'),evaluation_only=d.get('evaluation_only',False),
            antenna_up_offsets_fixed=d.get('antenna_up_offsets_fixed',False),
            antenna_bound_hits=d['antenna_bound_hits'],motion_bound_hits=d['motion_bound_hits'],height_shift_m=d['prior']['common_scatterer_height_shift_m'],
            antenna_offsets_enu_m=offset.tolist(),conditional_location_information=d['conditional_location_information'])
        if control is not None:
            explained=1-d['radar_rss_fraction'];baseline=1-control['radar_rss_fraction']
            summary.update(explained_power_change_from_control_db=float(10*np.log10(explained/baseline)) if explained>0 and baseline>0 else None,
                residual_power_change_from_control_db=float(10*np.log10(d['radar_rss_fraction']/control['radar_rss_fraction'])),
                posterior_improvement_from_control=control['objective']-d['objective'])
        summary['maximum_horizontal_antenna_offset_m']=float(np.linalg.norm(offset[:,:2],axis=1).max())
        summary['maximum_absolute_vertical_antenna_offset_m']=float(abs(offset[:,2]).max())
        records.append(summary)
        gain=summary.get('explained_power_change_from_control_db')
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [name,' / '.join(summary['assignment']),f"{d['radar_rss_fraction']:.9f}",f"{d['objective']:.9f}",f'{gain:+.3f}' if gain is not None else 'unavailable',d['numerical_convergence'],f"{d['maximum_linearized_feasible_improvement']:.2g}" if d['maximum_linearized_feasible_improvement'] is not None else 'unavailable',d['antenna_bound_hits'],d['motion_bound_hits']])+'</tr>')
    for i,label in enumerate(['East','North','Up']):ax[0,i].set(xticks=np.arange(6),xticklabels=NAMES,ylabel=f'{label} offset from assigned site prior (m)');ax[0,i].tick_params(axis='x',rotation=30);ax[0,i].axhline(0,color='grey',lw=.7)
    ax[1,0].set(xlabel='Capture time (s)',ylabel='East trajectory change from starting fit (m)');ax[1,1].set(xlabel='Capture time (s)',ylabel='North trajectory change from starting fit (m)')
    ax[1,2].set(xlabel='Raw radar RSS / fixed data energy',ylabel='Normalised motion/location/height prior')
    if records:fig.legend(*ax[1,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    fig.suptitle('Joint motion and antenna refinement; dashed curves hold nominal antenna up coordinates fixed');fig.savefig(a.output/'joint-site-motion-comparison.png',dpi=150);plt.close(fig)
    report=dict(complete=experiment['complete'],cases=records,running_cases=pending,accepted_as_calibration=False,
        scope='All native epochs. Same jerk/rate prior as the preceding fit; correlated antenna prior; unknown common height; four unknown constant relative band phases. Midpoint position/velocity are free. Four reference source coordinates remain fixed, so location inference is conditional on that scene.',
        interpretation='MAP objectives rank these local conditional fits; they are not marginalized site probabilities. Information eliminates brightness on its current positive active face, with source coordinates still fixed; it does not give calibrated location error bars. A low residual, solver success or smoother trajectory alone does not establish correct antenna positions.')
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Joint jerk-regularised site refinement</title><style>body{font:16px system-ui;max-width:1600px;margin:2em auto}img{width:100%}table{border-collapse:collapse}td,th{padding:.5em;border:1px solid #aaa}pre{white-space:pre-wrap}</style><h1>Joint motion and antenna refinement</h1>'
    hypotheses=[r for r in records if r['case'] in expected and r['case']!='control-00002']
    if experiment['complete'] and hypotheses:
        winner=min(hypotheses,key=lambda r:r['objective'])
        body+='<p><strong>Result of this pass:</strong> lowest returned objective among the five free-3D hypotheses: '+html.escape(winner['case'])+' ('+html.escape(' / '.join(winner['assignment']))+f"), {winner['objective']:.9f}. Its model-explained power changes by {winner['explained_power_change_from_control_db']:+.3f} dB relative to the matched fixed-location control, with {winner['antenna_bound_hits']} active location bounds."
        if not any(r['numerical_convergence'] for r in hypotheses):body+=' None of the five hypotheses passed the independent numerical convergence gate.'
        body+=' This pass does not establish new calibrated antenna coordinates.</p>'
    body+='<p>'+html.escape(report['scope'])+'</p><p>'+html.escape(report['interpretation'])+'</p><p>Site order: Seven / SBS / Nine / Ten / ABC. The fixed-location control releases the same motion, height and phase freedoms. Coordinates are not written into production metadata.</p>'
    if (a.output/'ais-check/index.html').exists():
        body+='<p><a href="ais-check/index.html">Consistency against original AIS reports</a>: frozen radar fits, with one rigid lever and constant timing offset profiled separately.</p>'
    if any(r['antenna_up_offsets_fixed'] for r in records):
        body+='<p><strong>Height sensitivity:</strong> the explicitly labelled fixed-up case (dashed curves) holds the six antenna local-up offsets at zero. Horizontal antenna positions and common source height remain free. It is a conditional sensitivity test, separate from the five hypotheses with free 3D antenna offsets.</p>'
    body+='<p><strong>Effect size:</strong> explained-power change compares the energy represented by the four-source model with the fixed-location control. It is not an isolated-target SNR or image-resolution measurement. Most field energy is outside that small source model, so total residual power alone is also an insensitive image-quality score.</p>'
    body+='<p><strong>Location priors:</strong> site standard deviations are 30 / 30 / 75 m (east / north / up), with independent per-channel offsets of 5 / 5 / 10 m and receiver uncertainty of 15 / 15 / 7.5 m. Bounds apply to correlated standardized parameters, not to physical ENU components. Common translation of the ship and every antenna is an exact radar-only gauge; the survey priors supply the absolute origin. No AIS position likelihood is used in this experiment.</p>'
    body+='<details><summary>Physical model and numerical checks</summary><p>The predicted complex profile is a sum over four fixed reference sources: each has its exact bistatic path, RF phase, path rate and known-waveform range response. Antenna changes update the transmitter-to-source and source-to-receiver paths and the transmitter-to-receiver reference path together. One smooth physical ship trajectory and orientation are shared by every band.</p><p>Real, nonnegative brightness envelopes have 2-second knots. Source phases are constant within declared 30-second aspects and common to the bands; four relative band phases are unknown constants for the whole capture, with Ten setting the phase gauge. The fit profiles the positive brightness coefficients and one stationary whole-record range vector. No fitted time-varying band phase screen, image registration or per-window Doppler notch is applied.</p><p>The objective is the original whitened complex residual energy plus the declared integrated squared jerk, angular acceleration, antenna-location and common-height Gaussian priors. The existing jerk strengths are unchanged. All original native epochs contribute. Source positions remain conditional, so this is a finite-source motion/location experiment rather than a complete ship-scattering model.</p><p>Tests cover all 13 physical waveform derivatives, their sparse chain rule, the profiled brightness/phase gradient, information after eliminating active brightness coefficients, exported world positions, and phase-coordinate round trips. A common translation of the ship and every antenna leaves the radar likelihood unchanged after the constant phase gauges are transformed. On the full recorded Seven field, the adjoint and fully materialized gradients agreed to 4.7e-15 relative error. Stationarity is checked independently by a feasible linearized-improvement calculation; the threshold remains 1e-7.</p></details>'
    if not experiment['complete']:
        body+='<p><strong>Computation in progress.</strong> Only fully completed cases appear in the comparison table. Optimizer callback trials can precede line-search acceptance.</p><pre>'+html.escape(json.dumps(pending,indent=2))+'</pre>'
    body+='<table><tr><th>Case</th><th>Sites</th><th>Radar RSS fraction</th><th>Posterior objective</th><th>Explained-power change (dB)</th><th>Numerically converged</th><th>Feasible first-order gap</th><th>Antenna bound hits</th><th>Motion bound hits</th></tr>'+''.join(rows)+'</table><img src="joint-site-motion-comparison.png"><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>'
    (a.output/'index.html').write_text(body);print(json.dumps(dict(complete=report['complete'],completed_cases=len(records),output=str(a.output)),indent=2))


if __name__=='__main__':main()
