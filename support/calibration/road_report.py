"""Reviewable progress report for the expanded vehicle/road calibration search."""
import argparse
from datetime import datetime,timezone
import html
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from road_search import OPTIONS
from road_geometry import RoadCurve
from road_model import RoadTrajectory
from identifiability_report import render as render_identifiability
from identifiability_report import render_revised


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('histories','search','fits','routes','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--labels',nargs='+',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    history={label:json.loads((args.histories/f'{label}.json').read_text()) for label in args.labels}
    colours=['#2373bb','#da8a15','#228d59','#9a58b5','#cf4b48']
    names=['Seven','SBS','Nine','Ten','ABC']
    fig,axes=plt.subplots(4,3,figsize=(14,12),layout='constrained')
    for ax,(label,d) in zip(axes.flat,history.items()):
        for ch,seed in enumerate(d['channels']):
            q=np.array(seed['coarse_observations'])
            if len(q):ax.plot(q[:,0]-d['start'],q[:,1]/1000,'.',ms=3,color=colours[ch],label=names[ch])
        ax.set(title=f'{label} · {d["start"]:.0f}–{d["end"]:.0f} s',xlabel='Time within interval (s)',ylabel='Excess path (km)')
        ax.grid(alpha=.12)
    axes.flat[0].legend(fontsize=8,ncol=3)
    fig.suptitle('Twelve candidate intervals · measured range peaks in each channel\nAssociation candidates, not accepted phase calibrators')
    fig.savefig(args.output/'vehicle-range-tracks.png',dpi=150);plt.close(fig)
    screens=[]
    for a in itertools.product(*OPTIONS):
        code=''.join(map(str,a));result=json.loads((args.search/f'sites-{code}.json').read_text())
        targets={t['label']:t['trials'] for t in result['targets']}
        screens.append((code,np.array([targets[label][0]['cost'] for label in args.labels])))
    screens.sort(key=lambda q:q[1].sum())
    matrix=np.array([v for _,v in screens]);difference=matrix-matrix.min(axis=0)
    fig,ax=plt.subplots(figsize=(11,12),layout='constrained')
    image=ax.imshow(difference,aspect='auto',cmap='viridis',vmin=0,vmax=np.percentile(difference,95))
    ax.set_xticks(range(len(args.labels)),args.labels);ax.set_yticks(range(len(screens)),[s for s,_ in screens],fontsize=7)
    ax.set(xlabel='Vehicle candidate',ylabel='Site choices: Seven, SBS, Nine, Ten, ABC',
           title=f'All {len(screens)} configured site assignments: coarse road-association screening\n0 Ornata · 1 Eyre · 2 Ridge; lower is better; not phase calibration evidence')
    fig.colorbar(image,ax=ax,label='Screening objective above best for this target (not log odds)')
    fig.savefig(args.output/'vehicle-sites-screening.png',dpi=150);plt.close(fig)
    completed=[]
    for a in itertools.product(*OPTIONS):
        code=''.join(map(str,a))
        files=[args.fits/f'{code}-{label}.json' for label in args.labels]
        if all(f.exists() for f in files):completed.append(code)
    selected=[]
    for label in args.labels:
        path=args.fits/f'00002-{label}.json'
        if path.exists():
            trial=json.loads(path.read_text())['trials'][0]
            selected.append({'label':label,**trial})
    routes=json.loads(args.routes.read_text())['routes']
    fig,ax=plt.subplots(figsize=(11,9),layout='constrained')
    lines=[np.array(r['enu_m'])[:,:2]/1000 for r in routes]
    ax.add_collection(LineCollection(lines,colors='#b3b7ba',linewidths=.3))
    for i,target in enumerate(selected):
        x=np.array(target['description']['midpoint_enu_m'])/1000
        ax.plot(*x[:2],'o',ms=6,color=plt.get_cmap('tab20')(i))
        ax.annotate(target['label'],x[:2],xytext=(5,4),textcoords='offset points',fontsize=10)
    ax.plot(0,0,'+',color='black',ms=10);ax.annotate('Receiver',(0,0),xytext=(7,-14),textcoords='offset points')
    locations=np.array([t['description']['midpoint_enu_m'][:2] for t in selected])/1000
    ax.set_xlim(min(-2.,locations[:,0].min()-1),max(3.,locations[:,0].max()+1))
    ax.set_ylim(min(-3.,locations[:,1].min()-1),max(3.,locations[:,1].max()+1))
    ax.set_aspect('equal');ax.set(xlabel='East of receiver (km)',ylabel='North of receiver (km)',
        title='Provisional road branches under the licensed main-site hypothesis\nThese are candidate associations, not recovered geographic target positions')
    fig.text(.01,.006,'Map data © OpenStreetMap contributors (ODbL); BBBike extract 2026-10-09.',fontsize=8)
    fig.savefig(args.output/'vehicle-road-hypotheses.png',dpi=150);plt.close(fig)
    preferred_directory=args.fits.parent/'road-fit-street-priors'
    preferred=[]
    for label in ['A','B','C','E','K','O','Q','R','S']:
        path=preferred_directory/f'00002-{label}.json'
        if not path.exists():continue
        choices=[t for t in json.loads(path.read_text())['trials'] if t['name'] in ['Albert Street','Victoria Parade']]
        if choices:preferred.append((label,choices))
    if preferred:
        fig,axes=plt.subplots(3,3,figsize=(14,6.5),layout='constrained',sharex=True,sharey=True)
        for ax,(label,choices) in zip(axes.flat,preferred):
            ax.add_collection(LineCollection(lines,colors='#d4d7da',linewidths=.4))
            for choice in choices:
                model=RoadTrajectory(history[label]['channels'][1],RoadCurve(routes[choice['route']]),
                                     choice['coarse_state'],[0,0,0,0,2],np.zeros((5,3)))
                time=np.linspace(history[label]['start'],history[label]['end'],101)
                position=model.positions(np.array(choice['state']),time,np.zeros((3,3)))[0]/1000
                colour='#2373bb' if choice['name']=='Albert Street' else '#b87413'
                ax.plot(position[:,0],position[:,1],color=colour,lw=2,label=choice['name'])
                ax.plot(position[0,0],position[0,1],'o',color=colour,ms=4)
                ax.annotate('',xy=position[-1,:2],xytext=position[-8,:2],
                            arrowprops={'arrowstyle':'->','color':colour,'lw':1.5,'mutation_scale':9})
            ax.plot(0,0,'+',color='black',ms=8)
            ax.set(title=f'{label} · {history[label]["start"]:.0f}–{history[label]["end"]:.0f} s',
                   xlim=(-1.6,.4),ylim=(-.2,.55),aspect='equal')
        for ax in axes[-1]:ax.set_xlabel('East of receiver (km)')
        for ax in axes[:,0]:ax.set_ylabel('North of receiver (km)')
        axes.flat[0].legend(fontsize=8,loc='lower left')
        fig.suptitle('Albert Street / Victoria Parade association hypotheses\nUser-prioritised plausible roads; these are not independently observed vehicle locations')
        fig.get_layout_engine().set(rect=(0,.035,1,.965))
        fig.text(.01,.005,'Map data © OpenStreetMap contributors (ODbL). Black cross: receiver; circle: interval start; arrow: interval end.',fontsize=8)
        fig.savefig(args.output/'vehicle-local-street-hypotheses.png',dpi=150);plt.close(fig)
    common_path=args.output/'vehicle-common-paths-road-seeded.json'
    if not common_path.exists():common_path=args.output/'vehicle-common-paths.json'
    common=json.loads(common_path.read_text()) if common_path.exists() else None
    joint_trials=[]
    for path in args.fits.glob('*-joint-roads6-all12.json'):
        trial=json.loads(path.read_text())
        joint_trials.append({'assignment':trial['assignment'],'cost':trial['cost'],'converged':trial['converged'],
            'evaluations':trial['evaluations'],'source':str(path)})
    joint_trials.sort(key=lambda t:t['cost'])
    phase_path=args.output/'vehicle-shared-phase-associated.json'
    phase=json.loads(phase_path.read_text()) if phase_path.exists() else None
    if phase:
        values=np.array([t['phase_contrast_degrees'] for t in phase['targets']])
        empirical=bool(phase.get('jackknife_block_seconds'))
        errors=np.array([t['delete_block_phase_sigma_degrees' if empirical else 'phase_standard_deviation_degrees_conditional'] for t in phase['targets']])
        ratios=np.array([t['shared_to_independent_amplitude_ratio'] for t in phase['targets']])
        fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained')
        for ax,array,limits,cmap,title in ((axes[0],values,(-180,180),'coolwarm','Residual phase relative to SBS'),
                (axes[1],ratios,(0,1),'viridis','Retained band amplitude in shared-phase fit')):
            image=ax.imshow(array,aspect='auto',cmap=cmap,vmin=limits[0],vmax=limits[1])
            ax.set_xticks(range(5),names);ax.set_yticks(range(len(values)),[t['label'] for t in phase['targets']])
            ax.set_title(title)
            for row in range(len(values)):
                for ch in range(5):
                    label=(f'{values[row,ch]:+.0f}°\n±{errors[row,ch]:.0f}°' if ax is axes[0] else f'{ratios[row,ch]:.2f}')
                    colour='white' if (abs(array[row,ch])>100 if ax is axes[0] else array[row,ch]<.45) else '#17212d'
                    ax.text(ch,row,label,ha='center',va='center',fontsize=9,color=colour)
            fig.colorbar(image,ax=ax,orientation='horizontal',shrink=.9,
                label=('Degrees; 2 s delete-block phase uncertainty at fixed path' if empirical else 'Degrees; errors condition on fitted path/noise model') if ax is axes[0] else 'Shared-model / independent-band amplitude')
        fig.suptitle('Five-band phase consistency under the fitted point-response model\nIn-sample diagnostic at fixed fitted trajectories; no coordinates adopted')
        fig.savefig(args.output/'vehicle-band-phase.png',dpi=150);plt.close(fig)
    withheld_path=args.output/'vehicle-main-roads6-withheld.json'
    withheld=json.loads(withheld_path.read_text()) if withheld_path.exists() else None
    receiver_path=args.fits/'00002-joint-receiver.json'
    receiver=json.loads(receiver_path.read_text()) if receiver_path.exists() else None
    stationary_path=args.output/'stationary-phase.json'
    stationary=json.loads(stationary_path.read_text()) if stationary_path.exists() else None
    street_trials=[]
    for path in sorted(preferred_directory.glob('?????-joint-streets9*.json')):
        trial=json.loads(path.read_text())
        if trial.get('amplitude_block_seconds'):continue
        street_trials.append({'assignment':trial['assignment'],'variant':path.stem.split('streets9',1)[1] or '-initial',
            'cost':trial['cost'],'converged':trial['converged'],'range_bias_m':trial.get('range_bias_m'),
            'source':str(path),'accepted':False})
    street_withheld_path=args.output/'vehicle-streets-delay-withheld.json'
    street_withheld=json.loads(street_withheld_path.read_text()) if street_withheld_path.exists() else None
    temporal_trials=[]
    for path in sorted(preferred_directory.glob('?????-joint-streets9-temporal.json')):
        trial=json.loads(path.read_text())
        temporal_trials.append({'assignment':trial['assignment'],'block_seconds':trial['amplitude_block_seconds'],
            'constant_control':trial['constant_control']['cost'],'variable_cost':trial['cost'],
            'converged':trial['converged'],'source':str(path),'accepted':False})
    temporal_withheld_path=args.output/'vehicle-streets-temporal-withheld.json'
    temporal_withheld=json.loads(temporal_withheld_path.read_text()) if temporal_withheld_path.exists() else None
    coverage_path=args.output/'vehicle-street-observability.json'
    coverage=json.loads(coverage_path.read_text()) if coverage_path.exists() else None
    mirror_path=args.output/'mirror-frequency-diagnostic.json'
    mirror=json.loads(mirror_path.read_text()) if mirror_path.exists() else None
    scatterer_path=args.output/'vehicle-scatterer-validation.json'
    scatterer=json.loads(scatterer_path.read_text()) if scatterer_path.exists() else None
    identifiability_section=render_identifiability(args.output)
    revised_section=render_revised(args.output)
    seed_audit_path=args.output/'seed-retention-audit.json'
    seed_audit=json.loads(seed_audit_path.read_text()) if seed_audit_path.exists() else None
    seed_notice=''
    if seed_audit:
        retained=min(b['old_independent_seed']['binning_and_gate_retained_fraction']
                     for t in seed_audit['targets'] for b in t['bands'])
        revised_status='revised conditional fits completed' if revised_section else 'revised fits pending'
        seed_notice=f'''<p class="status"><b>Calibration-input audit: {revised_status}.</b>
The old extraction independently reselected peaks in each band, including weak bands,
and gated around those seeds. For an ideal echo following the selected source track,
the retained information falls to {100*retained:.1f}% in the worst audited case.
This is a conditional analysis-operator result, not a measured production-pipeline loss.
Extraction now uses one source-track seed across bands; copied source measurements are
not counted as independent observations. Earlier fit, scattering and information results
below the revised-input section remain diagnostic comparisons under the old extraction and must be reevaluated
before drawing a final physical-calibration conclusion. No coordinates were adopted.
<a href="seed-retention-audit.json">Per-target retention audit and assumptions</a>.</p>'''
    antenna_trials=[]
    for path in sorted(args.fits.glob('?????-five-centres.json')):
        trial=json.loads(path.read_text())
        corrections=np.array(trial['channel_position_corrections_enu_m'])
        widths=3*np.sqrt(np.diag(np.array(trial['antenna_position_prior_covariance_m2']))).reshape(5,3)
        antenna_trials.append({'assignment':trial['assignment'],'cost':trial['cost'],
            'converged':trial['converged'],'labels':trial['labels'],
            'corrections_enu_m':corrections.tolist(),
            'coordinates_at_bound':np.argwhere(abs(corrections)>=widths-1e-5).tolist(),
            'zero_amplitude_channels_by_target':trial['zero_amplitude_channels_by_target'],
            'posterior_coordinate_sigma_m':trial['information'].get('channel_position_standard_deviation_m'),
            'source':str(path),'accepted':False})
    (args.output/'vehicle-five-centres-trials.json').write_text(json.dumps(antenna_trials,indent=2))
    phase_metrics=[]
    if common:
        for target in common['targets']:
            a=sum(x['heldout_focused_energy'] for x in target['common_metrics'])
            b=sum(x['heldout_focused_energy'] for x in target['independent_metrics'])
            phase_metrics.append({'label':target['label'],'combined_loss_db':float(10*np.log10(b/a)),
                'per_channel_loss_db':target['heldout_common_loss_db'],
                'converged':target['common_converged'] and all(target['independent_converged'])})
    summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'labels':args.labels,
        'site_combinations_screened':48,'independent_complex_combinations_completed':completed,
        'source':'captures/live-vhf-20261013-193334/capture.dvblive',
        'status':'Expanded road-constrained analysis in progress; no accepted calibration or geographic movie yet',
        'common_path_validation':phase_metrics,'provisional_main_site_roads':selected,
        'joint_twelve_interval_trials':joint_trials,
        'shared_band_phase_diagnostic':None if phase is None else str(phase_path),
        'stationary_phase_diagnostic':None if stationary is None else str(stationary_path),
        'compact_scatterer_diagnostic':None if scatterer is None else str(scatterer_path),
        'extended_vehicle_diagnostic':str(args.output/'vehicle-extended-response.json') if identifiability_section else None,
        'finite_geometry_diagnostic':str(args.output/'vehicle-finite-site-sensitivity.json') if identifiability_section else None,
        'slow_return_screen':str(args.output/'slow-search-summary.json') if (args.output/'slow-search-summary.json').exists() else None,
        'input_conditioning_audit':None if seed_audit is None else str(seed_audit_path),
        'refit_status':('Common-source extraction and ten-target held-out refits complete; no accepted calibration'
                        if revised_section else 'Shared-source extraction implemented; revised fits pending') if seed_audit else None,
        'revised_input_validation':str(args.output/'shared-seed-summary.json') if revised_section else None,
        'latest_revised_site_comparison':str(args.output/'shared-seed-site-summary.json')
            if (args.output/'shared-seed-site-summary.json').exists() else None,
        'five_electrical_phase_centres_trials':antenna_trials,
        'preferred_road_trials':street_trials,
        'preferred_road_withheld':None if street_withheld is None else str(street_withheld_path),
        'finite_duration_scattering_trials':temporal_trials,
        'finite_duration_withheld':None if temporal_withheld is None else str(temporal_withheld_path),
        'preferred_road_observability':None if coverage is None else str(coverage_path),
        'mirror_frequency_diagnostic':None if mirror is None else str(mirror_path),
        'user_road_priority':{'names':['Albert Street','Victoria Parade'],
            'basis':'User plausibility judgement from plots; explicitly not independently observed vehicle positions',
            'candidate_intervals':[label for label,_ in preferred],
            'status':'Separate road-association sensitivity fits in progress; no hard geographic anchor introduced'},
        'withheld_target_validation':None if withheld is None else str(withheld_path),
        'receiver_nuisance_trial':None if receiver is None else {
            'correction_enu_m':receiver['receiver_correction_enu_m'],'prior_sigma_m':receiver['receiver_prior_sigma_m'],
            'cost':receiver['cost'],'accepted':False,
            'reason':'The vertical receiver correction reaches its three-sigma bound. This is a model/conditioning diagnostic, not a measured roof position.'},
        'road_model':'Connected OSM ways; cubic along-road motion spline; 5 m cross-track map/lane offset prior, 3 m cross-track slope prior and 5 m reflecting-height/DEM prior. Licensed sites have 10 m coordinate priors. Receiver remains fixed in this initial trial.',
        'screening_only':'Coarse screening uses range/rate peak uncertainties and robust residuals; these samples are not counted again as independent data in the complex fit.'}
    (args.output/'vehicle-analysis-progress.json').write_text(json.dumps(summary,indent=2))
    rows=''.join(f'<tr><td>{html.escape(label)}</td><td>{d["start"]:.0f}–{d["end"]:.0f}</td><td>{d["end"]-d["start"]:.0f}</td></tr>' for label,d in history.items())
    metrics=''.join(f'<tr><td>{q["label"]}</td><td>{q["combined_loss_db"]:.2f}</td><td>{"yes" if q["converged"] else "no"}</td></tr>' for q in phase_metrics)
    withheld_rows='' if withheld is None else ''.join(f'<tr><td>{q["label"]}</td><td>{q["heldout_loss_db"]:.3f}</td><td>{q["phase_only_loss_db_same_trajectory"]:.3f}</td><td>{"yes" if q["calibration_converged"] and q["target_converged"] and q["independent_converged"] else "pending"}</td><td>{", ".join(q.get("excluded_calibration_intervals",[q["label"]]))}</td></tr>' for q in withheld['targets'])
    validation_section='' if withheld is None else f'''<h2>Withheld vehicles and time blocks</h2>
<p>A complete candidate is withheld from calibration. Its trajectory and magnitudes then use even one-second blocks; odd blocks assess focusing. Possible overlapping fragments E and O are withheld together. Roads and phase-search basins remain conditional on discovery in the complete recording. Negative losses mean the shared fit predicts the withheld samples better than the independent control, not an SNR gain above an oracle.</p>
<table><tr><th>Candidate</th><th>Held-out loss (dB)</th><th>Phase-only loss on same path (dB)</th><th>All fits converged</th><th>Intervals excluded from calibration</th></tr>{withheld_rows}</table>
<p>These combined statistics weight the stronger channels most heavily. A small combined loss alone does not certify the inter-band phases needed for a coherent multistatic geographic image.</p>'''
    phase_section='' if phase is None else '''<h2>Band-by-band phase check</h2><p>The left panel compares independent complex band responses to one transferable channel-phase calibration. Error bars use deletion of complete two-second blocks with the path held fixed, rather than thermal noise alone; they remain conditional estimates of temporal phase stability. The right panel shows how much amplitude the shared-phase model retains. Large phase disagreements can be hidden in an aggregate score when the fit reduces the affected band's amplitude. This is a test of the point-response hypothesis, not proof that a particular transmitter site or scattering mechanism is responsible.</p><img src="vehicle-band-phase.png" alt="Phase contrasts and fitted amplitude retention in each band">'''
    uncertainty_section='''<h2>Geographic registration versus defocus</h2><p>The earlier fixed-world-pixel uncertainty calculation mixed peak displacement with loss of focus. The analysis below separates them. It is conditional on the main-site branch, the selected roads, a fixed receiver and local Gaussian covariance; it is not an accepted calibration. The displacement panel clips at 500 m, and uncertainty is larger across much of the footprint.</p><img src="vehicle-image-registration.png" alt="Calibration uncertainty separated into fixed-pixel loss, residual phase loss after registration, and geographic displacement"><p>Exact terrain and bistatic-distance checks with the measured waveform envelope confirm that registration preserves nearly all coherent power at some locations, while appreciable loss remains in poorly conditioned areas. <a href="vehicle-image-registration.json">Full conditional calculation and finite-geometry trials</a>.</p>'''
    if receiver:
        uncertainty_section+=f'''<p>A separate joint receiver-position trial uses 15 m east/north and 7.5 m vertical prior standard deviations. It estimates a correction of {receiver['receiver_correction_enu_m'][0]:.2f}, {receiver['receiver_correction_enu_m'][1]:.2f}, {receiver['receiver_correction_enu_m'][2]:.2f} m. The vertical correction reaches the allowed bound; it is not adopted as a physical antenna location.</p>'''
    temporal_section=''
    if stationary:
        stable_rows=''.join(f'<tr><td>{q["name"]}</td><td>{q["regions"][0]["median_squared_coherence"]:.4f}</td><td>{q["regions"][0]["phase_p05_p50_p95_degrees"][0]:+.2f} to {q["regions"][0]["phase_p05_p50_p95_degrees"][2]:+.2f}</td></tr>' for q in stationary['channels'])
        temporal_section=f'''<h2>Measured stationary-return phase stability</h2>
<p>The retained static profiles provide a recording-level check for a common temporal phase drift. Three separate range regions are compared with their own 30–90 second reference profile, using fixed 3.00288 s integrations. In the 0.6–5 km region, the central 90% phase span is roughly one or two degrees over the ten-minute capture. That is far smaller than the tens or hundreds of degrees in the vehicle band contrasts. This rules against a large common temporal phase drift as their explanation; it does not establish absolute cross-channel phase or rule out every direction-dependent response.</p>
<table><tr><th>Channel</th><th>Median squared pattern coherence, 0.6–5 km</th><th>Phase 5th–95th percentile (degrees)</th></tr>{stable_rows}</table>
<img src="stationary-phase.png" alt="Phase and complex range-pattern correlation across the complete recording">
<p>No fitted correction is applied to the samples or calibration. The curves are empirical pattern comparisons, not thermal-error bounds. <a href="stationary-phase.json">Numerical data and scope</a>.</p>'''
    model_section=''
    if scatterer:
        scatterer_rows=''
        for q in scatterer['targets']:
            one,two=q['models'];a=one['held_out']['shared_prediction_error'];b=two['held_out']['shared_prediction_error']
            scatterer_rows+=f'<tr><td>{q["label"]}</td><td>{100*(1-b/a):+.2f}%</td><td>{two["held_out"]["design_condition_number"]:.1f}</td></tr>'
        model_section=f'''<h2>Checking model and optimiser limitations</h2>
<p>For fixed geometry, positive band amplitudes and one common target phase can be eliminated exactly. The implementation checks the stationary points of the piecewise sinusoidal likelihood, and its analytic derivatives pass numerical phase/range checks. The main-site fit still assigns exactly zero amplitude to several disagreeing bands. Slow convergence or a small positive amplitude floor therefore does not explain away that inconsistency.</p>
<p>A compact two-point vehicle test uses exact bistatic paths for rigidly attached scatterers, each with one complex reflectivity across all bands. It fits alternate one-second blocks and predicts the withheld blocks, conditional on the existing calibration and trajectory. The body is bounded to 8 × 3 × 3 m. This follows the point-scatterer forward-model structure used in <a href="https://doi.org/10.1049/rsn2.12263">multitransmitter passive-radar imaging</a>; it is a test hypothesis, not an identified vehicle model.</p>
<table><tr><th>Candidate</th><th>Held-out prediction-error reduction</th><th>Two-point design condition number</th></tr>{scatterer_rows}</table>
<p>These percentages describe prediction error, not radar SNR gain. Some targets improve modestly, others worsen, and several fits use nearly coincident, cancelling scatterers with a poorly conditioned design. The two-point extension has not been adopted. <a href="vehicle-scatterer-validation.json">Full comparison</a>.</p>'''
    if antenna_trials:
        antenna_rows=''.join(f'<tr><td>{"".join(map(str,q["assignment"]))}</td><td>{q["cost"]:.2f}</td><td>{len(q["coordinates_at_bound"])}</td><td>{sum(map(len,q["zero_amplitude_channels_by_target"]))}</td></tr>' for q in antenna_trials)
        model_section+=f'''<h2>Five electrical phase centres</h2>
<p>Channels on the same tower need not have exactly the same electrical phase centre. The expanded fit gives all five channels separate position corrections, with a correlated 10 m site-position uncertainty and independent within-tower standard deviations of 5 m horizontally and 10 m vertically. Those are explicit sensitivity assumptions, not measured antenna dimensions. All target trajectories still fit jointly; the receiver remains fixed in this test.</p>
<table><tr><th>Site assignment</th><th>Ten-interval objective</th><th>Coordinates at allowed bound</th><th>Zero fitted band responses</th></tr>{antenna_rows}</table>
<p>These trial objectives include their stated priors and are not calibrated odds for selecting a model. Added freedom improves the objective but has not yet produced a credible, fully retained five-band response. Boundary-reaching height changes are not adopted as antenna measurements. <a href="vehicle-five-centres-trials.json">Corrections, conditional uncertainties and trial details</a>.</p>'''
    preferred_section='' if not preferred else '''<h2>Prioritised local-street hypotheses</h2>
<p>The user considers Albert Street and Victoria Parade more plausible than several of the other plotted road associations. This is explicitly a plausibility judgement, not an independently observed vehicle position. Nine candidate intervals have alternatives on these two roads. Their complex-data fits are being tested separately, including alternatives that were below the first six coarse road candidates. The search retains the unrestricted trials for comparison; a preferred road is not counted as a measured geographic anchor.</p>
<img src="vehicle-local-street-hypotheses.png" alt="Nine candidate intervals with Albert Street and Victoria Parade alternatives; conditional road hypotheses">
<p>The two paths in each panel represent alternative explanations of the same echo. They do not represent two detected vehicles. The street geometry includes the mapped one-way carriageways and their speed limits.</p>'''
    if street_trials:
        trial_rows=''.join(f'<tr><td>{"".join(map(str,q["assignment"]))}</td><td>{html.escape(q["variant"])}</td><td>{q["cost"]:.2f}</td><td>{"yes" if q["converged"] else "no"}</td><td>{"—" if not q["range_bias_m"] else ", ".join(f"{x:+.1f}" for x in q["range_bias_m"])}</td></tr>' for q in street_trials)
        preferred_section+=f'''<h3>Preferred-road calibration trials</h3>
<p>These trials use the same nine intervals. A shared range-origin sensitivity test allows one constant envelope offset for Seven, Nine, Ten and ABC relative to SBS; it does not permit target-specific channel delays or phases. The measured offsets can reflect reference-response bias as well as geometry. They have not been identified as an SDR clock error or adopted in production. The carrier-phase ambiguity search evaluates every target's delay grid before exact physical-path refinement; this improves starting-point coverage but is not a proof of global optimality.</p>
<table><tr><th>Site choices</th><th>Trial</th><th>Objective</th><th>Converged</th><th>Range-origin offsets (m): Seven, Nine, Ten, ABC</th></tr>{trial_rows}</table>
<p>Objectives include their stated priors. Models with additional fitted parameters must be judged by withheld predictions, not objective reduction alone. E and O may be two intervals from one vehicle.</p>'''
    if street_withheld:
        band_rows=''.join(f'<tr><td>{q["label"]}</td><td>{q["heldout_loss_db"]:.3f}</td><td>{q["phase_contrast_degrees"][3]:+.0f}° / {q["shared_to_independent_amplitude_ratio"][3]:.2f}</td><td>{q["phase_contrast_degrees"][4]:+.0f}° / {q["shared_to_independent_amplitude_ratio"][4]:.2f}</td><td>{"yes" if q["calibration_converged"] and q["target_converged"] and q["independent_converged"] else "no"}</td></tr>' for q in street_withheld['targets'])
        preferred_section+=f'''<h3>Band-resolved withheld tests on the preferred roads</h3>
<p>Each complete candidate is excluded from calibration, with possible fragments E/O excluded together. Its path and band magnitudes use alternating one-second training blocks, with the remaining blocks reserved for prediction. The following table makes visible a failure that a combined score can hide: some measurable bands receive almost zero fitted amplitude. These are not accepted coherent calibrators.</p>
<table><tr><th>Candidate</th><th>Combined held-out loss (dB)</th><th>Ten phase / retained amplitude</th><th>ABC phase / retained amplitude</th><th>All fits converged</th></tr>{band_rows}</table>
<p>Phases are relative to SBS after the transferred calibration. Amplitude ratios compare the shared-phase model with an independent complex response on the same withheld trajectory. Road discovery and phase-branch initialization still used the full recording, so this remains a conditional test. <a href="vehicle-streets-delay-withheld.json">All five bands, signal energies and fit details</a>.</p>'''
    if (args.output/'vehicle-streets-time-phase.png').exists():
        preferred_section+='''<h3>Phase consistency over shorter intervals</h3>
<p>This plot holds one fitted path and channel calibration fixed and measures independent band responses in short, non-overlapping windows. Weak intervals and strong intervals remain visible. Phase becomes unreliable as focused signal energy falls. The observation window changes without changing the original deramp, RF phase origin or range covariance.</p>
<img src="vehicle-streets-time-phase.png" alt="Three-second band phase contrasts and focused signal energy for the nine road-priority intervals">
<p>A follow-up likelihood permits one common complex target reflectivity and positive band magnitudes per three-second block, while sharing geometry, trajectory continuity and calibration across the entire track. It introduces no independent band phase. Constant and variable-reflectivity controls use the identical samples, block projections and covariance; held-out prediction is required before adopting the extra freedom.</p>'''
    if temporal_trials:
        temporal_rows=''.join(f'<tr><td>{"".join(map(str,q["assignment"]))}</td><td>{q["constant_control"]:.2f}</td><td>{q["variable_cost"]:.2f}</td><td>{"yes" if q["converged"] else "no"}</td></tr>' for q in temporal_trials)
        preferred_section+=f'<table><tr><th>Site choices</th><th>Constant reflectivity, same block data</th><th>Variable reflectivity</th><th>Converged</th></tr>{temporal_rows}</table><p>These objectives use a different observation subspace from the whole-interval trials above and must not be compared numerically across the two tables.</p>'
    if temporal_withheld:
        transfer_rows=''
        for q in temporal_withheld['targets']:
            blocks=q['shared']['blocks']
            energy=np.array([b['independent_band_energy'] for b in blocks])
            phase=np.radians([b['phase_contrast_degrees'] for b in blocks])
            precision=energy*energy[:,1,None]/np.maximum(energy+energy[:,1,None],1e-30)
            resultant=(precision*np.exp(1j*phase)).sum(axis=0)/np.maximum(precision.sum(axis=0),1e-30)
            angles=np.angle(resultant,deg=True)
            transfer_rows+=f'<tr><td>{q["label"]}</td><td>{q["heldout_loss_db"]:.3f}</td><td>{100*q["prediction_error_reduction_fraction"]:+.2f}%</td><td>{angles[3]:+.0f}° / {abs(resultant[3]):.2f}</td><td>{angles[4]:+.0f}° / {abs(resultant[4]):.2f}</td></tr>'
        preferred_section+=f'''<h3>Does variable reflectivity transfer to withheld vehicles?</h3>
<p>The three-second model is now tested by excluding each complete candidate from calibration, fitting its path and scattering coefficients on alternating one-second blocks, and predicting the others. All fits in this test converged. Prediction-error changes below compare against independent per-band complex responses; positive values favour the shared-phase model. Phase averages weight each block by its conditional phase information; the following number is circular concentration, from 0 (dispersed) to 1 (consistent).</p>
<table><tr><th>Candidate</th><th>Held-out loss (dB)</th><th>Prediction-error reduction</th><th>Ten phase / concentration</th><th>ABC phase / concentration</th></tr>{transfer_rows}</table>
<p>The richer model helps some predictions, including A, but does not yield a transferable five-band point response across these candidates. Large phase offsets remain for several tracks. This is not yet a calibration suitable for adoption. <a href="vehicle-streets-temporal-withheld.json">Block-level withheld results and scope</a>.</p>
<img src="vehicle-streets-temporal-phase.png" alt="Short-window phase contrasts under the finite-duration scattering model">'''
    if coverage:
        information_rows=''
        labels={'known_paths_phases_and_range_origins':'Exact trajectories, known channel phase/origin',
                'known_paths_unknown_phase_and_range_origins':'Exact trajectories, unknown channel phase/origin',
                'unknown_paths_phase_and_range_origins':'Unknown trajectories, phase and origin'}
        for key,value in coverage['information_cases'].items():
            sigma=np.array(value['posterior_coordinate_sigma_with_10m_prior'])
            information_rows+=f'<tr><td>{labels[key]}</td><td>{sigma[:,0].min():.2f}–{sigma[:,0].max():.2f}</td><td>{sigma[:,1].min():.2f}–{sigma[:,1].max():.2f}</td><td>{sigma[:,2].min():.2f}–{sigma[:,2].max():.2f}</td></tr>'
        preferred_section+=f'''<h3>What the street hypotheses can constrain</h3>
<p>As viewed from Mt Dandenong, these nine intervals span only about <b>0.36° in azimuth and 0.09° in elevation</b>. Several vehicles along these roads therefore supply repeated measurements of a narrow set of departure directions. This statement concerns the proposed road associations, not every target in the recording.</p>
<img src="vehicle-street-observability.png" alt="Preferred-road angular coverage as viewed from each transmitter">
<p>An optimistic expected-information calculation uses all independently measured band powers and an ideal shared-phase point response, including bands rejected by the actual fit. Even treating the complete trajectories and channel phase/origins as known leaves the east coordinate almost entirely set by the 10 m prior. The table gives coordinate standard-deviation ranges across the three sites, in metres, after adding that explicit prior.</p>
<table><tr><th>Conditional assumptions</th><th>East σ (m)</th><th>North σ (m)</th><th>Up σ (m)</th></tr>{information_rows}</table>
<p>This is a sensitivity comparison under the stated point model and empirical covariance, not a certified experimental CRLB. Known-path cases are optimistic thought experiments, not independent vehicle observations. The actual fit also reaches several allowed coordinate bounds; a small numerical residual or optimizer convergence would not establish a surveyed phase centre. <a href="vehicle-street-observability.json">Calculation, matrices and geometric samples</a>.</p>'''
    audit_section=''
    if mirror:
        pairs='; '.join(f'{names[a]} / {names[b]}: median {q[0]:.1f} dB, maximum {q[2]:.1f} dB'
                       for (a,b),q in zip(mirror['pairs'],mirror['occupied_coherence_percentiles_db']))
        audit_section=f'''<h2>Phase-bookkeeping and mirror-channel controls</h2>
<p>A new synthetic archive test starts with exact bistatic point paths, five RF frequencies, fractional range coordinates, an asymmetric complex range response, different reversible deramp origins, nonzero archived window phases and FEC row erasures. The archived-profile extraction preserves phase to within its 0.01° test limit. This checks the analysis boundary after range compression; it complements the existing GPU tests and does not establish the physical scattering model.</p>
<p>Seven/ABC and SBS/Ten are mirror pairs about the 202 MHz capture LO. A nonconjugated cross-spectrum check uses four one-second intervals across the recording, 4096-sample FFTs and 125 kHz frequency groups. Measured occupied-band correlation: {pairs}. These are descriptive correlations, not a calibrated image-rejection bound: broadcast pilot structure and finite-sample uncertainty can contribute. No IQ correction has been applied. <a href="mirror-frequency-diagnostic.json">Frequency-resolved measurement</a>.</p>'''
    text=f'''<!doctype html><meta charset="utf-8"><title>Vehicle-constrained phase-centre analysis</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:36px auto;color:#23303b;line-height:1.55}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:6px 18px;border-bottom:1px solid #ddd}}.status{{background:#fff2ce;padding:18px}}a{{color:#13658c}}</style>
<h1>Expanded vehicle and transmitter-site analysis</h1>
<p><b>Latest revised-input comparison:</b> <a href="site-refits.html">all 48 licensed site
assignments, independent electrical centres, receiver uncertainty and withheld-band phase transfer</a>.
This comparison uses the corrected extraction. Historical sections below retain their original scope.</p>
{seed_notice}
{revised_section}
<p class="status">Calibration remains in progress. All 48 licensed site combinations have completed road-constrained range/rate screening; {len(completed)}/48 have completed an independent-band complex search. The broader joint search has saved {len(joint_trials)}/48 twelve-interval trials; some still require convergence or association refinement. No fitted coordinates have been adopted. Geographic imaging will use a coherent five-channel statistic.</p>
<p>The five complete channel videos are ready: <a href="../../../captures/live-vhf-20261013-193334/videos-3s-100hz/seven.mov">Seven</a> · <a href="../../../captures/live-vhf-20261013-193334/videos-3s-100hz/sbs.mov">SBS</a> · <a href="../../../captures/live-vhf-20261013-193334/videos-3s-100hz/nine.mov">Nine</a> · <a href="../../../captures/live-vhf-20261013-193334/videos-3s-100hz/ten.mov">Ten</a> · <a href="../../../captures/live-vhf-20261013-193334/videos-3s-100hz/abc.mov">ABC</a>. Each covers the complete recording at 30.64 fps, with a three-second coherent interval, ±100 Hz Doppler, 0–30 km excess range, a fixed colour scale and all-intra ProRes encoding. All 170 sampled QuickTime seek checks passed.</p>
<p>Twelve candidate intervals are included, lasting 14–30 seconds. Possible fragments E and O are not counted as independent vehicles in validation. The search follows connected OpenStreetMap roads, with actual one-way and speed-limit tags where available. Lane position, map error, reflecting height and motion remain uncertain; a plotted road match is a hypothesis.</p>
<table><tr><th>Candidate</th><th>Capture interval (s)</th><th>Length (s)</th></tr>{rows}</table>
<img src="vehicle-range-tracks.png" alt="Measured candidate range histories">
{preferred_section}
<p>Grey road lines below are geographic priors, not radar observations. Several candidate branches remain plausible. The current plot is conditional on main licensed transmitter sites and is not a calibrated target map.</p>
<img src="vehicle-road-hypotheses.png" alt="Provisional road associations">
<p>The 48 assignments include every permitted main/alternative/standby choice in the ACMA data: 2 × 2 × 2 × 3 × 2. The coarse scores below only select starting trajectories. The retained complex range/phase data, a shared channel-phase model, and held-out predictions decide calibration acceptance.</p>
<img src="vehicle-sites-screening.png" alt="All licensed site assignments screened">
<h2>Temporal coherence control</h2><p>This control tests a shared path history for the four nominally co-sited Ornata bands, retaining independent constant band phases. Alternating one-second blocks are withheld. It tests temporal phase consistency, not transferable absolute cross-channel phase. Association and starting branches are conditional on the full recording.</p>
<table><tr><th>Candidate</th><th>Combined held-out common-path loss (dB)</th><th>Fits converged</th></tr>{metrics}</table>
{validation_section}
{phase_section}
{temporal_section}
{model_section}
{identifiability_section}
{uncertainty_section}
{audit_section}
<h2>Acceptance work remaining</h2><p>All 48 site assignments have been explored, but several joint searches are not converged and the preferred-road fits fail full five-band phase transfer. A validated calibration still needs a transferable scattering model and enough independent geometric information. The <a href="../isar_20261013/analysis.html">container-ship analysis with historical AIS</a> supplies a different viewing direction, but its motion/scattering model also fails withheld prediction tests. No geographic or ship image is accepted as calibrated solely because its contrast improves. Independent information about a compact reflector's position or a calibration vehicle's trajectory would add constraints that the present road associations do not supply.</p>
<p>A further short-track screen outside the near-street group selected candidate T near 3.2 km excess range and U near 17.8 km. Their initial complex tests did not establish a common Seven/SBS/Nine path: withheld focused energies were only about 0.3–3.4 in the existing whitened statistic. They are not additional accepted calibrators, and greater excess range alone does not establish useful angular diversity. <a href="wider-angle-screen.json">Screening criteria and measurements</a>.</p>
<p>Sources: <a href="https://download.bbbike.org/osm/bbbike/Melbourne/">BBBike Melbourne extract</a>, map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors, ODbL</a>; <a href="https://www.acma.gov.au/radiocomms-licence-data">ACMA licence data</a>; <a href="https://services.ga.gov.au/gis/rest/services/DEM_SRTM_1Second_2024/MapServer">Geoscience Australia 1 arcsecond DEM</a>. Terrain is not a native DTED Level 2 product.</p>
<p><a href="vehicle-analysis-progress.json">Machine-readable analysis snapshot</a> · Updated {summary['generated_utc']}</p>'''
    (args.output/'vehicle-analysis.html').write_text(text)
    print('Report',args.output/'vehicle-analysis.html','complex assignments',len(completed),flush=True)


if __name__=='__main__':main()
