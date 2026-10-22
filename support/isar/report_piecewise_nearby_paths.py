"""Review piecewise measured phases and the co-sited relative-delay control."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import numpy as np

from cluster_phase import SplinePath
from antenna_motion import Antennas
from compare_ais_phase import NAMES,COLOURS
from fit_piecewise_nearby_paths import guarded_model
from measured_phase import range_rate_map
from motion import C
from nearby_echo_control import load
from phase_window import window_band
from relative_scatterer_delay import RelativeScattererDelay
from relative_delay_geometry import correlated_pair_bounds
from report_fixed_grid_guided_return import phase_comparison
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options


def histories(records,bands,interval):
    result={}
    for s in (1,4):
        native=[]
        for record in records:
            r=record[str(s)];z=np.asarray(r['complex_coefficient_real_imag'])
            native.append(dict(time_s=np.asarray(r['time_s']),physical_path_change_m=np.asarray(r['physical_path_change_m']),
                complex_coefficient=z[:,0]+1j*z[:,1],conditional_information=np.asarray(r['conditional_information'])))
        result[str(s)]=phase_comparison(native,bands,*interval)
    return result


def hull_pair_bound(control,interval):
    """Necessary double-difference bound across every licensed assignment."""
    mid=sum(interval)/2;row=control['bands'][0].field.data['row_time'];state=control['nominal'][0]
    centre=np.array([np.interp(mid,row,state['position'][:,i]) for i in range(3)])
    geo=json.loads((control['geometry']/'geometry-prior.json').read_text());rotation=np.asarray(geo['ecef_to_enu'])
    options=licensed_options(control['geometry']);records=[]
    for assignment in itertools.product(*[[o for o in options if o['channel']==c] for c in range(5)]):
        sites=[a['site_id'] for a in assignment];tx=np.asarray([a['transmitter_ecef_m'] for a in assignment])
        prior=Antennas(rotation,sites,receiver_sigma=(0.,0.,7.5))
        b=correlated_pair_bounds(centre,tx,rotation,prior.factor[:15],scene_radius_m=500.,
            maximum_speed_mps=0.,maximum_acceleration_mps2=0.,parameter_radius=3.)
        limit=500*np.max(b['maximum_transmitter_baseline_m']/b['minimum_propagation_distance_m'])
        records.append(dict(sites=sites,maximum_interband_pair_range_difference_m=float(limit)))
    return dict(scene_radius_m=500.,maximum_point_separation_m=500.,standardized_antenna_box_radius=3.,
        maximum_over_all_32_assignments_m=max(r['maximum_interband_pair_range_difference_m'] for r in records),
        assignments=records,scope='Necessary geometric feasibility, not a noise error bar. The common receiver leg cancels. '
            'Integrating the distance Hessian over transmitter and point segments bounds the interband pair double difference by point separation times transmitter baseline divided by the minimum distance. '
            'Site/individual antenna covariance and all 32 licensed assignments are included. Applies to a proposed common hull pair inside the stated generous ball, not arbitrary unrelated objects.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--relative',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    data=json.loads(args.source.read_text());relative=json.loads(args.relative.read_text()) if args.relative else None
    if not data['complete'] or relative and not relative['complete']:raise ValueError('Wait for every declared start')
    control=load();out=args.output;out.mkdir(exist_ok=True,parents=True)
    errors=[];reviews=[];selection_maps=[];selections=[];statistics=[];all_trials=[]
    for number,w in enumerate(data['windows']):
        first,last=w['interval_s'];bands=[window_band(b,first,last,complete_rows=True)[0] for b in control['bands']]
        curve=SplinePath(first,last,w['knot_spacing_s']);base=guarded_model(bands,control,curve,True)
        independent=[];records=[]
        for c,entry in enumerate(w['independent']):
            model=guarded_model([bands[c]],control,curve,False)
            for f in entry['trials']:
                r,_=model.evaluate(np.asarray(f['parameters']));errors.append(abs(r@r*model.energy-f['cost']))
                all_trials.append(dict(window=number,model='independent',channel=NAMES[c],seed=f['seed'],
                    gradient=f['maximum_projected_gradient'],cost=f['cost'],bounds=f['bound_indices']))
            fit=entry['trials'][entry['best_index']];independent.append(fit);records.append(entry['measured'][0])
            if c==0:selection_maps.append([]);selections.append([])
            b=bands[c];rates=np.linspace(-3,3,301)
            selection_maps[-1].append(range_rate_map(b.field.time,b.field.data['value'],b.field.data['information'],b.field.k,rates)/b.field.pulse.samples[64].real)
            q=np.asarray(fit['parameters']);local=q[model.indices[0]];chosen=[]
            for s in (1,4):
                arc=model.clusters[0].arcs[s];p=local[model.clusters[0].slices[s]]
                observer=-b.field.data['centre_omega']/(b.field.k*.001024)
                rate=arc.known_rate+arc.base.rate_basis@p[2:]+observer
                chosen.append(dict(point='B' if s==1 else 'E',offset_m=q[2*s]*C/8e6,
                    rate_mps=float(np.interp(curve.epoch,b.field.data['row_time'],rate))))
            selections[-1].append(chosen)
        for f in w['common']:
            r,_=base.evaluate(np.asarray(f['parameters']));errors.append(abs(r@r*base.energy-f['cost']))
            all_trials.append(dict(window=number,model='common paths',channel='all',seed=f['seed'],
                gradient=f['maximum_projected_gradient'],cost=f['cost'],bounds=f['bound_indices']))
        common=w['common'][w['best_common_index']]
        variants=[dict(label='Independent'+(' (unsettled band)' if max(f['maximum_projected_gradient'] for f in independent)>1e-5 else ''),records=records,parameters=None,measured=histories(records,bands,w['interval_s'])),
            dict(label='Common paths',records=w['common_measured'],parameters=common['parameters'],
                measured=histories(w['common_measured'],bands,w['interval_s']))]
        baseline=sum(f['prediction']['testing'][0]['heldout_error'] for f in independent)
        models_stats=[dict(model='Common paths',unused_percent=w['total_unused_error_change_percent'],
            gradient=common['maximum_projected_gradient'],bounds=common['bound_indices'],
            component_to_sum=w['common_local_B_E_component_to_sum_energy_ratio'],
            gaps_m=[(common['parameters'][10*c+8]-common['parameters'][10*c+2])*C/8e6 for c in range(5)])]
        if relative:
            rw=relative['windows'][number]
            if rw['interval_s']!=w['interval_s']:raise ValueError('Window identities differ')
            nested=RelativeScattererDelay(base,1,4,rw['groups'])
            for f in rw['trials']:
                r,_=nested.evaluate(np.asarray(f['parameters_reduced']));errors.append(abs(r@r*nested.energy-f['cost']))
                all_trials.append(dict(window=number,model='relative delay',channel='all',seed=f['seed'],
                    gradient=f['audit']['projected_gradient'],cost=f['cost'],bounds=f['audit']['bound_indices'],
                    mapped_guard=f['audit']['mapped_envelope_guard']))
            if 'best_index' in rw:
                fit=rw['trials'][rw['best_index']]
                variants.append(dict(label='Co-sited relative delay'+(' (unsettled)' if fit['audit']['projected_gradient']>1e-5 else ''),records=rw['measured'],parameters=fit['parameters'],
                    measured=histories(rw['measured'],bands,w['interval_s'])))
                models_stats.append(dict(model='Co-sited relative delay',unused_percent=rw['unused_change_from_independent_percent'],
                    gradient=fit['audit']['projected_gradient'],bounds=nested.kept[np.asarray(fit['audit']['bound_indices'],int)].tolist(),
                    component_to_sum=rw['local_B_E_component_to_sum_energy_ratio'],
                    gaps_m=[(fit['parameters'][10*c+8]-fit['parameters'][10*c+2])*C/8e6 for c in range(5)],
                    mapped_guard=fit['audit']['mapped_envelope_guard']))
                released=rw['released'];r,_=base.evaluate(np.asarray(released['parameters']));errors.append(abs(r@r*base.energy-released['cost']))
                variants.append(dict(label='Released control'+(' (unsettled)' if released['maximum_projected_gradient']>1e-5 else ''),records=rw['released_measured'],parameters=released['parameters'],
                    measured=histories(rw['released_measured'],bands,w['interval_s'])))
                models_stats.append(dict(model='Released control',unused_percent=100*(sum(v['heldout_error'] for v in released['prediction']['testing'])/baseline-1),
                    gradient=released['maximum_projected_gradient'],bounds=released['bound_indices'],
                    component_to_sum=rw['released_local_B_E_component_to_sum_energy_ratio'],
                    gaps_m=[(released['parameters'][10*c+8]-released['parameters'][10*c+2])*C/8e6 for c in range(5)]))
        bound=hull_pair_bound(control,w['interval_s'])
        for s in models_stats:
            s['gap_spread_m']=float(np.ptp(s['gaps_m']))
            s['inside_necessary_hull_pair_bound']=s['gap_spread_m']<=bound['maximum_over_all_32_assignments_m']
        statistics.append(dict(interval_s=w['interval_s'],models=models_stats,hull_pair_geometric_bound=bound,
            independent_selected_gradients=[f['maximum_projected_gradient'] for f in independent],
            independent_selected_bounds=[f['bound_indices'] for f in independent]))
        fig,axes=plt.subplots(2,len(variants),figsize=(4.7*len(variants),8),layout='constrained',sharex=True)
        fig2,axes2=plt.subplots(2,len(variants),figsize=(4.7*len(variants),8),layout='constrained',sharex=True)
        for col,v in enumerate(variants):
            for row,s in enumerate((1,4)):
                m=v['measured'][str(s)];time=m['time_s'];reference=m['channels'][2]['measured_path_change_m']
                for c,colour in enumerate(COLOURS):
                    value=m['channels'][c]['measured_path_change_m']
                    axes[row,col].plot(time,value,'.-',color=colour,ms=2,lw=.8,label=NAMES[c])
                    if c!=2:axes2[row,col].plot(time,100*(value-reference),'.-',color=colour,ms=2,lw=.8,label=NAMES[c])
                label='B' if s==1 else 'E'
                axes[row,col].set(title=f'{label} · {v["label"]}\norigin {m["anchor_time_s"]:.2f} s',ylabel='Measured path change (m)',xlim=(first,last))
                axes2[row,col].set(title=f'{label} · {v["label"]}',ylabel='Measured channel minus Nine (cm)',xlim=(first,last))
                axes2[row,col].axhline(0,color='#aaa',lw=.5)
                if 'unsettled' in v['label']:
                    axes[row,col].title.set_color('#b00020');axes2[row,col].title.set_color('#b00020')
                    for ax in (axes[row,col],axes2[row,col]):
                        ax.text(.03,.95,'Unsettled / cancelling model',transform=ax.transAxes,va='top',
                            color='#b00020',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        axes[0,0].legend(ncol=3,fontsize=8);axes2[0,0].legend(ncol=2,fontsize=8)
        for a in (axes,axes2):
            for ax in a.ravel():
                ax.xaxis.set_major_locator(MultipleLocator(.5));ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            for ax in a[-1]:ax.set_xlabel('Capture time (s)')
        fig.suptitle(f'{first:g}–{last:g} s · conditional phase measurements under each training-frozen scene\n'
            'Only constant origins are removed. Each RF phase is divided by its own wavenumber.\n'
            'Unsettled panels can contain enormous cancelling components; apparent agreement there is not valid point-association evidence.',fontsize=11)
        fig2.suptitle(f'{first:g}–{last:g} s · interband measured path-change discrepancies\n'
            'No slope or curvature correction. A co-sited delay constraint is a model hypothesis, not proof that selected components are material points.',fontsize=12)
        fig.savefig(out/f'{number+1:02d}-measured-paths.png',dpi=150);plt.close(fig)
        fig2.savefig(out/f'{number+1:02d}-interband-differences.png',dpi=150);plt.close(fig2)
        reviews.append(dict(interval_s=w['interval_s'],variants=variants))
    if max(errors)>1e-7:raise ValueError(f'Original likelihood changed: {max(errors)}')
    vmax=np.ceil(10*np.log10(max(m.max() for row in selection_maps for m in row)))
    fig,axes=plt.subplots(len(reviews),5,figsize=(19,4*len(reviews)),layout='constrained',sharex=True,sharey=True)
    for row,(review,maps,points) in enumerate(zip(reviews,selection_maps,selections)):
        for c,(power,chosen) in enumerate(zip(maps,points)):
            ax=axes[row,c];offset=control['bands'][c].field.offsets*C/8e6
            image=ax.imshow(10*np.log10(np.maximum(power,1e-8)),origin='lower',aspect='auto',cmap='magma',
                vmin=0,vmax=vmax,extent=[offset[0]-C/16e6,offset[-1]+C/16e6,-3,3],interpolation='nearest')
            for item,colour in zip(chosen,('#00e5ff','#72ff64')):
                xy=item['offset_m'],item['rate_mps'];ax.plot(*xy,'s',mfc='none',mec=colour,ms=9)
                ax.annotate(item['point'],xy,xytext=(6,5),textcoords='offset points',color=colour)
            ax.set(title=NAMES[c]+f' · {review["interval_s"][0]:g}–{review["interval_s"][1]:g} s',xlim=(-110,150))
            if row==len(reviews)-1:ax.set_xlabel('Fixed-reference offset (m)')
            if c==0:ax.set_ylabel('Excess-path rate (m/s)')
    fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.8,label='Original coherent cell power / conditional noise (dB)')
    fig.suptitle('Independent fitted centres in unchanged original channel maps\n'
        'B/E are candidate labels. The maps precede all fitted stationary/neighbour subtraction.',fontsize=12)
    fig.savefig(out/'00-selected-points.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(reviews),2,figsize=(14,4*len(reviews)),layout='constrained')
    for row,stat in enumerate(statistics):
        for m,style in zip(stat['models'],('o-','s--','^:')):
            axes[row,0].plot(NAMES,m['gaps_m'],style,label=m['model'])
            axes[row,1].plot(NAMES,m['component_to_sum'],style,label=m['model'])
        axes[row,0].set(title=f'{stat["interval_s"][0]:g}–{stat["interval_s"][1]:g} s · E−B envelope separation',ylabel='Fitted separation (m)')
        axes[row,1].set(title='Local B/E cancellation diagnostic',ylabel='Component energy / pair-sum energy',yscale='log')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Sharing temporal paths alone does not enforce a common physical point pair\n'
        'Co-sited relative gaps are equal by construction in the restricted model; predictive fit, conditioning and phase measurements must validate it.',fontsize=12)
    fig.savefig(out/'04-relative-range-and-cancellation.png',dpi=150);plt.close(fig)
    paths=[args.source,Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('report_fixed_grid_guided_return','phase_window','relative_scatterer_delay','joint_path_information',
            'relative_delay_geometry','antenna_motion','fit_piecewise_nearby_paths','nearby_echo_control')],
        *[control['geometry']/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')]]
    if args.relative:paths.append(args.relative)
    result=dict(complete=True,accepted_calibration=False,production_changed=False,reviews=reviews,statistics=statistics,
        selected_centres=selections,reproduced_objectives=len(errors),maximum_absolute_cost_error=max(errors),all_trials=all_trials,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    with (out/'measured-trajectories.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('window_start_s','window_end_s','model','point','channel','capture_time_s','measured_path_change_m','rf_phase_change_deg'))
        for review in reviews:
            for v in review['variants']:
                for s,label in ((1,'B'),(4,'E')):
                    m=v['measured'][str(s)]
                    for c,channel in enumerate(m['channels']):
                        for t,p in zip(m['time_s'],channel['measured_path_change_m']):
                            writer.writerow((*review['interval_s'],v['label'],label,NAMES[c],t,p,-np.degrees(control['bands'][c].field.k*p)))
    rows=''.join(f'<tr><td>{s["interval_s"]}</td><td>{m["model"]}</td><td>{m["unused_percent"]:+.3f}%</td><td>{m["gradient"]:.3g}</td><td>{m["bounds"]}</td><td>{m["gap_spread_m"]:.2f} / {s["hull_pair_geometric_bound"]["maximum_over_all_32_assignments_m"]:.2f} m</td></tr>' for s in statistics for m in s['models'])
    unsettled=''.join(f'<li>{s["interval_s"]}: {m["model"]}, projected gradient {m["gradient"]:.3g}, maximum local B/E component-to-sum energy ratio {max(m["component_to_sum"]):.3g}.</li>' for s in statistics for m in s['models'] if m['gradient']>1e-5)
    images=''.join(f'<h2>{r["interval_s"][0]:g}–{r["interval_s"][1]:g} s</h2><img src="{i+1:02d}-measured-paths.png" alt="Measured five-band B/E phase trajectories"><img src="{i+1:02d}-interband-differences.png" alt="Interband measured phase-change discrepancies">' for i,r in enumerate(reviews))
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Piecewise nearby-scatterer phase tests</title>
<style>body{{font:16px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px}}</style>
<h1>Piecewise nearby-scatterer phase analysis</h1>
<p>All three predeclared two-second controls are complete. Each compares two independent starts per band and three common-path starts. The co-sited relative-delay experiment adds the necessary equality of E−B range difference for the four nominal Ornata bands; ABC remains separate at Ridge. Per-band carrier phases and positive brightness remain free. A flat fitted gap in that restricted model is a constraint, not evidence of physical point identity. No antenna coordinates or production processing change.</p>
<p><b>Component labels are not cross-band track identities.</b> In the 163–165 s independent fits, B's midpoint path rates in Seven and SBS are −0.005 and +0.006 m/s, while E's are −1.03 and −1.16 m/s. Thus matching “B” to “B” can compare a near-zero-rate fitted component with the moving ship return in another band. The next physical-primary association must test the component choices and permit weak/absent bands, rather than force both labels to be common material points.</p>
<img src="00-selected-points.png" alt="Selected candidates in original maps for each matching window">
<table><tr><th>Window</th><th>Model</th><th>Unused squared-error change vs independent</th><th>Selected projected gradient</th><th>Guard indices in base coordinates</th><th>Fitted interband gap spread / conservative hull-pair bound</th></tr>{rows}</table>
<p>The last column checks a necessary physical condition across <b>all 32 licensed assignments</b>, allowing both points within 500 m of the AIS reference, up to 500 m point separation and the full ±3 standardized antenna-prior box. For D<sub>c</sub>=L<sub>c</sub>(x<sub>E</sub>)−L<sub>c</sub>(x<sub>B</sub>), the receiver cancels from D<sub>c</sub>−D<sub>d</sub>. The distance Hessian gives |D<sub>c</sub>−D<sub>d</sub>|≤point-separation×TX-baseline/minimum-distance. This is a feasibility bound, not a statistical error bar. Violating it disqualifies that fitted parameter state as a common hull pair; it does not prove which band association is wrong or exclude other admissible likelihood solutions.</p>
<p>Some alternative initializations fail local convergence or enter a cancelling decomposition; all trial gradients are retained in the data. In particular the lower-cost independent SBS trial in 164–166 s has a large gradient, so that window does not have a fully settled baseline. The smaller prediction percentages are exploratory diagnostics, not a certification of common scatterers or a calibrated significance test.</p>
<div style="background:#fff1d3;padding:14px"><b>Unsettled fits are failure diagnostics, not measured material-point trajectories.</b> The following selected cases have gradients above the explicit 1e−5 display warning level; their raw values remain in the table. A small gradient alone never establishes physical validity.<ul>{unsettled}</ul>Large opposing components can manufacture apparently matching conditional phase curves after subtraction. Those curves must not drive antenna calibration. The next association model needs physically consistent range support and must retain uncertainty in unresolved components.</div>
<img src="04-relative-range-and-cancellation.png" alt="Relative range mismatch and local component cancellation before and after the physical restriction">
{images}
<p>Observed coefficients follow training-frozen subtraction of other returns and the stationary profile. They are measured from the retained original bins, not replaced by the fitted phase. Adjacent 250 ms measurements at 50 ms steps are correlated. The overlapping two-second apertures also share original data and must not be multiplied as independent calibration observations.</p>
<p>The time-origin transformation is checked against every original-row complex response, moving the path constant into both envelope origin and carrier intercept. Only whole retained bins whose original rows lie inside the new temporal basis are used; boundary bins remain in the recording. The previous centre-bin attempt was rejected by this numerical gate before fitting. Brightness breakpoints and the original 125 ms train/test partition are retained.</p>
<p>A plotting-only correction gives two-second windows a nonempty interior phase-origin search. The former one-second margins could select the first sample because their intersection was empty. Six-second origins are unchanged, and no fitted likelihood changes. Missing phase connections remain missing.</p>
<p>Prior art: <a href="https://doi.org/10.1049/rsn2.12030">Zhu et al. (2021)</a> jointly treats multiband phase compensation and sparse scene reconstruction under an explicit frequency-independent point-scattering assumption; it does not validate that assumption for this ship or supply this bistatic geometry. <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4367330/">Gao et al. (2015)</a> demonstrates that independently resolving close range scatterers can create false centres and irregular apparent migration. Those results motivate joint original-data scene tests and explicit association controls here, rather than treating peak labels as material points.</p>
<p>{len(errors)} recorded likelihoods reproduce with maximum absolute error {max(errors):.3g}. <a href="analysis.json">Measured arrays, all trial diagnostics and source identities</a> · <a href="measured-trajectories.csv">CSV trajectories</a> · <a href="../piecewise-nearby-path-tests.log">Numerical tests</a> · <a href="../joint-main-return-review/index.html">Original six-second common-path control</a>.</p></html>''')
    snap=out/'sources';snap.mkdir(exist_ok=True)
    for p in paths:
        if p.suffix=='.py':shutil.copyfile(p,snap/p.name)
    print('report',out/'index.html','objectives',len(errors),'max error',max(errors),flush=True)


if __name__=='__main__':main()
