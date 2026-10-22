"""Report shared-antenna paired-echo fits, including physical and numerical limits."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common_physical_pair import CommonPhysicalPair
from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments,observables
from joint_physical_pairs import JointPhysicalPairs
from propagated_pair import primary_propagation
from ship_site_signature import licensed_options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/joint-ship-primary-pairs/fits.json'))
    args=parser.parse_args();out=args.source.parent;data=json.loads(args.source.read_text())
    if not data['complete']:raise ValueError('Finish every declared trial')
    case=prepare_problem();selected=[next(i for i,d in enumerate(case['descriptions']) if d['label']==label) for label in ('F01','Flash')]
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a
        for a in assignments(licensed_options(case['geometry_directory']))}
    source_path=out.parent/'ship-common-primary-pair/fits.json';separate=json.loads(source_path.read_text())
    error=lambda r:sum(p['heldout_error'] for arc in r['prediction'] for p in arc)
    family=data.get('primary_propagation','direct')
    summary=dict(scope=__doc__,accepted_calibration=False,primary_propagation=family,groups=[],objective_reproduction=[])
    models={};best={}
    for group in data['groups']:
        code=group['assignment'];arcs=[model_for_assignment(case,i,lookup[code]) for i in selected]
        if family!='direct':arcs=[primary_propagation(a,data['water'],family) for a in arcs]
        independent=JointPhysicalPairs(arcs);common=JointPhysicalPairs([CommonPhysicalPair(a) for a in arcs]);models[code]=common
        for model,rows in ((independent,[*group['independent_trials'],group['relaxed_control']]),(common,group['common_trials'])):
            for row in rows:
                difference=model.objective(np.asarray(row['parameters']))[0]*model.energy-row['cost']
                if abs(difference)>1e-7:raise ValueError('Saved joint objective did not reproduce')
                summary['objective_reproduction'].append(difference)
        control=min([*group['independent_trials'],group['relaxed_control']],key=lambda r:r['cost'])
        chosen=min(group['common_trials'],key=lambda r:r['cost']);best[code]=chosen,control
        summary['groups'].append(dict(assignment=code,common_branch=chosen['branch_index'],
            independent_cost=control['cost'],common_cost=chosen['cost'],
            training_penalty_percent=100*(chosen['cost']/control['cost']-1),
            independent_unused_error=error(control),common_unused_error=error(chosen),
            unused_error_change_percent=100*(error(chosen)/error(control)-1),
            common_strict=chosen['audit']['strict_local_minimum'],independent_strict=control['audit']['strict_local_minimum'],
            strict_common_trials=sum(r['audit']['strict_local_minimum'] for r in group['common_trials']),
            common_bounds=chosen['bound_indices'],
            maximum_component_to_sum_ratio=max(c['component_to_sum_energy_ratio'] for arc in chosen['decomposition'] for c in arc),
            antenna_fit=chosen['antenna_fit'],antenna_information_eigenvalues=chosen['information']['eigenvalues'],
            data_modes_above_unit_prior_precision=sum(v>1 for v in chosen['information']['eigenvalues'])))
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    for i,group in enumerate(data['groups']):
        code=group['assignment'];chosen,control=best[code]
        for j,row in enumerate(group['common_trials']):
            x=i+(j-(len(group['common_trials'])-1)/2)*.09;colour='#24699b' if row is chosen else '#9aa4ab';marker='o' if row['audit']['strict_local_minimum'] else 'x'
            axes[0].plot(x,100*(row['cost']/control['cost']-1),marker,color=colour)
            axes[1].plot(x,100*(error(row)/error(control)-1),marker,color=colour)
            axes[2].plot(x,max(c['component_to_sum_energy_ratio'] for arc in row['decomposition'] for c in arc),marker,color=colour)
    for ax in axes:ax.set_xticks(range(len(data['groups'])),[g['assignment'] for g in data['groups']])
    axes[0].set(ylabel='Training objective increase (%)',title='Common physical primary phase')
    axes[1].set(ylabel='Unused-sample error change (%)',title='Independent constant band phases as control');axes[1].axhline(0,color='#aaa',lw=.7)
    axes[2].set(yscale='log',ylabel='Largest component/sum energy ratio',title='Cancellation diagnostics');axes[2].axhline(1,color='#aaa',lw=.7)
    largest=max(c['component_to_sum_energy_ratio'] for g in data['groups'] for r in g['common_trials'] for arc in r['decomposition'] for c in arc)
    axes[2].set_ylim(.8,max(2.,largest*1.25))
    fig.suptitle('One set of antenna coordinates shared by F01 and Flash\nBlue: lowest training cost · ×: incomplete stationarity · all declared starts shown',fontsize=12)
    fig.savefig(out/'01-joint-prediction.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(len(data['groups']),3,figsize=(15,9),squeeze=False,layout='constrained')
    for row,group in enumerate(data['groups']):
        code=group['assignment'];chosen,control=best[code]
        status='' if chosen['audit']['strict_local_minimum'] else ' · incomplete fit'
        for record,name,colour in ((control,'Independent band phases','#bd6b22'),(chosen,'Common physical phase','#24699b')):
            tx=np.asarray(record['antenna_fit']['transmitter_offsets_enu_m'])
            for j,ax in enumerate(axes[row]):ax.plot(NAMES,tx[:,j],'o-',color=colour,label=name)
        for record in [g for g in separate['groups'] if g['assignment']==code]:
            r=min(record['common_trials'],key=lambda t:t['cost']);tx=np.asarray(r['antenna_fit']['transmitter_offsets_enu_m'])
            for j,ax in enumerate(axes[row]):ax.plot(NAMES,tx[:,j],':',color='#a2a8ad',label=record['interval']+' separately / direct')
        for j,ax in enumerate(axes[row]):
            ax.set(title=code+' · '+('East','North','Up')[j]+status,ylabel='Correction to supplied prior (m)');ax.tick_params(axis='x',labelrotation=30);ax.axhline(0,color='#bbb',lw=.6)
        axes[row,0].legend(fontsize=8)
    fig.suptitle('Waveform compatibility must not be mistaken for a credible antenna survey\nBlue/orange: jointly fitted · grey: separate interval estimates',fontsize=12)
    fig.savefig(out/'02-joint-antennas.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,len(data['groups']),figsize=(13,5),squeeze=False,layout='constrained')
    for ax,group in zip(axes[0],data['groups']):
        chosen,control=best[group['assignment']]
        for record,name in ((control,'Independent band phases'),(chosen,'Common physical phase')):
            values=np.asarray(record['information']['eigenvalues']);ax.semilogy(range(1,17),np.maximum(values,1e-14),'o-',label=name)
        ax.axhline(1,color='#888',ls='--',label='Unit standardized prior precision')
        status='' if chosen['audit']['strict_local_minimum'] else ' · incomplete, cancelling fit'
        ax.set(title=group['assignment']+status,xlabel='Antenna mode (largest first)',ylabel='Data-only conditional information eigenvalue',xticks=range(1,17,2));ax.legend(fontsize=8)
    fig.suptitle('Remove all primary motion, secondary motion/phase/brightness and stationary nuisance\nPriors and active bounds are excluded; thermal local information is not a calibrated confidence interval',fontsize=12)
    fig.savefig(out/'03-joint-information.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    motion_code=data['assignments'][0];chosen,_=best[motion_code];model=models[motion_code];parameters=np.asarray(chosen['parameters'])
    for row,(arc,index,i) in enumerate(zip(model.arcs,model.indices,selected)):
        g=arc.geometry;entry=case['descriptions'][i];midpoint=(entry['first_s']+entry['last_s'])/2
        p,v,a,tx,rx=g.state(parameters[index][:28]);time=np.linspace(entry['first_s'],entry['last_s'],101);dt=time-midpoint
        values=np.array([observables(p+t*v+.5*t*t*a,v+t*a,a,tx,rx) for t in dt])
        for ch,name in enumerate(NAMES):
            axes[row,0].plot(time,values[:,ch,1],label=name)
            axes[row,1].plot(time,values[:,ch,2],label=name)
        axes[row,0].set(ylabel='Excess-path rate (m/s)',title=entry['label']);axes[row,0].legend(fontsize=8)
        axes[row,1].set(ylabel='Excess-path curvature (m/s²)',title=entry['label'])
    for ax in axes[-1]:ax.set_xlabel('Capture time (s)')
    status='' if chosen['audit']['strict_local_minimum'] else ' · incomplete numerical fit'
    fig.suptitle(motion_code+status+' · short-interval apparent acceleration\nSeparate apparent reflectors are allowed; this is not yet a continuous rigid-vessel motion estimate',fontsize=12)
    fig.savefig(out/'04-primary-motion.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{g["assignment"]}</td><td>{g["common_cost"]:.2f}</td><td>{g["training_penalty_percent"]:+.3f}%</td><td>{g["common_unused_error"]:.2f}</td><td>{g["unused_error_change_percent"]:+.3f}%</td><td>{g["common_strict"]}</td><td>{g["data_modes_above_unit_prior_precision"]}/16</td></tr>' for g in summary['groups'])
    max_up={g['assignment']:float(np.max(abs(np.asarray(g['antenna_fit']['transmitter_offsets_enu_m'])[:,2]))) for g in summary['groups']}
    summary['maximum_transmitter_vertical_correction_m']=max_up
    total=sum(len(g['common_trials']) for g in data['groups']);strict=sum(g['strict_common_trials'] for g in summary['groups'])
    summary.update(common_trials=total,strict_common_trials=strict)
    height_text='; '.join(f'{code}: {height:.1f} m' for code,height in max_up.items())
    propagation_text=('Direct physical point.' if family=='direct' else
        'Direct point plus the declared '+('transmit-side sea reflection.' if family=='transmit' else 'three sea-assisted rays.')+
        ' Smooth spherical water, HH Fresnel coefficient with relative permittivity 80 and conductivity 4 S/m; no fitted reflected-ray phase or strength.')
    comparison='';comparison_paths=[]
    if family!='direct':
        path=out.parent/'joint-ship-primary-pairs/fits.json';direct=json.loads(path.read_text())
        if not direct['complete']:raise ValueError('Complete the direct shared-antenna control before comparing sea propagation')
        comparison_paths.append(path);summary['direct_comparison']=[]
        fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
        labels=[];training=[];unused=[];vertical=[]
        for code in data['assignments']:
            group=next(g for g in direct['groups'] if g['assignment']==code)
            control=min(group['common_trials'],key=lambda r:r['cost']);sea=best[code][0]
            summary['direct_comparison'].append(dict(assignment=code,direct_cost=control['cost'],sea_cost=sea['cost'],
                direct_unused_error=error(control),sea_unused_error=error(sea),
                unused_error_reduction_percent=100*(1-error(sea)/error(control)),
                direct_strict=control['audit']['strict_local_minimum'],sea_strict=sea['audit']['strict_local_minimum']))
            for name,r in (('Direct',control),('HH '+family,sea)):
                status='' if r['audit']['strict_local_minimum'] else '\nIncomplete stationarity'
                labels.append(code+'\n'+name+status);training.append(r['cost']);unused.append(error(r))
                vertical.append(np.max(abs(np.asarray(r['antenna_fit']['transmitter_offsets_enu_m'])[:,2])))
        for ax,values,title,ylabel in zip(axes,(training,unused,vertical),
            ('Training-selected common-phase fits','Unused complex samples','Largest transmitter height correction'),
            ('Squared objective','Squared prediction error','Absolute correction (m)')):
            ax.bar(labels,values,color=['#24699b','#287f5c']*len(data['assignments']));ax.set(title=title,ylabel=ylabel)
        fig.suptitle('Same observations, bounds, parameter counts and shared antenna architecture\nDifferent declared local starts; this is not a comparison of certified global optima',fontsize=12)
        fig.savefig(out/'05-direct-versus-sea.png',dpi=170);plt.close(fig)
        comparison='<h2>Controlled propagation comparison</h2><img src="05-direct-versus-sea.png" alt="Direct and physically tied HH sea-assisted primary comparison"><p>The sea model adds no fitted ray phase, amplitude or trajectory. Both models retain the independent secondary echo. This tests one declared smooth-water point-scattering hypothesis, not every possible ship-side or rough-water mechanism. Search histories differ and all convergence and plausibility checks still apply.</p>'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Joint two-interval ship geometry</title>
<style>body{{font:16px/1.55 system-ui;max-width:1260px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd4d9;padding:8px}}.note{{background:#fff1d3;padding:14px}}</style>
<h1>One antenna solution across two overlapping-echo intervals</h1>
<p>Primary propagation: {propagation_text}</p>
<p class="note">These tests share all transmitter and receiver corrections between F01 (159–162 s) and Flash (163–166 s). They do not yet justify adopting a calibration. The common-phase model can predict the observations, but physical plausibility, remaining carrier alternatives, parameter bounds and weak antenna modes still matter. A good predicted sum alone is insufficient to identify its constituent scatterers.</p>
<p>00002 means Seven/SBS/Nine/Ten at Ornata and ABC at Ridge; 10002 moves Seven to Eyre. Tested here: {', '.join(data['assignments'])}. Each interval has its own physical primary position, velocity, acceleration and scattering phase, and an explicit independent secondary echo in each band. Brightness and stationary coefficients use the unchanged original-row likelihood. Receiver latitude/longitude remain fixed at the user's supplied point. The prior is counted once for shared antennas. {data['common_starts']} starts are fitted from 36 local carrier initializers per site hypothesis, ranked by the actual training objective. This does not exhaust every carrier ambiguity or all 32 site hypotheses.</p>
<img src="01-joint-prediction.png" alt="Joint common phase cost, unused prediction and cancellation diagnostics">
<table><tr><th>Assignment</th><th>Common training cost</th><th>Phase constraint penalty</th><th>Unused error</th><th>Unused change</th><th>Selected strict minimum</th><th>Data modes above unit prior</th></tr>{rows}</table>
<p>The independent-phase model uses the same shared antenna coordinates, local motion and secondary structure but five constant primary phases per interval. Two starts and a relaxation from the best common-phase solution guard against comparing different local basins. No moving or stationary coefficients are refitted on unused samples. {strict} of {total} common starts pass the actual-objective KKT and tangent-curvature checks; all failures remain visible. Releasing the phase restriction reproduces the nested objective before optimization.</p>
<img src="02-joint-antennas.png" alt="Joint antenna corrections contrasted with separately estimated intervals">
<p>Largest absolute transmitter-height corrections: {height_text}. These are diagnostics of model/prior tension, not updated antenna heights. The supplied siting metadata is unchanged. A prior-bound solution can be a valid constrained numerical minimum without being a plausible physical calibration.</p>
<img src="03-joint-information.png" alt="Conditional antenna information modes with and without common physical primary phase">
<p>Information is computed from the physical mean derivative after removing stationary and active real brightness coefficients, all secondary coordinates, primary phase(s), and nine Cartesian target coordinates per interval. No geographical priors or active-bound constraints are counted as observations. The exact geographical rotation gauge has independent numerical tests. A large leading eigenvalue cannot replace the missing weak directions; these local thermal quantities also omit model mismatch and carrier-cycle uncertainty.</p>
<img src="04-primary-motion.png" alt="Apparent primary excess-path rate and curvature across both intervals">
<p>The short-interval trajectories are not assumed to be the same material point or a validated vessel translation. Their different apparent curvatures are an additional reason to test physical ship–sea propagation and stable alternative scatterers before extending a trajectory through the full capture. No arbitrary time-dependent inter-band phase correction, component-amplitude cap or minimum separation has been introduced.</p>
{comparison}
<p><a href="fits.json">All joint fits and declared initializers</a> · <a href="analysis.json">Derived comparisons and provenance</a> · <a href="../ship-common-primary-pair/index.html">Separate-interval common-phase control and tests</a> · <a href="../ship-amplitude-pair-F03/index.html">Third-interval decomposition check</a>.</p></html>'''
    (out/'index.html').write_text(page)
    paths=[args.source,source_path,Path(__file__),*comparison_paths]
    summary['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    (out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',out/'index.html','strict common',strict,'of',total,flush=True)


if __name__=='__main__':main()
