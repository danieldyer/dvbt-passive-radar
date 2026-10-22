"""Audit and plot integer-cycle search against the preceding local starts."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from carrier_cycles import cycle_constraints,joint_mapping
from common_physical_pair import CommonPhysicalPair
from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from joint_physical_pairs import JointPhysicalPairs
from pair_diagnostics import components
from propagated_pair import primary_propagation
from ship_site_signature import licensed_options


def unused(record):
    return sum(p['heldout_error'] for arc in record['prediction'] for p in arc)


def height(record):
    return max(abs(p[2]) for p in record['antenna_fit']['transmitter_offsets_enu_m'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/joint-carrier-cycles/fits.json'))
    args=parser.parse_args();data=json.loads(args.source.read_text());out=args.source.parent
    if not data['complete']:raise ValueError('Finish all declared fits before reporting')
    case=prepare_problem();labels=['F01','Flash']
    selected=[next(i for i,e in enumerate(case['descriptions']) if e['label']==label) for label in labels]
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a
        for a in assignments(licensed_options(case['geometry_directory']))}
    family=data['primary_propagation'];summary=dict(complete=True,accepted_calibration=False,groups=[],
        objective_reproduction=[],source_hash_checks=[])
    for name,digest in data['source_sha256'].items():
        path=Path(name);saved=out/'sources'/path.name
        checked=saved if path.suffix=='.py' else path
        actual=hashlib.file_digest(checked.open('rb'),'sha256').hexdigest()
        if actual!=digest:raise ValueError(f'Changed input: {checked}')
        summary['source_hash_checks'].append(str(checked))
    best={};models={};controls={}
    for group in data['groups']:
        code=group['assignment'];arcs=[model_for_assignment(case,i,lookup[code]) for i in selected]
        if family!='direct':arcs=[primary_propagation(a,data['water'],family) for a in arcs]
        independent=JointPhysicalPairs(arcs);model=JointPhysicalPairs([CommonPhysicalPair(a) for a in arcs])
        models[code]=model
        for current,rows in ((independent,[*group['independent_trials'],group['relaxed_control']]),
                             (model,[*group['common_trials'],group['previous_common_best']])):
            for row in rows:
                difference=current.objective(np.asarray(row['parameters']))[0]*current.energy-row['cost']
                if abs(difference)>1e-7:raise ValueError('Saved raw likelihood failed to reproduce')
                summary['objective_reproduction'].append(float(difference))
        chosen=min(group['common_trials'],key=lambda r:r['cost']);best[code]=chosen
        control=min([*group['independent_trials'],group['relaxed_control']],key=lambda r:r['cost']);controls[code]=control
        old=group['previous_common_best']
        origin=np.asarray(group['independent_trials'][0]['parameters'])
        mapped,_=joint_mapping(model,independent,np.asarray(chosen['parameters']))
        for index in independent.indices:
            phase=index[28:33]
            mapped[phase]=origin[phase]+np.angle(np.exp(1j*(mapped[phase]-origin[phase])))
        final_cycles=cycle_constraints(model,independent,mapped)[0]
        if abs(final_cycles-np.rint(final_cycles)).max()>1e-8:raise ValueError('Common phase closure failed')
        ratio=max(c['component_to_sum_energy_ratio'] for arc in chosen['decomposition'] for c in arc)
        linear=[r['bounded_linear_initializer'] for r in group['common_seeds'] if r.get('bounded_linear_initializer')]
        summary['groups'].append(dict(assignment=code,branch=chosen['branch_index'],cost=chosen['cost'],
            previous_cost=old['cost'],raw_objective_reduction=old['cost']-chosen['cost'],
            selected_strict=chosen['audit']['strict_local_minimum'],
            strict_trials=sum(r['audit']['strict_local_minimum'] for r in group['common_trials']),
            projection_evaluation_limit_hits=sum(r['projection_status']==0 for r in group['common_seeds']),
            projection_maximum_reported_optimality=max(r['projection_optimality'] for r in group['common_seeds']),
            bounded_quadratic_successes=sum(r['success'] for r in linear),
            bounded_quadratic_maximum_equality_error=max([r['equality_error'] for r in linear],default=0.),
            bounded_quadratic_maximum_bound_error=max([r['bound_error'] for r in linear],default=0.),
            unused_error=unused(chosen),previous_unused_error=unused(old),
            unused_error_change_percent=100*(unused(chosen)/unused(old)-1),
            independent_unused_error=unused(control),
            maximum_height_correction_m=height(chosen),previous_maximum_height_correction_m=height(old),
            maximum_horizontal_correction_m=float(np.linalg.norm(np.asarray(chosen['antenna_fit']['transmitter_offsets_enu_m'])[:,:2],axis=1).max()),
            bound_indices=chosen['bound_indices'],
            antenna_fit=chosen['antenna_fit'],maximum_component_to_sum_energy_ratio=ratio,
            antenna_data_information_eigenvalues=chosen['information']['eigenvalues'],
            antenna_modes_above_unit_prior_precision=sum(x>1 for x in chosen['information']['eigenvalues']),
            initial_cycles=chosen['carrier_cycles'],
            final_cycles_with_midpoint_phases_wrapped_nearest_independent_origin=np.rint(final_cycles).astype(int).tolist()))

    fig,axes=plt.subplots(len(data['groups']),2,figsize=(15,5*len(data['groups'])),squeeze=False,layout='constrained')
    for row,group in enumerate(data['groups']):
        code=group['assignment'];baseline=group['independent_trials'][0]['cost'];old=group['previous_common_best']
        records=group['common_seeds'];ranks=np.array([r['branch_index'] for r in records])
        axes[row,0].plot(ranks,[r['projected_surrogate_cost']-baseline for r in records],'.',label='Bounded geometry projection / local quadratic',color='#a0a4a7')
        axes[row,0].plot(ranks,[r['initial_training_cost']-baseline for r in records],'.',label='Same seed / original complex data',color='#4286b5')
        axes[row,0].axhline(old['cost']-baseline,color='#bf7733',ls='--',label='Previous refined common-phase fit')
        for r in group['common_trials']:
            marker='o' if r['audit']['strict_local_minimum'] else 'x'
            axes[row,0].plot(r['branch_index'],r['cost']-baseline,marker,color='#202020',ms=7)
        axes[row,0].set(title=code+' · all enumerated seeds',xlabel='Rank in unconstrained integer metric',ylabel='Objective above independent-phase origin')
        axes[row,0].set_yscale('symlog',linthresh=10)
        axes[row,0].legend(fontsize=8)
        for r in group['common_trials']:
            marker='o' if r['audit']['strict_local_minimum'] else 'x'
            axes[row,1].plot(r['cost']-baseline,100*(unused(r)/unused(old)-1),marker,
                color='#24699b' if r is best[code] else '#999999',ms=8)
            axes[row,1].annotate(str(r['branch_index']),(r['cost']-baseline,100*(unused(r)/unused(old)-1)),xytext=(5,5),textcoords='offset points',fontsize=8)
        axes[row,1].plot(old['cost']-baseline,0,'s',color='#bf7733',label='Previous selected fit')
        axes[row,1].axhline(0,color='#bbb',lw=.7)
        axes[row,1].set(title=code+' · refined original-data fits',xlabel='Training objective above independent-phase origin',ylabel='Unused error change from previous common fit (%)')
        axes[row,1].legend(fontsize=8)
    fig.suptitle('Carrier-cycle alternatives: training chooses the starts; unused coefficients stay frozen\n● strict local minimum · × numerical audit incomplete · blue lowest training cost',fontsize=12)
    fig.savefig(out/'01-cycle-search.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(len(data['groups']),3,figsize=(15,4.6*len(data['groups'])),squeeze=False,layout='constrained')
    for row,group in enumerate(data['groups']):
        code=group['assignment'];chosen=best[code]
        for r in group['common_trials']:
            offsets=np.asarray(r['antenna_fit']['transmitter_offsets_enu_m'])
            for axis in range(3):axes[row,axis].plot(range(5),offsets[:,axis],'-',color='#ccc',lw=.8)
        for r,label,colour,style in ((group['previous_common_best'],'Previous local search','#bf7733','--'),
                (controls[code],'Independent phase control','#777777',':'),
                (chosen,'Best new training fit','#24699b','-')):
            offsets=np.asarray(r['antenna_fit']['transmitter_offsets_enu_m'])
            for axis in range(3):axes[row,axis].plot(range(5),offsets[:,axis],'o'+style,color=colour,label=label)
        for axis in range(3):
            axes[row,axis].axhline(0,color='#bbb',lw=.7)
            status='' if chosen['audit']['strict_local_minimum'] else ' · audit incomplete'
            axes[row,axis].set(xticks=range(5),xticklabels=NAMES,ylabel='Correction to supplied prior (m)',title=code+' · '+('East','North','Up')[axis]+status)
            axes[row,axis].tick_params(axis='x',labelrotation=30)
        axes[row,0].legend(fontsize=8)
    fig.suptitle('Antenna corrections after broader carrier-cycle search\nAll new refined starts retained; height is not an independently surveyed quantity',fontsize=12)
    fig.savefig(out/'02-cycle-antennas.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,len(data['groups']),figsize=(14,5.5),squeeze=False,layout='constrained')
    for ax,group in zip(axes[0],data['groups']):
        code=group['assignment']
        for r,label,style in ((group['previous_common_best'],'Previous common phase','--'),
                (best[code],'New common phase','-'),(controls[code],'Independent phases',':')):
            values=np.array(r['information']['eigenvalues'])
            ax.semilogy(range(1,17),np.maximum(values,1e-14),'o'+style,label=label)
        ax.axhline(1,color='#888',lw=.8)
        status='' if best[code]['audit']['strict_local_minimum'] else ' · audit incomplete'
        ax.set(title=code+status,xlabel='Antenna information mode',ylabel='Data-only information / standardized prior coordinates',xticks=range(1,17,2))
        ax.legend(fontsize=8)
    fig.suptitle('Geometry information after removing primary and secondary scene nuisance\nPriors and active bounds excluded; conditional thermal metric, not a calibrated confidence interval',fontsize=12)
    fig.savefig(out/'03-cycle-information.png',dpi=170);plt.close(fig)

    # Project each observed range row onto the complete fitted two-echo row.
    # Static coefficients and echo brightness are fitted on training rows only.
    # This is a visualization diagnostic, never a correction or selection rule.
    code=data['assignments'][0];model=models[code];parameters=np.asarray(best[code]['parameters'])
    fig,axes=plt.subplots(5,2,figsize=(14,13),sharey=True,layout='constrained')
    for column,(arc,index,i) in enumerate(zip(model.arcs,model.indices,selected)):
        mapped,_=arc.mapping(parameters[index])
        for row,(pair,pindex) in enumerate(zip(arc.independent.pairs,arc.independent.indices)):
            b=pair.base;prediction=components(pair,mapped[pindex]).sum(axis=0);y=b.field.y.ravel()
            beta=b.nuisance.solve(y[b.selected]-prediction[b.selected])
            clean=(y-b.design@beta).reshape(b.field.y.shape);prediction=prediction.reshape(b.field.y.shape)
            product=np.sum(prediction.conj()*clean,axis=1)
            phase=np.degrees(np.angle(product));test=case['testing'][i][row]
            ax=axes[row,column]
            ax.plot(b.field.time[~test],phase[~test],'.',color='#aaa',ms=3,label='Training blocks')
            ax.plot(b.field.time[test],phase[test],'o',mfc='none',mec='#24699b',ms=4,label='Unused blocks')
            ax.axhline(0,color='#555',lw=.7);ax.set(ylim=(-180,180),yticks=[-180,-90,0,90,180],title=labels[column]+' · '+NAMES[row])
            if column==0:ax.set_ylabel('Observed − fitted phase (degrees)')
            if row==4:ax.set_xlabel('Capture time (s)')
            if row==0:ax.legend(fontsize=8)
    fig.suptitle(code+' · phase of the observed row projected onto the complete two-echo prediction\nFrozen training coefficients on unused blocks; this does not isolate the primary scatterer',fontsize=12)
    fig.savefig(out/'04-phase-prediction.png',dpi=170);plt.close(fig)

    rows=[]
    for group,s in zip(data['groups'],summary['groups']):
        rows.append(f"<tr><td>{s['assignment']}</td><td>{group['lattice']['visited_nodes']}</td><td>{s['strict_trials']}/{len(group['common_trials'])}</td><td>{s['previous_cost']:.3f} → {s['cost']:.3f}</td><td>{s['unused_error_change_percent']:+.3f}%</td><td>{s['previous_maximum_height_correction_m']:.2f} → {s['maximum_height_correction_m']:.2f} m</td><td>{s['maximum_horizontal_correction_m']:.2f} m</td><td>{s['maximum_component_to_sum_energy_ratio']:.3g}</td></tr>")
    selected_status='; '.join(f"{s['assignment']}: {'passes' if s['selected_strict'] else 'fails'} strict audit" for s in summary['groups'])
    initialization='The linear seed is clipped before nonlinear projection.'
    if data.get('bounded_initializer'):
        initialization='This control first solves the bounded equality-constrained quadratic in whitened coordinates, before the nonlinear projection. Its solver status and exact feasibility errors are retained for every candidate. It starts from the best independent-phase control available in its source report; that origin can differ from an earlier experiment.'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Broader carrier-cycle search</title>
<style>body{{font:16px/1.55 system-ui;max-width:1250px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:10px;border:1px solid #ddd;text-align:left}}code{{font-size:13px}}</style>
<h1>Broader carrier-cycle search</h1>
<p>No calibration is accepted from this experiment. It tests whether the preceding implausible height corrections were a consequence of considering only nearby phase-origin branches. {html.escape(selected_status)}.</p>
<table><tr><th>Assignment</th><th>Integer search nodes</th><th>Strict refined fits</th><th>Old → new training objective</th><th>Unused error change</th><th>Largest |TX up correction|</th><th>Largest TX horizontal correction</th><th>Largest component/sum energy</th></tr>{''.join(rows)}</table>
<p>Code order: Seven, SBS, Nine, Ten, ABC. 0 = Ornata, 1 = Eyre, 2 = Ridge. All antenna priors, receiver horizontal position, height bounds, original observations, training masks and unused-sample prediction rules are unchanged. This is previously inspected data, not blind external validation.</p>
<h2>What was searched</h2><p>The independent-phase joint fit supplies the full residual Jacobian, including motion, both echoes and the original priors. QR factorization gives a local Gauss–Newton quadratic. Eliminating its continuous coordinates gives an eight-dimensional integer metric: four relative carrier phases in each of two intervals. Integer basis reduction and shrinking-sphere enumeration find its {data['integer_candidates']} nearest lattice points. All are projected through the exact nonlinear geometric coordinate map and original bounds. Actual training likelihood ranks these seeds; the best {data['common_starts']} per assignment receive full complex-data refinement and the existing strict local-minimum audit.</p><p>{initialization}</p>
<details><summary>Reduction and its precise scope</summary><p>Let r and J be the unnormalized profiled residual and its exact Jacobian at the independent-phase fit. The residual includes the original priors once. For each interval, f contains the four physical primary phases relative to Ten, divided by 2π; f must be integer for a common phase. Let A be its geometric/phase Jacobian, and μ the unconstrained least-squares step minimizing ‖r + Jδ‖². With J = QR, define B = AR⁻¹, S = BBᵀ, and a = f + Aμ.</p><pre>integer cost:        (n − a)ᵀ S⁻¹ (n − a)
continuous step:    μ + R⁻¹ Bᵀ S⁻¹ (n − a)</pre><p>These identities include the correlations with all continuously fitted parameters. They are verified against the full constrained quadratic in the tests. S is an optimization metric, not an estimated confidence matrix. The original bound constraints and nonlinear geometry enter the subsequent projection; neither the local quadratic nor its integer ranking is assumed accurate far from its origin. Original complex-data fitting and unused prediction remain the final comparisons.</p></details>
<p>This reduction/search formulation follows the integer least-squares approach described by <a href="https://gnss.curtin.edu.au/wp-content/uploads/sites/21/2016/04/Teunissen1993Least.pdf">Teunissen</a> and <a href="https://www.cs.mcgill.ca/~chang/pub/ChangYangZhou.pdf">Chang, Yang and Zhou</a>. The implementation uses LLL basis reduction and sphere enumeration; it is not an implementation of MLAMBDA. The enumeration is complete for the supplied local quadratic, subject to floating-point arithmetic. It is not a proof of the global nonlinear optimum, nor an exhaustive test of sites or scattering models. Bounds are enforced in projection and raw refinement, not in the initial integer metric. The nonlinear surrogate projection is an initializer with a 150-evaluation limit; some projections stop without reaching stationarity. Every projection status is retained. Only the subsequent raw-data fit receives the strict numerical audit; no initializer is certified as a bounded global optimum.</p>
<img src="01-cycle-search.png" alt="All cycle candidates and prediction comparisons">
<img src="02-cycle-antennas.png" alt="New and previous antenna corrections">
<img src="03-cycle-information.png" alt="Conditional antenna information">
<img src="04-phase-prediction.png" alt="Original complex row phase relative to complete frozen two-echo prediction">
<h2>Checks and limitations</h2><p>The numerical tests compare the integer solver with brute force, check invariance under integer shifts and unimodular coordinate changes, recover known synthetic carrier cycles, and verify the constrained quadratic against the full residual. Saved original-data objectives and source hashes are rechecked when this report is built. A smaller height correction alone is not sufficient: horizontal displacement, remaining bound contacts and data-only information must also support the location. All bound indices and information spectra are retained in the numerical results. The phase constraints test an equal physical primary scattering phase across bands; they do not establish that a particular unresolved ship or sea-assisted return obeys that model. Nearly cancelling two-echo components remain a model-identifiability failure even if their sum fits well.</p>
<p>The final unwrapped cycle labels in <a href="analysis.json">analysis.json</a> use midpoint band phases wrapped nearest the independent origin; their absolute integers depend on that explicit gauge. No phase correction is applied to the observed samples. <a href="fits.json">Every candidate and fit</a> · <a href="tests.log">Numerical tests</a> · <a href="../index.html">Analysis index</a>.</p></html>'''
    (out/'index.html').write_text(page)
    (out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(dict(groups=summary['groups'],objectives_verified=len(summary['objective_reproduction']),
        source_hashes_verified=len(summary['source_hash_checks'])),indent=2))


if __name__=='__main__':main()
