"""Review every licensed assignment without turning a local minimum into a survey."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from ship_site_signature import licensed_options


def antenna_information(model,parameters):
    rows=[]
    for pair,index in zip(model.pairs,model.indices):
        local=pair.evaluate(parameters[index])['mean_derivative']*np.sqrt(2.)
        j=np.zeros((len(local),model.count));j[:,index]=local;rows.append(j)
    j=np.concatenate(rows);ranks=[]
    def remove(j,columns):
        norms=np.linalg.norm(columns,axis=0);good=norms>0
        if not np.any(good):return j,0
        a=columns[:,good]/norms[good]
        u,s,_=np.linalg.svd(a,full_matrices=False);keep=s>np.finfo(float).eps*max(a.shape)*s[0]
        q=u[:,keep];return j-q@(q.T@j),int(sum(keep))
    # Eliminate constant primary phases, all secondary coordinates, then the
    # nine independent Cartesian target coordinates. AIS time is redundant
    # with p/v/a at one epoch. Priors and active bounds supply no data here.
    for columns in (np.arange(28,33),np.arange(33,model.count),np.arange(18,27)):
        j,rank=remove(j,j[:,columns]);ranks.append(rank)
    antenna=j[:,np.r_[np.arange(15),17]]
    singular=np.linalg.svd(antenna,compute_uv=False)
    return dict(eigenvalues=(singular**2).tolist(),nuisance_ranks=ranks,
        eigenvalues_above_unit_prior=int(sum(singular**2>1)),
        scope='Conditional thermal information in standardized antenna-prior coordinates after all scene nuisance. Data only; no active-bound or prior information. One is unit prior precision, not a calibrated significance threshold.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/ship-physical-pair-sites-32/fits.json'))
    args=parser.parse_args();out=args.source.parent;source=json.loads(args.source.read_text())
    if not source['complete'] or len(source['trials'])!=64:raise ValueError('Finish all 64 declared fits')
    codes=sorted({r['assignment'] for r in source['trials']})
    if len(codes)!=32 or any(sum(r['assignment']==c for r in source['trials'])!=2 for c in codes):raise ValueError('Incomplete assignment comparison')
    best=[min([r for r in source['trials'] if r['assignment']==c],key=lambda r:r['cost']) for c in codes]
    ranked=sorted(best,key=lambda r:r['cost'])
    heldout=lambda row:sum(r['heldout_error'] for r in row['prediction'])
    prediction_order=sorted(best,key=heldout)
    test_log=out.parent/'ship-physical-pair-F01/tests.log';test_text=test_log.read_text()
    test_count=re.search(r'Ran (\d+) tests',test_text)
    if test_count is None or not test_text.rstrip().endswith('OK'):raise ValueError('Passing numerical tests required')
    summary=dict(scope=__doc__,accepted_calibration=False,complete_assignments=32,complete_trials=64,tests_passed=int(test_count[1]),
        strict_local_minima=sum(r['audit']['strict_local_minimum'] for r in source['trials']),
        lowest_training_assignment=ranked[0]['assignment'],lowest_heldout_assignment=prediction_order[0]['assignment'],
        selected_by_training=[dict(assignment=r['assignment'],association=r['initial_association'],cost=r['cost'],
            heldout_error=heldout(r),strict=r['audit']['strict_local_minimum'],bounds=r['bound_indices']) for r in ranked],
        data_information=[],objective_reproduction=[])
    fig,axes=plt.subplots(1,3,figsize=(17,12),layout='constrained')
    for index,best_row in enumerate(ranked):
        for row in [r for r in source['trials'] if r['assignment']==best_row['assignment']]:
            colour='#24699b' if row is best_row else '#a6adb3';marker='o' if row['audit']['strict_local_minimum'] else 'x'
            axes[0].plot(row['cost'],index,marker,color=colour,ms=5)
            axes[1].plot(heldout(row),index,marker,color=colour,ms=5)
        p=best_row['target_state']['position_correction_enu_m'];axes[2].plot(np.hypot(*p[:2]),index,'o',color='#bd6b22',ms=5)
    for ax in axes:ax.set_yticks(range(32));ax.invert_yaxis()
    axes[0].set_yticklabels([r['assignment'] for r in ranked]);axes[1].set_yticklabels([]);axes[2].set_yticklabels([])
    axes[0].set_xscale('log');axes[1].set_xscale('log')
    axes[0].set(xlabel='Training squared error + unchanged priors',title='Both declared associations retained')
    axes[1].set(xlabel='Unused-data squared error',title='No coefficients refitted on unused blocks')
    axes[2].set(xlabel='Horizontal displacement from shifted AIS (m)',title='Training-selected physical point')
    fig.suptitle('All 32 licensed assignments · five-band order Seven / SBS / Nine / Ten / ABC\n0 Ornata · 1 Eyre · 2 Ridge · blue selected by training · grey alternate · × incomplete stationarity',fontsize=12)
    fig.savefig(out/'01-all-sites.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
    colours=plt.cm.tab10(np.arange(min(8,len(ranked))))
    for row,colour in zip(ranked[:8],colours):
        p=np.asarray(row['target_state']['position_correction_enu_m']);v=np.asarray(row['target_state']['velocity_correction_enu_mps'])
        axes[0].plot(p[0],p[1],'o',color=colour,label=row['assignment']);axes[1].plot(v[0],v[1],'o',color=colour)
        axes[0].annotate(row['assignment'],p[:2],xytext=(4,4),textcoords='offset points',fontsize=8)
    for ax in axes:ax.axhline(0,color='#aaa',lw=.6);ax.axvline(0,color='#aaa',lw=.6);ax.set_aspect('equal',adjustable='datalim')
    axes[0].set(xlabel='East correction to shifted AIS (m)',ylabel='North correction (m)',title='Eight lowest-training-cost assignments')
    axes[1].set(xlabel='East velocity correction (m/s)',ylabel='North velocity correction (m/s)',title='Corresponding velocity corrections')
    axes[0].legend(fontsize=8);fig.savefig(out/'02-location-ambiguity.png',dpi=170);plt.close(fig)
    case=prepare_problem();i=next(i for i,e in enumerate(case['descriptions']) if e['label']=='F01')
    choices=assignments(licensed_options(case['geometry_directory']))
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a for a in choices}
    inspect=list(dict.fromkeys([ranked[0]['assignment'],prediction_order[0]['assignment'],'00002','02002']))
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    for code in inspect:
        row=next(r for r in best if r['assignment']==code);model=model_for_assignment(case,i,lookup[code]);p=np.asarray(row['parameters'])
        cost=model.objective(p)[0]*model.energy
        if abs(cost-row['cost'])>1e-7:raise ValueError('Assignment objective did not reproduce')
        summary['objective_reproduction'].append(dict(assignment=code,cost_difference=cost-row['cost']))
        information=antenna_information(model,p);summary['data_information'].append(dict(assignment=code,**information))
        ax.semilogy(range(1,17),np.maximum(information['eigenvalues'],1e-18),'o-',label=code)
    ax.axhline(1,color='#bd6b22',ls='--',label='Unit prior precision (reference only)')
    ax.set(xlabel='Antenna information mode, strongest first',ylabel='Conditional data-information eigenvalue',xticks=range(1,17),
        title='Information remaining after primary motion/phases and all secondary-echo nuisance')
    ax.legend();fig.savefig(out/'03-data-information.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{r["assignment"]}</td><td>{r["initial_association"]}</td><td>{r["cost"]:.2f}</td><td>{heldout(r):.2f}</td><td>{r["audit"]["strict_local_minimum"]}</td><td>{r["bound_indices"]}</td></tr>' for r in ranked)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>All licensed sites with overlapping ship returns</title>
<style>body{{font:16px/1.55 system-ui;max-width:1250px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px}}.note{{background:#fff1d3;padding:16px}}</style>
<h1>All 32 licensed assignments under the improved overlapping-echo model</h1>
<p class="note">All 64 declared fits have finished; {summary['strict_local_minima']} pass the strict local gradient/curvature audit. The smallest training objective belongs to {summary['lowest_training_assignment']}; the smallest unused-data error among training-selected solutions belongs to {summary['lowest_heldout_assignment']}. No site assignment or coordinate refinement is accepted from this single interval. Boundary contacts, competing associations and conditional data information remain explicit.</p>
<p>This repeats the <a href="../ship-physical-pair-review/index.html">F01 common-primary experiment</a> for every permitted combination, with two distinct primary/secondary associations per combination. Seven/Nine/Ten may use Ornata or Eyre; SBS/ABC may use Ridge or Ornata. The five-digit code is ordered Seven, SBS, Nine, Ten, ABC, with 0=Ornata, 1=Eyre, 2=Ridge. All physical priors, parameter bounds, original samples, positive brightness knots, secondary range support and training/test blocks are unchanged.</p>
<img src="01-all-sites.png" alt="Both association starts, training and prediction errors for every assignment">
<p>Circles certify a local constrained minimum of the tested model; they do not certify a global solution. Crosses retain incomplete numerical searches. A very poor but converged solution is not evidence that its assignment has been globally excluded. The blue fit is selected by training objective; unused-data errors are a diagnostic and do not select the start. No likelihood-ratio confidence or Bayes factor is inferred from these misspecified-scene residuals.</p>
<img src="02-location-ambiguity.png" alt="Position and velocity sensitivity among the eight smallest training objectives">
<img src="03-data-information.png" alt="Data-only antenna information after eliminating every scene nuisance">
<p>The information plot uses original-row mean-response derivatives. It removes static clutter and active real brightness, then constant primary phases, every secondary trajectory, and the nine independent Cartesian primary coordinates. The AIS time shift is already in that Cartesian span. Receiver horizontal coordinates are fixed, leaving 16 antenna directions. Gaussian location priors and active bounds are excluded. The dashed unit line is the precision of the declared standardized prior, not a detection threshold; the lower plot floor is 10⁻¹⁸. Thermal information is conditional on this two-echo scene being correct and can overstate certainty under coherent model error.</p>
<p>Some weak data-only directions are unavoidable geometric gauges. In particular, rotating every transmitter and target state about the receiver preserves every bistatic path. AIS and surveyed site priors supply the geographic orientation. A regression test verifies this invariance through the complete profiled complex-data operator for all three rotation axes; those directions must not acquire artificial radar information.</p>
<table><tr><th>Assignment</th><th>Association start</th><th>Training objective</th><th>Unused-data error</th><th>Strict local minimum</th><th>Bound indices</th></tr>{rows}</table>
<p>These remain antenna-refinement controls, not survey coordinates. The next decisive test is whether the same physical antennas predict other separately fitted ship intervals with those antennas frozen, while the common primary remains supported in all bands. Independently refitting coordinates on each interval would not provide that evidence.</p>
<p><a href="fits.json">All 64 fits</a> · <a href="analysis.json">Derived comparisons and provenance</a> · <a href="../ship-physical-pair-Flash-review/index.html">Separate flash model test</a> · <a href="../ship-physical-pair-F01/tests.log">{summary['tests_passed']} numerical tests</a>.</p></html>'''
    (out/'index.html').write_text(page)
    paths=[args.source,Path(__file__),Path('support/isar/physical_pair.py'),test_log,*set(case['field_paths'])]
    summary['sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    (out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',out/'index.html','strict',summary['strict_local_minima'],'train',summary['lowest_training_assignment'],
        'heldout',summary['lowest_heldout_assignment'],flush=True)


if __name__=='__main__':main()
