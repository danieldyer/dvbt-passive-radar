"""Compare physical common-phase paired echoes and expose cancellation limits."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common_physical_pair import CommonPhysicalPair
from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from pair_diagnostics import components,diagnose
from ship_site_signature import licensed_options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/ship-common-primary-pair/fits.json'))
    args=parser.parse_args();out=args.source.parent;data=json.loads(args.source.read_text())
    if not data['complete']:raise ValueError('Finish every declared trial')
    log=out/'tests.log';text=log.read_text();count=re.search(r'Ran (\d+) tests',text)
    if not count or not text.rstrip().endswith('OK'):raise ValueError('Passing numerical tests required')
    case=prepare_problem();lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a
        for a in assignments(licensed_options(case['geometry_directory']))}
    error=lambda r:sum(p['heldout_error'] for p in r['prediction'])
    summary=dict(scope=__doc__,accepted_calibration=False,tests_passed=int(count[1]),groups=[])
    models={}
    for group in data['groups']:
        label,code=group['interval'],group['assignment'];i=next(i for i,d in enumerate(case['descriptions']) if d['label']==label)
        model=CommonPhysicalPair(model_for_assignment(case,i,lookup[code]));models[label,code]=model
        rows=[]
        for row in group['common_trials']:
            p=np.asarray(row['parameters']);difference=model.objective(p)[0]*model.energy-row['cost']
            if abs(difference)>1e-7:raise ValueError('Common fit objective did not reproduce')
            rows.append(dict(branch_index=row['branch_index'],cost=row['cost'],unused_error=error(row),
                strict=row['audit']['strict_local_minimum'],bounds=row['bound_indices'],
                objective_reproduction_difference=difference,decomposition=diagnose(model,p)))
        best=min(rows,key=lambda r:r['cost']);strict=[r for r in rows if r['strict']]
        baseline=group['independent_baseline'];control=min([baseline,group['relaxed_control']],key=lambda r:r['cost'])
        summary['groups'].append(dict(interval=label,assignment=code,trials=rows,selected=best,
            best_strict=min(strict,key=lambda r:r['cost']) if strict else None,
            independent_cost=control['cost'],independent_unused_error=error(control),
            common_training_penalty_percent=100*(best['cost']/control['cost']-1),
            common_prediction_change_percent=100*(best['unused_error']/error(control)-1)))
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    labels=[]
    for i,g in enumerate(summary['groups']):
        labels.append(g['interval']+'\n'+g['assignment'])
        for j,r in enumerate(g['trials']):
            x=i+(j-2.5)*.09;marker='o' if r['strict'] else 'x'
            colour='#24699b' if r is g['selected'] else '#8d969c'
            axes[0].plot(x,100*(r['cost']/g['independent_cost']-1),marker,color=colour)
            axes[1].plot(x,100*(r['unused_error']/g['independent_unused_error']-1),marker,color=colour)
            axes[2].plot(x,max(d['component_to_sum_energy_ratio'] for d in r['decomposition']),marker,color=colour)
    for ax in axes:ax.set_xticks(range(len(labels)),labels)
    axes[0].set(ylabel='Training objective increase (%)',title='Cost of one physical primary phase')
    axes[1].set(ylabel='Unused-sample error change (%)',title='Frozen coefficients; no phase alignment')
    axes[1].axhline(0,color='#aaa',lw=.7)
    axes[2].set(yscale='log',ylabel='Largest component/sum energy ratio',title='Do echoes nearly cancel each other?')
    axes[2].axhline(1,color='#aaa',lw=.7)
    fig.suptitle('All 24 common-phase starts · blue: lowest training cost · ×: incomplete stationarity\nThe third panel is a diagnostic, not a penalty applied to the observations',fontsize=12)
    fig.savefig(out/'01-common-phase-controls.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for col,code in enumerate(('00002','10002')):
        group=next(g for g in data['groups'] if g['interval']=='F01' and g['assignment']==code)
        row=min(group['common_trials'],key=lambda r:r['cost']);model=models['F01',code]
        p=model.mapping(np.asarray(row['parameters']))[0];pair=model.independent.pairs[0];b=pair.base
        z=components(pair,p[model.independent.indices[0]]).reshape(2,*b.field.y.shape)
        strength=np.sqrt(np.sum(abs(z)**2,axis=2));summed=np.sqrt(np.sum(abs(z.sum(axis=0))**2,axis=1))
        for j,label in enumerate(('Primary','Secondary')):axes[0,col].plot(b.field.time,strength[j],label=label)
        axes[0,col].plot(b.field.time,summed,color='#111',lw=2,label='Complex sum')
        axes[0,col].set(yscale='log',ylabel='Whitened response norm',title='Seven · F01 · '+code);axes[0,col].legend(fontsize=8)
        cross=np.sum(z[0].conj()*z[1],axis=1);phase=np.degrees(np.angle(-cross))
        valid=np.prod(strength,axis=0)>0
        axes[1,col].plot(b.field.time[valid],phase[valid],color='#722d95')
        axes[1,col].set(xlabel='Capture time (s)',ylabel='Relative phase minus 180° (deg)')
    fig.suptitle('The low-cost Eyre alternative approaches two coincident, opposite-phase components\nA stable sum does not establish a stable phase or trajectory for either component',fontsize=12)
    fig.savefig(out/'02-cancelling-components.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for row,label in enumerate(('F01','Flash')):
        group=next(g for g in data['groups'] if g['interval']==label and g['assignment']=='00002')
        g=models[label,'00002'].geometry
        for trial in group['common_trials']:
            if not trial['audit']['strict_local_minimum']:continue
            p=np.asarray(trial['parameters']);colour='#24699b' if trial is min(group['common_trials'],key=lambda r:r['cost']) else '#a0a8ad'
            tx=np.asarray(trial['antenna_fit']['transmitter_offsets_enu_m'])
            for j in range(3):axes[row,j].plot(NAMES,tx[:,j],'o-',color=colour,alpha=.8)
        for j,axis in enumerate(('East','North','Up')):
            axes[row,j].set(ylabel=axis+' correction (m)',title=label+' · '+axis)
            axes[row,j].tick_params(axis='x',labelrotation=30)
    fig.suptitle('Antenna corrections across all converged Ornata/Ridge phase branches\nSeparate-interval fits; the next test must share these coordinates',fontsize=12)
    fig.savefig(out/'03-antenna-branches.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{g["interval"]}</td><td>{g["assignment"]}</td><td>{g["common_training_penalty_percent"]:+.3f}%</td><td>{g["common_prediction_change_percent"]:+.3f}%</td><td>{g["selected"]["strict"]}</td><td>{max(d["component_to_sum_energy_ratio"] for d in g["selected"]["decomposition"]):,.1f}</td></tr>' for g in summary['groups'])
    strict=sum(r['strict'] for g in summary['groups'] for r in g['trials']);summary['strict_trials']=strict
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Common primary phase with overlapping echoes</title>
<style>body{{font:16px/1.55 system-ui;max-width:1260px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd4d9;padding:8px}}.note{{background:#fff1d3;padding:14px}}</style>
<h1>Does a shared physical phase survive explicit overlapping echoes?</h1>
<p class="note">For the 00002 assignment, all twelve declared common-phase starts converge in the two intervals. A common physical primary phase costs only about a tenth of a percent in training error compared with independently phased primary echoes. This supports further joint testing, but the distinct carrier branches and per-interval antenna locations are not a conclusive calibration. In the competing 10002 assignment, most low-cost fits approach a cancellation singularity in Seven; those individual primary phases cannot be used as a stable location measurement.</p>
<p>00002 means Seven/SBS/Nine/Ten at Ornata and ABC at Ridge. 10002 moves Seven to Eyre, leaving the other four unchanged. These are two candidates from the previous 32-assignment comparison; other assignments are not excluded by this experiment. The two intervals are F01 at 159–162 s and Flash at 163–166 s.</p>
<img src="01-common-phase-controls.png" alt="Training, unused-data and cancellation diagnostics for all common phase branches">
<table><tr><th>Interval</th><th>Assignment</th><th>Training penalty</th><th>Unused error change</th><th>Best start converged</th><th>Largest component/sum energy ratio</th></tr>{rows}</table>
<p>Each primary uses one physical Cartesian path and one common physical scattering phase across the five carrier frequencies. The phase map includes the propagation-reference correction and its derivatives. This is an optimistic scattering hypothesis, not an arbitrary phase alignment of the observations. Each secondary retains an independent quadratic trajectory and constant phase. Both echoes retain the same original-symbol response, positive brightness knots and stationary nuisance as the independent-phase controls. Training-selected coefficients are frozen on unused samples. Releasing the common-phase restriction after fitting confirms that the nested, less constrained model does not have a larger optimized objective.</p>
<img src="02-cancelling-components.png" alt="Primary and secondary strengths and opposite phases in Seven for two site hypotheses">
<p>For the selected 10002 starts, the sum of the fitted component energies is about 111,102 times the summed-echo energy in F01 and 37,714 times in Flash. The two Seven responses are almost coincident and opposite in phase. This is a local model degeneracy, not evidence for huge physical sea echoes or a numerical convergence certificate. Close-component collisions and diverging amplitudes are a known inverse-problem pathology; see <a href="https://arxiv.org/abs/1301.1336">Batenkov and Yomdin, Geometry and Singularities of the Prony Mapping</a>. The analogy explains why apparently good total prediction does not identify either component separately; that paper does not prove a result for this specific radar operator.</p>
<p>No minimum separation, arbitrary amplitude cap or cancellation penalty has been imposed to force a preferred site. Every failed start remains in the source data and plot. The solver's actual-objective stationarity audit passes {strict} of 24 trials; several phase branches in 00002 are near-equivalent. A stable local result is necessary but does not establish globally unique antenna coordinates.</p>
<img src="03-antenna-branches.png" alt="Separate interval antenna estimates across converged carrier branches">
<p>{summary['tests_passed']} numerical tests pass. New checks cover the common-phase operator against the explicit physical carrier phase, exact nested likelihood, residual derivatives, recovery of known motion, reference-band invariance, rigid-rotation gauges, and shared-antenna priors counted once across intervals. All 24 saved common objectives reproduce from the original fields. Production processing and recording metadata are unchanged by these experiments.</p>
<p><a href="fits.json">All fits</a> · <a href="analysis.json">Derived diagnostics and hashes</a> · <a href="tests.log">Numerical test log</a> · <a href="../ship-pair-frozen-transfer/index.html">Previous frozen-antenna transfer</a> · <a href="../../isar_20261016/flash-reflection/index.html">HH sea-reflection geometry</a>.</p></html>'''
    (out/'index.html').write_text(page)
    summary['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (args.source,log,Path(__file__),Path(__file__).with_name('pair_diagnostics.py'))}
    (out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',out/'index.html','strict',strict,'tests',summary['tests_passed'])


if __name__=='__main__':main()
