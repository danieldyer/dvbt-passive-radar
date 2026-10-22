"""Review the shared physical echo, range-support control and remaining limits."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from amplitude_arc import AmplitudeArc
from amplitude_pair import AmplitudePair
from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem
from fit_ship_physical_pair import build_model
from motion import C
from ship_arc_coherence import archived_observables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-physical-pair-review'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);root=args.output.parent
    directories=['ship-physical-pair-F01','ship-physical-pair-F01-refined','ship-physical-pair-F01-wider']
    data=[json.loads((root/d/'fits.json').read_text()) for d in directories]
    audits=[json.loads((root/d/'stationarity.json').read_text()) for d in directories]
    old=json.loads((root/'ship-amplitude-pair-F01/fits.json').read_text())
    independent=json.loads((root/'ship-amplitude-pair-F01-wider/fits.json').read_text())
    if not all(x['complete'] for x in [*data,*audits,old,independent]):raise ValueError('Finish declared fits and audits first')
    log=root/directories[0]/'tests.log';tests=re.search(r'Ran (\d+) tests',log.read_text())
    if tests is None or not log.read_text().rstrip().endswith('OK'):raise ValueError('Passing tests required')
    case=prepare_problem();model,i=build_model(case,(-2.8,2.8))
    best=[min(rows['trials'],key=lambda r:r['cost']) for rows in audits]
    one=old['single_path']['channels'];two=[c['trials'][c['best_index']] for c in old['channels']]
    wide=[c['trials'][c['best_index']] for c in independent['channels']]
    labels=['Independent one echo','Independent two echoes / original support',
        'Shared primary / fixed antennas','Shared primary / refined antennas',
        'Shared primary / wider secondary support','Independent two echoes / wider support']
    errors=[sum(r['heldout_error'] for r in one),sum(r['prediction']['heldout_error'] for r in two),
        *[sum(p['heldout_error'] for p in r['prediction']) for r in best],
        sum(r['prediction']['heldout_error'] for r in wide)]
    summary=dict(scope=__doc__,accepted_calibration=False,tests_passed=int(tests[1]),labels=labels,
        heldout_errors=errors,all_physical_starts_strict=sum(t['audit']['strict_local_minimum'] for a in audits for t in a['trials']),
        selected_start_ranks=[r['start_rank'] for r in best],objective_reproduction=[])
    for source,audit in zip(data,audits):
        for row in audit['trials']:
            p=np.asarray(row['parameters']);cost=model.objective(p)[0]*model.energy
            if abs(cost-row['cost'])>1e-7:raise ValueError('Audited objective did not reproduce')
            summary['objective_reproduction'].append(cost-row['cost'])
    for pair,row in zip(model.pairs,wide):
        check=AmplitudePair(pair.secondary).evaluate(np.asarray(row['parameters']))
        if abs(check['residual']@check['residual']-row['objective'])>1e-11:raise ValueError('Independent wide fit did not reproduce')
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    ax.barh(labels,errors,color=['#9b5d36','#a8917e','#83a2b9','#608bad','#24699b','#334b5a'])
    ax.invert_yaxis();ax.set(xlabel='Squared prediction error on identical unused complex samples',
        title='One shared physical echo plus secondary echoes matches independent-band prediction')
    for j,error in enumerate(errors):ax.text(error+70,j,f'{error:,.1f}',va='center',fontsize=9)
    ax.set_xlim(0,max(errors)*1.14);fig.savefig(args.output/'01-heldout-comparison.png',dpi=170);plt.close(fig)
    parameters=np.asarray(best[2]['parameters']);g=model.geometry;primary=g.prediction(parameters[:g.count])
    nominal=g.prediction(np.zeros(g.count));secondary=[];fractions=[];cancellation=[]
    fig,axes=plt.subplots(5,2,figsize=(14,13),layout='constrained')
    for ch,(pair,index,test) in enumerate(zip(model.pairs,model.indices,case['testing'][i])):
        b=pair.base;base=pair.secondary;value=pair.evaluate(parameters[index]);k=len(b.knots)
        components=np.stack([value['design'][:,j*k:(j+1)*k]@value['amplitude'][j*k:(j+1)*k] for j in range(2)])
        energy=np.sum(abs(components[:,b.selected])**2,axis=1);fractions.append(float(energy[0]/energy.sum()))
        cancellation.append(float(energy.sum()/np.sum(abs(components[:,b.selected].sum(axis=0))**2)))
        beta=b.nuisance.solve(b.field.y.ravel()[b.selected]-components[:,b.selected].sum(axis=0))
        clean=(b.field.y.ravel()-b.design@beta).reshape(b.field.y.shape)
        first,second=components.reshape(2,*b.field.y.shape)
        before=np.sum(first.conj()*clean,axis=1);after=np.sum(first.conj()*(clean-second),axis=1)
        valid=np.sum(abs(first)**2,axis=1)>0
        for ax,values,title in zip(axes[ch],(before,after),('Before secondary subtraction','After secondary subtraction')):
            use=test&valid;ax.plot(b.field.time[use],np.degrees(np.angle(values[use])),'.',ms=4,color='#24699b')
            ax.axhline(0,color='#aaa',lw=.6);ax.set_ylim(-180,180)
            if ch==0:ax.set_title(title)
            if ch==4:ax.set_xlabel('Capture time (s)')
            ax.set_ylabel(NAMES[ch]+'\nPrimary phase residual (deg)')
        u=(b.midpoint-base.epoch)/base.scale
        transform=np.array([[C/8e6,0,u,u*u],[0,0,1/base.scale,2*u/base.scale],[0,0,0,2/base.scale**2]])
        secondary.append(archived_observables(b.field.data,b.midpoint)+transform@parameters[index[-4:]])
    fig.suptitle('Unused samples only · coefficients frozen from training · one physical trajectory\nSubtracting the fitted secondary is a conditional diagnostic, not independent proof of two material points',fontsize=12)
    fig.savefig(args.output/'02-heldout-phase.png',dpi=170);plt.close(fig)
    summary.update(primary_observables=primary.tolist(),secondary_observables=np.asarray(secondary).tolist(),
        primary_component_energy_fractions=fractions,component_to_sum_energy_ratios=cancellation)
    fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
    for ch,(one_row,two_row) in enumerate(zip(one,wide)):
        values=np.asarray(two_row['observables'])
        for j,ax in enumerate(axes):
            shift=nominal[ch,0] if j==0 else 0
            ax.plot(ch,one_row['observables'][j]-shift,'D',color='#bd6b22',ms=5,label='Independent one echo' if ch==0 else None)
            ax.scatter([ch-.12,ch+.12],values[:,j]-shift,color='#929a9f',s=28,label='Independent pair' if ch==0 else None)
    for j,ax in enumerate(axes):
        shift=nominal[:,0] if j==0 else 0
        ax.plot(range(5),primary[:,j]-shift,'o-',color='#24699b',label='Shared physical primary')
        ax.plot(range(5),np.asarray(secondary)[:,j]-shift,'x',color='#722d95',label='Secondary nuisance')
        ax.set_xticks(range(5),NAMES,rotation=30)
        ax.set_ylabel(['Excess path relative to nominal AIS (m)','Excess-path rate (m/s)','Excess-path curvature (m/s²)'][j])
    axes[0].legend(fontsize=8);fig.suptitle('A consistent primary can coexist with a very different single-echo phase fit\nThe secondary trajectories are not asserted to be common physical points',fontsize=12)
    fig.savefig(args.output/'03-component-motion.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for group,colour,label in zip(audits,['#83a2b9','#608bad','#bd6b22'],['Fixed antennas','Refined antennas','Wider secondary support']):
        for j,row in enumerate(group['trials']):
            p=np.asarray(row['parameters']);position=p[18:21]*g.nuisance_sigma[:3];velocity=p[21:24]*g.nuisance_sigma[3:6]
            axes[0].plot(*position[:2],'o',color=colour,label=label if j==0 else None)
            axes[0].annotate(str(row['start_rank']),position[:2],xytext=(4,4),textcoords='offset points',fontsize=8)
            axes[1].plot(*velocity[:2],'o',color=colour)
    offsets=g.antennas.describe(parameters[:18])
    for j,label in enumerate(('East','North','Up')):axes[2].plot(NAMES,np.asarray(offsets['transmitter_offsets_enu_m'])[:,j],'o-',label=label)
    axes[0].set(xlabel='East correction to AIS (m)',ylabel='North correction (m)',title='Apparent primary position');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='East velocity correction (m/s)',ylabel='North velocity correction (m/s)',title='Velocity correction')
    axes[2].set(ylabel='TX displacement from supplied prior (m)',title='Selected wider-support antenna fit');axes[2].legend(fontsize=8)
    axes[2].tick_params(axis='x',labelrotation=30)
    for ax in axes[:2]:ax.axhline(0,color='#aaa',lw=.6);ax.axvline(0,color='#aaa',lw=.6);ax.set_aspect('equal',adjustable='datalim')
    fig.suptitle('Model adequacy improves; a physical calibration still needs independent intervals and site discrimination',fontsize=12)
    fig.savefig(args.output/'04-location-sensitivity.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{label}</td><td>{error:,.2f}</td></tr>' for label,error in zip(labels,errors))
    gain=100*(errors[0]-errors[4])/errors[0];difference=100*(errors[4]/errors[5]-1)
    summary['heldout_error_reduction_from_one_echo_percent']=gain
    summary['shared_vs_independent_wider_error_difference_percent']=difference
    summary['selected_antenna_fit']=offsets
    summary['selected_position_correction_enu_m']=(parameters[18:21]*g.nuisance_sigma[:3]).tolist()
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Shared physical ship echo with overlapping returns</title>
<style>body{{font:16px/1.55 system-ui;max-width:1220px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}td,th{{padding:8px;border:1px solid #ccd4d9}}table{{border-collapse:collapse}}.note{{background:#fff1d3;padding:16px}}</style>
<h1>A shared physical ship echo survives the overlapping-return test</h1>
<p class="note">The F01 interval (159–162 s) now supports a substantially better common-motion model. One shared Cartesian primary trajectory plus explicit secondary echoes reduces unused-sample prediction error by {gain:.1f}% relative to independent single-echo fits. Its error differs from independent wider-support two-echo fits by only {difference:+.3f}%. This is prediction error, not an SNR gain or proof of a unique physical scene. Antenna refinement remains unaccepted.</p>
<p>The primary uses exact bistatic propagation, one position/velocity/acceleration state and one constant scattering phase per band. The secondary in each band is a quadratic range/phase trajectory, with its own constant phase and positive brightness. Both use the actual original-symbol response before averaging and range whitening. Brightness and stationary-clutter coefficients are estimated only from training blocks and frozen when predicting the alternating unused blocks. No per-band phase-rate correction or noise renormalization is introduced.</p>
<img src="01-heldout-comparison.png" alt="Prediction error on identical unused samples for six model controls">
<table><tr><th>Model</th><th>Unused-data squared error</th></tr>{rows}</table>
<p>Three declared starts were tested first at fixed antenna priors, then with the existing antenna priors released. Two secondary echoes initially reached the local range-search edge. Widening that search by two samples toward shorter paths (74.95 m), without changing any observations, removed those secondary bound contacts. Independent pair fits were repeated with the same wider support and both original and shared-trajectory starts, so this support improvement is not misattributed to a multichannel gain. Fits are selected by training objective, never prediction error.</p>
<img src="02-heldout-phase.png" alt="Conditional phase residual before and after secondary subtraction on unused samples">
<p>The phase plots show only unused samples, after the stationary coefficient fitted on training has been removed. A sample with exactly zero predicted primary strength has no defined phase and is omitted. Subtraction uses the frozen secondary prediction; no new phase fit is made on these samples. Because the scene decomposition is itself estimated, these are conditional diagnostics and must be read alongside the total complex prediction errors.</p>
<img src="03-component-motion.png" alt="Single echo, independent pair and common primary path rate and curvature">
<p>The common primary has path rate about −0.57 m/s in Seven/SBS/Nine/Ten and −0.60 m/s in ABC, with curvature about −0.334 m/s² in all bands. This explains why the earlier single-echo Nine fit, around −1.03 m/s, was misleading for association. It does not identify the secondary as sea multipath: at this geometry the simple smooth-sea excess paths are sub-metre to a few metres, whereas fitted secondary echoes can be much farther apart. See the separate <a href="../../isar_20261016/flash-reflection/index.html">HH sea-reflection geometry analysis</a>.</p>
<img src="04-location-sensitivity.png" alt="Position, velocity and antenna sensitivity across all nine starts">
<p>All nine physical solutions pass constrained first-order stationarity and positive actual-objective tangent-curvature checks. That verifies local numerical convergence, not uniqueness or physical correctness. The selected wider-support solution still reaches two correlated antenna-prior bounds and the target-height bound. Its apparent position is around 188 m from the time-shifted AIS position. Alternative associations and the 32 licensed primary/backup site assignments still need comparison and transfer to independent ship intervals. Receiver latitude/longitude stay at the supplied coordinates; no fitted location is written into recording metadata.</p>
<p>{summary['tests_passed']} tests pass, including known-truth two-component recovery across five frequencies, exact shared residual/Jacobian checks, counting the geometry prior once, fixed-antenna preservation, and protection against held-out-data leakage. All nine physical objectives and five selected independent wider-support objectives were rebuilt from the original fields. The actual-objective curvature audits are stored beside each source fit.</p>
<p><a href="analysis.json">Derived results and source hashes</a> · <a href="../ship-physical-pair-F01/tests.log">Tests</a> · <a href="../ship-physical-pair-F01-wider/fits.json">Wider physical fits</a> · <a href="../ship-physical-pair-F01-wider/stationarity.json">Curvature audits</a> · <a href="../ship-amplitude-pair-F01-wider/fits.json">Independent wider-support control</a> · <a href="../stationary-arc-phase-control/index.html">Stationary phase control</a>.</p></html>'''
    (args.output/'index.html').write_text(body)
    paths=[Path(__file__),log,*[root/d/f for d in directories for f in ('fits.json','stationarity.json')],
        root/'ship-amplitude-pair-F01/fits.json',root/'ship-amplitude-pair-F01-wider/fits.json']
    summary['sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    (args.output/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',args.output/'index.html','heldout gain',gain,'shared vs independent',difference,flush=True)


if __name__=='__main__':main()
