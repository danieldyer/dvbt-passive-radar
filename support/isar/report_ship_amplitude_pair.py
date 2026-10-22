"""Publish the short-arc overlapping-echo diagnostic without accepting geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from amplitude_arc import AmplitudeArc,linear_basis
from amplitude_pair import AmplitudePair
from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-amplitude-pair-F01'))
    args=parser.parse_args();source=args.output/'fits.json';data=json.loads(source.read_text())
    if not data['complete']:raise ValueError('All declared starts must finish')
    log=(args.output/'tests.log').read_text();tests=re.search(r'Ran (\d+) tests',log)
    if not tests or not log.rstrip().endswith('OK'):raise ValueError('Passing tests required')
    best=[c['trials'][c['best_index']] for c in data['channels']]
    single=data['single_path']['channels'];case=prepare_problem();entry=data['interval'];label=entry['label']
    index=next(i for i,e in enumerate(case['descriptions']) if e['label']==label)
    gain=np.array([100*(s['heldout_error']-p['prediction']['heldout_error'])/s['heldout_error'] for s,p in zip(single,best)])
    summary=dict(scope=__doc__,accepted_calibration=False,tests_passed=int(tests[1]),heldout_error_reduction_percent=gain.tolist(),
        numerically_regular=[p['numerically_regular'] for p in best],reproduction=[])
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    colours=['#24699b' if p['numerically_regular'] else '#bd6b22' for p in best]
    axes[0].bar(NAMES,gain,color=colours);axes[0].set(ylabel='Held-out squared-error reduction (%)',title='One echo → two echoes')
    axes[1].bar(NAMES,[100*p['prediction']['heldout_explained_fraction'] for p in best],color=colours)
    axes[1].set(ylabel='Held-out residual energy explained (%)',ylim=(0,100),title='Relative to stationary-only prediction')
    axes[2].bar(NAMES,[p['prediction']['component_to_sum_energy_ratio'] for p in best],color=colours)
    axes[2].set(yscale='log',ylabel='Sum of component energies / combined energy',title='Large values expose cancelling components')
    for ax in axes:ax.tick_params(axis='x',labelrotation=35)
    fig.suptitle(f'{label} · {entry["first_s"]:g}–{entry["last_s"]:g} s · same observations and original-row brightness knots\nBlue: full-rank stationary interior fit · orange: incomplete, rank-deficient or bound-limited',fontsize=12)
    fig.savefig(args.output/'01-prediction.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
    nominal=case['geometries'][index].prediction(np.zeros(28))[:,0]
    for ch,(row,s) in enumerate(zip(data['channels'],single)):
        shift=nominal[ch]
        for trial_index,trial in enumerate(row['trials']):
            points=np.asarray(trial['observables'])
            chosen=trial_index==row['best_index']
            for j,ax in enumerate(axes):
                values=points[:,j]-(shift if j==0 else 0)
                ax.scatter([ch-.1,ch+.1],values,s=45 if chosen else 25,marker='o' if trial['numerically_regular'] else 'x',
                    color='#24699b' if chosen else '#9da6ae',alpha=.9,zorder=4 if chosen else 2)
        for j,ax in enumerate(axes):
            val=s['observables'][j]-(shift if j==0 else 0)
            ax.plot(ch,val,'D',color='#bd6b22',ms=5,label='Single echo' if ch==0 else None)
    axes[0].set(ylabel='Path relative to nominal AIS geometry (m)',title='Same nominal antenna/AIS reference in every fit')
    axes[1].set(ylabel='Excess-path rate (m/s)',title='Both components from all three starts')
    axes[2].set(ylabel='Excess-path curvature (m/s²)',title='Numerical regularity is not physical association')
    for ax in axes:ax.set_xticks(range(5),NAMES,rotation=35)
    axes[0].legend();fig.suptitle('Independent band fits: multiple local interpretations remain\nBlue: lowest training cost · grey: other starts · ×: incomplete, rank-deficient or bound-limited',fontsize=12)
    fig.savefig(args.output/'02-component-trajectories.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(14,13),layout='constrained')
    for ch,(f,tr,te,knots,s,p) in enumerate(zip(case['fields'][index],case['training'][index],case['testing'][index],case['knots'][index],single,best)):
        base=AmplitudeArc(f,tr,knots);pair=AmplitudePair(base);parameters=np.asarray(p['parameters'])
        value=pair.evaluate(parameters)
        rebuilt=float(value['residual']@value['residual'])
        if abs(rebuilt-p['objective'])>1e-11:raise ValueError('Saved pair fit did not reproduce')
        summary['reproduction'].append(rebuilt-p['objective'])
        _,_,one_design,one_strength,_,_=base.evaluate(np.asarray(s['parameters']),True)
        for prediction,colour,label in [(one_design@one_strength,'#bd6b22','One echo'),(value['design']@value['amplitude'],'#24699b','Two echoes')]:
            beta=base.nuisance.solve(f.y.ravel()[base.selected]-prediction[base.selected])
            residual=(f.y.ravel()-base.design@beta-prediction).reshape(f.y.shape)
            power=np.sum(abs(residual)**2,axis=1)
            axes[ch,0].plot(f.time[te],power[te],'.-',color=colour,ms=3,lw=.6,label=label)
        amplitudes=linear_basis(f.time,knots)@value['amplitude'].reshape(2,-1).T
        axes[ch,1].plot(f.time,amplitudes[:,0],label='Component 1');axes[ch,1].plot(f.time,amplitudes[:,1],label='Component 2')
        axes[ch,0].set_ylabel(NAMES[ch]+'\nHeld-out residual energy');axes[ch,1].set_ylabel('Fitted real brightness')
        axes[ch,0].legend(fontsize=8);axes[ch,1].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Capture time (s)')
    fig.suptitle('Unused samples retain their original complex values and covariance\nAll moving and stationary coefficients are fitted only on alternating training blocks',fontsize=12)
    fig.savefig(args.output/'03-time-prediction.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{NAMES[ch]}</td><td>{gain[ch]:.2f}%</td><td>{p["information_rank"]}/8</td><td>{p["projected_gradient"]:.3g}</td><td>{p["bound_indices"]}</td><td>{p["prediction"]["component_to_sum_energy_ratio"]:.3f}</td></tr>' for ch,p in enumerate(best))
    regular=', '.join(NAMES[ch] for ch,p in enumerate(best) if p['numerically_regular']) or 'None'
    unresolved=', '.join(NAMES[ch] for ch,p in enumerate(best) if not p['numerically_regular']) or 'None'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{label} overlapping ship echoes</title>
<style>body{{font:16px/1.55 system-ui;max-width:1220px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccd4d9;text-align:left}}.note{{background:#fff1d3;padding:16px}}</style>
<h1>Overlapping echoes in {label}: {entry['first_s']:g}–{entry['last_s']:g} s</h1>
<p class="note"><b>An association diagnostic, not an accepted calibration.</b> Numerically regular training-selected pair fits: {regular}. Incomplete, rank-deficient or bound-limited fits: {unresolved}. Prediction improvements from irregular fits do not establish two identifiable physical echoes. All starts and failure diagnostics remain visible.</p>
<p>This follows the <a href="../joint-ship-band-phase-control/index.html">constant-phase control</a>, which established that the remaining disagreement includes phase rate and curvature. Here two quadratic range/phase trajectories are summed before the original symbol response, averaging and whitening. Each has a constant phase and nonnegative original-row brightness knots at the same 0.5 s spacing used by the single-echo model. Stationary complex coefficients are fitted jointly. There is no per-band time-varying phase correction or noise renormalization.</p>
<img src="01-prediction.png" alt="Prediction gains, explained energy and cancellation between two fitted components">
<table><tr><th>Band</th><th>Held-out error reduction</th><th>Information rank</th><th>Projected gradient</th><th>Bound indices</th><th>Component/sum energy</th></tr>{rows}</table>
<p>The same unused blocks score both models. Better complex prediction can expose single-echo model mismatch, but does not by itself establish a valid two-component decomposition or a measured SNR gain. The sum of individual component energies can exceed their combined energy because the complex fields interfere. Extremely large ratios expose nearly cancelling components whose individual phases and trajectories are not stably identified, even if their sum predicts reasonably well.</p>
<img src="02-component-trajectories.png" alt="All three starts and both component trajectories in every band">
<p>Every band has three declared starts. They combine the shared physical trajectory with the competing trajectory preferred by Nine, perturbing the latter's initial path rate by 0 or ±0.2 m/s. Each fit then moves freely within the same declared local range/rate search. Starts and numerical failures remain visible; no fit is selected using held-out error. This experiment tests association and model adequacy, not transmitter location, and the five bands are not yet constrained to describe the same two physical points.</p>
<img src="03-time-prediction.png" alt="Held-out residual energy and original-row fitted amplitudes over the three seconds">
<p>{summary['tests_passed']} tests passed. The new two-echo tests recover two known trajectories and time-varying strengths in strong stationary clutter, verify the full profiled-residual derivative with nonzero residual, verify component-exchange symmetry, and ensure withheld data cannot change the fitted likelihood. Capture checks report all eight conditional information modes, projected gradients, bound contacts and the actual objective Hessian's tangent curvature. All five selected objectives were reconstructed exactly.</p>
<p>Next, the useful component must survive a shared physical-motion fit and transfer to another interval before it can calibrate antennas. The current location/autofocus goal remains active. <a href="fits.json">All 15 fits</a> · <a href="analysis.json">Derived results</a> · <a href="tests.log">Numerical tests</a>.</p></html>'''
    (args.output/'index.html').write_text(page)
    paths=[source,Path(__file__),args.output/'tests.log']
    summary['sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    (args.output/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',args.output/'index.html',flush=True)


if __name__=='__main__':main()
