"""Review a completed common B/E path hypothesis against independent controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES,COLOURS
from joint_path_information import paths
from nearby_echo_control import load,shared_model
from report_fixed_grid_guided_return import phase_comparison
from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sites',type=Path,help='Completed frozen-path licensed-site sensitivity screen')
    args=parser.parse_args();data=json.loads(args.source.read_text())
    if not data['complete']:raise ValueError('Wait for every declared start')
    control=load();model=shared_model(control,data.get('assignment','00002'))
    errors=[]
    for f in data['trials']:
        r,_=model.evaluate(np.asarray(f['parameters']));errors.append(abs(r@r*model.energy-f['cost']))
    if max(errors)>1e-7:raise ValueError('Original joint likelihood changed')
    fit=data['trials'][data['best_index']];q=np.array(fit['parameters'])
    information=paths(model,q);first,last=data['interval_s']
    measured={}
    for s in (1,4):
        records=[]
        for channel in fit['measured']:
            rec=channel[str(s)];z=np.asarray(rec['complex_coefficient_real_imag'])
            records.append(dict(time_s=np.asarray(rec['time_s']),physical_path_change_m=np.asarray(rec['physical_path_change_m']),
                complex_coefficient=z[:,0]+1j*z[:,1],conditional_information=np.asarray(rec['conditional_information'])))
        measured[str(s)]=phase_comparison(records,control['bands'],first,last)
    independent_source=args.source.parent.parent/'main-return-five-band-review'/'analysis.json'
    independent=json.loads(independent_source.read_text())
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(18,9),layout='constrained',sharex=True)
    for row,s in enumerate((1,4)):
        current=measured[str(s)];time=current['time_s'];anchor=current['anchor_time_s']
        for c,colour in enumerate(COLOURS):
            previous=independent['measured_paths'][str(s)];old=previous['channels'][c]
            values=np.array(old['measured_path_change_m'],float)
            grid=np.array(previous['time_s'])
            axes[row,0].plot(grid,values,'.-',color=colour,ms=2,lw=.8,label=NAMES[c])
            current_values=current['channels'][c]['measured_path_change_m']
            axes[row,1].plot(time,current_values,'.-',color=colour,ms=2,lw=.8)
            ref=current['channels'][2]['measured_path_change_m']
            if c!=2:axes[row,2].plot(time,100*(current_values-ref),color=colour,lw=.8)
        target=control['curve'](time)@q[model.local_path_indices[0][s]]
        target-=np.interp(anchor,time,target)
        axes[row,1].plot(time,target,'k--',lw=1,label='Fitted Nine path')
        name='B' if s==1 else 'E'
        axes[row,0].set(title=f'{name} · independent scene · origin {previous["anchor_time_s"]:.2f} s',ylabel='Path change (m)')
        axes[row,1].set(title=f'{name} · common scene · origin {anchor:.2f} s',ylabel='Path change (m)')
        axes[row,2].set(title=name+' · common-scene channel minus Nine',ylabel='Measured interband difference (cm)')
        for ax in axes[row]:ax.set(xlim=(first,last),xlabel='Capture time (s)')
    axes[0,0].legend(ncol=3,fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Do two common temporal paths explain the original five-channel observations?\n'
        'Dots are measured complex coefficients after training-frozen scene subtraction. Dashed curve is a fit; it is not substituted for measured phases.',fontsize=12)
    fig.savefig(out/'01-measured-path-comparison.png',dpi=150);plt.close(fig)
    independent_error=np.array([v['heldout_error'] for v in data['independent_prediction']['testing']])
    common_error=np.array([v['heldout_error'] for v in fit['prediction']['testing']])
    changes=100*(common_error/independent_error-1)
    total=100*(common_error.sum()/independent_error.sum()-1)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    axes[0].bar(NAMES,changes,color=COLOURS);axes[0].axhline(0,color='#888',lw=.7)
    axes[0].set(title=f'Frozen unused prediction · total {total:+.3f}%',ylabel='Squared prediction-error change (%)')
    for label,value,colour in zip(('B','E','E−B'),information['standard_deviation_by_functional_m'],('#4c78a8','#e45756','#5c9e43')):
        axes[1].plot(information['time_s'],100*value,color=colour,label=label)
    axes[1].set(title='Conditional common-path data information',xlabel='Capture time (s)',ylabel='Path-change σ (cm)')
    axes[1].legend();fig.suptitle('Independent paths are the control; common-path precision alone does not validate association',fontsize=12)
    fig.savefig(out/'02-prediction-and-information.png',dpi=150);plt.close(fig)
    site_section='';site_review=None
    if args.sites:
        site_review=json.loads(args.sites.read_text())
        if (not site_review['complete'] or site_review['source_start']!=data['best_index'] or
                site_review['source_sha256'][str(args.source)]!=hashlib.sha256(args.source.read_bytes()).hexdigest()):
            raise ValueError('Site screen must use the identical completed common-path fit')
        trials=sorted(site_review['trials'],key=lambda f:f['cost'])
        original=next(f for f in trials if f['assignment']==site_review['reference_assignment'])
        fig,axes=plt.subplots(2,1,figsize=(15,8),layout='constrained',sharex=True)
        names=[f['assignment'] for f in trials]
        axes[0].bar(names,[f['cost']-original['cost'] for f in trials],color='#4c78a8')
        axes[0].set(ylabel='Conditional training cost difference',title='Original observation operator is fixed; only target-model geometry changes')
        baseline=sum(v['heldout_error'] for v in original['prediction']['testing'])
        axes[1].bar(names,[100*(sum(v['heldout_error'] for v in f['prediction']['testing'])/baseline-1) for f in trials],color='#e45756')
        axes[1].set(ylabel='Unused error change (%)',xlabel='Licensed assignment: Seven / SBS / Nine / Ten / ABC')
        axes[1].tick_params(axis='x',rotation=90,labelsize=8)
        for ax in axes:ax.axhline(0,color='#888',lw=.7)
        fig.suptitle('All 32 licensed-site hypotheses · conditional sensitivity only\n'
            'Common paths, envelope centres and constant phases are frozen; positive brightness/static nuisance are profiled. This is not a physical location fit.',fontsize=12)
        fig.savefig(out/'03-licensed-site-sensitivity.png',dpi=150);plt.close(fig)
        site_section=f'<img src="03-licensed-site-sensitivity.png" alt="Conditional original-data sensitivity to all 32 licensed transmitter assignments"><p>The best frozen-path training assignment is {trials[0]["assignment"]}. Codes: 0=Ornata, 1=Eyre, 2=Ridge; channel order Seven/SBS/Nine/Ten/ABC. These paths were optimized under the reference assignment, so this asymmetric screen cannot confirm its correctness. It neither optimizes every hypothesis nor supplies site probabilities. Reference-path rebasing is checked below one micrometre; original captured C0/C1 references never change.</p>'
    result=dict(complete=True,accepted_calibration=False,production_changed=False,source=str(args.source),
        selected_start=data['best_index'],assignment=data.get('assignment','00002'),
        reproduced_objectives=len(errors),maximum_absolute_reproduced_cost_error=max(errors),
        unused_error_change_percent=changes,total_unused_error_change_percent=total,
        local_B_E_component_to_sum_energy_ratio=fit['local_B_E_component_to_sum_energy_ratio'],
        measured=measured,conditional_data_information=information,
        licensed_site_sensitivity=site_review,
        scope='Two common temporal paths, nominal interband geometric increments, independent band envelopes/constant phases/positive brightness. '
            'A/C/D remain empirical nuisance returns. Shared paths are a tested restriction, not proof of common material points or calibrated antennas. '
            'Unused rows have been inspected in preceding exploratory analyses; prediction is a diagnostic, not an untouched final benchmark.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.source,independent_source,Path(__file__),Path('support/isar/joint_path_information.py'),Path('support/isar/nearby_echo_control.py'))})
    if args.sites:result['source_sha256'][str(args.sites)]=hashlib.sha256(args.sites.read_bytes()).hexdigest()
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    rows=''.join(f'<tr><td>{NAMES[c]}</td><td>{changes[c]:+.3f}%</td><td>{fit["local_B_E_component_to_sum_energy_ratio"][c]:.3f}</td></tr>' for c in range(5))
    starts=''.join(f'<tr><td>{f["start"]}: {f["name"]}</td><td>{f["cost"]:.3f}</td><td>{sum(v["heldout_error"] for v in f["prediction"]["testing"]):.3f}</td><td>{f["maximum_projected_gradient"]:.3g}</td><td>{f["bound_indices"]}</td></tr>' for f in data['trials'])
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Common nearby-return path test</title>
<style>body{{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px}}</style>
<h1>Two common nearby-return trajectories: original-data test</h1>
<p>All {len(data['trials'])} declared starts have completed. Training-selected start {data['best_index']} changes combined unused squared prediction error by {total:+.3f}% compared with independent five-return fits. A positive number is worse prediction. The nominal licensed assignment is {result['assignment']}; alternative licensed sites have not been excluded by this test. No phase-centre update is accepted.</p>
<p><a href="../main-return-five-band-review/00-selected-points.png">Selected B/E centres in all five original channel maps</a> · <a href="../main-return-five-band-review/index.html">Independent measured phases and pairwise deltas</a></p>
<p>For a common physical point x and a stationary receiver R, the excess path is L<sub>c</sub>=|x−T<sub>c</sub>|+|x−R|−|T<sub>c</sub>−R|. In L<sub>c</sub>−L<sub>d</sub>, the receiver-to-target leg cancels exactly; the receiver remains only in a time-independent baseline term. After removing constant origins, changing receiver coordinates cannot repair time-varying interband disagreement. This is why the common-trajectory and transmitter-hypothesis tests precede another physical receiver fit. A numerical regression independently checks this cancellation.</p>
<img src="01-measured-path-comparison.png" alt="Independent and common-scene measured phase histories across all five bands">
<p>The dashed curve is the fitted Nine reference path. Other channels retain their declared nominal interband geometric increment. Left and middle panels mark their own constant-origin epochs; compare their changes and curvature, not their absolute vertical intercepts. No measured rate or curvature is fitted out.</p>
<img src="02-prediction-and-information.png" alt="Unused prediction comparison and conditional common-path information">
{site_section}
<table><tr><th>Channel</th><th>Unused error change</th><th>Local B/E component energy / pair-sum</th></tr>{rows}</table>
<p>The information bound eliminates stationary profiles, active brightness and every other nonlinear scene parameter. Neither the white-jerk prior, fixed nuisance coefficients, active bounds nor an external guide is counted as measured information. Its zero at 165 s follows from defining path change relative to that epoch; it is not zero uncertainty in absolute carrier phase. Unknown geometry, model mismatch and carrier-branch ambiguity remain outside that local bound. Precision under a shared model is not evidence for the model.</p>
<table><tr><th>Start</th><th>Penalized training cost</th><th>Unused error</th><th>Projected gradient</th><th>Bound indices</th></tr>{starts}</table>
<p>Starts select minimum training cost only. The shared path prior is counted once per distinct temporal path; independent and shared MAP penalties have different dimensions and are not Bayes factors. Per-band RF phases and envelope origins remain free. Measured coefficients use 250 ms overlapping windows at 50 ms steps, so adjacent samples are correlated. Fades remain gaps; only constant phase origins are removed. Exploratory unused observations have already been inspected.</p>
<p><a href="analysis.json">Measured arrays, diagnostics and source hashes</a> · <a href="../current-nearby-phase-tests.log">54 numerical regression tests</a>. Production processing and approved geometry metadata are unchanged.</p></html>''')
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for name in ('report_joint_main_return.py','joint_path_information.py','nearby_echo_control.py','screen_nearby_sites.py'):
        p=Path('support/isar')/name;shutil.copyfile(p,snapshots/name)
    print('report',out/'index.html','unused change',total,'rank',information['target_rank'],flush=True)


if __name__=='__main__':main()
