"""Locate the remaining mismatch after freeing constant band phases."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from amplitude_arc import AmplitudeArc
from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem,build_arcs
from joint_physical_arcs import JointPhysicalArcs
from motion import C
from report_joint_sea_arcs import heldout
from ship_arc_coherence import archived_observables
from spherical_sea import geometry,horizontal_reflection


def sea_phase_observables(case,record,arc_index,step=.02):
    """Carrier phase rate/curvature of the summed rays, not just the direct ray.

    This is a diagnostic of the fitted propagation model. The fitted likelihood
    itself retains its analytic original-row range/Doppler derivatives.
    """
    g=case['geometries'][arc_index]
    p,v,a,tx,rx=g.state(np.asarray(record['prediction'][arc_index]['parameters'])[:28])
    dt=step*np.arange(-2,3)
    position=g.receiver+(p+dt[:,None]*v+.5*dt[:,None]**2*a)@g.rotation
    velocity=(v+dt[:,None]*a)@g.rotation
    result=[]
    for ch,f in enumerate(case['fields'][arc_index]):
        paths=geometry(position,velocity,g.receiver+tx[ch]@g.rotation,g.receiver+rx@g.rotation,
            np.asarray(case['water']['centre_ecef_m']),case['water']['radius_m'])
        length=paths['excess_path_m'];reflected=paths['reflected_legs'][0]
        gamma=horizontal_reflection(reflected['sin_grazing'],f.metadata['rf_hz'],relative_permittivity=80.,conductivity_s_per_m=4.)
        field=np.exp(-1j*f.k*(length[0]-length[0,2]))*(1+gamma*paths['relative_spreading'][1]*np.exp(-1j*f.k*(length[1]-length[0])))
        phase=np.angle(field*np.conj(field[2]))
        if abs(phase).max()>1:raise ValueError('Diagnostic derivative spans too much phase')
        coefficients=np.polynomial.polynomial.polyfit(dt,phase,4)
        result.append([-coefficients[1]/f.k,-2*coefficients[2]/f.k])
    return np.asarray(result)


def independent_paths(case):
    """Reproduce the existing fits on exactly the same fields and masks."""
    root=Path('docs/data/isar_20261017')
    continuity=json.loads((root/'ship-phase-continuity/fits.json').read_text())
    flash=next(r for r in continuity['intervals'] if r['first_s']==163 and r['last_s']==166)
    candidates=json.loads((root/'full-capture-ship-arc-validation/fits.json').read_text())['candidates']
    result=[]
    for description,fields,training,testing,knots in zip(case['descriptions'],case['fields'],case['training'],case['testing'],case['knots']):
        rows=[]
        for ch,(f,tr,te,k) in enumerate(zip(fields,training,testing,knots)):
            if description['label']=='Flash':
                channel=flash['channels'][ch];saved=channel['alternatives'][channel['chosen_index']]
            else:saved=next(c for c in candidates if c['label']==description['label'])['channels'][ch]
            model=AmplitudeArc(f,tr,k);p=np.asarray(saved['parameters'])
            value,_,design,amplitude,_,_=model.evaluate(p,True)
            if abs(value-saved['objective'])>1e-11:raise ValueError('Independent point fit did not reproduce')
            midpoint=(description['first_s']+description['last_s'])/2;u=(midpoint-model.epoch)/model.scale
            transform=np.array([[C/8e6,0.,u,u*u],[0.,0.,1/model.scale,2*u/model.scale],[0.,0.,0.,2/model.scale**2]])
            reference=archived_observables(f.data,midpoint)
            observables=reference+transform@p
            covariance=transform@np.asarray(saved['conditional_parameter_covariance'])@transform.T
            prediction=design@amplitude;y=f.y.ravel()
            static=model.nuisance.solve(y[model.selected]-prediction[model.selected]);use=np.repeat(te,len(f.offsets))
            error=float(np.sum(abs(y[use]-(prediction+model.design@static)[use])**2))
            rows.append(dict(channel=ch,training_error=value*model.energy,heldout_error=error,
                observables=observables.tolist(),covariance=covariance.tolist(),range_bound_hit=bool(abs(p[0]-description['centre_sample'])>.7999),
                gradient=saved['maximum_gradient'],parameters=p.tolist(),information_rank=saved['information_rank']))
        result.append(dict(label=description['label'],channels=rows))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/joint-ship-band-phase-control'))
    args=parser.parse_args();root=args.output.parent
    paths=[root/'joint-ship-sea-reference-phase/fits.json',args.output/'fits.json']
    common,control=[json.loads(p.read_text()) for p in paths]
    if not common['complete'] or not control['complete']:raise ValueError('Complete declared comparisons first')
    log=(args.output/'tests.log').read_text();tests=re.search(r'Ran (\d+) tests',log)
    if not tests or not log.rstrip().endswith('OK'):raise ValueError('Passing test log required')
    case=prepare_problem();free=independent_paths(case)
    comparisons=[]
    for source,row in [('common',r) for r in common['trials']]+[('independent',r) for r in control['trials']]:
        comparisons.append(dict(phase_model=source,family=row['family'],seed_cycle=row['seed_cycle'],cost=row['cost'],
            heldout_error=heldout(row),converged=row['audit']['converged'],kkt=row['audit']['polishing']['final_kkt_residual'],bounds=row['audit']['bound_indices']))
    best={family:min([r for r in control['trials'] if r['family']==family and r['audit']['converged']],key=lambda r:r['cost']) for family in ('direct','transmit')}
    summary=dict(scope=__doc__,accepted_calibration=False,new_focused_videos=False,tests_passed=int(tests[1]),
        comparisons=comparisons,independent_paths=free,reproduction=[],phase_rates=[],
        validation_scope='Same original complex observations, training masks and held-out blocks. No arbitrary time-dependent band phase or noise renormalization.')
    for row in control['trials']:
        joint=JointPhysicalArcs([a.independent for a in build_arcs(case,row['family'],3)])
        cost=joint.objective(np.asarray(row['parameters']))[0]*joint.energy
        if abs(cost-row['cost'])>1e-7:raise ValueError('Joint control objective failed reproduction')
        summary['reproduction'].append(dict(family=row['family'],seed_cycle=row['seed_cycle'],cost_difference=cost-row['cost']))
    for i in range(3):
        first=sea_phase_observables(case,best['transmit'],i)
        second=sea_phase_observables(case,best['transmit'],i,.01)
        if abs(first-second).max()>1e-4:raise ValueError('Sea phase diagnostic derivative is not resolved')
        summary['phase_rates'].append(dict(label=case['descriptions'][i]['label'],sea_effective_rate_curvature=first.tolist(),
            derivative_step_disagreement=float(abs(first-second).max())))
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    labels=[]
    for i,row in enumerate(comparisons):
        labels.append(f"{row['phase_model']} phase · {row['family']} · start {row['seed_cycle']}")
        style=dict(marker='o' if row['converged'] else 'x',color='#24699b' if row['phase_model']=='common' else '#bd6b22')
        axes[0].plot(row['cost'],i,**style);axes[1].plot(row['heldout_error'],i,**style)
    for ax in axes:ax.set_yticks(range(8));ax.invert_yaxis()
    axes[0].set_yticklabels(labels,fontsize=9);axes[1].set_yticklabels([])
    axes[0].set(xlabel='Training squared error + identical priors',title='Independent phase is a nested relaxation')
    axes[1].set(xlabel='Held-out squared prediction error',title='Constant band phases; no rate corrections')
    fig.suptitle('Both independent-phase starts converge to the same solution for each propagation model\nCircle: strict local minimum · cross: incomplete common-phase search',fontsize=12)
    fig.savefig(args.output/'01-phase-control.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for family,row in best.items():
        values=np.r_[row['antenna_fit']['transmitter_offsets_enu_m'],[row['antenna_fit']['receiver_offset_enu_m']]]
        for j,ax in enumerate(axes):ax.plot(range(6),values[:,j],'o-',label=family)
    for ax,label in zip(axes,['East','North','Up']):
        ax.axhline(0,color='#aaa',lw=.7);ax.set_xticks(range(6),[*NAMES,'RX'],rotation=35)
        ax.set(ylabel=f'{label} offset from user prior (m)',title=label)
    axes[0].legend();fig.suptitle('Releasing constant phases changes the antenna solution materially\nThe remaining shifts and motion-bound contacts still prevent accepting a calibration',fontsize=12)
    fig.savefig(args.output/'02-antenna-control.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for i,(description,g,point) in enumerate(zip(case['descriptions'],case['geometries'],free)):
        measurements=np.asarray([r['observables'] for r in point['channels']])
        sd=np.sqrt(np.asarray([np.diag(r['covariance']) for r in point['channels']]))
        nominal=g.prediction(np.zeros(28))
        for j,ax in enumerate(axes[i]):
            shift=nominal[:,0] if j==0 else np.zeros(5)
            ax.errorbar(range(5),measurements[:,j]-shift,yerr=sd[:,j],fmt='o',color='black',capsize=3,label='Independent path fits ±1 conditional σ')
            direct=np.asarray(best['direct']['prediction'][i]['observables'])[:,j]-shift
            ax.plot(range(5),direct,'s-',label='Shared physical motion / free band phases',color='#24699b')
            if j>0:ax.plot(range(5),np.asarray(summary['phase_rates'][i]['sea_effective_rate_curvature'])[:,j-1],'^--',label='Sea model effective carrier trajectory',color='#bd6b22')
            ax.set_xticks(range(5),NAMES,rotation=30)
            ax.set_ylabel(['Envelope path relative to nominal AIS (m)','Excess-path rate (m/s)','Excess-path curvature (m/s²)'][j])
            ax.set_title(f"{description['label']} · {description['first_s']:g}–{description['last_s']:g} s")
    axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=7)
    fig.suptitle('Remaining disagreement includes phase curvature, which a constant phase cannot fix\nConditional thermal error bars exclude target-model error; no Gaussian-significance claim',fontsize=12)
    fig.savefig(args.output/'03-motion-consistency.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for family,row in best.items():
        for state in row['target_states']:
            p=np.asarray(state['position_correction_enu_m']);v=np.asarray(state['velocity_correction_enu_mps'])
            axes[0].plot(p[0],p[1],'o' if family=='direct' else '^',label=family if state['label']=='Flash' else None)
            axes[0].annotate(state['label'],p[:2],xytext=(4,4),textcoords='offset points',fontsize=9)
            axes[1].plot(v[0],v[1],'o' if family=='direct' else '^')
            axes[1].annotate(state['label'],v[:2],xytext=(4,4),textcoords='offset points',fontsize=9)
    for ax in axes:ax.axhline(0,color='#aaa',lw=.7);ax.axvline(0,color='#aaa',lw=.7);ax.set_aspect('equal',adjustable='datalim')
    axes[0].set(xlabel='East correction to time-shifted AIS (m)',ylabel='North correction (m)',title='Apparent point position');axes[0].legend()
    axes[1].set(xlabel='East velocity correction (m/s)',ylabel='North velocity correction (m/s)',title='Apparent point velocity')
    fig.suptitle('These fitted points cannot yet be identified with fixed ship structures\nThe flash position reaches −450 m north and velocity reaches +1.5 m/s north',fontsize=12)
    fig.savefig(args.output/'04-target-state-control.png',dpi=170);plt.close(fig)
    table=''.join(f'<tr><td>{html.escape(r["phase_model"])}</td><td>{r["family"]}</td><td>{r["seed_cycle"]}</td><td>{r["cost"]:.3f}</td><td>{r["heldout_error"]:.3f}</td><td>{r["kkt"]:.3g}</td><td>{"yes" if r["converged"] else "no"}</td></tr>' for r in comparisons)
    gains={}
    for family in ('direct','transmit'):
        previous=min([r for r in common['trials'] if r['family']==family and r['audit']['converged']],key=lambda r:r['cost'])
        gains[family]=100*(heldout(previous)-heldout(best[family]))/heldout(previous)
    summary['constant_phase_heldout_error_reduction_percent']=gains
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Shared ship motion: constant band-phase control</title>
<style>body{{font:16px/1.55 system-ui;max-width:1220px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}td,th{{padding:8px;border:1px solid #ccd4d9;text-align:left}}table{{border-collapse:collapse}}.note{{background:#fff1d3;padding:16px}}</style>
<h1>Shared ship motion after releasing constant band phases</h1>
<p class="note"><b>The common scattering-phase assumption was part of the problem, but not all of it.</b> All four independent-phase searches converge, and both starts reach the same solution for each propagation model. Held-out squared error improves by {gains['direct']:.2f}% for direct propagation and {gains['transmit']:.2f}% for the sea model compared with their best converged common-phase counterparts. The fitted flash nevertheless lies about 530 m from its time-shifted AIS reference and reaches the permitted north-velocity correction. These are not accepted antenna or ship-motion estimates.</p>
<p>This is a nested control on the same three intervals and raw observations as the <a href="../joint-ship-sea-review/index.html">preceding joint sea-path experiment</a>. Each interval still has one physical Cartesian trajectory shared across all bands. The antenna coordinates are shared across intervals, and their prior is counted once. The only additional freedom is four independent constant band phases per interval. No independent phase slope, acceleration, timing resampling or empirical per-band noise scale has been added.</p>
<img src="01-phase-control.png" alt="Nested common versus independent phase comparisons">
<table><tr><th>Phase model</th><th>Propagation</th><th>Start</th><th>Training objective</th><th>Held-out error</th><th>KKT residual</th><th>Converged</th></tr>{table}</table>
<h2>What changed in the location fit</h2>
<img src="02-antenna-control.png" alt="Antenna offsets with independent constant band phases">
<p>The direct control no longer drives the receiver to ground level, and the sea control places it about 1.4 m above its specified prior. However, the sea control also moves the Ornata antennas downward by roughly 77–143 m and ABC upward by roughly 98 m. Agreement between numerical starts establishes a repeatable local minimum, not a correct physical calibration. Assignment 00002 alone is tested here; this is not a global site search.</p>
<h2>The remaining mismatch is visible in measured phase curvature</h2>
<img src="03-motion-consistency.png" alt="Independent envelope delay, path rate and curvature against the shared motion models">
<p>The black points reuse the existing independently fitted quadratic paths, with the exact same amplitude knots and training observations. Every stored likelihood was reconstructed exactly before comparison. For the flash, the four non-ABC bands have fitted curvature around 0.07–0.17 m/s², while ABC has about 0.59 m/s². A shared point trajectory predicts almost equal curvature for these closely spaced viewing directions. Releasing constant phases cannot remove this discrepancy. In the earlier F01 interval, the independent Nine fit favours a different phase trajectory from Ten and ABC; the mismatch is not confined to ABC. That suggests testing overlapping components, but does not yet identify distinct physical scatterers.</p>
<p>The sea curves show the phase rate and curvature of the coherently summed rays, including reflection phase and spreading, rather than just the geometric direct ray. Their numerical time derivatives were checked by halving the step; disagreements are below 10⁻⁴ m/s or m/s². Error bars are conditional thermal uncertainties, not calibrated confidence intervals in the presence of multiple unresolved scatterers. The independent path fits are diagnostics and are not being substituted for a physical five-band motion model.</p>
<img src="04-target-state-control.png" alt="Fitted target position and velocity relative to AIS">
<p>The AIS offset is a nuisance quantity, but a roughly 530 m displacement is too large to identify confidently with a fixed feature on this approximately 287 m ship. This supports revisiting the selected scatterer/model rather than enlarging antenna bounds or absorbing the discrepancy with arbitrary phase curves. It does not, by itself, distinguish unresolved ship structure from a changing specular contribution, propagation effects or another target.</p>
<h2>Validation</h2>
<p>{summary['tests_passed']} numerical tests pass. Added tests verify the independent-phase residual Jacobian, exact nesting of the physical model, one shared antenna prior, constant rather than time-varying phase freedom, and that releasing phases cannot increase the data information about antenna positions. All four saved control objectives and all fifteen independent point objectives were rebuilt. Test blocks never refit the moving or stationary coefficients.</p>
<p><a href="analysis.json">Derived quantities and source hashes</a> · <a href="fits.json">All four fits</a> · <a href="tests.log">Test log</a>. The goal remains active; this result is not adopted in production metadata, and no replacement focused video has been claimed.</p></html>'''
    (args.output/'index.html').write_text(page)
    inputs=[*paths,Path(__file__),args.output/'tests.log',root/'ship-phase-continuity/fits.json',root/'full-capture-ship-arc-validation/fits.json']
    summary['sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in inputs}
    (args.output/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',args.output/'index.html','held-out improvement',gains,flush=True)


if __name__=='__main__':main()
