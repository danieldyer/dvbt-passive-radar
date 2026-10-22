"""Show a paired physical-motion fit on a separate ship interval."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem
from fit_ship_physical_pair import build_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical',type=Path,default=Path('docs/data/isar_20261017/ship-physical-pair-Flash/fits.json'))
    parser.add_argument('--independent',type=Path,default=Path('docs/data/isar_20261017/ship-amplitude-pair-Flash/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-physical-pair-Flash-review'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.physical.read_text());independent=json.loads(args.independent.read_text())
    audit_path=args.physical.with_name('stationarity.json');audit=json.loads(audit_path.read_text())
    if not all(x['complete'] for x in (source,independent,audit)):raise ValueError('All declared fits and audits must finish')
    best=min(audit['trials'],key=lambda r:r['cost']);parameters=np.asarray(best['parameters'])
    case=prepare_problem();label=source['interval']['label']
    model,i=build_model(case,source['secondary_origin_bounds_samples'],label);g=model.geometry
    cost=model.objective(parameters)[0]*model.energy
    if abs(cost-best['cost'])>1e-7:raise ValueError('Saved physical fit did not reproduce')
    one=independent['single_path']['channels'];two=[c['trials'][c['best_index']] for c in independent['channels']]
    errors=np.array([[r['heldout_error'] for r in one],[r['prediction']['heldout_error'] for r in two],
        [r['heldout_error'] for r in best['prediction']]])
    primary=g.prediction(parameters[:g.count]);nominal=g.prediction(np.zeros(g.count))
    summary=dict(scope=__doc__,interval=source['interval'],accepted_calibration=False,selected_start=best['start_rank'],
        heldout_errors=errors.tolist(),primary_observables=primary.tolist(),cost_reproduction_error=cost-best['cost'],
        primary_component_energy_fraction=[],component_to_sum_energy_ratio=[],
        position_correction_enu_m=(parameters[18:21]*g.nuisance_sigma[:3]).tolist(),
        velocity_correction_enu_mps=(parameters[21:24]*g.nuisance_sigma[3:6]).tolist(),
        acceleration_correction_enu_mps2=(parameters[24:27]*g.nuisance_sigma[6:9]).tolist(),
        antenna_fit=g.antennas.describe(parameters[:18]))
    fig,axes=plt.subplots(1,4,figsize=(18,5),layout='constrained')
    for j,colour,name in zip(range(3),['#bd6b22','#929a9f','#24699b'],['Independent single echo','Independent pair','Shared physical primary + secondary']):
        axes[0].plot(NAMES,errors[j],'o-',color=colour,label=name)
    axes[0].set(ylabel='Unused-data squared prediction error',title='Identical unused samples');axes[0].legend(fontsize=7)
    for ch in range(5):
        for j,ax in enumerate(axes[1:]):
            shift=nominal[ch,0] if j==0 else 0
            ax.plot(ch,one[ch]['observables'][j]-shift,'D',color='#bd6b22',ms=4)
            ax.scatter([ch-.1,ch+.1],np.array(two[ch]['observables'])[:,j]-shift,color='#929a9f',s=25)
    for j,ax in enumerate(axes[1:]):
        shift=nominal[:,0] if j==0 else 0
        ax.plot(range(5),primary[:,j]-shift,'o-',color='#24699b');ax.set_xticks(range(5),NAMES)
        ax.set_ylabel(['Excess path relative to nominal AIS (m)','Excess-path rate (m/s)','Excess-path curvature (m/s²)'][j])
    for ax in axes:ax.tick_params(axis='x',labelrotation=35)
    fig.suptitle(f'{label} · {source["interval"]["first_s"]:g}–{source["interval"]["last_s"]:g} s · same operator, separate fits\nColours: independent one echo / independent two echoes / shared physical primary',fontsize=12)
    fig.savefig(args.output/'01-motion-and-prediction.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(14,13),layout='constrained')
    for ch,(pair,index,test) in enumerate(zip(model.pairs,model.indices,case['testing'][i])):
        b=pair.base;value=pair.evaluate(parameters[index]);k=len(b.knots)
        components=np.stack([value['design'][:,j*k:(j+1)*k]@value['amplitude'][j*k:(j+1)*k] for j in range(2)])
        energy=np.sum(abs(components[:,b.selected])**2,axis=1)
        summary['primary_component_energy_fraction'].append(float(energy[0]/energy.sum()))
        summary['component_to_sum_energy_ratio'].append(float(energy.sum()/np.sum(abs(components[:,b.selected].sum(axis=0))**2)))
        beta=b.nuisance.solve(b.field.y.ravel()[b.selected]-components[:,b.selected].sum(axis=0))
        clean=(b.field.y.ravel()-b.design@beta).reshape(b.field.y.shape)
        first,second=components.reshape(2,*b.field.y.shape);valid=test&(np.sum(abs(first)**2,axis=1)>0)
        for ax,remaining,title in zip(axes[ch],(clean,clean-second),('Before secondary subtraction','After secondary subtraction')):
            phase=np.angle(np.sum(first.conj()*remaining,axis=1));ax.plot(b.field.time[valid],np.degrees(phase[valid]),'.',ms=4)
            ax.axhline(0,color='#aaa',lw=.6);ax.set_ylim(-180,180);ax.set_ylabel(NAMES[ch]+'\nPhase residual (deg)')
            if ch==0:ax.set_title(title)
            if ch==4:ax.set_xlabel('Capture time (s)')
    fig.suptitle('Unused samples · no fitted test-time correction · phase undefined at zero predicted primary strength',fontsize=12)
    fig.savefig(args.output/'02-heldout-phase.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{NAMES[ch]}</td><td>{errors[0,ch]:.2f}</td><td>{errors[1,ch]:.2f}</td><td>{errors[2,ch]:.2f}</td><td>{summary["primary_component_energy_fraction"][ch]:.3f}</td></tr>' for ch in range(5))
    audits=''.join(f'<tr><td>{r["start_rank"]}</td><td>{r["cost"]:.2f}</td><td>{sum(p["heldout_error"] for p in r["prediction"]):.2f}</td><td>{r["audit"].get("final_kkt_residual",float("nan")):.3g}</td><td>{r["audit"]["strict_local_minimum"]}</td></tr>' for r in audit['trials'])
    bounds=np.flatnonzero((parameters<model.lower+1e-4)|(parameters>model.upper-1e-4)).tolist();summary['bound_indices']=bounds
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{label}: overlapping-echo physical fit</title>
<style>body{{font:16px/1.55 system-ui;max-width:1250px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:8px}}.note{{background:#fff1d3;padding:16px}}</style>
<h1>{label}: the same overlapping-echo model on a separate interval</h1>
<p class="note">This is a model-transfer test, not independent calibration validation: the physical coordinates and motion have been refitted on this interval's training samples. The site assignment remains 00002. No antenna or motion estimate is adopted. The selected solution contacts parameter bounds {bounds}; all other starts and convergence failures remain reported below.</p>
<p>One common Cartesian primary trajectory is combined with one independent quadratic secondary echo per band. Each component has nonnegative original-row brightness and a constant phase. Both pass through the actual original-symbol response before averaging/whitening; stationary coefficients are fitted jointly. All moving and stationary coefficients are frozen on unused samples. This is the same architecture tested in <a href="../ship-physical-pair-review/index.html">F01</a>.</p>
<img src="01-motion-and-prediction.png" alt="Physical phase trajectory compared with independent paths and held-out prediction errors">
<table><tr><th>Band</th><th>Single-echo error</th><th>Independent-pair error</th><th>Shared-physical error</th><th>Primary fraction of component energies</th></tr>{rows}</table>
<p>The energy fraction divides the primary component's energy by the sum of both component energies, before coherent interference. It is a decomposition diagnostic, not a target probability. Strong cancellation or a weak primary can make a common physical trajectory appear compatible without proving that one material point is observed in every band. Independent pairs are local fits with a different search history; their errors are not an oracle bound.</p>
<img src="02-heldout-phase.png" alt="Unused-sample phase residuals before and after subtracting the frozen secondary echo">
<p>These plots are conditional on the estimated two-echo decomposition. Flat residual phase after subtraction alone does not establish a ship-mounted point or a valid sea-path mechanism. The actual total complex prediction errors, parameter bounds, physical states and comparison with other intervals remain necessary.</p>
<table><tr><th>Start</th><th>Training objective</th><th>Unused-data error</th><th>Scaled KKT residual</th><th>Strict local minimum</th></tr>{audits}</table>
<p><a href="analysis.json">Derived results and provenance</a> · <a href="../{args.physical.parent.name}/fits.json">All physical fits</a> · <a href="../{args.physical.parent.name}/stationarity.json">Actual-curvature audits</a> · <a href="../{args.independent.parent.name}/fits.json">Independent pair fits</a>.</p></html>'''
    (args.output/'index.html').write_text(page)
    files=[args.physical,args.independent,audit_path,Path(__file__)]
    summary['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files}
    (args.output/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',args.output/'index.html','heldout totals',errors.sum(axis=1),'primary fractions',summary['primary_component_energy_fraction'],flush=True)


if __name__=='__main__':main()
