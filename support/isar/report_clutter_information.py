"""Known-clutter information bound for the fitted short ship intervals."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh

from clutter_information import pair_information
from common_physical_pair import CommonPhysicalPair
from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from joint_physical_pairs import JointPhysicalPairs
from pair_diagnostics import components
from ship_site_signature import licensed_options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/joint-carrier-cycles/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/clutter-information'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.source.read_text());group=next(g for g in source['groups'] if g['assignment']=='00002')
    record=min(group['common_trials'],key=lambda r:r['cost']);parameters=np.asarray(record['parameters'])
    case=prepare_problem();labels=['F01','Flash']
    selected=[next(i for i,e in enumerate(case['descriptions']) if e['label']==label) for label in labels]
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a
        for a in assignments(licensed_options(case['geometry_directory']))}
    arcs=[model_for_assignment(case,i,lookup['00002']) for i in selected]
    model=JointPhysicalPairs([CommonPhysicalPair(a) for a in arcs])
    if abs(model.objective(parameters)[0]*model.energy-record['cost'])>1e-7:raise ValueError('Source objective changed')
    result=dict(scope=__doc__,accepted_calibration=False,correction_applied=False,
        source=str(args.source),assignment='00002',source_branch=record['branch_index'],rows=[],
        interpretation='Conditional thermal information at a provisional fitted two-echo scene. Both cases profile active real brightness and all secondary trajectory coordinates. Only local stationary complex coefficients differ. Known clutter is an unattained upper bound, not a validated estimator.')
    for label,arc,index in zip(labels,model.arcs,model.indices):
        mapped,_=arc.mapping(parameters[index])
        for pair,pindex,name in zip(arc.independent.pairs,arc.independent.indices,NAMES):
            p=mapped[pindex];known=pair_information(pair,p,estimate_stationary=False)
            local=pair_information(pair,p,estimate_stationary=True)
            if np.linalg.eigvalsh(known).min()<=0 or np.linalg.eigvalsh(local).min()<=0:
                raise ValueError('Primary conditional information is singular; do not report inverse as a covariance')
            retained=eigvalsh(local,known)
            if retained.min()<-1e-7 or retained.max()>1+1e-7:raise ValueError('Information bound violated')
            kc=np.linalg.inv(known);lc=np.linalg.inv(local)
            loss=10*np.log10(np.diag(lc)/np.diag(kc))
            b=pair.base;h=components(pair,p)[0,b.selected];removed=b.nuisance.remove(h)
            energy_loss=-10*np.log10(np.vdot(removed,removed).real/np.vdot(h,h).real)
            result['rows'].append(dict(interval=label,channel=name,known_information=known.tolist(),
                local_information=local.tolist(),known_covariance=kc.tolist(),local_covariance=lc.tolist(),
                marginal_information_loss_db=loss.tolist(),generalized_retained_information=retained.tolist(),
                primary_energy_projection_loss_db=float(energy_loss),
                known_phase_bound_deg=float(np.degrees(np.sqrt(kc[1,1]))),
                local_phase_bound_deg=float(np.degrees(np.sqrt(lc[1,1]))),
                stationary_rank=b.nuisance.rank,stationary_columns=len(b.delays)))
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for row,label in enumerate(labels):
        records=[r for r in result['rows'] if r['interval']==label]
        for coordinate,name,colour in ((1,'Midpoint phase','#24699b'),(2,'Path rate','#bd6b22'),(3,'Path curvature','#527e47')):
            axes[row,0].plot(NAMES,[r['marginal_information_loss_db'][coordinate] for r in records],'o-',label=name,color=colour)
        axes[row,0].plot(NAMES,[r['primary_energy_projection_loss_db'] for r in records],'k:',label='Primary template energy only')
        axes[row,0].set(title=label,ylabel='Information loss relative to known clutter (dB)');axes[row,0].legend(fontsize=8)
        axes[row,1].plot(NAMES,[r['local_phase_bound_deg'] for r in records],'o-',label='Clutter estimated inside this interval')
        axes[row,1].plot(NAMES,[r['known_phase_bound_deg'] for r in records],'o-',label='Perfectly known clutter bound')
        axes[row,1].set(title=label,ylabel='Conditional midpoint phase standard deviation (degrees)');axes[row,1].legend(fontsize=8)
    fig.suptitle('Information cost of fitting stationary clutter within each three-second ship arc\nAt the provisional fitted scene; other echo and brightness nuisance included in both cases',fontsize=12)
    fig.savefig(out/'01-clutter-information.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    f=np.geomspace(.001,2,600);t=np.linspace(-1.5,1.5,301)
    retained=1-abs(np.mean(np.exp(2j*np.pi*t[:,None]*f),axis=0))**2
    ax.semilogx(f,-10*np.log10(retained))
    ax.set(xlabel='Constant target Doppler magnitude (Hz)',ylabel='Matched energy loss (dB)',title='Analytic control: a constant unknown clutter coefficient over three seconds')
    fig.savefig(out/'02-near-dc-bound.png',dpi=170);plt.close(fig)
    paths=[args.source,Path(__file__),Path('support/isar/clutter_information.py'),Path('support/isar/stationary.py'),
        Path('support/isar/amplitude_pair.py'),*set(case['field_paths'])]
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    table=''.join(f"<tr><td>{r['interval']}</td><td>{r['channel']}</td><td>{r['primary_energy_projection_loss_db']:.3f}</td><td>{r['marginal_information_loss_db'][1]:.3f}</td><td>{r['marginal_information_loss_db'][2]:.3f}</td><td>{r['marginal_information_loss_db'][3]:.3f}</td><td>{r['local_phase_bound_deg']:.3f} → {r['known_phase_bound_deg']:.3f}</td></tr>" for r in result['rows'])
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Short-window clutter information</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{border:1px solid #ddd;padding:9px}}table{{border-collapse:collapse}}</style>
<h1>Information removed by local stationary-clutter estimation</h1>
<p>No processing correction or new calibration is adopted. These calculations compare the same provisional primary/secondary echoes with perfectly known clutter versus stationary coefficients estimated independently in each three-second interval. Active real brightness coefficients, secondary phase and secondary motion are nuisance in both cases. The four primary coordinates are midpoint radial position, phase, radial velocity and acceleration. Covariance diagonals also account for uncertainty in the other three primary coordinates.</p>
<img src="01-clutter-information.png" alt="Phase and motion information lost to local clutter estimation">
<table><tr><th>Interval</th><th>Band</th><th>Template-energy loss (dB)</th><th>Phase-information loss (dB)</th><th>Rate-information loss (dB)</th><th>Curvature-information loss (dB)</th><th>Phase bound: local → known (degrees)</th></tr>{table}</table>
<p>The known-clutter comparison is an upper bound. It is not a demonstrated gain from an estimator: independent background estimates have noise, interference, scene evolution and possible target contamination. Those effects must be measured on disjoint times before using such estimates. The thermal covariance is conditional; these phase bounds are not calibrated error bars on the physical ship or antenna positions.</p>
<h2>Why near-zero Doppler is special</h2><p>For a constant-amplitude tone h with an unknown complex constant background, projecting out that background leaves h − mean(h). The fraction of matched energy retained is 1 − |mean(exp(j2πft))|². The numerical test verifies this exact finite-aperture identity, reproduces the existing profiled derivative, and checks that adding an unknown clutter nuisance cannot increase Fisher information. A perfectly stationary target cannot be separated from an unrestricted stationary background using that interval alone. No Doppler notch is involved.</p>
<img src="02-near-dc-bound.png" alt="Analytic near-zero-Doppler information loss">
<p><a href="analysis.json">Numerical matrices and source hashes</a> · <a href="../index.html">Analysis index</a></p></html>'''
    (out/'index.html').write_text(page);(out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for r in result['rows']:
        print(r['interval'],r['channel'],'energy dB',round(r['primary_energy_projection_loss_db'],3),
            'phase/rate/curvature loss',np.round(r['marginal_information_loss_db'][1:],3),flush=True)


if __name__=='__main__':main()
