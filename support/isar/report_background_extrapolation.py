"""Check a fitted stationary nuisance against surrounding recorded ranges.

The five-cell ship likelihood must not be used to assert that its fitted
stationary coefficients represent a physical background. This diagnostic
freezes every coefficient and predicts all seventeen existing range cells.
No new fit, phase correction, or covariance change is made.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

from cluster_phase import crop
from common_physical_pair import CommonPhysicalPair
from compare_ais_phase import NAMES
from field import Field
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from joint_physical_pairs import JointPhysicalPairs
from pair_diagnostics import components
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_pair import PhysicalNuisancePair
from ship_site_signature import licensed_options


def original_values(field,white):
    return solve_triangular(field.whiten,(white/field.root_information[:,None]).T,lower=True).T


def error_on_ranges(field,raw,columns):
    data=dict(field.data);data['value']=raw[:,columns];data['offset_samples']=field.offsets[columns]
    return float(np.sum(abs(Field(data,field.pulse.samples).y)**2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/joint-carrier-cycles/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/background-extrapolation'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.source.read_text());group=next(g for g in source['groups'] if g['assignment']=='00002')
    record=min(group['common_trials'],key=lambda r:r['cost']);parameters=np.asarray(record['parameters'])
    case=prepare_problem();labels=['F01','Flash']
    selected=[next(i for i,e in enumerate(case['descriptions']) if e['label']==label) for label in labels]
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a
        for a in assignments(licensed_options(case['geometry_directory']))}
    model=JointPhysicalPairs([CommonPhysicalPair(model_for_assignment(case,i,lookup['00002'])) for i in selected])
    if abs(model.objective(parameters)[0]*model.energy-record['cost'])>1e-7:raise ValueError('Source objective changed')
    result=dict(scope=__doc__,source=str(args.source),assignment='00002',source_branch=record['branch_index'],
        accepted_calibration=False,correction_applied=False,intervals=[])
    sources=[args.source,Path(__file__)];fig,axes=plt.subplots(5,2,figsize=(14,15),layout='constrained')
    for col,(label,arc,index,ci) in enumerate(zip(labels,model.arcs,model.indices,selected)):
        mapped,_=arc.mapping(parameters[index]);entry=case['descriptions'][ci]
        interval=dict(description=entry,channels=[]);result['intervals'].append(interval)
        for ch,(pair,pindex,name) in enumerate(zip(arc.independent.pairs,arc.independent.indices,NAMES)):
            p=mapped[pindex];b=pair.base;v=pair.evaluate(p);k=len(b.knots)
            old=components(pair,p).sum(axis=0)
            beta=b.nuisance.solve(b.field.y.ravel()[b.selected]-old[b.selected])
            path=Path(f'build/isar-20261015/full-fields/channel-{ch}.npz');sources.append(path)
            with np.load(path) as data:full=crop(data,entry['first_s'],entry['last_s'],np.arange(-8,9))
            base=PhysicalAmplitudeArc(full,b.training,b.knots,b.geometry,ch,b.midpoint)
            expanded=PhysicalNuisancePair(base)
            responses=[r for r,_ in expanded.response_blocks(p)]
            design=np.concatenate([base.reduced(r[:,:,None]*base.amplitude_basis[:,None,:]) for r in responses],axis=1)
            target=(design@v['amplitude']).reshape(full.y.shape)
            stationary=sum((full.stationary(delay)*coefficient for delay,coefficient in zip(b.delays,beta)),np.zeros_like(full.y))
            predicted=original_values(full,target+stationary);static=original_values(full,stationary)
            signal=original_values(full,target);observed=full.data['value']
            inside=np.isin(full.offsets,b.field.offsets);outside=~inside
            # Same fixed operator evaluated on the larger crop must reproduce
            # the original prediction before using any extrapolation evidence.
            old_raw=original_values(b.field,(old+b.design@beta).reshape(b.field.y.shape))
            error=float(np.max(abs(predicted[:,inside]-old_raw)))
            # Large opposing nuisance coefficients amplify float64 roundoff.
            # Retain the absolute discrepancy and an explicit operation-scale
            # bound, rather than judging it against the small cancelled sum.
            operation_scale=float(np.sum(abs(beta))*max(abs(full.pulse.samples))+np.sum(abs(v['amplitude'])))
            roundoff_bound=256*np.finfo(float).eps*operation_scale*np.linalg.cond(full.whiten)*np.linalg.cond(b.field.whiten)
            if error>max(2e-12,roundoff_bound):raise ValueError(f'Expanded operator changed the original prediction: {error} > {roundoff_bound}')
            norms={region:{key:error_on_ranges(full,value,mask) for key,value in
                [('observed',observed),('target',signal),('stationary',static),('prediction',predicted),('error',observed-predicted)]}
                for region,mask in [('inside',inside),('outside',outside)]}
            weights=full.data['information']
            power=lambda value:np.average(abs(value)**2,axis=0,weights=weights)
            curves={name:power(value) for name,value in [('Recorded',observed),('Two echoes',signal),('Fitted background',static),('Complete prediction',predicted)]}
            row=dict(channel=name,stationary_rank=b.nuisance.rank,stationary_columns=b.design.shape[1],
                stationary_condition=float(b.nuisance.singular_values[0]/b.nuisance.singular_values[b.nuisance.rank-1]),
                maximum_coefficient_magnitude=float(abs(beta).max()),reproduction_error=error,
                reproduction_roundoff_bound=roundoff_bound,energies=norms,
                offset_samples=full.offsets.tolist(),average_original_power={key:value.tolist() for key,value in curves.items()})
            interval['channels'].append(row)
            ax=axes[ch,col]
            for key,values in curves.items():ax.semilogy(full.offsets,values,'o-',ms=3,label=key)
            ax.axvspan(b.field.offsets[0]-.5,b.field.offsets[-1]+.5,color='grey',alpha=.1)
            ax.set(title=f'{label}: {name}',ylabel='Mean original-profile power')
            if ch==0:ax.legend(fontsize=8)
            print(label,name,'stationary condition',round(row['stationary_condition'],1),
                'outside prediction/data dB',round(10*np.log10(norms['outside']['prediction']/norms['outside']['observed']),3),flush=True)
    for ax in axes[-1]:ax.set_xlabel('Original field offset (range samples)')
    fig.suptitle('Frozen five-cell fit extrapolated to recorded surrounding ranges\nShading: fitted range cells; all motion, brightness and stationary coefficients frozen',fontsize=13)
    fig.savefig(out/'01-range-extrapolation.png',dpi=150);plt.close(fig)
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in set(sources)}
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    table=''.join(f"<tr><td>{v['description']['label']}</td><td>{r['channel']}</td><td>{r['stationary_rank']}</td><td>{r['stationary_condition']:.3g}</td><td>{10*np.log10(r['energies']['outside']['prediction']/r['energies']['outside']['observed']):+.2f}</td><td>{r['energies']['outside']['error']/r['energies']['outside']['observed']:.3g}</td></tr>" for v in result['intervals'] for r in v['channels'])
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Stationary background extrapolation</title>
<style>body{font:16px/1.55 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}img{width:100%}td,th{border:1px solid #ddd;padding:9px}table{border-collapse:collapse}</style>
<h1>Does the fitted stationary field predict surrounding ranges?</h1>
<p>The original five-cell fit is evaluated on all seventeen previously archived range cells, with every phase, motion, strength and background coefficient frozen. Inside the fitted crop, the larger operator is checked numerically against the original prediction. The plotted values undo only the diagnostic covariance whitening and retain the original measured profile units.</p>
<img src="01-range-extrapolation.png" alt="Recorded and extrapolated complex-field power">
<table><tr><th>Interval</th><th>Band</th><th>Stationary rank</th><th>Retained condition number</th><th>Outside prediction/data power (dB)</th><th>Outside error/data power</th></tr>'''+table+'''</table>
<p>Additional moving scatterers can make an incomplete model underpredict surrounding ranges. A greatly excessive predicted field is different: it demonstrates that the local fitted coefficients cannot be extrapolated as a measured physical background. The outer cells are correlated with the inner cells, so this is a deterministic consistency check, not an independent false-alarm test or a calibrated probability.</p>
<p><a href="analysis.json">Numerical results and source hashes</a> · <a href="../background-reference/index.html">Disjoint-time background comparison</a> · <a href="../index.html">Analysis index</a></p></html>''')


if __name__=='__main__':main()
