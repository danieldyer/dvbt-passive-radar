"""Test a disjoint-time stationary reference in the captured residual stream.

Reference times are fixed at 30--90 s. The 100--130 and 190--220 s controls
measure transfer without using either fitted ship interval. No calibration or
background correction is adopted by this diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from background_reference import profile_means,combine_means,field_reference,identity
from common_physical_pair import CommonPhysicalPair
from compare_ais_phase import NAMES
from extract_full_aperture import load
from field import FS
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from joint_physical_pairs import JointPhysicalPairs
from motion import C
from pair_diagnostics import components
from ship_site_signature import licensed_options


def energy(value):
    return float(np.vdot(value,value).real)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/joint-carrier-cycles/fits.json'))
    parser.add_argument('--aperture',type=Path,default=Path('build/isar-20261015/full-geographic-aperture'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/background-reference'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.source.read_text());group=next(g for g in source['groups'] if g['assignment']=='00002')
    record=min(group['common_trials'],key=lambda r:r['cost']);parameters=np.asarray(record['parameters'])
    case=prepare_problem();labels=['F01','Flash']
    selected=[next(i for i,e in enumerate(case['descriptions']) if e['label']==label) for label in labels]
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a
        for a in assignments(licensed_options(case['geometry_directory']))}
    model=JointPhysicalPairs([CommonPhysicalPair(model_for_assignment(case,i,lookup['00002'])) for i in selected])
    if abs(model.objective(parameters)[0]*model.energy-record['cost'])>1e-7:raise ValueError('Source objective changed')
    periods=[('Reference',30.,90.),('Control 1',100.,130.),('Control 2',190.,220.)]
    windows=[(t,t+5.) for _,start,stop in periods for t in np.arange(start,stop,5.)]
    groups=[np.array([start<=a<stop for a,_ in windows]) for _,start,stop in periods]
    result=dict(scope=__doc__,accepted_calibration=False,correction_applied=False,
        source=str(args.source),source_branch=record['branch_index'],assignment='00002',
        reference_periods=[dict(name=n,first_s=a,last_s=b) for n,a,b in periods],channels=[],
        observation='Original archived C0/C1 residual profiles, already subject to capture-time frozen stationary subtraction. No independently replayed IQ profiles are mixed into this analysis.',
        interpretation='Diagnostic transfer with no ship-model refit. Difference from a local fitted background may contain target model error or overfitting; it is not by itself a background error estimate. Block variation is empirical scene variation plus finite-sample noise, not a calibrated covariance.')
    sources=[args.source,Path(__file__),Path('support/isar/background_reference.py')]
    profiles=[]
    for ch,name in enumerate(NAMES):
        folder=args.aperture/f'channel-{ch}';aperture=load(folder);meta=identity(aperture)
        means=profile_means(aperture,windows)
        aggregate=[combine_means([m for m,keep in zip(means,g) if keep]) for g in groups]
        reference=aggregate[0];first=meta['first_range_sample']
        profiles.append((np.arange(reference.shape[1])+first)*C/FS/1000.)
        np.savez(out/f'channel-{ch}-means.npz',means=np.stack([r['mean'] for r in means]),
            intervals_s=np.asarray(windows),information=[r['information'] for r in means],
            aggregates=np.stack(aggregate),first_range_sample=first)
        channel=dict(channel=name,identity=meta,
            blocks=[{k:v for k,v in m.items() if k!='mean'} for m in means],intervals=[])
        for label,arc,index,ci in zip(labels,model.arcs,model.indices,selected):
            mapped,_=arc.mapping(parameters[index]);independent=arc.independent
            pair=independent.pairs[ch];p=mapped[independent.indices[ch]];b=pair.base;field=b.field
            target=components(pair,p);predicted=target.sum(axis=0)
            beta=b.nuisance.solve(field.y.ravel()[b.selected]-predicted[b.selected])
            local=(b.design@beta).reshape(field.y.shape)
            transferred=[field_reference(field,m,first) for m in aggregate]
            block_values=np.stack([field_reference(field,m['mean'],first) for m in means])
            training=b.training;testing=case['testing'][ci][ch]
            target=target.reshape(2,*field.y.shape);primary_energy=energy(target[0,testing])
            error_without_refit={}
            for name_static,static in [('Local fit',local),*zip([p[0] for p in periods],transferred),('Zero',np.zeros_like(local))]:
                residual=field.y-predicted.reshape(field.y.shape)-static
                error_without_refit[name_static]=dict(training=energy(residual[training]),unused=energy(residual[testing]))
            entry=dict(interval=label,primary_unused_energy=primary_energy,
                local_background_unused_energy=energy(local[testing]),
                reference_unused_energy=energy(transferred[0][testing]),
                local_minus_reference_unused_energy=energy((local-transferred[0])[testing]),
                control_minus_reference_unused_energy=[energy((v-transferred[0])[testing]) for v in transferred[1:]],
                block_minus_reference_unused_energy=[energy((v-transferred[0])[testing]) for v in block_values],
                error_without_refit=error_without_refit,
                full_field_unused_energy=energy(field.y[testing]))
            channel['intervals'].append(entry)
            print(name,label,'external/local/primary energy',np.round([entry['reference_unused_energy'],entry['local_background_unused_energy'],primary_energy],3),
                  'unused error local/reference',*[round(error_without_refit[n]['unused'],3) for n in ('Local fit','Reference')],flush=True)
        result['channels'].append(channel)
        sources += [folder/'metadata.json',folder/'coordinates.npz',Path(f'build/isar-20261015/full-fields/channel-{ch}.npz')]
    fig,axes=plt.subplots(5,2,figsize=(14,15),layout='constrained')
    colours=['#24699b','#c47721','#518b43']
    for ch,name in enumerate(NAMES):
        with np.load(out/f'channel-{ch}-means.npz') as data:mean=data['means'];aggregates=data['aggregates']
        ranges=profiles[ch];use=(ranges>=14.2)&(ranges<=15.3)
        for i,period in enumerate(periods):
            axes[ch,0].plot(ranges[use],abs(aggregates[i,0,use]),label=period[0],color=colours[i])
        peak=np.flatnonzero(use)[np.argmax(abs(aggregates[0,0,use]))]
        for i,(mask,period) in enumerate(zip(groups,periods)):
            times=np.asarray(windows).mean(axis=1)[mask]
            axes[ch,1].plot(times,mean[mask,0,peak].real,'o-',color=colours[i],label=f'{period[0]} real')
            axes[ch,1].plot(times,mean[mask,0,peak].imag,'x--',color=colours[i],alpha=.8,label=f'{period[0]} imaginary')
        axes[ch,0].set(title=f'{name}: stationary residual C0 mean',ylabel='Original complex-profile magnitude')
        axes[ch,1].set(title=f'{name}: fixed {ranges[peak]:.3f} km cell',ylabel='Original real / imaginary value')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=7,ncol=2)
    axes[-1,0].set_xlabel('Excess path (km)');axes[-1,1].set_xlabel('Capture time (s)')
    fig.suptitle('Independent residual-background transfer; no phase alignment or magnitude normalization',fontsize=13)
    fig.savefig(out/'01-background-transfer.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    rows=[]
    for i,label in enumerate(labels):
        entries=[next(e for e in c['intervals'] if e['interval']==label) for c in result['channels']]
        for key,title in [('local_background_unused_energy','Local fitted background'),('reference_unused_energy','Independent reference'),
                          ('local_minus_reference_unused_energy','Difference: local − reference')]:
            axes[i,0].plot(NAMES,[10*np.log10(e[key]/e['primary_unused_energy']) for e in entries],'o-',label=title)
        axes[i,0].axhline(0,color='grey',lw=.8);axes[i,0].set(title=label,ylabel='Energy / fitted primary energy (dB)');axes[i,0].legend(fontsize=8)
        for name_static in ('Reference','Control 1','Control 2','Zero'):
            change=[100*(e['error_without_refit'][name_static]['unused']/e['error_without_refit']['Local fit']['unused']-1) for e in entries]
            axes[i,1].plot(NAMES,change,'o-',label=name_static)
        axes[i,1].axhline(0,color='grey',lw=.8);axes[i,1].set(title=label,ylabel='Unused error change vs local fit (%)');axes[i,1].legend(fontsize=8)
        for ch,e in zip(NAMES,entries):
            base=e['error_without_refit']['Local fit']['unused']
            rows.append(f"<tr><td>{label}</td><td>{ch}</td><td>{e['local_background_unused_energy']/e['primary_unused_energy']:.3g}</td><td>{e['reference_unused_energy']/e['primary_unused_energy']:.3g}</td><td>{100*(e['error_without_refit']['Reference']['unused']/base-1):+.3f}%</td></tr>")
    fig.suptitle('Background replacement with the previous ship fit frozen\nDiagnostic only: a proper alternative fit must estimate target motion and brightness again',fontsize=12)
    fig.savefig(out/'02-ship-reference-comparison.png',dpi=170);plt.close(fig)
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sources}
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Independent stationary reference</title>
<style>body{font:16px/1.55 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}img{width:100%}td,th{border:1px solid #ddd;padding:9px}table{border-collapse:collapse}</style>
<h1>Independent stationary residual reference</h1>
<p>The reference uses 30–90 seconds of the original archived C0/C1 stream. Separate controls use 100–130 and 190–220 seconds. These times exclude both fitted ship intervals and were declared before inspecting their background estimates. No complex phase alignment, fitted normalization or new production correction is applied. The archive already contains capture-time frozen stationary subtraction: this estimates the remaining stationary residual, not the original total clutter.</p>
<img src="01-background-transfer.png" alt="Complex stationary residual across independent periods">
<p>The left column shows the magnitude of each time-averaged C0 profile near the ship range. The right column retains both real and imaginary parts at the strongest reference-mean cell in that range. A 5-second block may contain moving returns; stability of a mean alone does not prove absence of contamination or validate its covariance.</p>
<img src="02-ship-reference-comparison.png" alt="Comparison to locally fitted stationary nuisance">
<table><tr><th>Interval</th><th>Band</th><th>Local-background / primary energy</th><th>Reference / primary energy</th><th>Unused error change with reference</th></tr>'''+''.join(rows)+'''</table>
<p>The old target motion and brightness coefficients remain frozen in this comparison. A difference between local and independently learned background can arise from an incorrect target model, local-background overfitting, or a changing physical background. It is not by itself evidence for any one of those mechanisms. Any subsequent alternative fit must propagate reference uncertainty and retain unused-data validation.</p>
<p><a href="analysis.json">Numerical results and hashes</a> · <a href="../clutter-information/index.html">Conditional information bound</a> · <a href="../index.html">Analysis index</a></p></html>''')


if __name__=='__main__':main()
