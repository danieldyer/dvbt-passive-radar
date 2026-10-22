"""Compare selected-point phase sharing in adjacent flash windows."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES,COLOURS
from fit_shared_echo_cluster import prepare
from report_measured_scatterer_phases import load_fields,serial
from report_shared_echo_cluster import component_phase
from shared_echo_cluster import SharedEchoCluster


def main():
    root=Path('docs/data/isar_20261017');out=root/'adjacent-scatterer-phases';out.mkdir(exist_ok=True)
    geometry=Path('docs/data/calibration_20261016_sites')
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    records=[];reproduction=[];sources=[Path(__file__),Path('support/isar/shared_echo_cluster.py')]
    for epoch,folder,selected_folder in ((163,'shared-echo-cluster-context-163','selected-echo-motion-163'),
            (164,'shared-echo-cluster-context','selected-echo-groups-164'),
            (165,'shared-echo-cluster-context-165','selected-echo-motion-165')):
        source=root/folder/'fits.json';selected_source=root/selected_folder/'fits.json'
        full=json.loads(source.read_text());chosen=json.loads(selected_source.read_text())
        if not full['complete'] or not chosen['complete']:raise ValueError('Wait for completed fits')
        sources += [source,selected_source]
        tests=list(chosen['tests'])
        if epoch==164:
            extra_source=root/'selected-echo-motion-164/fits.json'
            extra=json.loads(extra_source.read_text())
            if not extra['complete']:raise ValueError('Individual flash controls incomplete')
            sources.append(extra_source);tests+=extra['tests']
        bands,partitions=prepare(fields,geometry,*full['interval_s'],range_extent=full['range_extent_samples'])
        independent=[]
        for c,entry in enumerate(full['independent']):
            model=SharedEchoCluster([bands[c]],full['selected_origins_samples'],origin_radius=full['origin_radius_samples'])
            best=entry['trials'][entry['best_index']];p=np.asarray(best['parameters'])
            for fit in entry['trials']:
                r,_=model.evaluate(np.asarray(fit['parameters']))
                reproduction.append(abs(r@r*model.energy-fit['cost'])/fit['cost'])
            independent.append(dict(model=model,parameters=p,fit=best,
                phases=component_phase(model.clusters[0],p[model.indices[0]],partitions[c]==3)))
        all_model=SharedEchoCluster(bands,full['selected_origins_samples'],origin_radius=full['origin_radius_samples'])
        for fit in full['common']:
            r,_=all_model.evaluate(np.asarray(fit['parameters']))
            reproduction.append(abs(r@r*all_model.energy-fit['cost'])/fit['cost'])
        alternatives=[]
        for test in tests:
            selected=[full['labels'].index(v) for v in test['label'].split(',')]
            model=SharedEchoCluster(bands,full['selected_origins_samples'],origin_radius=full['origin_radius_samples'],shared_components=selected)
            for fit in test['trials']:
                r,_=model.evaluate(np.asarray(fit['parameters']))
                reproduction.append(abs(r@r*model.energy-fit['cost'])/fit['cost'])
            best=test['trials'][test['best_index']];p=np.asarray(best['parameters'])
            phase=[component_phase(cluster,p[index],part==3)
                for cluster,index,part in zip(model.clusters,model.indices,partitions)]
            alternatives.append(dict(label=test['label'],model=model,parameters=p,fit=best,phases=phase))
        group=next(a for a in alternatives if a['label']=='A,B,C')
        summaries=[]
        for s,label in enumerate(full['labels'][:3]):
            summaries.append(dict(label=label,
                independent_phase_rms_deg=[r['phases'][s]['test_phase_rms_deg'] for r in independent],
                common_phase_rms_deg=[r[s]['test_phase_rms_deg'] for r in group['phases']],
                common_residual_path_coefficients_m=group['parameters'][group['model'].phase_indices[0,s]]))
        old=sum(f['fit']['prediction']['testing'][0]['heldout_error'] for f in independent)
        all_shared=full['common'][full['best_common_index']]
        records.append(dict(epoch=epoch,source=source,full=full,bands=bands,partitions=partitions,
            independent=independent,group=group,alternatives=alternatives,
            summary=dict(interval_s=full['interval_s'],independent_unused_error=old,
                all_shared_unused_error_ratio=sum(f['heldout_error'] for f in all_shared['prediction']['testing'])/old,
                selected_shared_unused_error_ratio=group['fit']['unused_error_ratio_to_independent'],
                component_to_sum_energy_ratio=group['fit']['component_to_sum_energy_ratio'],
                active_brightness_condition=group['fit']['active_brightness_condition'],
                projected_gradient=group['fit']['maximum_projected_gradient'],bound_indices=group['fit']['bound_indices'],
                components=summaries)))
    if max(reproduction)>1e-9:raise ValueError('Saved cost reproduction failed')
    fig,axes=plt.subplots(3,3,figsize=(18,12),layout='constrained',squeeze=False)
    for col,record in enumerate(records):
        first,last=record['full']['interval_s'];grid=np.linspace(first,last,180);group=record['group']
        for s,label in enumerate(record['full']['labels'][:3]):
            ax=axes[s,col]
            for c,(entry,b,colour) in enumerate(zip(record['independent'],record['bands'],COLOURS)):
                local=entry['parameters'][entry['model'].indices[0]].reshape(4,4)[s]
                u=(grid-(first+last)/2)/b.scale;curve=local[2]*u+local[3]*u*u
                ax.plot(grid,curve*100,color=colour,lw=1.2,label=NAMES[c])
                measurement=entry['phases'][s];t=np.asarray(measurement['time_s']);test=np.asarray(measurement['testing'])
                u=(t-(first+last)/2)/b.scale
                observed=local[2]*u+local[3]*u*u-np.radians(measurement['phase_residual_deg'])/b.field.k
                ax.plot(t[test],observed[test]*100,'.',color=colour,ms=2.3,alpha=.5)
            coefficients=group['parameters'][group['model'].phase_indices[0,s]]
            u=(grid-(first+last)/2)/record['bands'][0].scale
            ax.plot(grid,100*(coefficients[0]*u+coefficients[1]*u*u),'k--',lw=1.6,label='A/B/C shared; D independent')
            ax.set(title=f'{label} · {first:g}–{last:g} s',xlabel='Capture time (s)',ylabel='Residual path relative to midpoint (cm)')
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Adjacent-window phase trajectories after jointly resolving four returns\n'
        'Colour: independent band fits and unused measured circular innovations. Black: A/B/C share paths; D keeps independent band motion.',fontsize=13)
    fig.savefig(out/'01-adjacent-trajectories.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(17,11),layout='constrained',squeeze=False)
    for col,record in enumerate(records):
        first,last=record['full']['interval_s'];grid=np.linspace(first,last,180)
        for s,label in enumerate(record['full']['labels'][:3]):
            ax=axes[s,col];u=(grid-(first+last)/2)/record['bands'][0].scale
            reference=record['independent'][2];rp=reference['parameters'][reference['model'].indices[0]].reshape(4,4)[s,2:]
            for c,(entry,colour) in enumerate(zip(record['independent'],COLOURS)):
                if c==2:continue
                p=entry['parameters'][entry['model'].indices[0]].reshape(4,4)[s,2:]-rp
                ax.plot(grid,100*(p[0]*u+p[1]*u*u),color=colour,label=NAMES[c])
            ax.axhline(0,color='#555',lw=.6)
            ax.set(title=f'{label} · {first:g}–{last:g} s',xlabel='Capture time (s)',ylabel='Independent residual path minus Nine (cm)')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Interband phase-change differences over matching windows\n'
        'Only midpoint constants removed; plotted curves are independently fitted quadratic paths, with nominal geometric interband changes removed.',fontsize=13)
    fig.savefig(out/'02-interband-differences.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,record in zip(axes,records):
        labels=['Independent','All A/B/C/D shared']+[a['label']+' shared' for a in record['alternatives']]
        ratio=[1.,record['summary']['all_shared_unused_error_ratio']]+[a['fit']['unused_error_ratio_to_independent'] for a in record['alternatives']]
        ax.bar(range(len(labels)),100*(np.asarray(ratio)-1),color=['#999','#d55e00']+['#0072b2']*(len(labels)-2))
        ax.axhline(0,color='#555',lw=.6);ax.set_xticks(range(len(labels)),labels,rotation=40,ha='right',fontsize=8)
        ax.set(title=str(record['full']['interval_s'])+' seconds',ylabel='Change in total unused squared prediction error (%)')
    fig.suptitle('Which phase-sharing hypothesis causes the prediction penalty?\n'
        'Other returns stay independently fitted. Total error includes the whole 33-cell field; conditional target phase errors are reported separately.',fontsize=13)
    fig.savefig(out/'03-sharing-controls.png',dpi=170);plt.close(fig)
    summary=dict(accepted_calibration=False,production_changed=False,multipath_model=False,
        windows=[r['summary'] for r in records],reproduced_objectives=len(reproduction),maximum_relative_cost_error=max(reproduction),
        overlapping_window_policy='These overlapping windows are consistency controls, not independent observations for a combined calibration likelihood. A subsequent physical fit must use the original rows once or explicitly retain their cross-window covariance.',
        scope='Internal phase-association controls. Positive brightness, exact C0/C1 response, averaging and range covariance are unchanged. A/B/C names denote local components, not established material points. Conditional measured innovations subtract other frozen training-fitted returns and use each independent polynomial phase branch. Total-field prediction ratios are not bounds on target-only phase loss or location uncertainty.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'analysis.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False))
    rows=''
    for record in records:
        for item in record['summary']['components']:
            rows+=f'<tr><td>{record["full"]["interval_s"]}</td><td>{item["label"]}</td><td>'+', '.join(f'{name}: {value:.1f}' for name,value in zip(NAMES,item['independent_phase_rms_deg']))+'</td><td>'+', '.join(f'{name}: {value:.1f}' for name,value in zip(NAMES,item['common_phase_rms_deg']))+'</td></tr>'
    changes=[100*(r['summary']['selected_shared_unused_error_ratio']-1) for r in records]
    delayed=next(a for a in records[1]['alternatives'] if a['label']=='D')
    delayed_change=100*(delayed['fit']['unused_error_ratio_to_independent']-1)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Adjacent-window multiple-return phases</title>
<style>body{{font:16px/1.55 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:8px;border:1px solid #ddd}}table{{border-collapse:collapse}}.note{{padding:16px;background:#eef4f8}}</style>
<h1>Separate the selected points’ phase tests from the trailing return</h1>
<p class="note">Three overlapping two-second windows now resolve four returns jointly. The new controls share A/B/C trajectories while leaving the trailing D component’s trajectory independent in every band. This tests the main selected features without assuming that every range feature represents the same point across channels.</p>
<p>At 164–166 s, sharing A/B/C changes total unused error by {changes[1]:+.3f}%, while sharing D alone changes it by {delayed_change:+.2f}%. Sharing A/B/C gives {changes[0]:+.3f}% at 163–165 s and {changes[2]:+.3f}% at 165–167 s. Thus the large earlier all-return penalty is primarily the D association. The central B trajectory is the strongest phase measurement; A remains weak and unstable. Small total-field error changes do not prove precise phase agreement for weak components.</p>
<p>No new background, multipath model or arbitrary time-dependent phase correction is present. Range origins and constant phases remain independent per band. Brightness is real nonnegative and fitted on training only. Each quadratic trajectory drives both carrier phase and range migration before exact original-symbol C0/C1 averaging.</p>
<p>The windows overlap. They test local consistency and must not be counted as independent measurements in a combined physical-location fit.</p>
<img src="01-adjacent-trajectories.png" alt="Jointly resolved measured phase changes and independent fits in three adjacent windows">
<p>Each window has its own constant phase gauge. Dots are unused measured circular innovations after subtracting other training-fitted returns, displayed on each independent fit’s phase branch. They do not establish a long phase connection. <a href="../measured-scatterer-phases/index.html">Direct single-cell measurements and their fade flags</a> remain available separately.</p>
<img src="02-interband-differences.png" alt="Independently fitted interband phase-change differences">
<img src="03-sharing-controls.png" alt="Prediction penalty for different sets of shared trajectories">
<p>The total-field prediction error can be dominated by stronger nuisance returns. A small percentage change alone is insufficient to establish a precise target phase. The conditional unused target phase RMS values below are therefore shown explicitly; they do not include model mismatch or uncertain physical associations.</p>
<table><tr><th>Window (s)</th><th>Component</th><th>Independent phase RMS (degrees)</th><th>A/B/C shared phase RMS (degrees)</th></tr>{rows}</table>
<p>All {len(reproduction)} saved objectives reproduced, maximum relative discrepancy {max(reproduction):.3g}. Bound contacts, incomplete local starts, brightness conditioning and opposing-component ratios remain in the <a href="analysis.json">numerical summary</a> and original fit files. No location update or hull-point association is accepted here.</p>
<p><a href="../phase-neighbourhood-tests.log">Numerical tests</a> · <a href="../phase-trajectories-review/index.html">Selected-point review</a> · <a href="../index.html">Analysis index</a></p></html>''')
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for p in sources:
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
