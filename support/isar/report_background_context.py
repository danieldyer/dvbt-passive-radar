"""Publish the corrected range-context diagnostic and invalidate unsafe fits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES
from background_reference import field_reference
from extract_full_aperture import load
from fit_joint_sea_arcs import prepare_problem
from fit_reference_background_pairs import build_model
from gaussian_background import predictive_factors
from cluster_phase import crop
from recorded_field_context import expand
from report_background_extrapolation import original_values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/background-context-refits'))
    args=parser.parse_args();out=args.output;source=out/'fits.json';data=json.loads(source.read_text())
    if not data['complete'] or not data['range_context']:raise ValueError('A complete context comparison is required')
    test_log=(out/'tests.log').read_text();match=re.search(r'Ran (\d+) tests',test_log)
    if not match or not test_log.rstrip().endswith('OK'):raise ValueError('A passing regression log is required')
    reference=Path(data['reference']);case=prepare_problem()
    old=json.loads(Path('docs/data/isar_20261017/background-extrapolation/analysis.json').read_text())
    summary=dict(scope=__doc__,accepted_calibration=False,production_changed=False,tests_passed=int(match[1]),
        prior_antenna_fit_interpretation='Superseded: the five-cell local stationary model extrapolates 154–198 dB too much power to surrounding recorded ranges. Antenna positions and physical scene conclusions drawn from those fits are not reliable.',
        reproduction=[],intervals=[],
        remaining_limit='The new background system is well conditioned, but some two-echo decompositions still coalesce into cancelling components or reach bounds. A correct nuisance operator does not certify a common scatterer or solve antenna geometry.')
    plots,axes=plt.subplots(5,2,figsize=(14,15),layout='constrained')
    for col,interval in enumerate(data['intervals']):
        description=interval['description'];label=description['label']
        ci=next(i for i,e in enumerate(case['descriptions']) if e['label']==label)
        output=dict(description=description,channels=[]);summary['intervals'].append(output)
        for ch,channel in enumerate(interval['channels']):
            with np.load(Path(f'build/isar-20261015/full-fields/channel-{ch}.npz')) as stored:
                f=crop(stored,description['first_s'],description['last_s'],np.arange(-8,9))
            core=case['fields'][ci][ch];tr=case['training'][ci][ch];ks=case['knots'][ci][ch]
            aperture=load(Path(f'build/isar-20261015/full-geographic-aperture/channel-{ch}'))
            extended=expand(f,aperture,np.arange(-16,17))
            inside=np.isin(extended.offsets,f.offsets);outside=~inside
            if not np.array_equal(extended.data['value'][:,inside],f.data['value']):raise ValueError('Archived context does not reproduce the saved field exactly')
            with np.load(reference/f'channel-{ch}-means.npz') as loaded:ref=dict(loaded)
            row=dict(channel=NAMES[ch],modes=[]);output['channels'].append(row)
            for group in channel['modes']:
                mode=group['mode'];delays=group['stationary_delays_m']
                model,mean=build_model(f,tr,ks,ref,mode,delays)
                maximum=0.
                for trial in group['trials']:
                    r=model.evaluate(np.asarray(trial['parameters']))['residual']
                    error=abs(float(r@r)-trial['objective']);maximum=max(maximum,error)
                    if error>2e-9:raise ValueError('A saved objective did not reproduce')
                summary['reproduction'].append(dict(interval=label,channel=ch,mode=mode,trials=len(group['trials']),maximum_error=maximum))
                selected=group['trials'][group['best_index']];p=np.asarray(selected['parameters'])
                value=model.evaluate(p);b=model.base;k=len(ks);prediction=value['design']@value['amplitude']
                beta=b.nuisance.solve(f.y.ravel()[b.selected]-mean.ravel()[b.selected]-prediction[b.selected])
                # Extend the frozen response, not the nuisance fitter. A new
                # default delay grid would add unsupported columns and could
                # exceed the stored pulse support; no new columns are needed.
                extra=copy.copy(model);eb=copy.copy(b);extra.base=eb
                eb.field=extended;eb.base=copy.copy(b.base);eb.base.field=extended
                if mode=='local':
                    extra_mean=np.zeros_like(extended.y)
                    extra_design=np.column_stack([extended.stationary(delay).ravel() for delay in delays])
                else:
                    references=np.stack([field_reference(extended,m,int(ref['first_range_sample'])).ravel() for m in ref['means'][:12]])
                    extra_mean,extra_design=predictive_factors(references,ref['information'][:12])
                    extra_mean=extra_mean.reshape(extended.y.shape)
                raw=[r for r,_ in extra.response_blocks(p)]
                design=np.concatenate([eb.reduced(r[:,:,None]*eb.amplitude_basis[:,None,:]) for r in raw],axis=1)
                target=(design@value['amplitude']).reshape(extended.y.shape)
                background=(extra_design@beta).reshape(extended.y.shape)+extra_mean
                predicted=original_values(extended,target+background)
                static=original_values(extended,background);observed=extended.data['value']
                old_prediction=original_values(f,(prediction+b.design@beta+mean.ravel()).reshape(f.y.shape))
                discrepancy=float(np.max(abs(predicted[:,inside]-old_prediction)))
                weighted=lambda z:np.average(abs(z)**2,axis=0,weights=extended.data['information'])
                ratio=float(np.sum(weighted(predicted)[outside])/np.sum(weighted(observed)[outside]))
                entry=dict(mode=mode,selected_regular=selected['numerically_regular'],
                    regular_starts=sum(t['numerically_regular'] for t in group['trials']),starts=len(group['trials']),
                    stationary_condition=group['stationary_condition'],expanded_prediction_reproduction_error=discrepancy,
                    outside_prediction_to_observed_power_db=float(10*np.log10(ratio)),
                    core_unused_error=selected['decomposition']['core_unused_error'],
                    total_component_cancellation_ratio=selected['decomposition']['component_plus_adjustable_background_energy_ratio'],
                    parameters=selected['parameters'],observables=selected['observables'])
                row['modes'].append(entry)
                if mode=='local':
                    ax=axes[ch,col]
                    for title,z in [('Recorded',observed),('Two echoes',original_values(extended,target)),
                                    ('Background',static),('Complete prediction',predicted)]:
                        ax.semilogy(extended.offsets,weighted(z),'o-',ms=2.5,label=title)
                    ax.axvspan(-8.5,8.5,color='grey',alpha=.1)
                    ax.set(title=f'{label}: {NAMES[ch]}'+(' — pair not identified' if not selected['numerically_regular'] else ''),ylabel='Mean original-profile power')
                    if ch==0:ax.legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Original field offset (range samples)')
    plots.suptitle('Seventeen-cell fits checked on thirty-three original recorded cells\nShaded: fitted ranges; all parameters frozen; original profile units',fontsize=13)
    plots.savefig(out/'01-expanded-prediction.png',dpi=150);plt.close(plots)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    conditions=[]
    for ax,v,prior in zip(axes,summary['intervals'],old['intervals']):
        before=[c['stationary_condition'] for c in prior['channels']]
        after=[next(m for m in c['modes'] if m['mode']=='local')['stationary_condition'] for c in v['channels']]
        conditions+=after
        ax.semilogy(NAMES,before,'o-',label='Five fitted range cells')
        ax.semilogy(NAMES,after,'o-',label='Seventeen cells; same nuisance columns')
        ax.set(title=v['description']['label'],ylabel='Retained nuisance condition number');ax.legend(fontsize=9)
    fig.suptitle('The surrounding measurements resolve the previously unsupported background modes')
    fig.savefig(out/'02-conditioning.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for i,v in enumerate(summary['intervals']):
        ci=next(i for i,e in enumerate(case['descriptions']) if e['label']==v['description']['label'])
        nominal=case['geometries'][ci].prediction(np.zeros(28))[:,0]
        for ch,c in enumerate(v['channels']):
            for mode,shift,colour in [('local',-.11,'#24699b'),('finite',.11,'#c47721')]:
                m=next(m for m in c['modes'] if m['mode']==mode);values=np.asarray(m['observables'])
                for j,ax in enumerate(axes[i]):
                    ax.scatter(np.repeat(ch+shift,2),values[:,j]-(nominal[ch] if j==0 else 0),
                        color=colour,marker='o' if m['selected_regular'] else 'x',s=48,
                        label=('Free background with range context' if mode=='local' else 'Independent mean + finite covariance') if ch==0 else None)
        for ax,title,y in zip(axes[i],['Excess path','Excess-path rate','Excess-path curvature'],
            ['Path relative to nominal AIS (m)','Rate (m/s)','Curvature (m/s²)']):
            ax.set(title=f"{v['description']['label']}: {title}",ylabel=y);ax.set_xticks(range(5),NAMES)
        axes[i,0].legend(fontsize=7)
    fig.suptitle('Both component trajectories remain hypotheses\nCircles: numerically regular local fit; crosses: degenerate, boundary-limited or incomplete',fontsize=12)
    fig.savefig(out/'03-component-trajectories.png',dpi=170);plt.close(fig)
    summary['nuisance_condition_range']=list(map(float,[min(conditions),max(conditions)]))
    summary['fits_reproduced']=sum(r['trials'] for r in summary['reproduction'])
    summary['maximum_objective_reproduction_error']=max(r['maximum_error'] for r in summary['reproduction'])
    table=''.join(f"<tr><td>{v['description']['label']}</td><td>{c['channel']}</td><td>{m['mode']}</td><td>{m['regular_starts']}/{m['starts']}</td><td>{m['selected_regular']}</td><td>{m['core_unused_error']:.2f}</td><td>{m['outside_prediction_to_observed_power_db']:+.2f}</td><td>{m['total_component_cancellation_ratio']:.3g}</td></tr>"
        for v in summary['intervals'] for c in v['channels'] for m in c['modes'])
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Correcting the calibration background model</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{border:1px solid #ddd;padding:8px}}table{{border-collapse:collapse}}.note{{background:#fff1d3;padding:18px}}</style>
<h1>A flaw in the short-range calibration background model</h1>
<p class="note"><b>The preceding antenna fits are not reliable physical calibrations.</b> The five-cell analysis admitted enormous opposing stationary coefficients which cancel inside that crop. Their frozen prediction exceeds the measured surrounding field by 154–198 dB. The previous unused-time checks used the same five cells and could not expose this failure. The <a href="../background-extrapolation/index.html">explicit extrapolation plots</a> demonstrate it. Captured data, production processing and user-supplied antenna priors are unchanged.</p>
<p>The local least-squares calculation itself was numerically implemented and differentiated consistently; its unconstrained nuisance model was inappropriate for the available range support. A successful solver termination and matching synthetic tests did not justify treating those coefficients, derived antenna locations or inferred physical components as measured reality. In particular, old direct-versus-sea model comparisons cannot settle the physical scattering hypothesis.</p>
<h2>Correction tested on the recorded data</h2>
<p>The same background delay columns are now constrained by all seventeen existing range cells. This reduces the condition number to {min(conditions):.1f}–{max(conditions):.1f}, with no arbitrary singular-value threshold change. A second control uses the independently measured 30–90 s residual mean and its finite block covariance. The Gaussian covariance is applied consistently to data, templates and derivatives; a test verifies equality with augmented least squares. Independent control periods show drift, so that covariance is a sensitivity model rather than a calibrated background uncertainty.</p>
<img src="02-conditioning.png" alt="Nuisance conditioning with genuine surrounding observations">
<p>Every fitted parameter is then frozen and evaluated on thirty-three range cells extracted from the original C0/C1 archive. The inner seventeen measured cells reproduce the previously saved data bit for bit. The new predicted background no longer explodes immediately outside the fitted range support.</p>
<img src="01-expanded-prediction.png" alt="Frozen expanded-range prediction in original measured units">
<h2>What remains unresolved</h2>
<p>There is still no accepted common point-scatterer or antenna solution. Some two-echo fits collapse into almost identical, opposing components; these are marked explicitly. The finite-background control and range-context fit also retain differences in apparent motion. A distributed ship, several unresolved echoes, sea-assisted rays, and unmodelled waveform effects must be distinguished with the corrected observation model. More restarts of the old five-cell likelihood would not resolve its defective background interpretation.</p>
<img src="03-component-trajectories.png" alt="Both fitted echo trajectories under the corrected background controls">
<table><tr><th>Interval</th><th>Band</th><th>Background</th><th>Regular starts</th><th>Selected regular?</th><th>Unused error in original five cells</th><th>Further outside prediction/data (dB)</th><th>Component/background cancellation ratio</th></tr>{table}</table>
<p>Each model selects its minimum training cost. Unused data and numerical diagnostics remain independent of selection; a degenerate minimum is not silently replaced with a prettier alternative. Differently weighted training objectives are not compared across covariance models. Original unused-data errors are comparable, but an over-flexible unphysical model can still win that limited prediction test. Numerical regularity is conditional on the model, not a physical identification certificate.</p>
<p>{summary['fits_reproduced']} saved objectives reproduce, maximum normalized discrepancy {summary['maximum_objective_reproduction_error']:.3g}. <a href="tests.log">All {summary['tests_passed']} numerical tests pass</a>. New tests cover the known-background information bound, phase and C0/C1 bookkeeping, finite covariance, exact profiled derivatives, and known two-echo recovery after adding surrounding background constraints.</p>
<p>Related prior art: Colone et al. discuss how short adaptive cancellation intervals distort slowly moving echoes in <a href="https://iris.uniroma1.it/handle/11573/906920">Sliding extensive cancellation algorithm for disturbance removal in passive radar (2016)</a>. That motivates checking slow-target information, but the extreme spatial extrapolation identified here is demonstrated directly from this code and recording.</p>
<p><a href="fits.json">All fits</a> · <a href="analysis.json">Reproduction and extrapolation results</a> · <a href="../background-reference/index.html">Independent background measurements</a> · <a href="../clutter-information/index.html">Conditional information bound</a> · <a href="../index.html">Analysis index</a></p></html>''')
    sources=[source,Path(__file__),*map(Path,data['source_sha256']),Path('support/isar/recorded_field_context.py')]
    summary['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sources}
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for p in sources:
        if str(p).startswith('support/isar/') and p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    (out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',out/'index.html','reproduced',summary['fits_reproduced'],flush=True)


if __name__=='__main__':main()
