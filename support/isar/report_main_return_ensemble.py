"""Compare completed independent B/E measurements across all five bands."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES,COLOURS
from motion import C
from measured_phase import range_rate_map,path_increments,interpolate_connected
from nearby_echo_control import load
from report_fixed_grid_guided_return import phase_comparison
from report_measured_scatterer_phases import serial
from scatterer_phase_deltas import compare_pair


def main():
    root=Path('docs/data/isar_20261017');out=root/'main-return-five-band-review';out.mkdir(exist_ok=True)
    folders=[f'main-return-extra-{n}-guided' for n in NAMES[:4]]+['main-return-extra-ABC-flexible-wide']
    sources=[root/f/'fits.json' for f in folders];saved=[json.loads(p.read_text()) for p in sources]
    if any(not d['complete'] for d in saved):raise ValueError('Wait for all declared independent starts')
    first,last=saved[0]['interval_s'];bands=[];fits=[];records={1:[],4:[]};statistics=[]
    for c,d in enumerate(saved):
        if d['channel']!=c or d['interval_s']!=[first,last] or d['extra_curve']!='flexible':raise ValueError('Matching ordered independent fields required')
        with np.load(f'build/isar-20261015/full-fields/channel-{c}.npz') as a:meta=json.loads(str(a['metadata_json']))
        k=2*np.pi*meta['rf_hz']/C;bands.append(SimpleNamespace(field=SimpleNamespace(k=k)))
        fit=d['trials'][d['best_index']];fits.append(fit)
        for s in (1,4):
            r=fit['measured'][str(s)];z=np.array(r['complex_coefficient_real_imag'])
            records[s].append(dict(time_s=np.array(r['time_s']),physical_path_change_m=np.array(r['physical_path_change_m']),
                complex_coefficient=z[:,0]+1j*z[:,1],conditional_information=np.array(r['conditional_information'])))
        statistics.append(dict(channel=NAMES[c],selected_start=d['best_index'],training_cost=fit['cost'],
            unused_error_change_percent=100*(fit['prediction']['testing'][0]['heldout_error']/d['original_prediction']['testing'][0]['heldout_error']-1),
            projected_gradient=fit['maximum_projected_gradient'],bounds=fit['bound_indices'],
            origins_m=np.array(fit['parameters'])[[2,8]]*C/8e6))
    # Mark the independently fitted centres against the unchanged observations,
    # before any fitted stationary or neighbouring return is subtracted.
    control=load();rates=np.linspace(-2.,.5,251);maps=[];selections=[]
    for c,(b,model,fit) in enumerate(zip(control['bands'],control['independent_models'],fits)):
        maps.append(range_rate_map(b.field.time,b.field.data['value'],b.field.data['information'],
            b.field.k,rates)/b.field.pulse.samples[64].real)
        q=np.asarray(fit['parameters']);local=q[model.indices[0]];chosen=[]
        for s in (1,4):
            arc=model.clusters[0].arcs[s];parameters=local[model.clusters[0].slices[s]]
            observer=-b.field.data['centre_omega']/(b.field.k*.001024)
            rate=arc.known_rate+arc.base.rate_basis@parameters[2:]+observer
            chosen.append(dict(point='B' if s==1 else 'E',offset_m=q[2*s]*C/8e6,
                physical_rate_mps=float(np.interp(control['curve'].epoch,b.field.data['row_time'],rate))))
        selections.append(chosen)
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained',sharex=True,sharey=True)
    vmax=float(np.ceil(10*np.log10(max(m.max() for m in maps))))
    for c,(ax,b,power,chosen) in enumerate(zip(axes.ravel(),control['bands'],maps,selections)):
        offset=b.field.offsets*C/8e6
        image=ax.imshow(10*np.log10(np.maximum(power,1e-8)),origin='lower',aspect='auto',
            extent=[offset[0]-C/16e6,offset[-1]+C/16e6,rates[0],rates[-1]],
            cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
        for item,colour in zip(chosen,('#00e5ff','#72ff64')):
            xy=(item['offset_m'],item['physical_rate_mps'])
            ax.plot(*xy,'s',mfc='none',mec=colour,ms=11,mew=1.5)
            ax.annotate(item['point'],xy,xytext=(8,6),textcoords='offset points',color=colour,weight='bold')
        ax.set(title=NAMES[c],xlim=(-110,150),xlabel='Offset from fixed range reference (m)',ylabel='Excess-path rate (m/s)')
    axes.ravel()[-1].axis('off')
    fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.85,label='Original coherent cell power / conditional noise (dB)')
    fig.suptitle('Selected nearby components in every channel · 162–168 s\n'
        'Original measured fields before scene subtraction; marks show independent fitted centres at 165 s. B/E are provisional associations.',fontsize=12)
    fig.savefig(out/'00-selected-points.png',dpi=150);plt.close(fig)
    measured={str(s):phase_comparison(records[s],bands,first,last) for s in records}
    fig,axes=plt.subplots(2,3,figsize=(18,9),layout='constrained',sharex=True)
    for row,s in enumerate((1,4)):
        m=measured[str(s)];time=m['time_s'];reference=m['channels'][2]
        for c,(v,colour) in enumerate(zip(m['channels'],COLOURS)):
            axes[row,0].plot(time,v['measured_path_change_m'],'.-',ms=2,lw=1,color=colour,label=NAMES[c])
            if c!=2:axes[row,1].plot(time,100*(v['measured_path_change_m']-reference['measured_path_change_m']),color=colour)
            axes[row,2].plot(v['time_s'],10*np.log10(np.maximum(v['power_to_noise'],1e-10)),color=colour)
        name='B' if s==1 else 'E'
        axes[row,0].set(title=name+' · measured physical path change',ylabel='Path change (m)')
        axes[row,1].set(title=name+' · channel minus Nine',ylabel='Interband path-change difference (cm)')
        axes[row,2].set(title=name+' · fade evidence',ylabel='Power / conditional thermal noise (dB)')
        for ax in axes[row]:ax.set(xlim=(first,last),xlabel='Capture time (s)')
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle('Two nearby return components, measured independently in all five bands\n'
        'ABC supplies initialization only. Every phase, envelope and path is free afterward; only constant origins are removed.',fontsize=12)
    fig.savefig(out/'01-independent-measured-paths.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True)
    for row,s in enumerate((1,4)):
        m=measured[str(s)]
        for c,(v,colour) in enumerate(zip(m['channels'],COLOURS)):
            axes[row].plot(m['time_s'],-np.degrees(bands[c].field.k*v['measured_path_change_m']),'.-',ms=2,lw=1,color=colour,label=NAMES[c])
        axes[row].set(title=('B' if s==1 else 'E')+' · measured physical RF phase',ylabel='RF phase change (degrees)',xlim=(first,last))
    axes[0].legend(ncol=5,fontsize=8);axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('Measured carrier phases use each channel\'s own RF frequency\nConstant origins only; equal physical paths have different phase slopes at different frequencies.',fontsize=12)
    fig.savefig(out/'04-measured-rf-phases.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True)
    for row,s in enumerate((1,4)):
        for c,(fit,colour) in enumerate(zip(fits,COLOURS)):
            info=fit['information'][str(s)]
            axes[row].plot(info['time_s'],[np.nan if v is None else v for v in info['phase_standard_deviation_deg']],color=colour,label=NAMES[c])
        axes[row].set(title=('B' if s==1 else 'E')+' · local data information after all neighbours',
            ylabel='Conditional phase-change σ (degrees)',xlim=(first,last))
    axes[0].legend(ncol=5,fontsize=8);axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('Some individual decompositions are weakly determined despite strong summed returns\n'
        'No motion prior, range bound or guide counted as measured phase information. This does not include unknown scene mismatch.',fontsize=12)
    fig.savefig(out/'02-phase-information.png',dpi=150);plt.close(fig)
    time=measured['1']['time_s'];paths=np.array([[v['measured_path_change_m'] for v in measured[str(s)]['channels']] for s in (1,4)])
    power=np.array([[np.interp(time,v['time_s'],v['power_to_noise']) for v in measured[str(s)]['channels']] for s in (1,4)])
    try:pair=compare_pair(paths,time,power)
    except ValueError:pair=None
    fig,axes=plt.subplots(4,2,figsize=(14,12),layout='constrained',sharex=True)
    pairwise=[]
    for row,c in enumerate((0,1,3,4)):
        try:local=compare_pair(paths[:,[c,2]],time,power[:,[c,2]],reference_band=1)
        except ValueError:local=None
        pairwise.append(dict(channel=NAMES[c],reference='Nine',comparison=local))
        if local:
            for j,name,colour in ((0,NAMES[c],COLOURS[c]),(1,'Nine',COLOURS[2])):
                axes[row,0].plot(time,100*local['pair_path_change_m'][j],color=colour,label=name)
            axes[row,1].plot(time,100*local['double_difference_m'][0],color=COLOURS[c])
            for ax in axes[row]:ax.axvline(local['anchor_time_s'],color='#aaa',ls=':',lw=.7)
            axes[row,0].legend(fontsize=8,ncol=2)
        else:
            for ax in axes[row]:ax.text(.5,.5,'No shared connected phase origin',ha='center',transform=ax.transAxes)
        axes[row,0].set(title=NAMES[c]+' / Nine: E−B within each band',ylabel='Within-band E−B change (cm)')
        axes[row,1].set(title=NAMES[c]+' minus Nine: (E−B) double difference',ylabel='Interband discrepancy (cm)')
        for ax in axes[row]:ax.set(xlim=(first,last));ax.axhline(0,color='#aaa',lw=.5)
    for ax in axes[-1]:ax.set_xlabel('Capture time (s)')
    fig.suptitle('Compare overlapping phase pieces; do not require both features in all five bands simultaneously\n'
        'Each pair removes a constant at its own marked common epoch. Absolute offsets between panels are not comparable; slopes are untouched.',fontsize=12)
    fig.savefig(out/'03-between-point-differences.png',dpi=150);plt.close(fig)
    # Short differences retain useful disconnected pieces and do not infer a
    # carrier cycle through a fade. A constant component phase cancels exactly.
    lag=4;increment_time=(time[lag:]+time[:-lag])/2;increment_rates=[];alias_limits=[]
    for s in (1,4):
        rates_for_point=[];limits=[]
        for c,v in enumerate(measured[str(s)]['channels']):
            native=np.asarray(v['time_s']);z=np.asarray(v['averaged_real'])+1j*np.asarray(v['averaged_imag'])
            centre,delta,_=path_increments(native,z,np.asarray(v['variance']),bands[c].field.k,np.zeros(len(native)),lag)
            dt=native[lag:]-native[:-lag]
            rates_for_point.append(interpolate_connected(centre,delta/dt,increment_time))
            limits.append(float(np.min(np.pi/(bands[c].field.k*dt))))
        increment_rates.append(rates_for_point);alias_limits.append(limits)
    increments=np.asarray(increment_rates);pair_increment=increments[1]-increments[0]
    fig,axes=plt.subplots(3,2,figsize=(15,11),layout='constrained',sharex=True)
    for row,(values,label) in enumerate(((increments[0],'B'),(increments[1],'E'),(pair_increment,'E−B'))):
        for c,colour in enumerate(COLOURS):
            axes[row,0].plot(increment_time,values[c],color=colour,lw=.9,label=NAMES[c])
            if c!=2:axes[row,1].plot(increment_time,100*(values[c]-values[2]),color=colour,lw=.9)
        axes[row,0].set(title=label+' · measured short-interval rate',ylabel='Path change / interval (m/s)')
        axes[row,1].set(title=label+' · channel minus Nine',ylabel='Short-rate discrepancy (cm/s)')
        for ax in axes[row]:ax.set(xlim=(first,last),xlabel='Capture time (s)');ax.axhline(0,color='#aaa',lw=.5)
    axes[0,0].legend(ncol=5,fontsize=8)
    fig.suptitle('Local 0.2 s phase increments: compare pieces without a global phase origin\n'
        'Principal complex phase differences, divided by each carrier wavenumber and actual interval. Fades stay gaps; no rate alignment.',fontsize=12)
    fig.savefig(out/'05-local-phase-increments.png',dpi=150);plt.close(fig)
    with (out/'measured-trajectories.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('capture_time_s','point','channel','physical_path_change_m','rf_phase_change_deg'))
        for s in (1,4):
            m=measured[str(s)]
            for c,v in enumerate(m['channels']):
                for t,value in zip(m['time_s'],v['measured_path_change_m']):writer.writerow((t,'B' if s==1 else 'E',NAMES[c],value,-np.degrees(bands[c].field.k*value)))
    result=dict(complete=True,accepted_calibration=False,production_changed=False,
        completed_independent_starts=sum(len(d['trials']) for d in saved),statistics=statistics,
        selected_centres=selections,selection_epoch_s=control['curve'].epoch,
        reproduced_original_objectives=control['reproduced_objectives'],
        maximum_reproduced_objective_error=control['maximum_absolute_cost_error'],
        measured_paths=measured,between_point_comparison=pair,pairwise_between_point_comparisons=pairwise,
        short_increments=dict(time_s=increment_time,lag_samples=lag,nominal_interval_s=.2,
            rates_mps=increments,pair_rate_difference_mps=pair_increment,
            minimum_principal_phase_unambiguous_rate_mps=alias_limits,
            scope='No global phase unwrapping or cycle connection through fades. Principal phase increments retain their wavelength ambiguity; rates beyond the stated limits would alias. Measurements overlap and are correlated.'),
        scope='Independent per-band five-return fits initialized from two completed ABC decompositions. '
            'No interband path or phase restriction is imposed. Labels may exchange physical identities, and fades/weak conditional '
            'information are explicit. These plots do not yet establish a common set of physical points or phase-centre coordinates.',
        reproduction_scope='This page reads the native measured complex histories saved by the completed original-row fits. '
            'The ABC source/start likelihoods and stored histories are separately reproduced in its detailed review. '
            'Every per-band producer checks original baseline identity and original-row directional derivatives before fitting.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources,Path(__file__)]})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True);manifest={}
    for name in ('report_main_return_ensemble','scatterer_phase_deltas','measured_phase',
            'report_fixed_grid_guided_return','nearby_echo_control'):
        p=Path('support/isar')/(name+'.py');shutil.copyfile(p,snapshots/p.name)
        manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'plot-source-manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=''.join(f'<tr><td>{s["channel"]}</td><td>{s["selected_start"]}</td><td>{s["unused_error_change_percent"]:+.3f}%</td><td>{s["projected_gradient"]:.3g}</td><td>{s["bounds"]}</td></tr>' for s in statistics)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Five-band nearby return phases</title><style>body{{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px}}</style>
<h1>Two nearby features: independent five-band phase measurements</h1>
<p>All ten declared independent starts are complete. The extra nearby component improves unused prediction in every band at the training-selected starts, most strongly in ABC and Ten. ABC supplied initialization only; no channel shares a fitted temporal path or phase. B/E labels are therefore provisional identities. The same two features cannot be assumed merely from those labels.</p>
<a href="00-selected-points.png"><img src="00-selected-points.png" alt="Independent B/E selections marked in all five original range and path-rate maps"></a>
<img src="01-independent-measured-paths.png" alt="Measured B/E paths, interband differences and strength in all five bands">
<p><a href="04-measured-rf-phases.png">Physical RF-phase version of these same measurements</a>; the path plot divides phase by each carrier wavenumber for comparison.</p>
<img src="02-phase-information.png" alt="Conditional phase information showing weak individual decompositions">
<img src="03-between-point-differences.png" alt="Within-band differences between nearby candidate features">
<img src="05-local-phase-increments.png" alt="Local measured B/E phase increments and between-point rate differences across all five channels">
<p>The local-increment plot retains disconnected pieces without a common phase origin. It takes the complex phase difference over about 0.2 s, then divides by each channel’s carrier wavenumber and the actual interval. It does not unwrap through fades or fit out any rate. These principal increments retain a wavelength ambiguity (limits are recorded in the data); mixture-induced phase swings can still occur. The 250 ms measurement windows overlap, so these are correlated diagnostics.</p>
<p>No single epoch has both features phase-connected in all five bands. The between-point panels therefore use overlapping band/Nine pairs. Each panel marks its own magnitude-selected common constant origin; slopes and curvature remain untouched. These are overlapping measurements, not independent calibration constraints to multiply together.</p>
<table><tr><th>Channel</th><th>Training-selected start</th><th>Unused prediction-error change</th><th>Projected gradient</th><th>Bounds</th></tr>{rows}</table>
<p>Ten's bound index 4 is the earlier C nuisance envelope, not a B/E location bound. Other B/E fits are interior. Several isolated per-band decompositions have substantial phase uncertainty or fades. This is the motivation for a joint original-data common-trajectory test, not evidence to force the curves into agreement. That test must predict unused observations and retain the independent-band model as its control.</p>
<p>These measured coefficients follow training-frozen subtraction of other returns/static terms. Measurements use overlapping 250 ms windows at 50 ms steps, so adjacent plotted samples are correlated. Constant origins only; no fitted rate/curvature alignment. Missing lines stay missing. The detailed ABC review checks local B/E cancellation rather than a whole-scene ratio dominated by the distant D return.</p>
<p><a href="analysis.json">All measured paths, source identities and scope</a> · <a href="measured-trajectories.csv">CSV phases and paths</a> · <a href="../main-return-extra-ABC-flexible-review/index.html">Detailed ABC replay</a> · <a href="../measured-scatterer-phases/index.html">Ten raw-cell candidates in four matching windows</a> · <a href="../current-nearby-phase-tests.log">54 numerical checks</a></p></html>''')
    print(out/'index.html',statistics,flush=True)


if __name__=='__main__':main()
