"""Review independently measured B/D phases from the longer known-kernel fit."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_joint_kernel_phase import known_band
from cluster_phase import SplinePath
from compare_ais_phase import NAMES,COLOURS
from component_phase import component_measurements
from fit_fixed_grid_phases import prepare_fixed
from measured_phase import range_rate_map
from motion import C
from release_fixed_grid_phase_guide import path_information
from report_fixed_grid_guided_return import phase_comparison
from report_measured_scatterer_phases import load_fields,serial
from scatterer_phase_deltas import compare_pair
from smooth_echo_cluster import SmoothEchoCluster
from waveform_kernels import read_kernels


def phase_delta_plots(measured,bands,first,last,out):
    """Plot physical RF phase and translation-suppressed measured differences."""
    time=np.asarray(measured['1']['time_s'])
    paths=np.asarray([[c['measured_path_change_m'] for c in measured[str(s)]['channels']] for s in (1,3)])
    powers=np.asarray([[c['power_to_noise'] for c in measured[str(s)]['channels']] for s in (1,3)])
    # Effective epochs differ very slightly with matched-filter information.
    # Interpolate the strengths only for magnitude-based origin selection.
    for row,s in enumerate((1,3)):
        for c,record in enumerate(measured[str(s)]['channels']):
            powers[row,c]=np.interp(time,record['time_s'],record['power_to_noise'])
    delta=compare_pair(paths,time,powers)
    origin=delta['anchor_time_s'];paths=delta['centred_paths_m']
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True)
    for row,name in enumerate(('B','D')):
        for c,(band,colour) in enumerate(zip(bands,COLOURS)):
            axes[row].plot(time,-np.degrees(band.field.k*paths[row,c]),'.-',ms=2,lw=1,color=colour,label=NAMES[c])
        axes[row].axvline(origin,color='#999',ls=':',lw=.7)
        axes[row].set(title=f'{name} · measured physical carrier phase',ylabel='RF phase change (degrees)',xlim=(first,last))
    axes[0].legend(ncol=5,fontsize=9);axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('Physical RF phases at the five different carrier frequencies\n'
        '0.25 s complex measurements, every 0.05 s; frozen training-only neighbour/static subtraction. Constant origins only.',fontsize=12)
    fig.savefig(out/'05-measured-rf-phase.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,5.4),layout='constrained')
    for c,colour in enumerate(COLOURS):
        axes[0].plot(time,100*delta['pair_path_change_m'][c],'.-',ms=2,lw=1,color=colour,label=NAMES[c])
        if c!=2:axes[1].plot(time,100*delta['double_difference_m'][c],'.-',ms=2,lw=1,color=colour)
    for ax in axes:
        ax.axvline(origin,color='#999',ls=':',lw=.7);ax.axhline(0,color='#aaa',lw=.5)
        ax.set(xlim=(first,last),xlabel='Capture time (s)',ylabel='Path-change difference (cm)')
    axes[0].set_title('D minus B, measured independently within each band')
    axes[1].set_title('(D minus B) in each band, minus (D minus B) in Nine')
    axes[0].legend(ncol=5,fontsize=9)
    factor=360*200e6/C/100
    axes[1].secondary_yaxis('right',functions=(lambda cm:-factor*cm,lambda phase:-phase/factor)).set_ylabel('Equivalent phase difference at 200 MHz (degrees)')
    fig.suptitle('Remove common translation, then compare the remaining phase trajectory between channels\n'
        'Only one constant per curve is removed at the marked common epoch. No slope, curvature, heading or site correction.',fontsize=12)
    fig.savefig(out/'06-between-scatterer-deltas.png',dpi=150);plt.close(fig)
    with (out/'measured-phase-trajectories.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('capture_time_s','point','channel','rf_hz','measured_path_change_m',
            'measured_rf_phase_change_deg','power_to_conditional_noise','D_minus_B_m','D_minus_B_minus_Nine_m'))
        for row,name in enumerate(('B','D')):
            for c,band in enumerate(bands):
                for i,t in enumerate(time):
                    writer.writerow((t,name,NAMES[c],band.field.metadata['rf_hz'],paths[row,c,i],
                        -np.degrees(band.field.k*paths[row,c,i]),powers[row,c,i],
                        delta['pair_path_change_m'][c,i],delta['double_difference_m'][c,i]))
    return delta


def main():
    root=Path('docs/data/isar_20261017');source=root/'longer-known-phases-6s/fits.json'
    saved=json.loads(source.read_text())
    if not saved['complete']:raise ValueError('Wait for every declared independent/shared start')
    first,last=saved['interval_s'];out=root/'longer-known-phases-6s-review';out.mkdir(exist_ok=True)
    baseline=json.loads(Path(saved['source']).read_text());origins=baseline['selected_origins_samples']
    geometry=Path('docs/data/calibration_20261016_sites')
    mean_bands,parts,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    for c,b in enumerate(mean_bands):
        with np.load(source.parent/'observations'/f'channel-{c}.npz') as original:
            for key in ('value','row_time','row_information','information'):
                np.testing.assert_array_equal(original[key],b.field.data[key])
    kernel=Path('build/isar-20261016/waveform-kernels-30s');_,records,samples=read_kernels(kernel/'response.kernels')
    with np.load(kernel/'rows.npz') as grids:
        bands=[known_band(b,grids[f'channel_{c}'],records[c],samples[c]) for c,b in enumerate(mean_bands)]
    curve=SplinePath(first,last,saved['knot_spacing_s']);errors=[]
    for group in saved['groups']:
        for c,e in enumerate(group['independent']):
            model=SmoothEchoCluster([bands[c]],origins,(1,3),curve,shared=False,jerk_std=group['jerk_std'])
            for fit in e['trials']:
                r,_=model.evaluate(np.asarray(fit['parameters']));errors.append(abs(r@r*model.energy-fit['cost']))
    std=saved['validation_selected_jerk_std'];group=next(g for g in saved['groups'] if g['jerk_std']==std)
    independent=[];measurements={1:[],3:[]};infos=[]
    for c,e in enumerate(group['independent']):
        fit=e['trials'][e['best_index']];p=np.asarray(fit['parameters'])
        model=SmoothEchoCluster([bands[c]],origins,(1,3),curve,shared=False,jerk_std=std)
        local=component_measurements(model.clusters[0],p[model.indices[0]])
        for s in measurements:measurements[s].append(local[s])
        infos.append({str(s):path_information(model,p,component=s) for s in (1,3)})
        independent.append(dict(model=model,fit=fit,parameters=p))
    common=SmoothEchoCluster(bands,origins,(1,3),curve,shared_components=(1,),jerk_std=std)
    for fit in saved['common']:
        r,_=common.evaluate(np.asarray(fit['parameters']));errors.append(abs(r@r*common.energy-fit['cost']))
    if max(errors)>1e-7:raise ValueError('Saved objective changed')
    chosen=saved['common'][saved['best_common_index']];common_p=np.asarray(chosen['parameters'])
    measured={str(s):phase_comparison(records,bands,first,last) for s,records in measurements.items()}
    phase_delta=phase_delta_plots(measured,bands,first,last,out)
    fig,axes=plt.subplots(2,3,figsize=(18,8.4),layout='constrained')
    for row,s in enumerate((1,3)):
        m=measured[str(s)];t=m['time_s'];nine=m['channels'][2]
        for c,(ch,colour) in enumerate(zip(m['channels'],COLOURS)):
            axes[row,0].plot(t,ch['measured_path_change_m'],color=colour,label=NAMES[c])
            if c!=2:axes[row,1].plot(t,100*(ch['measured_path_change_m']-nine['measured_path_change_m']),color=colour)
            axes[row,2].plot(ch['time_s'],10*np.log10(np.maximum(ch['power_to_noise'],1e-10)),color=colour)
        label='B' if s==1 else 'D'
        axes[row,0].set(title=label+' · independently measured paths',ylabel='Physical path change (m)')
        axes[row,1].set(title=label+' · band minus Nine',ylabel='Interband path-change difference (cm)')
        axes[row,2].set(title=label+' · conditional strength',ylabel='Power / conditional thermal noise (dB)')
        for ax in axes[row]:ax.set(xlim=(first,last),xlabel='Capture time (s)')
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle('Six-second independent phase measurements, all five channels\n'
        'Measured after frozen neighbour/static subtraction; original-weight known kernels. Constant phase origins only; '
        'no cross-channel path constraint.',fontsize=12)
    fig.savefig(out/'01-independent-measured-paths.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(20,5),layout='constrained',sharex=True,sharey=True)
    rates=np.linspace(-2.5,1.5,301);maps=[range_rate_map(b.field.time,b.field.data['value'],b.field.data['information'],b.field.k,rates)/b.field.pulse.samples[64].real for b in bands]
    vmax=np.ceil(10*np.log10(max(m.max() for m in maps)))
    for c,(ax,b,power,item) in enumerate(zip(axes,bands,maps,independent)):
        offsets=b.field.offsets*C/8e6
        image=ax.imshow(10*np.log10(np.maximum(power,1e-8)),origin='lower',aspect='auto',
            extent=[offsets[0]-C/16e6,offsets[-1]+C/16e6,rates[0],rates[-1]],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
        for s,colour in ((1,'#00e5ff'),(3,'#55dd99')):
            origin=item['parameters'][2*s]*C/8e6
            d=item['fit']['phase'][0][0 if s==1 else 1]
            rate=np.interp(165,d['time_s'],np.gradient(d['physical_path_change_m'],d['time_s']))
            ax.plot(origin,rate,'s',mfc='none',mec=colour,ms=11)
            ax.annotate('B' if s==1 else 'D',(origin,rate),xytext=(6,6),textcoords='offset points',color=colour)
        ax.axhline(0,color='#ccc',ls='--',lw=.7)
        ax.set(title=NAMES[c],xlabel='Range offset from fixed observer reference (m)')
    axes[0].set_ylabel('Physical excess-path rate (m/s)')
    fig.colorbar(image,ax=axes,label='Original coherent cell power / conditional noise (dB)',shrink=.7)
    fig.suptitle('B and D selections in original 162–168 s range/rate observations\n'
        'Markers are fitted local return centres; association with material hull points is not assumed.',fontsize=12)
    fig.savefig(out/'02-selected-points.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(14,12),layout='constrained',sharex=True)
    for c,ax in enumerate(axes):
        before=independent[c]['fit']['phase'][0][0];after=chosen['phase'][c][0]
        for d,label,colour in ((before,'Independent B',COLOURS[c]),(after,'One shared B','#555')):
            use=np.asarray(d['testing']);time=np.asarray(d['time_s']);phase=np.asarray(d['conditional_phase_residual_deg'])
            ax.plot(time[use],phase[use],'.',color=colour,ms=3,label=label,alpha=.8)
        ax.set(title=NAMES[c],ylabel='Unused phase innovation (°)',ylim=(-180,180));ax.axhline(0,color='#aaa',lw=.5)
    axes[0].legend(ncol=2,fontsize=9);axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('Does one common B trajectory predict the unused phase samples?\n'
        'Conditional neighbour/static subtraction; no test-time refit. The declared 00002 geometry supplies only nominal interband differences.',fontsize=12)
    fig.savefig(out/'03-shared-path-innovations.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True)
    for row,s in enumerate((1,3)):
        for c,colour in enumerate(COLOURS):
            info=infos[c][str(s)]
            if 'phase_standard_deviation_deg' in info:
                axes[row].plot(info['time_s'],info['phase_standard_deviation_deg'],color=colour,label=NAMES[c])
            else:axes[row].plot([],[],color=colour,label=NAMES[c]+' (rank deficient)')
        axes[row].set(title=('B' if s==1 else 'D')+' · neighbour-marginalized data information',ylabel='σ of phase change relative to 165 s (°)')
    axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('Conditional data-only uncertainty; no motion prior counted as measured information\n'
        'Unknown scene components, interference and physical association remain outside these local bounds.',fontsize=12)
    fig.savefig(out/'04-phase-information.png',dpi=150);plt.close(fig)
    before=np.array([e['fit']['prediction']['testing'][0]['heldout_error'] for e in independent])
    after=np.array([p['heldout_error'] for p in chosen['prediction']['testing']])
    changes=100*(after/before-1)
    rows=''.join(f'<tr><td>{NAMES[c]}</td><td>{before[c]:.2f}</td><td>{after[c]:.2f}</td><td>{changes[c]:+.3f}%</td></tr>' for c in range(5))
    total=100*(after.sum()/before.sum()-1)
    (out/'analysis.json').write_text(json.dumps(serial(dict(complete=True,accepted_calibration=False,production_changed=False,
        source=str(source),reproduced_objectives=len(errors),maximum_absolute_cost_error=max(errors),
        selected_jerk_std=std,independent_measured_paths=measured,information=infos,between_scatterer_differences=phase_delta,
        measurement_policy='Conditional measured complex coefficients after training-frozen neighbour/static subtraction. '
            '0.25 s complex measurements overlap at 0.05 s steps. Data curves retain independent phase slopes and curvature; '
            'no cross-channel trajectory is imposed. Local-return identities are not known physical hull-point identities.',
        independent_unused_errors=before,common_unused_errors=after,unused_error_change_percent=changes,
        total_unused_error_change_percent=total,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source,Path(__file__),Path('support/isar/component_phase.py'),Path('support/isar/release_fixed_grid_phase_guide.py'),
             Path('support/isar/scatterer_phase_deltas.py'))})),indent=2,allow_nan=False))
    pair_rows=''.join(f'<tr><td>{name}</td><td>{100*phase_delta["rms_against_reference_m"][c]:.1f}</td>'
        f'<td>{100*phase_delta["peak_to_peak_against_reference_m"][c]:.1f}</td></tr>' for c,name in enumerate(NAMES) if c!=2)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Longer multichannel phase analysis</title><style>body{{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:7px;border:1px solid #ddd}}.note{{background:#edf3f7;padding:16px}}</style>
<h1>Independent B/D phase trajectories over six seconds</h1>
<p class="note">The original four-second observations reproduce exactly inside this longer aperture; forward templates agree within {saved['maximum_overlap_forward_error']:.3g}. Both B and its brighter D neighbour now use flexible trajectories, and known original-weight waveform kernels follow original symbol rows. Two starts per channel and two continuity scales are complete. Validation selects white-jerk scale {std:g}; unused blocks choose neither starts nor scale.</p>
<p>These are local return labels, not known material points on one hull. The same time-constant observed static profile is retained. Its possible known-filter variation and unknown scene remain modelling limitations. Only the previously unidentifiable Ten C nuisance path stays fixed at its earlier training estimate. No external ABC phase trajectory is imposed on the independent fits.</p>
<a href="02-selected-points.png"><img src="02-selected-points.png" alt="Selected B and D points in all five original range-Doppler maps"></a>
<a href="01-independent-measured-paths.png"><img src="01-independent-measured-paths.png" alt="Measured multichannel physical path changes and interband differences for B and D"></a>
<h2>Measured RF phase and between-point phase differences</h2>
<p>The RF phase plot uses each band's own wavelength: the same path change has different phase change at different RF frequencies. The D−B comparison divides phase by that wavenumber first, subtracts B within each band, and then subtracts Nine for a direct interband comparison. Only constants are removed, at {phase_delta['anchor_time_s']:.2f} s. No heading or antenna correction is fitted. These are measured coefficients after conditional neighbour/static subtraction, not just the optimized spline paths.</p>
<a href="05-measured-rf-phase.png"><img src="05-measured-rf-phase.png" alt="Measured physical RF phases for B and D in all five channels"></a>
<a href="06-between-scatterer-deltas.png"><img src="06-between-scatterer-deltas.png" alt="Measured D minus B phase trajectory and between-channel double differences"></a>
<table><tr><th>Band</th><th>D−B difference from Nine, observed RMS (cm)</th><th>Peak to peak (cm)</th></tr>{pair_rows}</table>
<p>These describe discrepancies over the shared connected interval. They are not confidence intervals: the local averages overlap, and the scene decomposition is conditional. <a href="measured-phase-trajectories.csv">All plotted phase and differential samples (CSV)</a>. <a href="../scatterer-phase-comparison/index.html">Raw-cell phases at ten selections in four windows</a> provide the comparison without this additional subtraction; local labels differ between those two analyses.</p>
<h2>Shared B control</h2><p>At the declared nominal 00002 site assignment, fitting one B trajectory across all five channels changes total unused squared prediction error by {total:+.3f}%. Other return paths stay independent. This is whole-field prediction error, not target SNR or proof of antenna positions. Alternative licensed site assignments and actual target-motion differences need physical geometric treatment before interpreting a small phase discrepancy as a position error.</p>
<table><tr><th>Channel</th><th>Independent unused error</th><th>Shared B unused error</th><th>Change</th></tr>{rows}</table>
<a href="03-shared-path-innovations.png"><img src="03-shared-path-innovations.png" alt="Unused measured phase innovations with independent and shared B trajectories"></a>
<a href="04-phase-information.png"><img src="04-phase-information.png" alt="Conditional path precision after marginalizing neighbouring nonlinear and amplitude parameters"></a>
<p>Every phase extraction uses neighbours and static coefficients fitted on training data and frozen on unused samples. Only constant origins are removed from the measured histories; no per-band rate or curvature alignment is applied. Lines stop at fades. The uncertainty plots are local conditional data bounds, not global calibration confidence intervals or phase-error estimates for a distributed ship.</p>
<p>All {len(errors)} completed objectives reproduce against the original observations. <a href="analysis.json">Full measurements and diagnostics</a> · <a href="../longer-known-phases-6s/fits.json">Every start and prior control</a> · <a href="../longer-known-phases-6s/tests.log">Numerical checks</a> · <a href="../joint-kernel-phase/index.html">Known-kernel neighbour-coupling audit</a></p>
<p>No coordinate calibration or replacement focused video is accepted from this diagnostic alone.</p></html>''')
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for name in (Path(__file__).name,'scatterer_phase_deltas.py','test_scatterer_phase_deltas.py'):
        shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
