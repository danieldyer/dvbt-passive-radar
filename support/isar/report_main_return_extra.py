"""Reproduce and plot measured phases after adding a nearby return."""
import argparse
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
from report_fixed_grid_guided_return import phase_comparison
from report_measured_scatterer_phases import load_fields,serial
from smooth_echo_cluster import SmoothEchoCluster
from waveform_kernels import read_kernels


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.source.read_text())
    if not data['complete']:raise ValueError('Wait for every declared extra-component start')
    c=data['channel']
    if c!=4:raise ValueError('This initial cross-channel review compares an ABC replacement with the other four bands')
    first,last=data['interval_s'];source=Path(data['source']);saved=json.loads(source.read_text())
    baseline=json.loads(Path(saved['source']).read_text());std=saved['validation_selected_jerk_std']
    group=next(g for g in saved['groups'] if g['jerk_std']==std)
    geometry=Path('docs/data/calibration_20261016_sites')
    mean,parts,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    kernel=Path('build/isar-20261016/waveform-kernels-30s');_,records,pulses=read_kernels(kernel/'response.kernels')
    with np.load(kernel/'rows.npz') as grids:bands=[known_band(b,grids[f'channel_{j}'],records[j],pulses[j]) for j,b in enumerate(mean)]
    curve=SplinePath(first,last,saved['knot_spacing_s']);old=[];errors=[]
    for j,b in enumerate(bands):
        fit=group['independent'][j]['trials'][group['independent'][j]['best_index']]
        model=SmoothEchoCluster([b],baseline['selected_origins_samples'],(1,3),curve,shared=False,jerk_std=std)
        q=np.array(fit['parameters']);r,_=model.evaluate(q);errors.append(abs(r@r*model.energy-fit['cost']))
        old.append(component_measurements(model.clusters[0],q[model.indices[0]])[1])
    selected=(1,3,4) if data.get('extra_curve','quadratic')=='flexible' else (1,3)
    model=SmoothEchoCluster([bands[c]],data['origins_samples'],selected,curve,shared=False,jerk_std=std)
    diagnostics=[]
    for fit in data['trials']:
        q=np.array(fit['parameters']);r,_=model.evaluate(q);errors.append(abs(r@r*model.energy-fit['cost']))
        rec=component_measurements(model.clusters[0],q[model.indices[0]])
        for s in (1,4):
            saved_rec=fit['measured'][str(s)];z=rec[s]['complex_coefficient']
            np.testing.assert_allclose(np.c_[z.real,z.imag],saved_rec['complex_coefficient_real_imag'],rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(rec[s]['physical_path_change_m'],saved_rec['physical_path_change_m'],rtol=0,atol=1e-11)
        pair=np.array([rec[s]['frozen_component_prediction'][bands[c].training] for s in (1,4)])
        ratio=float(np.sum(abs(pair)**2)/np.sum(abs(pair.sum(axis=0))**2))
        gap=float(abs(q[8]-q[2])*C/8e6)
        diagnostics.append(dict(start=fit['start'],local_B_E_component_to_sum_energy_ratio=ratio,
            B_E_envelope_separation_m=gap,testing_error=fit['prediction']['testing'][0]['heldout_error'],bounds=fit['bound_indices']))
    if max(errors)>1e-7:raise ValueError('Original-data likelihood changed')
    best=data['trials'][data['best_index']];q=np.array(best['parameters'])
    rec=component_measurements(model.clusters[0],q[model.indices[0]])
    comparisons=[]
    for label,abc in (('Original ABC main B',old[c]),('After split: ABC B',rec[1]),('After split: ABC E',rec[4])):
        result=phase_comparison([*old[:4],abc],bands,first,last);time=result['time_s']
        anchor=int(np.argmin(abs(time-165.)))
        for band in result['channels']:
            values=band['measured_path_change_m']
            if np.isfinite(values[anchor]):values-=values[anchor]
        comparisons.append(dict(label=label,measurement=result))
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(18,5.5),layout='constrained',sharex=True,sharey=True)
    for ax,comparison in zip(axes,comparisons):
        m=comparison['measurement']
        for j,(band,colour) in enumerate(zip(m['channels'],COLOURS)):
            ax.plot(m['time_s'],band['measured_path_change_m'],'.-',ms=2,lw=1,color=colour,label=NAMES[j])
        ax.set(title=comparison['label'],xlabel='Capture time (s)',xlim=(first,last));ax.axvline(165,color='#999',lw=.6,ls=':')
    axes[0].set_ylabel('Measured physical path change from 165 s (m)');axes[0].legend(ncol=3,fontsize=8)
    fig.suptitle('Which ABC feature resembles the independently measured main return in the other four bands?\n'
        'Same original data. Training-frozen neighbour/static subtraction; 0.25 s complex measurements. No shared temporal paths or slopes.',fontsize=12)
    fig.savefig(out/'01-measured-phase-associations.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5.3),layout='constrained')
    for comparison,colour in zip(comparisons,('#777',COLOURS[4],'#10746a')):
        m=comparison['measurement'];difference=m['channels'][4]['measured_path_change_m']-m['channels'][2]['measured_path_change_m']
        axes[0].plot(m['time_s'],100*difference,color=colour,label=comparison['label'])
    axes[0].axhline(0,color='#aaa',lw=.5);axes[0].legend(fontsize=8)
    axes[0].set(xlabel='Capture time (s)',ylabel='ABC minus Nine measured path change (cm)',xlim=(first,last))
    for s,label,colour in ((1,'ABC B',COLOURS[4]),(4,'ABC E','#10746a')):
        info=best['information'][str(s)]
        axes[1].plot(info['time_s'],[np.nan if v is None else v for v in info['phase_standard_deviation_deg']],color=colour,label=label)
    axes[1].legend();axes[1].set(xlabel='Capture time (s)',ylabel='Conditional data-only phase-change σ (degrees)',xlim=(first,last))
    fig.suptitle('Association discrepancy and precision after every neighbouring parameter is marginalized\n'
        'Bounds and the motion prior do not count as data information. Model/association errors remain outside these local bounds.',fontsize=12)
    fig.savefig(out/'02-differences-and-information.png',dpi=150);plt.close(fig)
    b=bands[c];rates=np.linspace(-2,.5,251)
    power=range_rate_map(b.field.time,b.field.data['value'],b.field.data['information'],b.field.k,rates)/b.field.pulse.samples[64].real
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    offsets=b.field.offsets*C/8e6
    image=ax.imshow(10*np.log10(np.maximum(power,1e-8)),origin='lower',aspect='auto',
        extent=[offsets[0]-C/16e6,offsets[-1]+C/16e6,rates[0],rates[-1]],cmap='magma',vmin=0,vmax=np.ceil(10*np.log10(power.max())),interpolation='nearest')
    for s,label,colour in ((1,'B',COLOURS[4]),(4,'E','#00e5ff')):
        record=rec[s];rate=float(np.interp(165,record['time_s'],np.gradient(record['physical_path_change_m'],record['time_s'])))
        ax.plot(q[2*s]*C/8e6,rate,'s',mfc='none',mec=colour,ms=12)
        ax.annotate(label,(q[2*s]*C/8e6,rate),xytext=(8,5),textcoords='offset points',color=colour)
    ax.set(xlim=(-110,150),xlabel='Offset from fixed original range reference (m)',ylabel='Physical excess-path rate (m/s)',title='ABC B/E locations in the original 162–168 s data')
    fig.colorbar(image,ax=ax,label='Original coherent cell power / conditional noise (dB)')
    fig.savefig(out/'03-selected-points.png',dpi=150);plt.close(fig)
    improvement=100*(best['prediction']['testing'][0]['heldout_error']/data['original_prediction']['testing'][0]['heldout_error']-1)
    result=dict(complete=True,accepted_calibration=False,source=str(args.source),reproduced_objectives=len(errors),
        maximum_absolute_cost_error=max(errors),unused_error_change_percent=improvement,
        local_component_diagnostics=diagnostics,measured_association_comparisons=comparisons,
        phase_information=best['information'],basis=data.get('extra_curve','quadratic'),bound_indices=best['bound_indices'],
        scope='Conditional association/mismatch test. An additional near-main-return echo is not automatically a known material point; bounds and branch alternatives remain explicit. No coordinate update or image-phase correction is adopted.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.source,source,Path(__file__),Path('support/isar/component_phase.py'))})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    rows=''.join(f'<tr><td>{v["start"]}</td><td>{v["local_B_E_component_to_sum_energy_ratio"]:.4f}</td><td>{v["B_E_envelope_separation_m"]:.2f}</td><td>{v["testing_error"]:.2f}</td><td>{v["bounds"]}</td></tr>' for v in diagnostics)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>ABC near-main-return phase separation</title><style>body{{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px}}</style>
<h1>Add a nearby echo before interpreting ABC's main-return phase</h1>
<p>The additional feature was acquired using training rows near the main return. Full original-row five-return fits change unused prediction error by {improvement:+.2f}%. It is not an antenna correction or forced phase alignment. All seven original/start objectives reproduce, and both stored measured B/E complex histories reproduce. The additional path basis here is {data.get('extra_curve','quadratic')}.</p>
<p>Selected fit bound indices: {best['bound_indices']}. The initial quadratic experiment reaches its additional envelope's lower search boundary, so its location is not accepted. A separately recorded flexible/wider-range control follows from that evidence. Phase information shown here does not count bound contacts as measurements.</p>
<img src="03-selected-points.png" alt="Selected B/E candidate centres in original ABC observations">
<img src="01-measured-phase-associations.png" alt="Five-band measured main-return phases before and after local ABC separation">
<img src="02-differences-and-information.png" alt="ABC versus Nine measured differences and conditional B/E phase precision">
<table><tr><th>Start</th><th>B/E component energy / pair-sum energy</th><th>B/E delay separation (m)</th><th>Unused error</th><th>Bounds</th></tr>{rows}</table>
<p>Unlike a whole-scene energy ratio dominated by distant D, the cancellation diagnostic uses only B and E. Large opposing-component decompositions must not be mistaken for physical points. Conditional Fisher precision marginalizes every other local phase, envelope, trajectory and active brightness parameter; it excludes unknown interference and association errors.</p>
<p>The four non-ABC curves stay at their independent original estimates. ABC is shown under two different feature choices after the split; similarity alone does not establish the same material point. Only constant origins are removed at 165 s. The 250 ms windows overlap, so visual RMS differences are not independent confidence intervals.</p>
<p><a href="analysis.json">All measured curves, local information and reproducibility</a> · <a href="../main-return-extra-ABC-flexible-wide/fits.json">Flexible, wider-range control; check its completion flag</a> · <a href="../main-return-residual-acquisition/index.html">Training-only acquisition</a> · <a href="../current-phase-and-geometry-tests.log">39 numerical checks</a></p></html>''')
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True);shutil.copyfile(Path(__file__),snapshots/Path(__file__).name)
    print(out/'index.html',improvement,diagnostics,flush=True)


if __name__=='__main__':main()
