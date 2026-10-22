"""Review the nested constant-phase constraint against unused complex data."""
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
from relative_scatterer_phase import RelativeScattererPhase
from report_measured_scatterer_phases import load_fields,serial
from smooth_echo_cluster import SmoothEchoCluster
from waveform_kernels import read_kernels
from motion import C


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();saved=json.loads(args.source.read_text())
    if not saved['complete']:raise ValueError('Wait for all declared carrier-branch starts')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    geometry=Path('docs/data/calibration_20261016_sites');first,last=saved['interval_s']
    mean,all_parts,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    channels=saved['channels'];kernel=Path('build/isar-20261016/waveform-kernels-30s')
    _,records,samples=read_kernels(kernel/'response.kernels')
    with np.load(kernel/'rows.npz') as grids:
        bands=[known_band(mean[c],grids[f'channel_{c}'],records[c],samples[c]) for c in channels]
    curve=SplinePath(first,last,saved['knot_spacing_s'])
    base=SmoothEchoCluster(bands,saved['origins_samples'],(1,3),curve,shared=False,jerk_std=saved['jerk_std'])
    model=RelativeScattererPhase(base,1,3,reference_band=channels.index(2))
    before=np.array(saved['original_independent_parameters']);errors=[]
    r,_=base.evaluate(before);errors.append(abs(r@r*base.energy-saved['original_independent_cost']))
    for fit in saved['trials']:
        p=np.array(fit['parameters']);np.testing.assert_allclose(model.mapping(p),fit['base_parameters'],rtol=2e-15,atol=2e-12)
        r,_=model.evaluate(p);errors.append(abs(r@r*base.energy-fit['cost']))
    if max(errors)>1e-7:raise ValueError('Original-data objective changed')
    best=saved['trials'][saved['best_index']];after=np.array(best['base_parameters'])
    old=np.array([v['heldout_error'] for v in saved['independent_prediction']['testing']])
    new=np.array([v['heldout_error'] for v in best['prediction']['testing']])
    changes=100*(new/old-1);total_change=100*(new.sum()/old.sum()-1)
    origin_changes=(after-before)[model.origin_indices]*C/8e6
    measured=[]
    fig,axes=plt.subplots(2,len(channels),figsize=(4.4*len(channels),8),layout='constrained',squeeze=False,sharex=True,sharey=True)
    for c,original in enumerate(channels):
        band=bands[c];cluster=base.clusters[c];index=base.indices[c];entry=[]
        for row,s in enumerate((1,3)):
            diagnostics=[]
            for parameters,label,colour in ((before,'Independent phases','#999'),(after,'Constant-phase constraint',COLOURS[original])):
                record=component_measurements(cluster,parameters[index])[s]
                phase=np.angle(record['complex_coefficient']*np.exp(1j*band.field.k*record['physical_path_change_m']))
                power=np.sum(abs(record['frozen_component_prediction'])**2,axis=1);unused=(all_parts[original]==3)&(power>0)
                rms=float(np.degrees(np.sqrt(np.average(phase[unused]**2,weights=power[unused]))))
                axes[row,c].plot(record['time_s'][unused],np.degrees(phase[unused]),'.',ms=3,color=colour,label=label,alpha=.65)
                diagnostics.append(dict(label=label,time_s=record['time_s'],testing=unused,
                    phase_residual_deg=np.degrees(phase),unused_conditional_phase_rms_deg=rms))
            entry.append(dict(component=s,label='B' if s==1 else 'D',diagnostics=diagnostics))
            axes[row,c].set(title=NAMES[original]+' · '+('B' if s==1 else 'D'),
                ylabel='Unused conditional phase innovation (degrees)',xlim=(first,last),ylim=(-180,180))
            axes[row,c].axhline(0,color='#aaa',lw=.5)
        axes[-1,c].set_xlabel('Capture time (s)');measured.append(entry)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('The constant-phase constraint is checked on actual unused complex observations\n'
        'Other returns and stationary coefficients are training-frozen. No shared temporal path or test-time phase alignment.',fontsize=12)
    fig.savefig(out/'01-unused-phase-innovations.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    x=np.arange(len(channels));width=.35
    for s,label,offset in ((0,'B',-.5),(1,'D',.5)):
        axes[0].bar(x+width*offset,100*origin_changes[:,s],width,label=label)
    axes[0].set_xticks(x,[NAMES[c] for c in channels]);axes[0].set(ylabel='Envelope-origin change (cm)',title='Delay changes paid for by the original likelihood');axes[0].legend()
    axes[1].bar(x,changes,color=[COLOURS[c] for c in channels]);axes[1].set_xticks(x,[NAMES[c] for c in channels])
    axes[1].set(ylabel='Unused squared prediction error change (%)',title='Negative means improved unused prediction')
    for ax in axes:ax.axhline(0,color='#999',lw=.6)
    fig.suptitle('Use envelope delay and relative carrier phase together; measured data stay unchanged',fontsize=12)
    fig.savefig(out/'02-delay-and-prediction.png',dpi=150);plt.close(fig)
    bounds=[int(model.kept[i]) for i in best['audit']['bound_indices']]
    result=dict(complete=True,accepted_calibration=False,production_changed=False,source=str(args.source),
        completed_starts=len(saved['trials']),reproduced_objectives=len(errors),maximum_absolute_cost_error=max(errors),
        channels=channels,training_cost_change=best['cost']-saved['original_independent_cost'],
        unused_error_change_percent=changes,total_unused_error_change_percent=total_change,
        envelope_origin_changes_m=origin_changes,unused_phase=measured,
        best_projected_gradient=best['audit']['projected_gradient'],best_base_bound_indices=bounds,
        scope='Common intrinsic constant phase between two candidates across the listed bands is a nested scattering hypothesis. '
            'Independent temporal trajectories, magnitudes and origins remain. This necessary phase/delay consistency test does not '
            'establish physical point identity, site assignment, inter-window phase transfer or a geographic motion fit.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.source,Path(__file__),Path('support/isar/relative_scatterer_phase.py'),Path('support/isar/component_phase.py'))})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    rows=''.join(f'<tr><td>{f["start"]}</td><td>{f["cost"]:.5f}</td><td>{sum(v["heldout_error"] for v in f["prediction"]["testing"]):.5f}</td><td>{f["audit"]["projected_gradient"]:.3g}</td><td>{f["audit"]["bound_indices"]}</td></tr>' for f in saved['trials'])
    names=', '.join(NAMES[c] for c in channels)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Multiscatterer relative-phase constraint</title><style>body{{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:7px}}</style>
<h1>Can the selected returns share a physical relative phase across bands?</h1>
<p>This tests {names} over {first:g}–{last:g} s. It changes only the B/D constant-phase relation: their relative carrier phase plus carrier wavenumber times their relative envelope delay must equal one intrinsic scatterer-phase difference. An arbitrary constant band phase cancels. Each return's temporal trajectory and positive brightness remain independent. The recorded samples and their noise metric do not change.</p>
<p>Selected training winner: start {saved['best_index']}. Training objective change {best['cost']-saved['original_independent_cost']:+.5f}; total unused prediction error change {total_change:+.5f}%. These are original complex-data likelihood/prediction comparisons, not a target SNR or antenna precision claim. Conditional innovations below are measured from unused samples after training-frozen subtraction.</p>
<img src="02-delay-and-prediction.png" alt="Delay shifts and actual unused prediction change">
<img src="01-unused-phase-innovations.png" alt="Measured unused B and D phase innovations before and after the restriction">
<p>The displayed phase residuals are not uncertainty bounds and include noise and scene mismatch. The relation itself is imposed, so plotting its zero residual would not be evidence of success. Instead the plots show actual unused complex observations and the cost of moving envelope delays to satisfy it. All {len(errors)} source/start objectives reproduce within {max(errors):.3g}.</p>
<table><tr><th>Start</th><th>Training objective</th><th>Unused squared error</th><th>Projected gradient</th><th>Reduced bound indices</th></tr>{rows}</table>
<p>Best bound coordinates mapped to the original independent model: {bounds}. The previously unidentifiable Ten C nuisance path remains fixed at its earlier training estimate; this conditional restriction is retained from the independent control. No new calibration or phase correction is adopted here. A shared intrinsic phase is a testable scattering assumption, not guaranteed by close antenna positions. It needs cross-channel and inter-window prediction before being used in coherent image formation.</p>
<p><a href="../longer-known-phases-6s-review/02-selected-points.png">Selected B/D candidates</a> · <a href="analysis.json">All measured diagnostics and source identities</a> · <a href="../relative-phase-first-four-tests.log">35 numerical checks</a> · <a href="../index.html">Analysis index</a></p></html>''')
    sources=out/'sources';sources.mkdir(exist_ok=True)
    for name in (Path(__file__).name,'relative_scatterer_phase.py'):
        shutil.copyfile(Path(__file__).with_name(name),sources/name)
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
