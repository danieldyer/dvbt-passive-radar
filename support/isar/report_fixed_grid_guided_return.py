"""Plot measured phases for both ABC association branches and their controls."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cluster_phase import SplinePath
from compare_ais_phase import NAMES,COLOURS
from component_phase import component_measurements
from field import average
from fit_fixed_grid_phases import prepare_fixed
from flexible_echo_cluster import FlexibleEchoCluster
from measured_phase import anchored_history,interpolate_connected,local_average,range_rate_map
from motion import C
from report_measured_scatterer_phases import load_fields,serial
from release_fixed_grid_phase_guide import path_information
from smooth_echo_cluster import SmoothEchoCluster


def histories(model,parameters,selected):
    records=[]
    for cluster,index in zip(model.clusters,model.indices):
        records.append(component_measurements(cluster,np.asarray(parameters)[index])[selected])
    return records


def phase_comparison(records,bands,first,last):
    grid=np.arange(first+.2,last-.199,.05);measurements=[]
    for rec,b in zip(records,bands):
        path=rec['physical_path_change_m'];rate=float(np.interp((first+last)/2,rec['time_s'],np.gradient(path,rec['time_s'])))
        time,z,var=local_average(rec['time_s'],rec['complex_coefficient'],rec['conditional_information'],grid,.25,b.field.k,rate)
        measurements.append(dict(time_s=time,coefficient=z,variance=var,k=b.field.k,power_to_noise=abs(z)**2/var))
    margin=min(1.,.25*(last-first))
    middle=(grid>=first+margin)&(grid<=last-margin)
    anchor=int(np.argmax(np.where(middle,np.min([m['power_to_noise'] for m in measurements],axis=0),-np.inf)))
    for m in measurements:
        zero=np.zeros(len(grid));h=anchored_history(m['time_s'],m['coefficient'],m['variance'],m['k'],zero,zero,anchor)
        path=interpolate_connected(m['time_s'],h['measured_path_m'],grid)
        if np.isfinite(path[anchor]):path-=path[anchor]
        m['measured_path_change_m']=path
        value=m.pop('coefficient')
        m['averaged_real']=value.real;m['averaged_imag']=value.imag
        # Keep complex phase data through the original row coefficients in the
        # provenance; this compact file needs just the plotted paths and power.
    return dict(time_s=grid,anchor_time_s=grid[anchor],channels=measurements)


def main():
    root=Path('docs/data/isar_20261017');out=root/'fixed-grid-guided-abc-D-review';out.mkdir(exist_ok=True)
    source=root/'fixed-grid-guided-abc-D-fixed-static-C/fits.json';saved=json.loads(source.read_text())
    if not saved['complete']:raise ValueError('Wait for all declared starts')
    previous=json.loads(Path(saved['source']).read_text());first,last=saved['interval_s']
    geometry=Path('docs/data/calibration_20261016_sites')
    bands,part,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    for c,b in enumerate(bands):
        with np.load(source.parent/'observations'/f'channel-{c}.npz') as original:
            for key in ('value','row_time','row_information','information'):
                np.testing.assert_array_equal(original[key],b.field.data[key])
    origins=previous['selected_origins_samples'];curve=SplinePath(first,last,saved['guide_knot_spacing_s'])
    guide=SmoothEchoCluster(bands[:4],origins,3,curve,jerk_std=saved['guide_jerk_std'])
    costs=[]
    for fit in saved['guide_fits']:
        r,_=guide.evaluate(np.asarray(fit['parameters']));costs.append(abs(r@r*guide.energy-fit['cost']))
    chosen=saved['guide_fits'][saved['guide_best_index']]
    guide_p=np.asarray(chosen['parameters']);guide_records=histories(guide,guide_p,3)
    abc=FlexibleEchoCluster([bands[4]],[*origins,14.],4,curve)
    for fit in saved['trials']:
        r,_=abc.evaluate(np.asarray(fit['parameters']));costs.append(abs(r@r*abc.energy-fit['cost']))
    if max(costs)>1e-7:raise ValueError('Saved original-data objective changed')
    branches=[]
    for name,start in (('Near / training winner',saved['best_index']),('Far / alternative',1)):
        fit=saved['trials'][start];p=np.asarray(fit['parameters'])
        rec=histories(abc,p,4)[0]
        measurement=phase_comparison([*guide_records,rec],bands,first,last)
        branches.append(dict(name=name,start=start,parameters=p,measurement=measurement,
            conditional_phase_rms_deg=fit['selected_diagnostics'][0]['selected_unused_phase_rms_deg'],
            training_error=fit['cost'],validation_error=fit['prediction']['validation'][0]['heldout_error'],
            unused_error=fit['prediction']['testing'][0]['heldout_error'],
            brightness_condition=fit['active_brightness_condition'][0],
            component_to_sum_energy_ratio=fit['component_to_sum_energy_ratio'][0],
            abc_origin_m=p[8]*C/8e6,nearest_abc_return_separation_m=min(abs(p[8]-p[:8:2]))*C/8e6))
    fig,axes=plt.subplots(2,3,figsize=(17,8.4),layout='constrained')
    for row,branch in enumerate(branches):
        m=branch['measurement'];t=m['time_s'];reference=m['channels'][2]
        for c,(ch,colour) in enumerate(zip(m['channels'],COLOURS)):
            path=ch['measured_path_change_m']
            axes[row,0].plot(t,path,color=colour,label=NAMES[c])
            if c!=2:axes[row,1].plot(t,100*(path-reference['measured_path_change_m']),color=colour)
            axes[row,2].plot(ch['time_s'],10*np.log10(np.maximum(ch['power_to_noise'],1e-10)),color=colour)
        axes[row,0].set(title=branch['name'],ylabel='Measured physical path change (m)')
        axes[row,1].set(title='Channel minus Nine',ylabel='Interband path-change difference (cm)')
        axes[row,2].set(title='Conditional strength / fade evidence',ylabel='Power / conditional noise (dB)')
        for ax in axes[row]:ax.set(xlabel='Capture time (s)',xlim=[first,last]);ax.axvline(m['anchor_time_s'],color='#aaa',ls=':',lw=.6)
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle('Measured D-region phases: four-band guide and two ABC association branches\n'
        'Neighbours and static profile use frozen training coefficients. ABC guide is fixed only for acquisition; '
        'the plotted innovations come from data.',fontsize=11)
    fig.savefig(out/'01-measured-paths.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(20,4.8),layout='constrained',sharex=True,sharey=True)
    rates=np.linspace(-2,1,241);maps=[range_rate_map(b.field.time,b.field.data['value'],b.field.data['information'],b.field.k,rates)/b.field.pulse.samples[64].real for b in bands]
    vmax=np.ceil(10*np.log10(max(m.max() for m in maps)))
    for c,(ax,b,power) in enumerate(zip(axes,bands,maps)):
        offsets=b.field.offsets*C/8e6
        image=ax.imshow(10*np.log10(np.maximum(power,1e-8)),origin='lower',aspect='auto',
            extent=[offsets[0]-C/16e6,offsets[-1]+C/16e6,rates[0],rates[-1]],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
        if c<4:
            local=guide_p[guide.indices[c]];cluster=guide.clusters[c];q=local[cluster.slices[3]]
            path=cluster.arcs[3].known_path+curve(b.field.data['row_time'])@q[2:]
            rate=float(np.interp((first+last)/2,b.field.data['row_time'],np.gradient(path,b.field.data['row_time'])))
            ax.plot(q[0]*C/8e6,rate,'s',mfc='none',mec='#00e5ff',ms=11)
        else:
            for branch,marker in zip(branches,('s','o')):
                p=branch['parameters'];path=histories(abc,p,4)[0]['physical_path_change_m']
                rate=float(np.interp((first+last)/2,b.field.time,np.gradient(path,b.field.time)))
                ax.plot(branch['abc_origin_m'],rate,marker,mfc='none',mec='#00e5ff',ms=11)
        ax.set(title=NAMES[c],xlabel='Range offset from fixed reference (m)',xlim=(400,600))
    axes[0].set_ylabel('Physical excess-path rate (m/s)')
    fig.colorbar(image,ax=axes,label='Original cell power / conditional noise (dB)',shrink=.75)
    fig.suptitle('Selected D-region components in the original fixed-grid observations\nABC square: near branch; circle: farther alternative. Markers are fitted component centres, not identified hull points.',fontsize=12)
    fig.savefig(out/'02-selected-points.png',dpi=150);plt.close(fig)
    result=dict(complete=True,accepted_calibration=False,production_changed=False,source=str(source),
        reproduced_objectives=len(costs),maximum_absolute_cost_error=max(costs),branches=branches,
        limits='Conditional phase extraction freezes neighbouring returns and stationary coefficients fitted on training. It excludes their uncertainty and does not identify a physical hull point. One near-static Ten C nuisance path is held at its earlier training estimate. Two branch predictions disagree. Independent ABC path release is required before using this association for antenna calibration.')
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    release_path=root/'fixed-grid-guided-abc-D-independent/fits.json'
    released=json.loads(release_path.read_text())
    if not released['complete']:raise ValueError('Wait for the independent ABC path tests')
    independent=SmoothEchoCluster([bands[4]],[*origins,14.],4,curve,jerk_std=saved['guide_jerk_std'])
    grid=np.linspace(first,last,201);qguide=np.asarray(saved['fixed_guide_path_coefficients'])
    fig,axes=plt.subplots(2,3,figsize=(17,8.2),layout='constrained')
    summaries=[];information_audits=[]
    for row,fit in enumerate(released['trials']):
        p=np.asarray(fit['parameters']);r,_=independent.evaluate(p)
        error=abs(r@r*independent.energy-fit['cost'])
        if error>1e-7:raise ValueError('Independent original-data objective changed')
        q=p[independent.local_path_indices[0][4]]
        difference=curve(grid)@(q-qguide)
        axes[row,0].plot(grid,curve(grid,1)@qguide,label='External four-band guide',color='#555')
        axes[row,0].plot(grid,curve(grid,1)@q,label='ABC independently fitted',color=COLOURS[4])
        axes[row,0].set(title=f"Released start {fit['source_start']}",ylabel='Reference path rate (m/s)')
        axes[row,1].plot(grid,100*difference,color=COLOURS[4])
        axes[row,1].axhline(0,color='#999',lw=.7)
        axes[row,1].set(title='ABC minus guide; common phase origin',ylabel='Fitted path-change difference (cm)')
        before=path_information(independent,np.asarray(saved['trials'][fit['source_start']]['parameters']))
        after=path_information(independent,p)
        information_audits.append(dict(source_start=fit['source_start'],at_guided_solution=before,at_independent_solution=after))
        for info,label,style in ((before,'At constrained guide','--'),(after,'After independent fit','-')):
            if info['full_rank']:
                axes[row,2].plot(info['time_s'],info['phase_standard_deviation_deg'],style,label=label)
            else:
                axes[row,2].text(.04,.95,'Constrained-guide information is rank deficient',va='top',transform=axes[row,2].transAxes,fontsize=8)
        axes[row,2].set(title='Neighbour-marginalized conditional uncertainty',ylabel='σ of phase change relative to 165 s (°)')
        for ax in axes[row]:ax.set(xlim=(first,last),xlabel='Capture time (s)')
        summaries.append(dict(source_start=fit['source_start'],maximum_path_difference_m=float(max(abs(difference))),
            abc_origin_m=float(p[8]*C/8e6),training_cost=fit['cost'],unused_error=fit['prediction']['testing'][0]['heldout_error'],
            brightness_condition=fit['active_brightness_condition'][0],component_to_sum_energy_ratio=fit['component_to_sum_energy_ratio'][0],
            maximum_absolute_reproduced_cost_error=error))
    axes[0,0].legend(fontsize=8);axes[0,2].legend(fontsize=8)
    fig.suptitle('Release the ABC trajectory: does it independently recover the four-band path?\n'
        'Same original data and motion prior. Data-only information marginalizes brightness, neighbour paths, origins and constant phases.',fontsize=12)
    fig.savefig(out/'03-independent-path-test.png',dpi=150);plt.close(fig)
    (out/'independent-analysis.json').write_text(json.dumps(serial(dict(source=str(release_path),summaries=summaries,
        information_audits=information_audits,information_rank_policy='Pre-nuisance target column normalization, verified against full Gaussian covariance and an exactly unidentifiable path column. Original fit objectives/parameters are unchanged.',
        accepted_calibration=False,scope='Local conditional model comparison; branch ambiguity and unmodelled interference remain outside Fisher bounds.')),indent=2,allow_nan=False))
    rows=''.join(f'<tr><td>{b["name"]}</td><td>{b["abc_origin_m"]:.2f}</td><td>{b["nearest_abc_return_separation_m"]:.2f}</td><td>{b["training_error"]:.1f}</td><td>{b["validation_error"]:.1f}</td><td>{b["unused_error"]:.1f}</td><td>{b["brightness_condition"]:.2f}</td><td>{b["component_to_sum_energy_ratio"]:.2f}</td></tr>' for b in branches)
    baseline=saved['baseline']['prediction']['testing'][0]['heldout_error']
    gain=100*(1-branches[0]['unused_error']/baseline)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Corrected-grid D-region phase association</title>
<style>body{{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:6px;border:1px solid #ddd}}.note{{background:#edf3f7;padding:16px}}</style>
<h1>Does ABC contain the four-band D phase trajectory?</h1>
<p class="note"><b>The independent test does not validate this as a shared material-point trajectory.</b> Fixing the four-band guide gives {gain:.2f}% less unused prediction error, but releasing the ABC path gives a different trajectory. The near split becomes a strongly cancelling decomposition; the better-predicting far branch remains smooth but disagrees with the external path. These points are not used to refine antenna coordinates.</p>
<h2>Independent path release and actual phase precision</h2><a href="03-independent-path-test.png"><img src="03-independent-path-test.png" alt="Independent ABC path versus the four-band guide and marginalized phase uncertainty"></a>
<p>A conditional phase residual around 1.8° in the near guided branch is not its trajectory uncertainty. Once neighbour coordinates and brightness are marginalized, its conditional data-only phase-change standard deviation reaches about 59°. The far guided branch has a rank-deficient information direction. After release, the near branch has component/sum energy ratio about 50; it is not an identified isolated point. The far branch has ratio about 0.98 and improves unused prediction further, but follows a different phase trajectory. Fisher bounds here are local to each fitted model; they do not cover branch ambiguity or unmodelled interference.</p>
<h2>What the guide-constrained measurement looked like</h2>
<p>Seven/SBS/Nine/Ten estimate one shared 1 s-knot D path with a declared white-jerk prior. ABC has no role in that estimate. Its new component keeps the guide path fixed, while constant phase, range origin and positive brightness use ABC training rows. Four other ABC returns remain independent nuisance components. All coefficients stay frozen on unused rows. The plots show phase measured from those observations after conditional neighbour/static subtraction.</p>
<a href="02-selected-points.png"><img src="02-selected-points.png" alt="Selected components in all five measured range-rate maps"></a>
<a href="01-measured-paths.png"><img src="01-measured-paths.png" alt="Five-channel measured path histories for near and far ABC branches"></a>
<p>The training and validation blocks prefer the near branch; final unused blocks prefer the farther alternative. Its conditional phase residual is larger. Conditional phase residual alone is therefore insufficient to accept a physical decomposition. The independent path-release control must check whether ABC recovers the guide without fixing it, and must marginalize neighbouring parameters when reporting phase precision.</p>
<table><tr><th>Branch</th><th>ABC origin (m)</th><th>Nearest component separation (m)</th><th>Training error</th><th>Validation error</th><th>Unused error</th><th>Brightness condition</th><th>Component/sum energy ratio</th></tr>{rows}</table>
<p><b>Numerical control:</b> the unrestricted guide was stopped after Ten C approached a singular near-stationary solution. Finite differences disagreed with its derivatives. This repeat freezes only its two path coefficients at their earlier independent training estimate; all other parameters remain free. Both guide starts converge without bound contacts and agree in objective to 1.2×10⁻⁷. All seven completed objectives reproduce from the original observations.</p>
<p><a href="analysis.json">Measurements and diagnostics</a> · <a href="../fixed-grid-guided-abc-D-fixed-static-C/fits.json">All fits</a> · <a href="../fixed-grid-guided-abc-D-fixed-static-C/tests.log">26 numerical checks</a> · <a href="../fixed-grid-guided-abc-D/stopped.json">Stopped singular control</a> · <a href="../flash-kernel-phase/index.html">Known-filter phase audit</a></p>
<p><a href="../fixed-grid-guided-abc-D-independent/fits.json">All independent fits and information matrices</a> · <a href="independent-analysis.json">Independent comparison values</a></p>
<p>No transmitter/receiver coordinates, production processing or focused videos are accepted from this conditional comparison.</p></html>''')
    sources=[source,release_path,Path(__file__),Path('support/isar/component_phase.py'),Path('support/isar/measured_phase.py'),
        Path('support/isar/release_fixed_grid_phase_guide.py'),Path('support/isar/test_path_information.py')]
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for p in sources:
        if p.suffix=='.py':shutil.copyfile(p,snapshots/p.name)
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
