"""Review physical primary fits without confusing local focus with a survey."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from compare_ais_phase import NAMES,COLOURS
from field import average
from fit_physical_echo_cluster import geometry_for,make_model
from fit_shared_echo_cluster import prepare
from report_measured_scatterer_phases import load_fields,serial
from report_shared_echo_cluster import parts
from ship_site_signature import licensed_options
from flash_geometry import assignments


def primary_diagnostic(cluster,parameters,testing):
    b=cluster.base;signal=parts(cluster,parameters);pred=signal.sum(axis=0)
    residual=b.field.y-(pred-signal[0]).reshape(b.field.y.shape)
    raw,_=cluster.response_blocks(parameters)[0]
    unit=b.reduced(raw[:,:,None])[:,0].reshape(b.field.y.shape)
    norm=np.sum(abs(unit)**2,axis=1)
    z=np.sum(unit.conj()*residual,axis=1)/np.sqrt(np.maximum(norm,1e-300))
    mu=np.sum(unit.conj()*signal[0].reshape(b.field.y.shape),axis=1)/np.sqrt(np.maximum(norm,1e-300))
    phase=np.angle(z*mu.conj());weight=abs(mu)**2
    use=testing&(weight>0)
    rms=float(np.degrees(np.sqrt(np.average(phase[use]**2,weights=weight[use])))) if use.any() else None
    time=b.field.data['row_time']-b.midpoint;g=b.geometry;q=parameters[:g.count]
    if hasattr(g,'trajectory'):local,_,_=g.trajectory(q,time);tx,rx=g.antennas_at(q)
    else:
        p,v,a,tx,rx=g.state(q);local=p+time[:,None]*v+.5*time[:,None]**2*a
    path=np.linalg.norm(local-tx[b.channel],axis=1)+np.linalg.norm(local-rx,axis=1)-np.linalg.norm(tx[b.channel]-rx)
    curve=path-b.field.data['centre_path_m']
    # Empirical nuisance uses the current nominal AIS geometry as a known
    # increment. Recover that same fixed reference for the plotted comparison.
    curve-=cluster.empirical.known_path
    curve-=np.interp(b.midpoint,b.field.data['row_time'],curve)
    curve=average(curve,b.field.data['row_information'],b.field.metadata['block_rows'])[0]
    measured=curve-phase/b.field.k
    k=len(b.knots);pieces=signal[:,b.selected]
    return dict(time_s=b.field.time,testing=testing,phase_residual_deg=np.degrees(phase),
        predicted_information=weight,conditional_phase_rms_deg=rms,
        residual_path_m=curve,measured_path_innovation_m=measured,
        component_to_sum_energy_ratio=float(np.sum(abs(pieces)**2)/np.sum(abs(pieces.sum(axis=0))**2)))


def main():
    root=Path('docs/data/isar_20261017');out=root/'physical-flash-review';out.mkdir(exist_ok=True)
    geometry=Path('docs/data/calibration_20261016_sites');geo=json.loads((geometry/'geometry-prior.json').read_text())
    receiver=np.asarray(geo['receiver_ecef_m']);rotation=np.asarray(geo['ecef_to_enu'])
    source=root/'shared-echo-cluster-flash-four-seconds/fits.json';baseline=json.loads(source.read_text())
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    first,last=baseline['interval_s'];midpoint=(first+last)/2
    bands,partition=prepare(fields,geometry,first,last,range_extent=baseline['range_extent_samples'])
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    choices=assignments(licensed_options(geometry));records=[];models=[];errors=[];inputs=[];completion=[]
    for folder in ('physical-four-echo-flash','ais-course-four-echo-flash','ais-fixed-four-echo-flash','ais-envelope-four-echo-flash'):
        path=root/folder/'fits.json'
        if not path.exists():continue
        text=path.read_text();data=json.loads(text);snapshot=out/(folder+'.json');snapshot.write_text(text);inputs.append(snapshot)
        completion.append(bool(data['complete']))
        for group in data['groups']:
            if not group['trials']:continue
            assignment=next(a for a in choices if ''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a)==group['assignment'])
            g=geometry_for(assignment,track,midpoint,receiver,rotation,
                np.asarray(data['measured_initialization_observables']),np.asarray(data['conditional_initialization_covariance']),data.get('motion_model','quadratic'))
            model=make_model(bands,g,midpoint,data['nuisance_origins_samples'],data.get('envelope_bias',False))
            for trial in group['trials']:
                p=np.asarray(trial['parameters']);r,_=model.evaluate(p)
                error=abs(r@r*model.energy-trial['cost'])/trial['cost'];errors.append(error)
                if error>1e-9:raise ValueError('Saved objective reproduction failed')
                diagnostics=[primary_diagnostic(c,p[i],part==3)
                    for c,i,part in zip(model.clusters,model.indices,partition)]
                unused=sum(r['heldout_error'] for r in trial['prediction'])
                record=dict(model=data.get('motion_model','quadratic'),assignment=group['assignment'],
                    free_antennas=trial['free_antennas'],source_complete=data['complete'],
                    cost=trial['cost'],prior_cost=trial['prior_cost'],
                    unused_error=unused,unused_error_ratio=unused/data['reference_independent_unused_error'],
                    projected_gradient=trial['audit']['projected_gradient'],bounds=trial['audit']['bound_indices'],
                    primary_range_origin_m=trial['primary_range_origin_m'],antenna_fit=trial['antenna_fit'],
                    envelope_offset_m=trial.get('envelope_offset_m'),
                    phase_rms_deg=[d['conditional_phase_rms_deg'] for d in diagnostics],
                    component_to_sum_energy_ratio=[d['component_to_sum_energy_ratio'] for d in diagnostics])
                if hasattr(g,'follow_course_change'):
                    h=np.asarray(model.information(p)['matrix'])
                    variable=np.arange(g.count) if trial['free_antennas'] else np.arange(18,g.count)
                    covariance=np.linalg.solve(h[np.ix_(variable,variable)]+np.eye(len(variable)),np.eye(len(variable)))
                    target=covariance[-4:,-4:];sd=np.sqrt(np.diag(target))
                    record['ais_offset_time']=dict(lever_arm_enu_m=p[18:21]*g.nuisance_sigma[:3],
                        time_shift_s=p[21]*g.nuisance_sigma[3],
                        conditional_standard_deviation_m_m_m_s=sd*g.nuisance_sigma,
                        conditional_correlation=target/sd[:,None]/sd[None,:],
                        data_information_eigenvalues=np.linalg.eigvalsh(h[np.ix_(variable,variable)]),
                        scope='Local proper-complex mean information with declared real priors, eliminating phase, positive brightness and all other returns. Active bounds are not extra information. This optimistic covariance excludes coherent model mismatch, wrong associations and missing true heading; it is not a measured AIS latency uncertainty or a conclusive physical survey.')
                records.append(record)
                if group['assignment']=='00002' and trial is group['trials'][-1]:
                    models.append((record,diagnostics))
    if not models:raise ValueError('Wait for at least one completed physical control')
    fig,axes=plt.subplots(len(models),3,figsize=(18,4.5*len(models)),layout='constrained',squeeze=False)
    for row,(record,diagnostics) in enumerate(models):
        for c,(d,colour) in enumerate(zip(diagnostics,COLOURS)):
            t=np.asarray(d['time_s']);test=np.asarray(d['testing'])
            axes[row,0].plot(t,100*np.asarray(d['residual_path_m']),color=colour,lw=1.3,label=NAMES[c])
            axes[row,0].plot(t[test],100*np.asarray(d['measured_path_innovation_m'])[test],'.',color=colour,ms=2.5,alpha=.6)
            axes[row,1].plot(t[test],np.asarray(d['phase_residual_deg'])[test],'.',color=colour,ms=3)
        axes[row,0].set(ylabel='Residual path change (cm)',xlabel='Capture time (s)')
        axes[row,1].set(ylabel='Conditional unused phase innovation (degrees)',xlabel='Capture time (s)',ylim=(-180,180))
        empirical=np.array([c['trials'][c['best_index']]['parameters'][2] for c in baseline['independent']])*299792458/8e6
        axes[row,2].plot(range(5),empirical,'ko-',label='Independent envelope fits')
        axes[row,2].plot(range(5),record['primary_range_origin_m'],'s-',color='#d55e00',label='Physical primary')
        axes[row,2].set_xticks(range(5),NAMES,rotation=25);axes[row,2].set_ylabel('Range origin from saved AIS reference (m)')
        antennas='refined' if record['free_antennas'] else 'fixed'
        axes[row,0].set_title(record['model']+' · antennas '+antennas+(' · envelope control' if record['envelope_offset_m'] is not None else ''))
        axes[row,1].set_title('Other three returns subtracted using frozen training fits')
        axes[row,2].set_title('Envelope location versus physical point delay')
        if record['envelope_offset_m'] is not None:
            axes[row,2].plot(range(5),np.asarray(record['primary_range_origin_m'])+record['envelope_offset_m'],'.-',color='#0072b2',label='Envelope with fixed band offsets')
        axes[row,0].legend(fontsize=8,ncol=3);axes[row,2].legend(fontsize=8)
    fig.suptitle('Physical flash-point controls on one four-second aperture\n'
        'No additional stationary or multipath model. Curves are bin-averaged physical paths; dots are measured circular innovations on that model branch.',fontsize=13)
    fig.savefig(out/'01-physical-phase-controls.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{r["model"]}</td><td>{r["assignment"]}</td><td>{r["free_antennas"]}</td><td>{r["envelope_offset_m"] is not None}</td><td>{r["cost"]:.2f}</td><td>{100*(r["unused_error_ratio"]-1):+.3f}%</td><td>{r["projected_gradient"]:.2g}</td><td>{r["bounds"]}</td></tr>' for r in records)
    phase_rows=''.join(f'<tr><td>{r["model"]}</td><td>{r["assignment"]}</td><td>{r["free_antennas"]}</td><td>{r["envelope_offset_m"] is not None}</td><td>'+', '.join(f'{name}: {v:.1f}' for name,v in zip(NAMES,r['phase_rms_deg']))+'</td></tr>' for r in records)
    ais_rows=''.join(f'<tr><td>{r["model"]}</td><td>{r["assignment"]}</td><td>{r["free_antennas"]}</td><td>{r["envelope_offset_m"] is not None}</td><td>{r["ais_offset_time"]["time_shift_s"]:+.2f}</td><td>'+', '.join(f'{v:+.1f}' for v in r['ais_offset_time']['lever_arm_enu_m'])+'</td><td>'+', '.join(f'{v:+.3f}' for v in np.asarray(r['ais_offset_time']['conditional_correlation'])[:3,3])+'</td></tr>' for r in records if 'ais_offset_time' in r)
    summary=dict(accepted_calibration=False,source_jobs_complete=all(completion),records=records,reproduced_objectives=len(errors),maximum_relative_cost_error=max(errors),
        scope='Snapshot of completed controls only. Source jobs may still be active. Initialization covers 32 licensed assignments, but the original-data fits cover only the explicitly listed finalists. Point-envelope fits and carrier-phase agreement are not an accepted physical survey. AIS-course prescribes interpolated course change, not measured true heading; no yaw is fitted.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__),*inputs]})
    (out/'analysis.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False))
    job_status='All source control jobs are complete.' if all(completion) else 'Some source control jobs remain in progress.'
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Physical flash primary with independent neighbours</title>
<style>body{{font:16px/1.55 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:8px;border:1px solid #ddd}}table{{border-collapse:collapse}}.note{{padding:16px;background:#eef4f8}}</style>
<h1>Physical flash-point motion with A/C/D kept separate</h1>
<p class="note">This snapshot contains {len(records)} completed controls. {job_status} No locations or motion solution have been accepted. Every value below is reproduced from a frozen copy of the completed fit records.</p>
<p>All five channels and 33 original range cells enter one 163–167 s likelihood. The main B return has a physical two-leg propagation model with the direct-path baseline removed. A/C/D each retain independent range origins, constant phases and quadratic trajectories in each band. Real nonnegative brightness is fitted on training only. Each original row is used once; overlapping window summaries are not treated as independent observations.</p>
<p>The quadratic model has free local position/velocity/acceleration corrections. AIS-fixed instead uses the interpolated AIS track plus one constant N/E/up offset and a constant time shift. AIS-course rotates that same lever arm by the prescribed course change, without fitting yaw. There are no true-heading reports in this AIS export: matching course change to heading change is a sensitivity hypothesis. Neither AIS model is adopted merely because it has fewer free parameters.</p>
<img src="01-physical-phase-controls.png" alt="Physical path and measured phase innovations, with independent versus physical envelope centres">
<p>Frozen nuisance subtraction makes these conditional target diagnostics, not independent raw phase unwraps. The envelope comparison matters: a common phase trend does not prove that independently fitted band envelope centres are unbiased measurements of one point. Antenna shifts must not be accepted simply because they absorb that mismatch. The original carrier deramp remains immutable throughout.</p>
<table><tr><th>Motion</th><th>Sites</th><th>Antennas free</th><th>Envelope offsets free</th><th>Training data NLL + prior penalty</th><th>Unused error change versus independent empirical fits</th><th>Projected gradient</th><th>Bound coordinates</th></tr>{rows}</table>
<p>0 = Ornata, 1 = Eyre, 2 = Ridge, ordered Seven/SBS/Nine/Ten/ABC. Summary proposals cover all 32 licensed assignments; these raw-data controls are not yet an exhaustive final site comparison. The proper-complex data NLL plus half the real standardized prior squared is used once across all channels. Site/antenna prior scales, bounds and fitted offsets are retained in the numerical records.</p>
<table><tr><th>Motion</th><th>Sites</th><th>Antennas free</th><th>Envelope offsets free</th><th>Conditional unused primary phase RMS (degrees)</th></tr>{phase_rows}</table>
<h2>AIS time shift and lever-arm coupling</h2>
<p>The fitted AIS shift is a nuisance parameter, not a measured reporting latency. Along-track lever arm, time shift and antenna corrections can trade against one another. The table exposes that tradeoff using conditional local correlations; full matrices and prior scales are in the numerical summary. Thermal-only covariance and bound contacts do not establish a calibrated geographic uncertainty.</p>
<table><tr><th>Motion</th><th>Sites</th><th>Antennas free</th><th>Envelope offsets free</th><th>Time shift (s)</th><th>Lever arm E/N/up (m)</th><th>Conditional E/time, N/time, up/time correlations</th></tr>{ais_rows}</table>
<p>The optional envelope-offset case is a sensitivity test: each band gets one constant envelope offset over the entire interval, while geometric carrier phase and Doppler remain unchanged. It tests whether point-envelope mismatch is driving antenna corrections. No time-dependent phase screen or change to the measured samples is introduced, and this freedom is not adopted as a production correction.</p>
<p>All {len(errors)} saved objectives reproduce; maximum relative discrepancy {max(errors):.3g}. <a href="analysis.json">Metrics and snapshot provenance</a> · <a href="../physical-four-echo-tests.log">Numerical tests</a> · <a href="../adjacent-scatterer-phases/index.html">Adjacent-window phase controls</a> · <a href="../index.html">Analysis index</a></p>
<p>Method context: <a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5233">Garry and Smith, Passive ISAR part I</a> separates translation compensation, apparent rotation and backprojection onto a chosen image plane. The <a href="https://cris.tau.ac.il/en/publications/joint-self-calibration-of-receiver-geometry-timing-and-target-pos/">2026 multistatic self-calibration study</a> distinguishes measurement information from prior constraints and explicitly tests geometric ambiguities. Neither source establishes identifiability or model adequacy for this recording; those require the present data tests.</p></html>''')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
