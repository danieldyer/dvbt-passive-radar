"""Review stationary separation, phase uncertainty and original-data controls."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cluster_phase import crop
from compare_ais_phase import NAMES,COLOURS,path_geometry
from field import Field
from fixed_range_field import stationary_profile_design
from phase_noise import phase_variance
from report_measured_scatterer_phases import load_fields,serial
from shared_echo_cluster import ReferencedAmplitudeArc,SharedEchoCluster
from ship_site_signature import licensed_options


def reload_bands(folder,group,data,fields,geometry):
    options=licensed_options(geometry);geo=json.loads((geometry/'geometry-prior.json').read_text())
    rx=np.asarray(geo['receiver_ecef_m']);reference=np.asarray(next(o for o in options if o['channel']==2 and o['site_id']=='12013')['transmitter_ecef_m'])
    first,last=data['interval_s'];mid=(first+last)/2;bands=[];parts=[];paths=[]
    for c,record in enumerate(fields):
        path=folder/(group['mode']+'-observations')/f'channel-{c}.npz';paths.append(path)
        with np.load(path) as saved:f=Field(dict(saved),saved['pulse_samples'])
        old=crop(record['data'],first,last,np.arange(-8,9));d=old.data
        if not np.array_equal(d['row_time'],f.data['row_time']):raise ValueError('Original row epochs changed')
        tx=np.asarray(next(o for o in options if o['channel']==c and o['site_id']==('12014' if c==4 else '12013'))['transmitter_ecef_m'])
        p,v=path_geometry(d['centre_ecef_m'],d['centre_velocity_ecef_mps'],tx,rx)
        pref,vref=path_geometry(d['centre_ecef_m'],d['centre_velocity_ecef_mps'],reference,rx)
        known=p-pref;known-=np.interp(mid,d['row_time'],known)
        observer=float(np.interp(mid,d['row_time'],v));known_rate=v-vref-observer
        if not np.all(f.data['centre_omega']==-f.k*.001024*observer):raise ValueError('Observer Doppler reference changed')
        part=np.floor((f.time-first)/.125).astype(int)%4;training=part%2==0
        knots=np.r_[d['row_time'][0],np.arange(first+.25,last,.25),d['row_time'][-1]]
        design=stationary_profile_design(f) if group['mode']=='stationary' else None
        bands.append(ReferencedAmplitudeArc(f,training,knots,first,last,known,known_rate,stationary_design=design));parts.append(part)
    return bands,parts,paths


def phase_statistics(model,p,partitions):
    stats=[]
    for cluster,index,part in zip(model.clusters,model.indices,partitions):
        b=cluster.base;local=p[index];value=cluster.evaluate(local);mu=value['design']@value['amplitude']
        y=b.field.y.ravel();beta=b.nuisance.solve(y[b.selected]-mu[b.selected]);static=b.design@beta
        clean=(y-static).reshape(b.field.y.shape);prediction=mu.reshape(b.field.y.shape)
        raw,_=b.response(local);unit=b.reduced(raw[:,:,None])[:,0].reshape(b.field.y.shape)
        norm=np.sqrt(np.maximum(np.sum(abs(unit)**2,axis=1),1e-300))
        z=np.sum(unit.conj()*clean,axis=1)/norm;expected=np.sum(unit.conj()*prediction,axis=1)/norm
        phase=np.angle(z*expected.conj());snr=abs(expected)**2;test=(part==3)&(snr>0)
        rms=float(np.degrees(np.sqrt(np.average(phase[test]**2,weights=snr[test]))))
        thermal=float(np.degrees(np.sqrt(np.average([phase_variance(v) for v in snr[test]],weights=snr[test]))))
        local_information=np.asarray(value['information']);scale=np.sqrt(np.diag(local_information))
        normalized=local_information/scale[:,None]/scale[None,:];eigen=np.linalg.eigvalsh(normalized)
        uncertainty=dict(identifiable=False,normalized_information_eigenvalues=eigen)
        if eigen[0]>eigen[-1]*1e-10:
            covariance=np.linalg.solve(normalized,np.eye(4))/scale[:,None]/scale[None,:]
            uncertainty.update(identifiable=True,covariance=covariance,
                rate_standard_deviation_mps=np.sqrt(covariance[2,2])/b.scale,
                curvature_standard_deviation_mps2=2*np.sqrt(covariance[3,3])/b.scale**2)
        t=b.field.time;u=(t-b.epoch)/b.scale
        known=np.interp(t,b.field.data['row_time'],b.known_path)
        curve=known+local[2]*u+local[3]*u*u
        measured=curve-phase/b.field.k
        relative_standard_deviation=None
        if uncertainty['identifiable']:
            basis=np.column_stack([np.zeros(len(t)),np.zeros(len(t)),u,u*u])
            relative_standard_deviation=np.sqrt(np.maximum(0.,np.einsum('ni,ij,nj->n',basis,covariance,basis)))
        primary=mu[b.selected];projected=b.nuisance.remove(primary)
        retained=float(np.vdot(projected,projected).real/np.vdot(primary,primary).real)
        singular=b.nuisance.singular_values
        stats.append(dict(time_s=t,testing=part==3,path_m=curve,measured_innovation_path_m=measured,
            phase_residual_deg=np.degrees(phase),conditional_phase_rms_deg=rms,
            known_mean_thermal_phase_rms_deg=thermal,median_test_bin_snr_db=10*np.log10(np.median(snr[test])),
            total_test_matched_snr_db=10*np.log10(snr[test].sum()),
            conditional_information=uncertainty,relative_path_standard_deviation_m=relative_standard_deviation,
            reference_rate_mps=local[2]/b.scale,range_origin_m=local[0]*299792458/8e6,
            retained_fitted_signal_energy=retained,stationary_rank=b.nuisance.rank,
            stationary_condition=float(singular.max()/singular.min()) if len(singular) else 1.))
    return stats


def main():
    root=Path('docs/data/isar_20261017');out=root/'fixed-grid-phase-review';out.mkdir(exist_ok=True)
    geometry=Path('docs/data/calibration_20261016_sites');fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    records=[];sources=[Path(__file__),Path('support/isar/phase_noise.py')];errors=[]
    for folder,mode in (('fixed-grid-phase-AIS3-exact','raw'),('fixed-grid-phase-AIS3-stationary','stationary'),
            ('fixed-grid-phase-Flash-stationary','stationary'),('fixed-grid-phase-AIS2-stationary','stationary')):
        directory=root/folder;source=directory/'fits.json';data=json.loads(source.read_text())
        group=next(g for g in data['groups'] if g['mode']==mode)
        if len(group['independent'])!=5 or any('best_index' not in c for c in group['independent']) or 'best_common_index' not in group:
            raise ValueError('Wait for completed independent and common controls')
        bands,partitions,observations=reload_bands(directory,group,data,fields,geometry)
        sources += [source,*observations];independent=[]
        for c,entry in enumerate(group['independent']):
            model=SharedEchoCluster([bands[c]],[data['selected_origin_samples']],origin_radius=1.5)
            for fit in entry['trials']:
                r,_=model.evaluate(np.asarray(fit['parameters']));errors.append(abs(r@r*model.energy-fit['cost'])/fit['cost'])
            fit=entry['trials'][entry['best_index']];p=np.asarray(fit['parameters'])
            independent.append(dict(fit=fit,statistics=phase_statistics(model,p,[partitions[c]])[0]))
        common=SharedEchoCluster(bands,[data['selected_origin_samples']],origin_radius=1.5)
        for fit in group['common']:
            r,_=common.evaluate(np.asarray(fit['parameters']));errors.append(abs(r@r*common.energy-fit['cost'])/fit['cost'])
        fit=group['common'][group['best_common_index']];p=np.asarray(fit['parameters'])
        records.append(dict(window=data['window'],mode=mode,interval_s=data['interval_s'],
            bands=bands,independent=independent,common=fit,common_statistics=phase_statistics(common,p,partitions)))
    if max(errors)>1e-9:raise ValueError('Saved likelihood reproduction failed')
    raw,stat=records[:2]
    for a,b in zip(raw['bands'],stat['bands']):
        np.testing.assert_array_equal(a.field.data['value'],b.field.data['value'])
        np.testing.assert_array_equal(a.field.data['information'],b.field.data['information'])
    fig,axes=plt.subplots(2,5,figsize=(20,9),layout='constrained',sharex=True,sharey=True)
    for row,record in enumerate((raw,stat)):
        for c,entry in enumerate(record['independent']):
            d=entry['statistics'];t=np.asarray(d['time_s']);test=np.asarray(d['testing']);ax=axes[row,c]
            ax.plot(t,np.asarray(d['path_m']),color=COLOURS[c],lw=1.5)
            ax.plot(t[test],np.asarray(d['measured_innovation_path_m'])[test],'.',color=COLOURS[c],ms=3,alpha=.6)
            sigma=d['relative_path_standard_deviation_m']
            if sigma is not None:ax.fill_between(t,np.asarray(d['path_m'])-np.asarray(sigma),np.asarray(d['path_m'])+np.asarray(sigma),color=COLOURS[c],alpha=.15)
            common=record['common_statistics'][c];ax.plot(t,common['path_m'],'k--',lw=1.2)
            ax.set(title=NAMES[c]+' · '+record['mode'],xlabel='Capture time (s)')
            if c==0:ax.set_ylabel('Path change from midpoint (m)')
    fig.suptitle('Same AIS3 observations: independent measured moving trajectories before/after stationary separation\n'
        'Colour: independent fits and unused measured circular innovations. Shading: conditional thermal 1σ of relative path. Black: shared-trajectory control.',fontsize=13)
    fig.savefig(out/'01-before-after-trajectories.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5.5),layout='constrained')
    x=np.arange(5)
    for row,record in enumerate((raw,stat)):
        measured=[e['statistics']['conditional_phase_rms_deg'] for e in record['independent']]
        predicted=[e['statistics']['known_mean_thermal_phase_rms_deg'] for e in record['independent']]
        axes[0].bar(x+(row-.5)*.35,measured,.35,label=record['mode'])
        axes[0].plot(x+(row-.5)*.35,predicted,'kx',ms=8)
        rates=[e['statistics']['reference_rate_mps'] for e in record['independent']]
        sigma=[e['statistics']['conditional_information'].get('rate_standard_deviation_mps',np.nan) for e in record['independent']]
        axes[1].errorbar(x+(row-.5)*.1,rates,yerr=sigma,fmt='o',capsize=3,label=record['mode'])
    axes[0].set(ylabel='Unused per-bin phase RMS (degrees)',title='Black crosses: known-mean thermal reference')
    axes[1].set(ylabel='Reference-band path rate (m/s)',title='Independent rates and conditional thermal 1σ')
    retention=[e['statistics']['retained_fitted_signal_energy'] for e in stat['independent']]
    axes[2].bar(x,-10*np.log10(retention),color=COLOURS)
    axes[2].set(ylabel='Conditional fitted-signal information loss (dB)',title='Accounted projection loss, not a gain normalization')
    for ax in axes:ax.set_xticks(x,NAMES,rotation=25)
    axes[0].legend();axes[1].legend();fig.savefig(out/'02-noise-rates-retention.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(20,9),layout='constrained',sharey=True)
    for row,record in enumerate(records[2:]):
        for c,entry in enumerate(record['independent']):
            d=entry['statistics'];t=np.asarray(d['time_s']);test=np.asarray(d['testing']);ax=axes[row,c]
            ax.plot(t,d['path_m'],color=COLOURS[c],lw=1.3)
            ax.plot(t[test],np.asarray(d['measured_innovation_path_m'])[test],'.',color=COLOURS[c],ms=3,alpha=.6)
            ax.plot(t,record['common_statistics'][c]['path_m'],'k--',lw=1.2)
            ax.set(title=record['window']+' · '+NAMES[c],xlabel='Capture time (s)')
            if c==0:ax.set_ylabel('Path change from midpoint (m)')
    fig.suptitle('Independent measured phase trajectories at two other intervals\n'
        'A fixed-grid stationary profile is fitted jointly; no assumed common AIS velocity or acceleration enters.',fontsize=13)
    fig.savefig(out/'03-other-intervals.png',dpi=170);plt.close(fig)
    summaries=[];rows=''
    for record in records:
        independent_error=sum(e['fit']['prediction']['testing'][0]['heldout_error'] for e in record['independent'])
        common_error=sum(v['heldout_error'] for v in record['common']['prediction']['testing'])
        summaries.append(dict(window=record['window'],mode=record['mode'],interval_s=record['interval_s'],
            independent=[e['statistics'] for e in record['independent']],common_statistics=record['common_statistics'],
            independent_unused_error=independent_error,common_unused_error=common_error,
            common_to_independent_unused_error_ratio=common_error/independent_error,
            common_projected_gradient=record['common']['maximum_projected_gradient'],common_bounds=record['common']['bound_indices']))
        for name,e in zip(NAMES,record['independent']):
            d=e['statistics'];sigma=d['conditional_information'].get('rate_standard_deviation_mps')
            rows+=f'<tr><td>{record["window"]}</td><td>{record["mode"]}</td><td>{name}</td><td>{d["reference_rate_mps"]:.4f}</td><td>{sigma if sigma is not None else "unidentified"}</td><td>{d["conditional_phase_rms_deg"]:.1f}</td><td>{d["known_mean_thermal_phase_rms_deg"]:.1f}</td><td>{d["stationary_condition"]:.6f}</td><td>{100*d["retained_fitted_signal_energy"]:.2f}%</td></tr>'
    result=dict(accepted_calibration=False,production_changed=False,summaries=summaries,
        reproduced_objectives=len(errors),maximum_relative_cost_error=max(errors),same_observations_verified=True,
        uncertainty_scope='Fisher matrices condition on one moving point, active real brightness, known averaged range covariance and a constant stationary profile. They exclude model mismatch, other moving targets, target association and uncertain geometry. Shading and error bars are conditional thermal bounds, not calibrated physical confidence intervals. The Rician noise comparison holds the fitted mean known and omits training-parameter uncertainty; no noise scale is fitted.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    abc_raw=raw['independent'][4]['fit']['prediction']['testing'][0]['heldout_error'];abc_stat=stat['independent'][4]['fit']['prediction']['testing'][0]['heldout_error']
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Fixed-grid stationary and moving phase separation</title>
<style>body{{font:16px/1.55 system-ui;max-width:1550px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:7px;border:1px solid #ddd}}table{{border-collapse:collapse}}.note{{background:#eef4f8;padding:16px}}</style>
<h1>Separate a stationary profile before comparing moving phase trajectories</h1>
<p class="note">The AIS3 comparison uses bit-identical original observations and noise information. A jointly fitted, time-constant stationary profile recovers moving components in Ten/ABC that the unseparated single-point fits interpreted as nearly stationary. ABC unused squared prediction error improves {100*(1-abc_stat/abc_raw):.2f}%. No antenna coordinates, recording data or production processing are changed.</p>
<p>The fixed absolute range grid and fixed carrier/Doppler reference use original C0/C1 rows, preserving their epochs, weights and exact averaging. Every observed range cell has one constant complex nuisance coefficient. In whitened coordinates, DᴴD = Σ information · I: the design has condition number one. There are no unobserved delay columns or extrapolated FIR coefficients. The moving template receives exactly the same nuisance projection as the data; all nuisance/brightness/phase parameters are frozen on unused blocks.</p>
<img src="01-before-after-trajectories.png" alt="Independent measured trajectories before and after stationary separation in all five channels">
<p>The quadratic describes radar-measured reference-channel path change directly. AIS supplies only the small nominal interband geometric differences, evaluated at each channel’s own original epochs. Co-sited bands have exactly zero supplied difference. No common AIS velocity, acceleration or Hermite-knot curvature is imposed.</p>
<img src="02-noise-rates-retention.png" alt="Conditional rate uncertainties, phase-noise reference and moving information loss">
<p>Per-bin phase RMS is not the uncertainty of the fitted trajectory. Individual 17 ms bins can have SNR below 0 dB even when a multi-second trajectory is measurable. Black crosses integrate the known-mean proper-complex Gaussian phase density at the fitted SNR; they are a conditional thermal reference, not a fitted noise floor or a goodness-of-fit certificate. Rate bounds marginalize the full local envelope/constant-phase/trajectory covariance after stationary and real-brightness nuisance. Neither diagnostic includes all model/association error.</p>
<p>For a target template h and stationary design D, the retained information is ‖P⊥h‖²/‖h‖²; the efficient conditional mean information is 2 Re[JᴴP⊥J]. These are Schur-complement results. Applying a projection to data while using the original unprojected template energy would give a wrong likelihood. Loss near zero Doppler is not assumed away: the per-fit retained information is listed explicitly. No blanket production DC notch is introduced.</p>
<img src="03-other-intervals.png" alt="Additional flash and AIS2 measured phase trajectories">
<table><tr><th>Window</th><th>Mode</th><th>Band</th><th>Reference path rate (m/s)</th><th>Conditional rate σ (m/s)</th><th>Observed bin phase RMS (°)</th><th>Known-mean thermal RMS (°)</th><th>Stationary design condition</th><th>Fitted signal information retained</th></tr>{rows}</table>
<p>All {len(errors)} saved start objectives reproduce; maximum relative discrepancy {max(errors):.3g}. Original fixed-grid raw/stationary samples are bit-identical. Independent Fourier-waveform tests verify both symbol moments and carrier signs; a synthetic moving point is recovered against a 100-times-stronger arbitrary stationary profile, including frozen unused predictions and the analytic information loss.</p>
<p>This is component-separation evidence, not a conclusive location survey or proof that every selected resolution cell is one material point. Other moving returns and frequency-dependent unresolved structure remain possible. The raw exploratory source contains completed raw controls and explicitly recorded partial stationary controls; only its completed raw group is used here. The separate stationary jobs complete the comparison.</p>
<p>Literature: <a href="https://doi.org/10.1109/78.301849">Scharf and Friedlander, Matched subspace detectors</a> gives the nuisance-subspace likelihood construction. <a href="https://iris.uniroma1.it/handle/11573/906920">Colone et al., Sliding extensive cancellation algorithm</a> discusses slow-target attenuation when disturbance adaptation is too rapid. The present fixed-profile model is a simpler specialization, with its own information loss retained explicitly; it does not implement a sliding ECA-S filter.</p>
<p><a href="analysis.json">All trajectory measurements, conditional covariance and provenance</a> · <a href="tests.log">Numerical tests</a> · <a href="../ais-epoch-phase-candidates/index.html">Unseparated raw measurements</a> · <a href="../index.html">Analysis index</a></p></html>''')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
