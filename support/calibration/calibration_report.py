"""Build the reviewable offline calibration report from measured artifacts."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES=['Seven','SBS','Nine','Ten','ABC']
LABELS=['A','B','C','E']
COLOURS=['#275dad','#d47a00','#16877f','#a84277']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--html',type=Path,required=True);args=p.parse_args();directory=args.data
    def read(name): return json.loads((directory/name).read_text())
    main=read('sites-main-physical.json');old=read('sites-ozdigitaltv-physical.json')
    validation=read('site-heldout-validation-physical.json')
    delta=[]
    for label in LABELS:
        values=[]
        for name in ['main','ozdigitaltv']:
            record=next(r for r in validation if r['hypothesis']==name and r['target']==label)
            values.append(np.array([ch['heldout_focused_energy'] for ch in record['channels']]))
        delta.append(10*np.log10(values[0]/values[1]))
    delta=np.array(delta)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for i,label in enumerate(LABELS):
        ax.bar(np.arange(5)+(i-1.5)*.18,delta[i],width=.18,label=label,color=COLOURS[i])
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(5),NAMES)
    ax.set(ylabel='Held-out focused energy change (dB)',title='Licensed main sites versus the previous site assignments')
    ax.legend(title='Target');fig.savefig(directory/'site-validation.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,model,title in zip(axes,[old,main],['Previous assignment','Licensed main-site hypothesis']):
        for i,target in enumerate(model['targets']):
            x=np.array(target['curve']['position_enu_m'])
            ax.plot(x[:,0]/1000,x[:,1]/1000,label=target['label'],color=COLOURS[i])
            ax.scatter(x[0,0]/1000,x[0,1]/1000,s=15,color=COLOURS[i])
        ax.plot(0,0,'k+',label='Receiver');ax.set_aspect('equal',adjustable='datalim')
        ax.set(xlabel='East of receiver (km)',ylabel='North of receiver (km)',title=title)
        ax.legend()
    fig.suptitle('Alternative continuous target trajectories — none independently located')
    fig.savefig(directory/'trajectory-ambiguity.png',dpi=160);plt.close(fig)

    joint=read('licensed-joint-main.json')
    bias=read('licensed-joint-main-bias.json')
    # The original fit archives retain their objective-Jacobian diagnostics.
    # Report expected signal information, recomputed without residual clutter.
    joint['information']=read('licensed-joint-main-information.json')
    bias['information']=read('licensed-joint-main-bias-information.json')
    scale=np.tile([30,30,75],3)
    sigma=np.sqrt(np.diag(joint['information']['data_posterior_covariance_normalized']))*scale
    motion_sigma=np.sqrt(np.diag(joint['information']['data_and_motion_posterior_covariance_normalized']))*scale
    indices=[0,1,2,6,7,8]
    fitted_sites=np.array(joint['site_positions_enu_m'])+np.array(joint['best']['transmitter_corrections_enu_m'])
    difference=fitted_sites[0]-fitted_sites[2]
    baseline_gradient=np.zeros(9);baseline_gradient[:3]=difference/np.linalg.norm(difference)
    baseline_gradient[6:]=-baseline_gradient[:3]
    covariance=np.array(joint['information']['data_posterior_covariance_normalized'])*scale[:,None]*scale[None,:]
    baseline_sigma=float(np.sqrt(baseline_gradient@covariance@baseline_gradient))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    singular=np.array(joint['information']['data_normalized_singular_values'])[:6]
    axes[0].semilogy(range(1,7),singular,'o-',label='Shared channel phase')
    axes[0].semilogy(range(1,7),np.array(bias['information']['data_normalized_singular_values'])[:6],'s-',label='Also fit shared range offsets')
    axes[0].set(xlabel='Active-site parameter mode',ylabel='Singular value (30/30/75 m parameter scaling)',title='Data after eliminating target and response variables')
    axes[0].legend()
    x=np.arange(6)
    axes[1].bar(x-.18,sigma[indices],width=.36,label='Data + 10 m site prior')
    axes[1].bar(x+.18,motion_sigma[indices],width=.36,label='Also continuity / ground constraints')
    axes[1].axhline(10,color='black',linestyle='--',label='Prior only')
    axes[1].set_xticks(x,['Ornata E','Ornata N','Ornata U','Ridge E','Ridge N','Ridge U'],rotation=35,ha='right')
    axes[1].set(ylabel='Conditional standard deviation (m)',ylim=(0,11),title='The fit retains almost all of the prior uncertainty')
    axes[1].legend(fontsize=8);fig.savefig(directory/'calibration-conditioning.png',dpi=160);plt.close(fig)

    cosited=read('cosited-paths.json');coarser=read('cosited-paths-3s.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    cosited_labels=[t['label'] for t in coarser['targets']]
    for ch in range(4):
        axes[0].bar(np.arange(len(cosited_labels))+(ch-1.5)*.18,
            [t['heldout_common_loss_db'][ch] for t in coarser['targets']],width=.18,label=NAMES[ch])
    axes[0].set_xticks(range(len(cosited_labels)),cosited_labels);axes[0].axhline(0,color='black',linewidth=.7)
    axes[0].set(ylabel='Common-path held-out energy loss (dB)',title='Per band, 3 s spline knots')
    axes[0].legend(fontsize=8)
    aggregate=[]
    for index,study in enumerate([cosited,coarser]):
        values=[];valid=[]
        for t in study['targets']:
            c=sum(r['heldout_focused_energy'] for r in t['common_metrics'])
            i=sum(r['heldout_focused_energy'] for r in t['independent_metrics'])
            values.append(float(10*np.log10(i/c)))
            valid.append(t['common_converged'] and all(t['independent_converged']))
        bars=axes[1].bar(np.arange(len(cosited_labels))+(index-.5)*.32,values,width=.32,
                         label=f'{study["targets"][0]["knot_interval_s"]:g} s knots')
        for bar,ok in zip(bars,valid):
            if not ok: bar.set_hatch('///');bar.set_alpha(.4)
        aggregate.append(values)
    axes[1].set_xticks(range(len(cosited_labels)),cosited_labels);axes[1].axhline(0,color='black',linewidth=.7)
    axes[1].set(ylabel='Combined held-out energy loss (dB)',title='Trajectory flexibility sensitivity')
    axes[1].legend();fig.savefig(directory/'cosited-common-path.png',dpi=160);plt.close(fig)

    anchors=read('anchor-information.json');cases=anchors['cases']
    anchor_names=['Unknown',*['Known '+c['exact_known_tracks'][0] for c in cases[1:-1]],'All known']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    axes[0].bar(range(len(cases)),[c['ornata_ridge_separation_posterior_sigma_m'] for c in cases])
    axes[0].axhline(anchors['baseline_prior_sigma_m'],color='black',linestyle='--',label='Prior only')
    axes[0].set_xticks(range(len(cases)),anchor_names,rotation=35,ha='right')
    axes[0].set(ylabel='Conditional standard deviation (m)',title='Ornata–Ridge separation');axes[0].legend()
    for i,c in enumerate([cases[0],cases[-1]]):
        axes[1].bar(np.arange(6)+(i-.5)*.34,np.array(c['site_coordinate_posterior_sigma_m'])[indices],
                    width=.34,label=['Unknown tracks','Four exactly known tracks'][i])
    axes[1].set_xticks(range(6),['Ornata E','Ornata N','Ornata U','Ridge E','Ridge N','Ridge U'],rotation=35,ha='right')
    axes[1].set(ylabel='Conditional standard deviation (m)',title='Ideal anchors still leave weak directions')
    axes[1].legend(fontsize=8);fig.savefig(directory/'anchor-information.png',dpi=160);plt.close(fig)
    imaging_risk=read('calibration-imaging-risk.json')

    source=read('acma-transmitters.json')
    rows=[]
    for channel,call in zip(NAMES,['HSV6','SBS7','GTV8','ATV11','ABC12']):
        records=[d for d in source['devices'] if d['CALL_SIGN']==call]
        selected=next(d for d in records if d['STATION_TYPE']=='Main')
        names={'12013':'Ornata Road','12014':'Ridge Road','12015':'Eyre Road'}
        alternatives=', '.join(f"{names[d['SITE_ID']]} ({d['STATION_TYPE'].lower()})" for d in records if d!=selected)
        rows.append(f'<tr><td>{channel}</td><td>{float(selected["FREQUENCY"])/1e6:g}</td><td>{names[selected["SITE_ID"]]}</td><td>{selected["HEIGHT"]} m</td><td>{alternatives}</td></tr>')
    numeric={'accepted_calibration':False,'reason':'The measured-data information remains weak after target trajectories and channel responses are eliminated.',
        'main_site_mapping':['Ornata','Ornata','Ornata','Ornata','Ridge'],
        'heldout_focus_gain_db':delta.tolist(),'heldout_labels':LABELS,'channels':NAMES,
        'active_site_data_posterior_sigma_m':sigma[indices].tolist(),
        'active_site_data_and_motion_posterior_sigma_m':motion_sigma[indices].tolist(),
        'site_prior_sigma_m':10.,'covariance_status':'conditional local diagnostic using empirical projected covariance; not a certified CRLB',
        'ornata_ridge_distance_prior_sigma_m':float(np.sqrt(200.)),
        'ornata_ridge_distance_data_posterior_sigma_m':baseline_sigma,
        'information_construction':joint['information']['construction'],
        'cosited_labels':cosited_labels,'cosited_common_path_combined_loss_db_1p5s':aggregate[0],
        'cosited_common_path_combined_loss_db_3s':aggregate[1],
        'exact_known_tracks_separation_sigma_m':cases[-1]['ornata_ridge_separation_posterior_sigma_m'],
        'support_tests_passed':24,
        'conditional_calibration_imaging_loss_db_quantiles':imaging_risk['calibration_loss_db_quantiles'],
        'conditional_calibration_imaging_quantiles_percent':imaging_risk['quantiles_percent'],
        'main_joint_cost':joint['best']['cost'],'main_joint_converged':joint['best']['converged'],
        'range_offset_trial_converged':bias['best']['converged'],
        'backprojection_gpu_double_reference_relative_rms':9.06891e-8,
        'backprojection_analytic_point_max_complex_error':1.43855e-5,
        'production_metadata_updated':False,'calibrated_geographic_video_generated':False}
    (directory/'analysis-summary.json').write_text(json.dumps(numeric,indent=2))
    prefix='data/'+directory.name+'/'
    def figure(name,caption): return f'<figure><a href="{prefix+name}"><img src="{prefix+name}" alt="{html.escape(caption)}"></a><figcaption>{caption}</figcaption></figure>'
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DVB-T phase-centre calibration review — 13 October 2026</title>
<style>body{{font:17px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#17212d;background:#fafbfc}}h1,h2{{line-height:1.2}}h2{{margin-top:2em}}a{{color:#1856a5}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:9px;border-bottom:1px solid #cdd5df;text-align:left}}figure{{margin:26px 0}}img{{width:100%;height:auto}}figcaption,.small{{font-size:14px;color:#465466}}.status{{padding:18px 22px;background:#fff0d2;border-left:5px solid #b96f00}}code{{font-size:.9em}}pre{{white-space:pre-wrap;background:#eaf0f6;padding:16px}}li{{margin:.4em 0}}</style>
<h1>Phase-centre calibration review</h1><p>Updated 14 October 2026 · Capture <code>live-vhf-20261013-193334</code> · 601.87 seconds</p>
<div class="status"><b>No accepted phase-centre calibration yet.</b> The licensed transmitter configuration differs materially from the earlier assumption, and the recorded targets do not yet give a well-conditioned refinement. The best joint fit converges, but its antenna coordinates remain controlled primarily by the physical priors. A calibrated geographic movie is therefore still outstanding.</div>
<h2>The transmitter assignment needed correction</h2>
<p>The ACMA technical register explicitly distinguishes main from standby/alternative installations. Its current main configuration uses <b>two masts</b>: four services at Ornata Road and ABC at Ridge Road. There are several possible RF phase centres on Ornata: the licensed mounting heights span 160–164 m. This is different from treating five frequencies as three widely separated active transmitters. A licence does not prove which backup was active during a recording; the signal comparison below supports the main configuration.</p>
<table><thead><tr><th>Channel</th><th>MHz</th><th>Licensed main site</th><th>Mounting height</th><th>Other licensed installations</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="small">Based on Australian Communications and Media Authority information. Sources: <a href="https://www.acma.gov.au/radiocomms-licence-data">ACMA RRL</a>, downloaded 13 October 2026; <a href="{prefix}acma-transmitters.json">extracted technical records</a>; <a href="https://www.txaustralia.com.au/tvcoverage/metropolitan/melbourne/tower/monbulk">TX Australia site specifications</a>. RRL coordinates are GDA94, declared within 10 m; they are not an RF phase-centre survey.</p>
<h2>What the capture actually supports</h2>
<p>A, B and C are the reviewed moving candidates; E adds another interval. Each uses a 30-second complex range/slow-time patch from all five channels. The model has one continuous three-dimensional trajectory: its exact bistatic excess length generates the envelope position, carrier phase and Doppler. The likelihood projects the data and model through the same Doppler gate and uses the complete projected covariance. The range template uses the reconstructed waveform autocorrelation, including pilots and the estimated response. It is averaged over each candidate interval; residual template variability remains a modelling limitation.</p>
<p>The site comparison fits alternating one-second blocks and evaluates the others. Branch selection used the full-data search, so these are conditional validation results, not an unbiased global model-selection probability. Antenna coordinates are fixed at their respective licensed hypotheses for this comparison. Above-ground feasibility and the same continuity penalties apply to both. The main-site hypothesis improves ABC's withheld focused energy by <b>{delta[:,4].min():.2f}–{delta[:,4].max():.2f} dB</b> across the four candidates.</p>
{figure('site-validation.png','Positive values favour the licensed main-site model. Each bar uses held-out complex samples; no target-dependent colour normalization is applied.')}
<p>The corresponding trajectories differ by hundreds of metres to kilometres. These are alternative fitted explanations, not known aircraft or vehicle tracks. The main E fixed-site search reached its evaluation limit, so its best result remains a trial rather than a convergence claim.</p>
{figure('trajectory-ambiguity.png','Both models enforce continuous motion and terrain feasibility. Different plausible geometric branches survive; the dots mark interval starts.')}
<h2>A simpler co-sited phase check</h2>
<p>Before attributing the ambiguity to target scattering or a phase-bookkeeping fault, I fitted one common excess-path history to the four Ornata services. Each band keeps an independent constant complex gain. This requires agreement of temporal phase after carrier-frequency scaling, without imposing a geographic position or a shared absolute scattering phase. The comparison uses alternating one-second blocks for fitting and evaluation, with both 1.5 s and 3 s cubic-spline knot spacings. Association and some initialization branches came from the full-data search, so this is conditional validation.</p>
<p>For A, B and C, the combined held-out loss of the common path is between <b>{min(aggregate[0][:3]+aggregate[1][:3]):.2f} and {max(aggregate[0][:3]+aggregate[1][:3]):.2f} dB</b>. Negative values mean the shared fit predicts better than the independent fits, consistent with reduced fitting variance. Individual weak bands can still disagree: Ten loses 4.01 dB for C with the coarser spline, although it contributes little to the total signal. E is inconclusive with the finer spline because some optimizations reach their evaluation limit; the coarser trial converges. The ship-like J loses {aggregate[1][-1]:.2f} dB in total and 4.41–9.12 dB in the three lower-frequency bands with the coarser spline. It remains unsuitable as a common point calibrator under this model.</p>
{figure('cosited-common-path.png','Positive values favour independent band histories. Hatched bars contain unconverged trials and are not acceptance results. The 3 s cases all converged. These are empirical held-out signal scores, not certified SNR-loss bounds.')}
<p>This supports using A–C as temporal phase references; it does not locate them. The test treats the Ornata phase centres as coincident and does not model their 0–4 m mounting-height differences or the receiver antenna pattern. Frequency-dependent scattering, residual response errors and association uncertainty are possible explanations of disagreement; this diagnostic does not uniquely identify its cause.</p>
<h2>Why a converged optimizer is not sufficient</h2>
<p>The joint model shares transmitter positions and channel phase offsets between targets. Each target retains its own complex scattering amplitude and positive per-channel magnitudes. This tests the non-dispersive point-scatterer hypothesis rather than assuming it for arbitrary distributed objects. The container-ship-like return D was inspected across range and time: it contains changing, frequency-dependent components and was not accepted as a single calibration point. Other distant candidates also failed the phase/kinematic checks.</p>
<p>The additional ±200 Hz, 3–30 km searches used both one- and three-second coherence with a 9 dB screening threshold. Persistent detections were concentrated around the same 14–15 km return, rather than providing a new spatially diverse set. Many had Doppler inconsistent with their envelope migration. An early interval (candidate J, 5–35 s) was also extracted and fitted across all five channels. Its best point-target explanation required 342 m/s speed, 9.13 m/s² RMS acceleration and a climb of over 6 km in 30 s; this was not accepted as a calibration of the ship-like return. These screens do not prove that no weaker useful echo exists in the recording. <a href="{prefix}wider-target-search.json">Search measurements</a> and <a href="{prefix}sites-main-J-physical.json">J fit</a> are retained.</p>
<p>For the licensed main configuration, the best joint trial converged at objective {joint['best']['cost']:.2f}. However, eliminating the unknown target and response parameters leaves very little information about the antenna coordinates. Starting from a 10 m coordinate prior, the data-only conditional posterior standard deviations remain approximately <b>{sigma[indices].min():.3f}–{sigma[indices].max():.3f} m</b>. Even the strongest singular value is only {singular[0]:.3f} on the stated 30/30/75 m parameter scaling. The unused Eyre parameters are omitted from the plot; their exact zero-information modes are not the reason for this conclusion.</p>
{figure('calibration-conditioning.png','Left: measured-data sensitivity after nuisance elimination. Right: posterior uncertainty compared with the explicit site prior. Motion/terrain rows are shown separately from measured-data information.')}
<p>This is not just an absolute-orientation gauge issue: the rotation-invariant Ornata–Ridge separation has a prior uncertainty of {np.sqrt(200):.3f} m and a data-plus-prior conditional uncertainty of {baseline_sigma:.3f} m. The measurements barely constrain even that relative dimension.</p>
<p>These are conditional local information calculations with empirical noise weights, not a certified CRLB. Correlated clutter, waveform-response uncertainty, glint and association errors can make the real uncertainty larger. A separate shared range-offset trial lowers the residual further but remains unconverged and weakly conditioned; its offsets were not adopted as calibration. Such response biases also mean the site-comparison result is evidence under the current point-response model, not definitive proof of which antennas were transmitting.</p>
<p>The 14 October calculation uses the <b>expected signal Jacobian</b>, holding the fitted amplitudes and covariance fixed, before eliminating nuisance parameters. The original objective-Jacobian diagnostic also contained a term driven by the unfitted residual. Removing that term changes the separation uncertainty by only about 0.00002 m and leaves the conclusion unchanged. A synthetic regression verifies that adding clutter orthogonal to the fitted target signal cannot manufacture expected information.</p>
<p>There is also an exact geometric ambiguity before geographic anchors are supplied. With the receiver as the origin,</p>
<pre>Lᵢ(t) = ‖x(t) − Tᵢ‖ + ‖x(t)‖ − ‖Tᵢ‖
fD,ᵢ(t) = −(fᵢ / c) dLᵢ(t)/dt.

For any rigid rotation Q:
Lᵢ[Qx, QTᵢ] = Lᵢ[x, Tᵢ].
Speed, acceleration and jerk norms are also unchanged.</pre>
<p>Thus kinematic continuity is useful but does not itself provide an absolute geographic orientation. Range migration and Doppler must agree—as they do in this implementation—but are not independent geometric observations. Survey priors or an independently located target anchor the geographic frame. Multiple targets with genuinely different transmitter look directions can improve the remaining relative parameters; target count alone does not ensure conditioning.</p>
<h2>What an independent track could add</h2>
<p>A design calculation holds one or all four trial trajectories fixed in world coordinates, as if their positions were known exactly. Channel phase offsets and amplitudes remain unknown, and no motion prior contributes information. An individual ideal anchor reduces the conditional Ornata–Ridge separation uncertainty to 8.4–10.6 m. Even four exact anchors at these trial positions only reduce it to <b>{cases[-1]['ornata_ridge_separation_posterior_sigma_m']:.2f} m</b>; east-coordinate uncertainties remain close to 10 m. This is a hypothetical sensitivity calculation, not new observed information, and real track/time/scattering-centre uncertainty would weaken it.</p>
{figure('anchor-information.png','All cases use the same signal strengths, empirical covariance and 10 m site priors. Known means an ideal zero-error world trajectory at the current trial positions. The remaining weak directions motivate wider angular diversity or a physical antenna survey, not stronger arbitrary continuity penalties.')}
<h2>Does the coordinate uncertainty actually hurt coherent imaging?</h2>
<p>Large coordinate uncertainty alone is not a sufficient rejection criterion: correlated antenna and channel-phase errors could cancel in the predicted image. I therefore retained their joint covariance, eliminated target trajectories and amplitude parameters, and propagated the phase contrasts to fixed points on the ground DEM. For five equally weighted, non-dispersive channels and locally Gaussian calibration errors, the expected coherent power relative to perfect calibration is</p>
<pre>G = [5 + 2 Σᵢ&lt;ⱼ exp(−Var(δθᵢ − δθⱼ)/2)] / 25.
δθᵢ includes both the antenna-position error and fitted channel-phase error.
Their correlations are retained; a common phase error cancels exactly.</pre>
<p>Under that conditional model, the expected calibration loss ranges from <b>{imaging_risk['calibration_loss_db_quantiles'][0]:.2f} to {imaging_risk['calibration_loss_db_quantiles'][-1]:.2f} dB</b> across the sampled 0.6–30 km ground footprint, with median {imaging_risk['calibration_loss_db_quantiles'][4]:.2f} dB. No sampled ground point meets a 1 dB loss criterion. A 100,000-draw nonlinear-geometry check at seven representative points agrees with the linear phase propagation within {max(abs(c['loss_difference_db']) for c in imaging_risk['nonlinear_geometry_checks']):.3f} dB.</p>
{figure('calibration-imaging-risk.png','A conditional calibration-risk map, not observed radar power or a measured SNR loss. It assumes a local single-branch Gaussian posterior, equal band strengths, a common point scatterer and a fixed receiver. Additional scattering, terrain, receiver and alternative-branch uncertainty is not included.')}
<p>This checks the relevant imaging quantity rather than insisting that every antenna coordinate be individually precise. It still does not support adopting the present fit for a coherent geographic movie. The Gaussian model is a local diagnostic, not a proof of a global uncertainty bound or a guarantee about losses in an eventual calibrated image.</p>
<h2>The ideal aperture also has strong ambiguities</h2>
<p>An independent synthetic check assumes perfect calibration, a non-dispersive point scatterer, equal channel amplitudes and noise, the occupied DVB-T bandwidth, and a flat local surface. Exact GPU backprojection then evaluates the point response at 1 m spacing. For this example, a false response more than 300 m away in cross-range is only 0.042 dB below the true peak with the licensed main configuration (0.509 dB for the previous three-mast assumption). This is a conditional point-response example, not a universal resolution bound, but it shows why even perfect calibration will leave strong spatial ambiguity in a simple real-aperture image.</p>
{figure('ideal-aperture-psf.png','The true point is at zero displacement. Bright repeated peaks are aperture ambiguities, with no noise, motion or calibration error in this test.')}
<h2>Geographic backprojection is ready for validation</h2>
<p>The saved channel IQ was replayed through the production reconstruction and matched filter with static terrain retained and the fitted direct path cancelled. The export spans the full recording. The Vulkan point-backprojection kernel uses double-precision precomputed geometry and carrier phasors, then 64-tap bandlimited complex profile interpolation on the GPU. It passed a random-input double-reference comparison (relative RMS error 9.07×10⁻⁸) and independent analytic fractional-delay/RF-phase tests (maximum complex error 1.44×10⁻⁵).</p>
<p>The image below is an explicitly <b>uncalibrated independent-channel diagnostic</b>, using a 3.00288-second interval and 0.6–30 km excess-path support. It profiles independent complex channel responses; it is not a coherently focused multistatic terrain image. Elliptical ridges demonstrate the ambiguity of projecting range-only static returns. They must not be read as geographically located reflectors.</p>
{figure('geographic-uncalibrated.png','A diagnostic of the projection machinery, not the requested calibrated geographic product. Static integration attenuates moving targets.')}
<p>For this image the terrain is the authoritative <a href="https://services.ga.gov.au/gis/rest/services/DEM_SRTM_1Second_2024/MapServer">Geoscience Australia one-arcsecond SRTM-derived DEM</a>, with the <a href="https://www.ga.gov.au/bigobj/GA18067.pdf">documented EGM96 vertical datum</a> converted to ellipsoidal height for ECEF geometry. This has the posting interval requested for DTED level 2, but is <b>not labelled as a native DTED2 product</b>. Earlier fit diagnostics used the provisional one-arcsecond terrain mosaic; both this and the approximate receiver roof height remain physical uncertainty sources.</p>
<h2>Acceptance status and next constraint</h2>
<ul><li>Passed: 24 support tests covering continuous range/phase/Doppler bookkeeping, analytic Jacobians, nuisance projection, expected signal information, co-sited history recovery and held-out isolation, fixed-world phase propagation and common-phase cancellation, multichannel timing, interpolation, DEM pixel-centre conventions and fixed-CPI gap checks, plus the GPU/reference and analytic point-backprojection checks.</li>
<li>Supported: the licensed main-site assignment explains withheld observations better than the previous site selection.</li>
<li>Not passed: well-conditioned antenna refinement, stable held-out geographic target locations, and transferable coherent phase calibration over the full imaging area.</li>
<li>Outstanding: calibrated full-recording geographic video. Production phase-centre metadata has not been updated.</li></ul>
<p>The useful next constraint is an independently located point or trajectory, or clear calibration returns spanning a wider angular sector as seen from Mt Dandenong. A known track could separate geometric error from target-dependent scattering phase and response bias. Tightening priors solely to force a small covariance would not supply that evidence. The approximate receiver position and any frequency/direction dependence of its roof-antenna phase response also remain outside this point-receiver model.</p>
<p class="small">Reproducible artifacts: <a href="{prefix}analysis-summary.json">summary JSON</a> · <a href="{prefix}site-heldout-validation-physical.json">held-out measurements</a> · <a href="{prefix}licensed-joint-main.json">joint fit archive</a> · <a href="{prefix}licensed-joint-main-information.json">expected information</a> · <a href="{prefix}licensed-joint-main-bias.json">range-offset trial</a> · <a href="{prefix}cosited-paths-3s.json">co-sited path check</a> · <a href="{prefix}anchor-information.json">ideal anchor calculation</a> · <a href="{prefix}calibration-imaging-risk.json">imaging-phase uncertainty</a> · <a href="{prefix}geographic-uncalibrated.json">geographic diagnostic metadata</a>. Support implementation: <code>support/calibration/</code>. Capture IQ remains intact.</p>
</html>'''
    args.html.write_text(report)
    print(args.html)


if __name__=='__main__':
    main()
