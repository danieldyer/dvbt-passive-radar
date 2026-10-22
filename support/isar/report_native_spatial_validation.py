"""Review spatial calibration, native PSF controls and continuous replay."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from recovery_experiment import NAMES
from report_measured_scatterer_phases import serial


def main():
    root=Path('docs/data/isar_20261018/native-spatial-validation')
    names=['image-training','image-testing','point-spread-training','point-expectation-training']
    images=[]
    for name in names:
        with np.load(root/name/'images.npz') as z:images.append(dict(z))
    actual,test,constant,expected=images
    for z in (constant,expected):
        np.testing.assert_array_equal(actual['information'],z['information'])
        np.testing.assert_array_equal(actual['gains'],z['gains'])
    x=np.argmin(abs(actual['east_m']));y=np.argmin(abs(actual['north_m']))
    reference=actual['power'][5,y,x]
    gain=actual['gains'];normal=np.sum(abs(gain)**2*actual['information'])
    residual=8192*normal*abs(actual['combined_amplitude']-expected['combined_amplitude'])**2
    extent=[actual['east_m'][0],actual['east_m'][-1],actual['north_m'][0],actual['north_m'][-1]]
    fig,axes=plt.subplots(1,4,figsize=(17,6),sharex=True,sharey=True,layout='constrained')
    values=[actual['power'][5],constant['power'][5],expected['power'][5],residual]
    titles=['Measured training image','One constant-strength point','One point; fitted real brightness','Data minus point expectation']
    for ax,power,title in zip(axes,values,titles):
        im=ax.imshow(10*np.log10(np.maximum(power/reference,1e-30)),origin='lower',extent=extent,
            cmap='magma',vmin=-40,vmax=5,aspect='equal')
        ax.set(title=title,xlabel='East from fitted point (m)',ylabel='North (m)')
    fig.colorbar(im,ax=axes.tolist(),label='Power relative to measured training anchor (dB)',shrink=.7)
    fig.suptitle('Identical native symbols, waveform, motion, frozen band coefficients and coordinates\nSynthetic point controls contain no second scatterer; the residual can contain clutter and model error as well as other targets')
    fig.savefig(root/'measured-versus-point-psf.png',dpi=160);plt.close(fig)
    refined=json.loads((root/'continuous-point-transfer/analysis.json').read_text())
    phases=np.asarray([[r['phase_error_deg'] for r in b] for b in refined['component_transfer']])
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for f,(ax,label) in enumerate(zip(axes,['Training','Excluded'])):
        im=ax.imshow(phases[:,f,:],cmap='twilight',vmin=-180,vmax=180,aspect='auto')
        ax.set(title=label+' phase after frozen component prediction',xlabel='Candidate',yticks=range(5),yticklabels=NAMES)
        for c in range(5):
            for j in range(phases.shape[2]):ax.text(j,c,f'{phases[c,f,j]:.0f}°',ha='center',va='center',fontsize=10,color='white')
    fig.colorbar(im,ax=axes.tolist(),label='Diagnostic complex multiplier phase (degrees)',shrink=.8)
    fig.suptitle('Continuous coordinates and native positive brightness; band phases remain frozen\nIndependent multipliers are measured for diagnosis and are never applied to the data or image')
    fig.savefig(root/'refined-spatial-phase-transfer.png',dpi=160);plt.close(fig)
    full=json.loads((root/'full-replay-calibration-transfer.json').read_text())
    clock=json.loads((root/'stationary-phase-controls/analysis.json').read_text())
    old=np.asarray(refined['points_east_north_up_m']);new=np.asarray(refined['refined_points_east_north_up_m'])
    hashes={}
    for path in [Path('build/injection-20261018/deterministic-baseline/profiles.bin'),Path('build/injection-20261018/baseline-no-source/profiles.bin')]:
        with path.open('rb') as stream:hashes[str(path)]=hashlib.file_digest(stream,'sha256').hexdigest()
    if len(set(hashes.values()))!=1:raise ValueError('Default replay payload identity changed')
    comparison=json.loads((root/'rigid-brightness-fit-v2/analysis.json').read_text())['operator_comparison']
    maximum_loss=max(v['matched_snr_loss_db'] for v in comparison)
    maximum_phase=max(abs(v['constant_phase_difference_degrees']) for v in comparison)
    result=dict(complete=True,focused_ship_certified=False,full_duration_videos_complete=False,
        full_continuous_replay='build/isar-20261018/current-native-full/profiles.bin',
        default_replay_sha256=hashes,
        replay_without_unused_source_byte_identical=True,
        full_replay_excluded_coherent_ratio=full['folds'][1]['coherent_to_independent_power'],
        continuous_point_corrections_m=new-old,
        refined_component_phase_errors_deg=phases,
        stationary_phase_controls=clock['channels'],
        data_or_image_alignment_applied=False)
    (root/'review.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    stationary_rows=''.join('<tr><td>'+c['name']+'</td><td>'+', '.join(f'{q["excluded_quarter_second_phase_rms_deg"]:.1f}°' for q in c['cells'])+'</td></tr>' for c in clock['channels'])
    model_sections=[]
    for name in ['rigid-brightness-fit-v2','two-point-rigid-fit','two-point-rigid-30s']:
        path=root/name/'analysis.json'
        if not path.exists():continue
        d=json.loads(path.read_text())
        if not d.get('complete'):model_sections.append(f'<p>{name}: still running; no conclusion is drawn from an incomplete fit.</p>');continue
        f=d['fit'];model_sections.append(f'<p><a href="{name}/analysis.json">{name}</a>: objective {f["objective"]:.6f}; active bounds {f["active_bounds"]}; maximum scaled projected gradient {f["maximum_scaled_projected_gradient"]:.3g}. Excluded complex-error reductions: '+', '.join(f'{v["explained_fraction"]:.1%}' for v in d['prediction'])+'. A bound or a low training cost does not establish physical motion or focus.</p>')
    trials=[]
    for name,label in [('two-point-rigid-fit','Two points; cubic translation and angular acceleration'),
            ('constant-velocity-two-point-6s','Two points; strict Cartesian constant velocity'),
            ('constant-translation-two-point-6s','Two points; constant translation with constant angular rate'),
            ('residual-three-point-wide','Residual-selected third candidate; motion held fixed'),
            ('residual-three-point-motion','Residual-selected third candidate; shared motion refined')]:
        path=root/name/'analysis.json'
        if path.exists():
            d=json.loads(path.read_text())
            if d.get('complete'):trials.append((name,label,d))
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');x=np.arange(5);width=.8/max(len(trials),1)
    for i,(_,label,d) in enumerate(trials):
        ax.bar(x-.4+width*(i+.5),[v['explained_fraction'] for v in d['prediction']],width,label=label)
    ax.set(xticks=x,xticklabels=NAMES,ylabel='Excluded complex-energy reduction against stationary-only fit',ylim=(0,.5))
    ax.legend(fontsize=8);ax.set_title('Exploratory six-second model comparisons; no target-SNR or geographic-localisation claim')
    fig.savefig(root/'native-model-comparison.png',dpi=160);plt.close(fig)
    history_sections=[]
    for name,label in [('two-point-phase-histories','Original two-point motion fit'),
            ('residual-three-point-wide','Third residual-selected candidate, original motion frozen'),
            ('residual-three-point-motion','Three candidates, shared motion refined')]:
        picture=root/name/'phase-histories/phase-histories.png'
        if picture.exists():history_sections.append(f'<h3>{label}</h3><img src="{picture.relative_to(root)}">')
    trial_rows=''.join(f'<tr><td><a href="{name}/analysis.json">{label}</a></td><td>{d["fit"]["objective"]:.6f}</td><td>{d["fit"]["active_bounds"]}</td><td>'+', '.join(f'{v["explained_fraction"]:.1%}' for v in d['prediction'])+'</td></tr>' for name,label,d in trials)
    native_picture=root/'residual-three-point-wide/native-image-6s/point-images.png'
    native_section=(f'<p><a href="{native_picture.relative_to(root)}">Current six-second native per-channel and combined image</a>. This uses the frozen shared motion and observed data; the additional fitted point is not inserted into the image. The short-aperture image remains unresolved and is not accepted as a focused ship.</p>' if native_picture.exists() else '')
    long_sections=[]
    for name,label in [('two-point-rigid-30s','Two-point 30-second fit'),('residual-three-point-30s','Three-candidate 30-second fit'),
            ('residual-three-point-30s-native','Three-candidate native refinement')]:
        path=root/name/'analysis.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        if not data.get('complete'):
            long_sections.append(f'<p>{label}: still running.</p>');continue
        f=data['fit'];long_sections.append(f'<h3>{label}</h3><p>Native normalised residual {f["objective"]:.6f}; active bounds {f["active_bounds"]}. Excluded complex-error reductions: '+', '.join(f'{v["explained_fraction"]:.1%}' for v in data['prediction'])+'. This statistic does not demonstrate coherent resolution throughout the aperture.</p>')
        for picture in [root/name/'native-image/point-images.png',root/name/'phase-histories/phase-histories.png']:
            if picture.exists():long_sections.append(f'<img src="{picture.relative_to(root)}">')
    support_path=root/'two-point-30s-aperture-support/analysis.json'
    if support_path.exists():
        support=json.loads(support_path.read_text())
        if support.get('complete'):
            s=support['phase_histories']['conditional_aperture_support']
            rows=''.join(f'<tr><td>{NAMES[c]}</td><td>{j}</td><td>{v["rectangular_moment_equivalent_aperture_s"]:.2f} s</td><td>{-10*np.log10(v["constant_strength_template_efficiency_after_stationary"]):.2f} dB</td><td>{-10*np.log10(v["target_information_retention_after_stationary"]):.3f} dB</td></tr>' for c,band in enumerate(s) for j,v in enumerate(band) if v['supported'])
            long_sections.append('<h3>Which part of the nominal aperture carries fitted information?</h3><img src="two-point-30s-aperture-support/phase-histories/aperture-support.png"><table><tr><th>Band</th><th>Candidate</th><th>Moment-equivalent duration</th><th>Constant-strength template mismatch</th><th>Stationary nuisance loss</th></tr>'+rows+'</table>')
            long_sections.append('<p>The duration is √(12 Var<sub>w</sub>(t)), with weights equal to each native component’s expected conditional energy. It equals the duration of a rectangular aperture with the same time variance; it is not a measured resolution or CRLB. The anchor has only about 4.7–9.8 seconds of moment-equivalent support within this 30-second interval.</p><p>The template comparison uses the exact original-row forward responses and the same known covariance. Let P remove the full-aperture stationary nuisance in whitened coordinates, h be the unit-strength motion template and s its training-fitted positive-brightness counterpart. The constant-strength efficiency is |hᴴPs|²/[(hᴴPh)(sᴴPs)]. The stationary retention is (sᴴPs)/(sᴴs). Both templates receive the same projection. The resulting 5–7 dB anchor mismatch is a conditional prediction from the fitted brightness, not a demonstrated recoverable gain or a true-target SNR bound. Motion/scene error and brightness uncertainty remain outside this pure-template comparison. Stationary projection costs under 0.04 dB for these fitted components, so it does not explain the poor long-aperture result.</p>')
            long_sections.append('<img src="two-point-30s-aperture-support/phase-histories/phase-histories.png"><p>Small average phase errors conceal large local residuals, particularly in Nine. The raw native image also includes stationary scene energy and uses constant-strength temporal weighting. It should not be interpreted as the likelihood image of the fitted variable-brightness scene. A numerical fit concentrated on a brief flash is insufficient evidence for a 30-second coherent ship image.</p>')
    branch_section=''
    branch_path=root/'branch-geometry-audit/analysis.json'
    direct_path=root/'direct-registration-audit/analysis.json'
    if branch_path.exists() and direct_path.exists():
        branch=json.loads(branch_path.read_text());direct=json.loads(direct_path.read_text())
        result['strong_branch_identification_certified']=False
        result['strong_branch_geometry_audit']=branch
        result['direct_registration_audit']=direct
        (root/'review.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
        branch_section='''<h2>Echo identity must be checked before calling these fits ship motion</h2>
<p><strong>The recent stronger branch is about 450 m beyond the AIS-predicted excess path.</strong> The conditional radial-fit point is 244 m from the AIS antenna, with 243 m of that displacement across the course direction. That point is not a certified geographic localisation. More directly, its equal-range contour misses the broad course-aligned hull hypotheses below.</p>
<img src="branch-geometry-audit/branch-geometry.png">
<p>The deliberately generous hull model allows ±300 m along the ship, ±25 m across it and 0–60 m of height above the AIS antenna. At the nominal course direction, the maximum added path is about 95 m for Ornata. Allowing ±20° heading departure and testing all three supplied transmitter sites still leaves the stronger branch 116–175 m beyond the maximum hull path. Course is not heading; these are conditional bounds, not an assertion that heading is known. Neither a multipath explanation nor a large receiver-position correction has been adopted.</p>
<img src="branch-geometry-audit/native-branch-profiles.png">
<p>The native data show a separate flash near the AIS-predicted range around 158–162 s. The stronger, more delayed flash used in the recent point fits occurs later. The right column removes only the training mean as a labelled display diagnostic; the inference observations are unchanged.</p>
<img src="direct-registration-audit/direct-registration.png">
<p>A new production replay with direct-path cancellation disabled puts every checked native peak at lag zero: 27,573 rows across all five channels. One lag is 37.47 m. This rules out the approximately 12-sample common processing shift needed to account for the 450 m discrepancy, at the integer-lag resolution of this check. It does not certify antenna locations or a sub-sample delay calibration.</p>'''
        near=root/'ais-near-brightness-4s/analysis.json'
        if near.exists():
            n=json.loads(near.read_text())
            if n.get('complete'):
                fractions=', '.join(f'{c["name"]} {c["fading"]["prediction"]:.1%}' for c in n['channels'])
                branch_section+='<p>The first independent-band four-second test restricts acquisition and refinement to ±3 samples (±112 m) around the AIS path. Its excluded complex-error reductions are '+fractions+'. These support investigating this branch; the independent fits do not yet establish one common physical scatterer or shared motion.</p><img src="ais-near-brightness-4s/brightness-and-phase.png">'
        near_shared=root/'ais-near-shared-point-4s/analysis.json'
        near_transfer=root/'ais-near-shared-point-4s/calibration-transfer.json'
        if near_shared.exists() and near_transfer.exists():
            s=json.loads(near_shared.read_text());t=json.loads(near_transfer.read_text())
            if s.get('complete'):
                fraction=', '.join(f'{name} {v["test_complex_prediction_fraction"]:.1%}' for name,v in zip(NAMES,s['prediction']))
                delta=np.asarray(s['physical_state_enu'])[0]-np.asarray(s['prior_mean_state_enu'])[0]
                power=t['folds'][1]['coherent_to_independent_power'];db=t['folds'][1]['difference_db']
                branch_section+=f'<h3>A physically feasible common short-arc hypothesis</h3><p>The shared radial fit converges without active bounds. Its midpoint is {delta[0]:+.1f} m east and {delta[1]:+.1f} m north of the AIS antenna, rather than 244 m away on the more delayed branch. Excluded complex-error reductions are {fraction}. The withheld coherent/independently-phased matched-power ratio is {power:.2%} ({db:.3f} dB), with band phases and stationary coefficients frozen from training. This is a local transfer statistic, not proof of unique geographic localisation or true-target SNR efficiency.</p><img src="ais-near-shared-point-4s/native-image/point-images.png"><p>The four-second image remains unresolved. It uses constant north/east axes and zero physical rotation because that point fit does not estimate rotational motion. The next scene test uses the checked Vulkan residual proposals around this physically feasible branch, rather than inheriting the old branch’s calibration or support.</p>'
        near_scene=root/'ais-near-three-point-4s-native/analysis.json'
        if near_scene.exists():
            scene=json.loads(near_scene.read_text())
            if scene.get('complete'):
                f=scene['fit'];fraction=', '.join(f'{name} {v["explained_fraction"]:.1%}' for name,v in zip(NAMES,scene['prediction']))
                branch_section+=f'<h3>Three-candidate short-arc scene</h3><p>The two added candidates are selected by the training residual adjoint. Native refinement converges with maximum scaled projected gradient {f["maximum_scaled_projected_gradient"]:.3g}, no active bounds, and excluded complex-error reductions {fraction}. Common physical motion is held fixed; the angular rate is a declared +0.2268°/s counterclockwise course-derived hypothesis, not a heading measurement. Point positions and one constant phase per point are shared across bands.</p><img src="ais-near-three-point-4s-native/phase-histories/phase-histories.png"><img src="ais-near-three-point-4s-native/native-image/point-images.png"><p>The first optimiser stop was premature despite its success flag. Central differences verify the real-data gradient and the NNLS KKT residuals are below 2 × 10<sup>−15</sup>; restarting native refinement reduces the scaled gradient from 1.3 × 10<sup>−3</sup> to 2.6 × 10<sup>−9</sup>. These checks establish numerical convergence, not material-point identity. The longer 30-second shared-motion test now uses this AIS-near reference and explicitly declared support.</p>'
    weighted=root/'residual-three-point-30s-native/brightness-image-0/point-images.png'
    if weighted.exists():long_sections.append(f'<h3>Conditional real-brightness matched rendering</h3><img src="{weighted.relative_to(root)}"><p>This forms the observed data with one component’s real temporal envelope. That envelope multiplies the coherent numerator once and its thermal-noise information twice; no time-dependent phase is applied. Original native-bin identities are preserved without extrapolating the brightness model. The image remains a conditional echo diagnostic, not a certified ship image or an optimum template for every other point.</p>')
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Spatial calibration and native imaging controls</title>
<style>body{{font:16px/1.6 system-ui;max-width:1350px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}th,td{{padding:8px 12px;border-bottom:1px solid #ccc;text-align:left}}</style>
<h1>Spatial calibration and native imaging controls</h1>
<p class="note"><a href="projected-likelihood-review/index.html">Latest: covariance-corrected images, independent native point response and fine physical cuts</a>, with <a href="projected-likelihood-review/anchor-transfer.html">excluded-data interband phase transfer</a>. This supersedes the unchanged-information-denominator display control below.</p>
<p class="note"><a href="motion-controls-review/index.html">Latest motion and image controls</a>: constant-velocity comparisons, independent finite-surface tests, and a quantitative explanation of the dominant northeast image ridge as stationary leakage. The matched zero-Doppler locus is predicted from physical geometry; no image alignment is fitted.</p>
<p><strong>The full requested ship image/video remains unproved.</strong> This report separates instrumental coherence, point-spread structure, coordinate error and unsupported motion fits. No per-band time-dependent phase correction, image registration or local power normalisation is introduced.</p>
{branch_section}
<h2>Continuous current-production replay</h2><p>The entire capture has been reprocessed continuously through the current decoder and range compressor. The previous six-second motion and band phase calibration transfer to it: the excluded-data coherent/independent matched-power ratio is {full['folds'][1]['coherent_to_independent_power']:.2%}, a {abs(full['folds'][1]['difference_db']):.3f} dB difference. This is a local observed-power transfer statistic, not a target-SNR bound. The standalone replay’s unused decoded-source-file dependency was removed; its original 30-second default export is byte-identical (SHA-256 in <a href="review.json">review.json</a>). The wider continuous export preserves all native C0/C1 epochs, known-waveform kernels, weights and explicit FEC erasures.</p>
<p>A bounded archive reader retains original global 16-symbol group identities. Round-trip payload and training/test partition tests cover this. Reading a 30-second aperture materialises approximately 0.80 GiB of complex profiles rather than the 19 GiB archive.</p>
<h2>A point is not an isolated image pixel</h2><img src="measured-versus-point-psf.png">
<p>The constant-point control is an actual finite-waveform increment through the production compressor. The second control applies the real brightness knots estimated from training only. Both retain native symbol identities and all C0/C1 moments. Their prominent sidelobes and elongation show why selecting bright neighbouring pixels alone does not certify independent scatterers. The measured data also contain substantial structure the single-point control does not explain. Subtraction here is a diagnostic linear difference, never a change to production observations.</p>
<h2>Coarse-grid phase comparison was insufficient</h2><p>The initial fixed-coordinate comparison below uses candidates selected on the training-only image. Constant complex coefficients are deblended jointly, avoiding raw bright-cell phase readings, but neither coarse coordinates nor constant strength are guaranteed adequate.</p>
<img src="spatial-transfer/spatial-phase-transfer.png">
<p>Across these carrier frequencies, a few metres of excess-path error creates a large relative phase slope. Continuous common position refinement shifts the second candidate by {new[1,0]-old[1,0]:+.3f} m east and {new[1,1]-old[1,1]:+.3f} m north. Coupled range migration and RF phase use these same coordinates. Real brightness knots are also deblended at original symbol epochs; the improvement cannot be attributed to coordinates alone.</p>
<img src="refined-spatial-phase-transfer.png">
<p>The anchor and second candidate retain useful excluded-data phase agreement with frozen band calibration; ABC still has a {abs(phases[4,1,1]):.1f}° discrepancy at the second candidate. The weaker third candidate has poor transfer and reaches its coordinate bound. It is not accepted as a persistent material scatterer. These results neither establish absolute transmitter locations nor certify geographic localisation. All component multiplier estimates in the diagnostic are excluded from image formation.</p>
<h2>Stationary phase controls</h2><img src="stationary-phase-controls/stationary-phase-controls.png">
<table><tr><th>Band</th><th>Excluded quarter-second phase RMS in three training-selected cells</th></tr>{stationary_rows}</table>
<p>Several separated stationary-like cells retain much smaller phase variation than the problematic moving-target histories. This argues against attributing all moving-target phase structure to a single common instrumental phase error. It is not a universal phase-noise bound: the cells can contain noise, residual interference and unresolved returns.</p>
<h2>Motion trials and remaining work</h2>{''.join(model_sections)}
<h2>Constant velocity and residual-selected support</h2>
<p>The strict constant-velocity control removes Cartesian acceleration, jerk and physical rotation. Its translation direction is fixed in Cartesian coordinates; it does not silently follow a changing bistatic gradient. Exact bistatic geometry still produces the range curvature implied by constant Cartesian velocity. A separate control permits constant angular rate while retaining constant translation. The numerical regressions verify both the physical velocity and renderer/inference agreement.</p>
<img src="native-model-comparison.png"><table><tr><th>Six-second hypothesis</th><th>Normalised training residual</th><th>Active bounds</th><th>Excluded reductions: Seven / SBS / Nine / Ten / ABC</th></tr>{trial_rows}</table>
<p>The third candidate is selected by the training residual adjoint, after the fitted native moving signals and stationary coefficients are removed. The GPU/CPU adjoint relative disagreement is 3.42 × 10<sup>−5</sup>; the representative/native matched-template discrepancy is below 4.2 × 10<sup>−8</sup> dB at the checked probes. This validates the proposal computation, not the identity of a material scatterer. The wider coordinate fit keeps the original two coordinates and all motion fixed, so its additional explanatory power can be traced to the new candidate. Its original 6 m search-bound result is retained separately; the wider fit is not treated as certification based on its lower training residual.</p>
<p><a href="two-point-residual-search/residual-proposals/residual-proposals.png">Training residual proposal map</a>. These comparisons remain exploratory. The old weak third candidate in the earlier coarse-support section is a different location from this residual-selected candidate.</p>
{''.join(history_sections)}
<p>These histories jointly deblend the components in quarter-second blocks, using geometry, positive brightness envelopes, stationary coefficients and band calibration frozen from training. The measured complex multipliers are diagnostic only and are never applied to image formation. Marker area follows expected component energy; phase during a fade carries little information. A small whole-aperture average phase error can coexist with large local swings. In particular, adding the nearby residual candidate markedly reduces Nine’s large residual phase swings. Prediction improvements are reductions in complex error over the selected range cells, not a demonstrated image-SNR gain.</p>
{native_section}
<h2>Long-aperture validation</h2>{''.join(long_sections)}
<h2>Relevant prior work and its limits here</h2>
<p><a href="https://iris.uniroma1.it/handle/11573/1339039">Pisciottano, Cristallini and Pastina (2019), Maritime target imaging via simultaneous DVB-T and DVB-S passive ISAR</a>, compares measured maritime images with point-model simulations using known motion and backprojection. It supports the physical backprojection/synthetic-control approach, but does not establish our unknown motion or transmitter locations.</p>
<p><a href="https://doi.org/10.1109/TCI.2017.2750330">Xu et al. (2017), Enhanced ISAR Imaging and Motion Estimation With Parametric and Dynamic Sparse Bayesian Learning</a>, alternates structured sparse scene reconstruction and motion-dependent dictionary refinement. That motivates expanding the validated shared scene before drawing conclusions from a few bright cells. The present candidate fits are not an implementation or validation of that paper’s Bayesian algorithm.</p>
<p>The representative-epoch approximation is checked against original-symbol templates at the fitted motion, using identical coefficients. The six-second trial’s largest matched loss is {maximum_loss:.3g} dB; its largest constant phase difference is {maximum_phase:.3g}°. This validates that acceleration of the numerical search at that aperture, not the physical scattering model.</p>
<p>The next gate is a longer-aperture shared motion/scene fit with physical bounds, independent complex-data prediction and point-spread checks. Only then can it be extended to the full capture and the requested north-up comoving and fixed-geographic videos. <a href="../native-motion-review/index.html">Previous native motion review</a> · <a href="regressions.log">Numerical regressions</a>.</p></html>'''
    (root/'index.html').write_text(text)
    print('Wrote',root/'index.html',flush=True)


if __name__=='__main__':main()
