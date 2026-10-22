"""Publish nested fixed-N/E scene, translation and conditional efficiency tests."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fit_distributed_motion import aggregate
from fit_mixed_motion import changes
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


NAMES=['Seven','SBS','Nine','Ten','ABC']


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path('build/isar-20261016'))
    p.add_argument('--output',type=Path,default=Path('docs/data/isar_20261015'))
    p.add_argument('--durations',type=int,nargs='+',default=[30,45])
    p.add_argument('--videos',type=Path)
    args=p.parse_args()
    sources,experiments=[],[]
    for duration in args.durations:
        paths=[args.work/f'mixed-ne-cv-{duration}s/experiment.json',
            args.work/f'mixed-ne-motion-{duration}s.json',args.work/f'mixed-ne-motion-{duration}s-grid-audit.json']
        training,motion,grid=[read(path) for path in paths]
        if not all(v['complete'] for v in (training,motion,grid)) or not motion['fit']['converged']:
            raise ValueError('Finish all scene controls, original-row fits and grid checks')
        if motion['training_sha256']!=fingerprint(paths[0]) or grid['motion_sha256']!=fingerprint(paths[1]):
            raise ValueError('The experiment source identities disagree')
        if np.any(SplineMotion(motion).yaw_coefficients):
            raise ValueError('Yaw must remain exactly zero')
        sources.extend({'path':str(path),'sha256':fingerprint(path)} for path in paths)
        experiments.append({'duration_s':duration,'training':training,'motion':motion,'grid':grid})
    numerical_path=args.work/'mixed-ne-numerical-audit.json'
    numerical=read(numerical_path)
    if not numerical['complete'] or numerical['training_sha256']!=fingerprint(Path(numerical['training'])):
        raise ValueError('The numerical audit must use the completed, unchanged training experiment')
    sources.append({'path':str(numerical_path),'sha256':fingerprint(numerical_path)})
    test_path=args.work.parent/'isar-20261016-tests-mixed.log'
    log=test_path.read_text();count=re.search(r'Ran (\d+) tests',log)
    if not count or not log.rstrip().endswith('OK'):
        raise ValueError('The support suite has not passed')
    args.output.mkdir(parents=True,exist_ok=True)
    def link(path,label):
        return f'<a href="{escape(os.path.relpath(Path(path).resolve(),args.output.resolve()))}">{escape(label)}</a>'
    fig,axes=plt.subplots(1,len(experiments),figsize=(7*len(experiments),5),squeeze=False,layout='constrained')
    results=[]
    for ax,experiment in zip(axes.ravel(),experiments):
        m=experiment['motion']
        labels=['Added scene, original motion','Added scene, refitted motion','Original scene, refitted motion']
        keys=['before','after','sparse_at_new_motion']
        for j,(key,label) in enumerate(zip(keys,labels)):
            gain=[changes(a,b)['matched_energy_change_db'] for a,b in zip(m['baseline'],m[key])]
            ax.bar(np.arange(5)+(j-1)*.25,gain,width=.25,label=label)
        ax.axhline(0.,color='#333',lw=.8)
        ax.set(xticks=np.arange(5),xticklabels=NAMES,ylabel='Withheld matched-energy change (dB)',
            title=f'{experiment["duration_s"]} s fit; common yaw-free trajectory')
        ax.legend(fontsize=8)
        baseline=m['summaries']['baseline']
        results.append({'duration_s':experiment['duration_s'],
            'frozen_motion_change':changes(baseline,m['summaries']['before']),
            'total_change':m['baseline_comparison'],'motion_change':m['motion_change'],
            'sparse_motion_change':m['sparse_motion_comparison'],
            'maximum_kkt_residual':m['fit']['maximum_kkt_residual'],
            'penalties':m['fixed_penalties'],
            'after_explained_fraction':m['summaries']['after']['heldout_explained_fraction']})
    fig.suptitle('A nested model retains every original reflector\nAdded coefficients and motion are trained without outer test blocks')
    fig.savefig(args.output/'mixed-ne-prediction.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(experiments),2,figsize=(14,4.6*len(experiments)),squeeze=False,layout='constrained')
    for row,experiment in enumerate(experiments):
        m=experiment['motion'];old=read(Path(m['source_motion']))
        time=np.linspace(*m['interval_s'],601)
        a,b=[SplineMotion(v).components(time) for v in (old,m)]
        for j,label in [(0,'Reference path coordinate'),(-1,'Transverse coordinate')]:
            axes[row,0].plot(time,b[:,j]-a[:,j],label=label)
        axes[row,0].set(xlabel='Capture time (s)',ylabel='Change from original fit (m)',
            title=f'{experiment["duration_s"]} s: physical translation update')
        axes[row,0].legend(fontsize=8)
        for key,label,style in [('baseline','Original sparse model','o--'),('after','Mixed model and refitted motion','x-')]:
            axes[row,1].plot(NAMES,[v['phase_rms_degrees'] for v in m[key]],style,label=label)
        axes[row,1].set(ylabel='One-second prediction phase RMS (degrees)',title='Includes noise and scattering mismatch')
        axes[row,1].legend(fontsize=8)
    fig.savefig(args.output/'mixed-ne-motion-phase.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),layout='constrained')
    for experiment in experiments:
        for spacing,style in [(20.,'o-'),(10.,'x--')]:
            trials=[next(t for t in c['trials'] if t['spacing_m']==spacing) for c in experiment['training']['channels']]
            label=f'{experiment["duration_s"]} s / {spacing:g} m grid'
            axes[0].plot(NAMES,[t['tuning']['effective_scene_rank'] for t in trials],style,label=label)
            axes[1].plot(NAMES,[t['heldout_matched_energy_gain_db'] for t in trials],style,label=label)
    axes[0].axhline(8,color='#444',lw=.7,ls=':')
    axes[0].set(ylabel='Effective scene rank',title='Eight original reflectors are always retained')
    axes[1].set(ylabel='Withheld matched-energy change (dB)',title='Fixed-motion grid convergence')
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(args.output/'mixed-ne-grid-controls.png',dpi=150);plt.close(fig)
    video_html=''
    video_checks=None
    quicktime_checks=[]
    if args.videos:
        validation=args.videos/'validation.json'
        video_checks=read(validation)
        manifest_path=args.videos/'videos.json'
        manifest=read(manifest_path)
        if not video_checks['complete'] or not manifest['complete']:
            raise ValueError('Finish encoding and validating both videos')
        if manifest['scientific_motion_sha256']!=fingerprint(Path(manifest['scientific_motion'])):
            raise ValueError('The video motion source has changed')
        if manifest['scientific_motion_sha256'] not in [fingerprint(args.work/f'mixed-ne-motion-{d}s.json') for d in args.durations]:
            raise ValueError('The video must use a reported motion fit')
        for name in ('ship-following-north-up-30s-30fps.mp4','ship-geographic-north-up-30s-30fps.mp4'):
            path=args.videos/name
            entries=[v for v in video_checks['files'] if Path(v['path']).resolve()==path.resolve()]
            if not path.exists() or len(entries)!=1 or not entries[0]['full_decode_passed']:
                raise ValueError('Complete and fully decode both requested views')
            if entries[0]['frames']!=manifest['frames'] or entries[0]['fps']!=manifest['fps']:
                raise ValueError('The validation and manifest frame grids disagree')
        for name in ('following','geographic'):
            path=args.videos/f'quicktime-{name}.json'
            check=read(path)
            if (check['framework']!='AVFoundation' or len(check['seeks'])<34
                    or abs(check['duration_s']-manifest['frames']/manifest['fps'])>1e-8
                    or any(abs(s['actual_s']-s['requested_s'])>1e-8 for s in check['seeks'])):
                raise ValueError('Both views must pass exact forward/backward AVFoundation seeks')
            quicktime_checks.append(check)
            sources.append({'path':str(path),'sha256':fingerprint(path)})
        sources.extend({'path':str(path),'sha256':fingerprint(path)} for path in (validation,manifest_path))
        video_html='<h2>New diagnostic previews</h2><p>'+link(args.videos/'ship-following-north-up-30s-30fps.mp4','Ship-following, north-up')+' · '+link(args.videos/'ship-geographic-north-up-30s-30fps.mp4','Fixed geographic frame, north-up')+'''</p>
<p>The 45-second fit supplies a sliding 30-second aperture. Both views use the original five-channel profiles, zero yaw, no aperture DC notch, fixed colour scale and cyan interpolated AIS. Complex channel sums precede power detection. The learned reflectivity is not substituted for the data. The preserved provisional channel calibration remains unvalidated; these previews are a motion-model comparison, not a resolved-hull claim.</p>
<p>Each preview is 14.9 seconds long at 30 fps (447 frames), covering aperture centres 158.333–173.200 seconds into the capture. Both pass full decoding and 34 exact forward/backward AVFoundation seeks. The fixed scale is 21.875–49.6 dB. The visible broad diagonal responses remain unresolved; these images do not demonstrate improved hull focusing.</p>'''
        for name,label in [('following','Ship-following'),('geographic','Fixed geographic')]:
            path=args.videos/f'{name}-frame-00223.png'
            video_html+=f'<figure><img src="{escape(os.path.relpath(path.resolve(),args.output.resolve()))}" alt="{label}, middle frame"><figcaption>{label}, capture time 165.767 s; north up and cyan AIS.</figcaption></figure>'
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'scope':__doc__,'sources':sources,
        'results':results,'experiments':experiments,'numerical_audit':numerical,
        'support_tests_passed':int(count[1]),'test_log':str(test_path),'video_validation':video_checks,
        'quicktime_validation':quicktime_checks,
        'videos':str(args.videos) if args.videos else None,'yaw_fitted':False,'phase_centres_validated':False,
        'production_changed':False,'independent_band_phase_calibration_validated':False,
        'literature':[{'url':'https://www.cs.umd.edu/~oleary/software/varpro.pdf','title':'O’Leary and Rust, Variable Projection for Nonlinear Least Squares Problems'},
            {'url':'https://gaussianprocess.org/gpml/chapters/RW2.pdf','title':'Rasmussen and Williams, Gaussian Processes for Machine Learning, section 2.7'}]}
    (args.output/'mixed-ne-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=''.join(f'<tr><td>{r["duration_s"]} s</td><td>{r["frozen_motion_change"]["matched_energy_change_db"]:+.3f} dB</td><td>{r["total_change"]["matched_energy_change_db"]:+.3f} dB</td><td>{r["total_change"]["prediction_error_change_percent"]:+.3f}%</td><td>{r["sparse_motion_change"]["matched_energy_change_db"]:+.3f} dB</td><td>{r["maximum_kkt_residual"]:.3g}</td></tr>' for r in results)
    grid_cases=[c for e in experiments for c in e['grid']['channels'] if not c.get('core_only')]
    max_grid=max(abs(v['matched_energy_change_db']) for v in grid_cases)
    max_operator=max(c['same_coefficient_prediction']['matched_snr_loss_db'] for e in experiments for c in e['motion']['search_approximation_check'])
    efficiency=numerical['conditional_efficiency']
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Nested fixed N/E scene and translation validation</title>
<style>body{{font:17px/1.55 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}pre{{padding:16px;background:#f4f6f8;overflow:auto}}.note{{padding:18px;background:#fff0ce}}</style>
<h1>Keep the reflectors; regularize only the additional scene</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="mixed-ne-validation.json">Full measurements and identities</a> · <a href="distributed-ne-validation.html">Previous distributed-only control</a> · <a href="information-autofocus-audit.html">Stage information audit</a></p>
<p class="note">Yaw is exactly zero. This model retains the original eight unpenalized reflectors in each band and adds an optional distributed component. It improves prediction relative to the original fixed-N/E baseline in the reported apertures. The conditional linear estimator passes its theoretical efficiency check, but the physical motion and cross-band phase calibration remain unverified. The results do not establish a resolved hull or a real-data CRLB certificate.</p>
<p>For the original question, <a href="fixed-ne-validation.html">the direct yaw-removal comparison</a> remains negative: matched energy fell by 1.159 and 3.924 dB in two overlapping 30-second apertures, with withheld error rising 22.45% and 40.98%. The improvements below are relative to those fixed-N/E models, not gains over the rotating controls. A predictive advantage for rotation does not establish that the fitted yaw is the vessel's true heading.</p>
<h2>The nested model and validation protocol</h2>
<pre>minimize over physical translation p and complex responses a,b:
Σc ||Pc [yc − A0,c(p) ac − A1,c(p) Dc bc]||² + λc ||bc||²

A0: the original eight-point scene, with no coefficient penalty
A1: the added fixed east/north distributed scene
Dc: square root of quadrature cell area
Pc: the unchanged stationary-nuisance projector
λ = infinity: exactly the original eight-point model</pre>
<p>Unlike the previous distributed-only control, a strongly regularized channel keeps its moving-target response and its motion information. No channel is discarded. The added support covers east −120 to +120 m and north −180 to +200 m at 20 m relative height; 20 m and 10 m grids approximate the same per-unit-area prior. The rectangle is support for a model, not an inferred hull boundary.</p>
<p>Three inner splits within alternating 0.5-second training blocks select λ from 65 finite strengths and the exact baseline limit. The outer blocks never select the penalty. The subsequent physical-motion search freezes λ and re-solves all complex coefficients at every trial. Delay, carrier phase and within-symbol Doppler change together. Yaw, roll, pitch, antenna locations, hardware range origins and provisional channel gains stay fixed.</p>
<p>The 30-second aperture is 150.772–180.772 s; the 45-second aperture is 143.272–188.272 s. They overlap, so they are consistency controls rather than independent replication. Existing reflector support and coarse motion already used these observations in earlier work; this is conditional validation of the added model and refit, not independent target discovery.</p>
<img src="mixed-ne-grid-controls.png" alt="Training-selected complexity and fixed-motion prediction under spatial refinement">
<h2>Measured prediction changes</h2>
<table><tr><th>Fit aperture</th><th>Added scene, fixed motion</th><th>Scene + refitted motion</th><th>Squared error change</th><th>Old scene at new motion</th><th>Original-row KKT residual</th></tr>{rows}</table>
<p>All changes use the original sparse fixed-N/E fit as baseline. Matched energy is <code>|predictionᴴ y|² / ||prediction||²</code> after the common stationary projection. Squared prediction error retains the fitted amplitude and phase without adjusting them on test data. The aggregate concatenates band predictions and observations; it does not assert a common reflectivity or independently calibrated coherent spatial sum.</p>
<p>The last comparison deliberately removes the diffuse component when evaluating the new motion. It distinguishes benefit of the combined model from a claim that the new trajectory alone improves every scene representation. Physical truth cannot be established from a lower penalized training objective or a brighter image peak.</p>
<img src="mixed-ne-prediction.png" alt="Per-channel matched-energy changes with frozen and refitted motion, including the original-scene control">
<img src="mixed-ne-motion-phase.png" alt="Physical translation changes and withheld prediction-phase statistics">
{video_html}
<h2>Numerical checks and the precise theoretical result</h2>
<p>All {int(count[1])} support tests pass. The new tests compare core elimination against an independent augmented SVD, including exact overlap between core and diffuse columns; check the complete residual Jacobian; verify original-row physical derivatives; preserve the complete baseline value, gradient and curvature in the infinite-penalty limit; and confirm that arbitrary corruption of outer test samples cannot affect penalty selection.</p>
<p>An actual-data directional derivative agrees with the normalized analytic gradient to {min(v['absolute_error'] for v in numerical['gradient_checks']):.3g}. Every final motion fit passes original-row stationarity and feasibility. The largest composite-template loss from the representative-epoch search approximation is {max_operator:.3g} dB. Refining the final diffuse scene from 20 m to 10 m, with motion and penalties frozen, changes per-band matched energy by at most {max_grid:.5f} dB. These are numerical checks, not evidence that the physical scattering assumptions are exact.</p>
<p>For known motion and λ, unit projected thermal noise and a true diffuse prior <code>b ∼ CN(0, λ⁻¹I)</code>, the linear step is the best linear unbiased predictor with an unknown deterministic core mean. Its augmented matrix and prediction risk are:</p>
<pre>H = [ A0  A1D ]       V = (HᴴH + Λ)⁻¹
Λ = diag(0 for core, λ for diffuse)
predicted held-out mean = B V Hᴴ y
expected squared prediction error = trace(B V Bᴴ)</pre>
<p>The core complex response is preserved to {efficiency['maximum_unbiased_core_complex_response_error']:.3g}. On the actual 30-second Ten operator, the calculated risk divided by this conditional bound is {efficiency['calculated_prediction_risk_over_conditional_bound']:.12f}. Across {efficiency['trials']:,} stipulated-model simulations, measured risk divided by the bound is {efficiency['measured_prediction_risk_over_conditional_bound']:.6f}, with Monte Carlo standard error {efficiency['monte_carlo_standard_error_of_ratio']:.6f}. This certifies the linear calculation under its declared model; it does not validate the recording's noise covariance, diffuse prior, motion or antenna calibration.</p>
<p>The physical-motion step is a constrained penalized fit. It is not asserted to maximize a marginalized random-scene likelihood: that likelihood would additionally include the diffuse covariance log determinant. The grid audit records its change separately. In particular, conditional linear MMSE optimality does not establish joint motion efficiency, and optimizer curvature is not substituted for a real-data Fisher information bound.</p>
<h2>Literature connection and remaining scope</h2>
<p><a href="https://www.cs.umd.edu/~oleary/software/varpro.pdf">O’Leary and Rust, Variable Projection for Nonlinear Least Squares Problems</a> explains elimination of linear coefficients and retaining both terms of the residual Jacobian. This implementation applies that method to the augmented, partially penalized complex system, with the existing exact bistatic operator; it uses SVD rather than normal equations.</p>
<p><a href="https://gaussianprocess.org/gpml/chapters/RW2.pdf">Rasmussen and Williams, Gaussian Processes for Machine Learning, section 2.7</a> develops a Gaussian residual process alongside explicit mean basis functions, including uncertain mean coefficients. That supplies the statistical structure for the core-plus-diffuse control. Its usefulness for this ship still requires the measured transfer tests; its prior is not established by the mathematical equivalence.</p>
<p>The current result is an improvement in conditional signal prediction while preserving the requested fixed orientation. Accepted physical phase centres, validated cross-band scattering phase and full-capture coherent hull imaging remain open. Production capture and range-compression code are unchanged.</p>
<p>{link(numerical_path,'Numerical and conditional-efficiency audit')} · {link(test_path,'Full test log')}</p></html>'''
    (args.output/'mixed-ne-validation.html').write_text(page)
    print(json.dumps({'report':str(args.output/'mixed-ne-validation.html'),'results':results,'tests':int(count[1])}))


if __name__=='__main__':
    main()
