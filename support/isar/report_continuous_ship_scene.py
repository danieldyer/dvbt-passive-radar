"""Review continuous-aperture real-data fits and their point controls."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('docs/data/isar_20261018/native-spatial-validation')
    out=root/'continuous-ship-review';out.mkdir(parents=True,exist_ok=True)
    model=json.loads((root/'ais-near-four-target-full-aperture-30s/analysis.json').read_text())
    image=json.loads((root/'four-target-continuous-refined-image/analysis.json').read_text())
    closure=json.loads((root/'ais-near-four-target-full-aperture-30s/component-closure/analysis.json').read_text())
    fit=model['fit'];ratios=closure['folds'][0]['positive_amplitude_to_independent_ratio']
    sources=''.join(f'<tr><td>{j}</td><td>{q[0]:+.2f}, {q[1]:+.2f}</td><td>{r:.2%}</td></tr>'
        for j,(q,r) in enumerate(zip(model['refined_points_east_north_up_m'],ratios)))
    rows=[];comparisons={}
    for name,label in [('ais-near-cv-calibrated-30s','Free position/velocity; straight centre translation'),
                       ('ais-near-coordinated-calibrated-30s','Free position/velocity; coordinated turn')]:
        path=root/name/'analysis.json'
        data=json.loads(path.read_text()) if path.exists() else {}
        if data.get('complete'):
            f=data['fit'];comparisons[name]=dict(complete=True,fit=f,physical_state_enu=data['physical_state_enu'])
            status=f'Native residual {f["objective"]:.6f}; bounds {f["active_bounds"]}; projected gradient {f["maximum_scaled_projected_gradient"]:.3g}'
        else:
            comparisons[name]=dict(complete=False);status='Real-data optimisation running; not a result'
        rows.append(f'<tr><td>{label}</td><td>{status}</td></tr>')
    metrics=dict(full_aperture_fit=fit,initial_full_aperture_objective=model['initial_native_objective'],
        native_image_checks=image['native_checks'],conditional_component_phase_ratios=ratios,
        physical_model_comparisons=comparisons,full_capture_focused_ship_verified=False)
    (out/'analysis.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    html=f'''<!doctype html><html><meta charset="utf-8"><title>Continuous-aperture ship analysis</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;line-height:1.5}}img{{width:100%}}td,th{{padding:9px;text-align:left;vertical-align:top}}.note{{background:#f5f5f5;padding:16px}}</style>
<h1>Continuous-aperture real ship analysis</h1>
<p class="note">This is real recording data, 142–172 s, with all available samples. The full-capture focused ship video remains unfinished. None of these images is certified as a resolved hull or a unique geographic reconstruction.</p>
<p>The four-source shared-motion fit converged in {fit['iterations']} native iterations, without active bounds; maximum scaled projected gradient {fit['maximum_scaled_projected_gradient']:.3g}. Its full-data normalized residual fell from {model['initial_native_objective']:.9f} to {fit['objective']:.9f}. Both numbers use the same complete aperture and nuisance model. The aggregate residual change is not a target-SNR gain.</p>
<p>The current radial-cubic model still conditions anchor position, transverse motion and three support coordinates on earlier radar/AIS estimates. It does not prove that motion or geometry is unique. The separate physical comparisons below instead free east/north position and velocity while holding relative physical band calibration fixed. A common carrier-phase coordinate gauge is inverted on export; it cannot absorb inter-band phase when position changes.</p>
<h2>Individual channels and a coherent combination</h2>
<p>Five individual matched images and a coherent complex sum using the reference reflector's estimated band spectrum. That spectrum is conditional; it need not be optimal for every reflector. Native forward checks put the score error below {max(c['native_observed_score_relative_error'] for c in image['native_checks']):.2g} relative at the audited physical pixels. These are numerical operator checks, not a validation of the ship motion hypothesis.</p>
<img src="../four-target-continuous-refined-image/observed-fixed-spectrum-image.png">
<p><a href="../four-target-continuous-refined-image/observed-image.png">Alternative common-phase statistic with positive band amplitudes profiled</a> · <a href="../four-target-continuous-refined-image/analysis.json">Native image checks</a></p>
<h2>Single-point response under the same aperture</h2>
<img src="../four-target-continuous-refined-image/point-fixed-spectrum-image.png">
<p>Real-image extent must be compared with this response. Sparse-frequency sidelobes and ambiguity remain even after the periodic validation mask is removed. No fitted image rotation, sharpening, magnitude blending or image-derived noise normalization is used.</p>
<h2>Why the bright-flash weighting is not hull evidence</h2>
<p>A conditional temporal template using the reference reflector's fitted visibility gives more weight to its bright flash. The corresponding single-point response below is elongated too: the nonuniform exposure broadens the effective aperture response. A similar elongated shape in the recording cannot establish a resolved extended ship. This comparison uses the assessed full-data seed before the small motion refinement above.</p>
<img src="../four-target-continuous-visibility-seed-image/point-fixed-spectrum-image.png">
<p><a href="../four-target-continuous-visibility-seed-image/observed-fixed-spectrum-image.png">Recording with that visibility template</a></p>
<h2>Multiple candidate phase histories</h2>
<table><tr><th>Candidate</th><th>East, north from anchor (m)</th><th>Common / independent phase statistic</th></tr>{sources}</table>
<p>The ratios retain exact component-overlap uncertainty. They are in-sample conditional model diagnostics, not independent recovery tests, target-SNR bounds or proof of scatterer identity.</p>
<img src="../ais-near-four-target-full-aperture-30s/phase-histories/phase-histories.png">
<h2>Physical motion comparisons on the recording</h2><table>{''.join(rows)}</table>
<p>The coordinated-turn hypothesis couples centre course change and rigid rotation. It is a model comparison, not an assumption that AIS course is vessel heading. The constant-velocity hypothesis leaves rigid spin independent of the straight centre trajectory.</p>
<p><a href="../continuous-aperture-review/index.html">Sampling-mask analysis</a> · <a href="../multiple-component-review/index.html">Earlier multiple-component evidence</a> · <a href="analysis.json">Current numerical results</a></p></html>'''
    (out/'index.html').write_text(html)
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
