"""Review current native-row motion controls and the strong-return association."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from render_injected_point import cartesian_state
from recovery_experiment import SCALE,NAMES
from report_measured_scatterer_phases import serial


def main():
    root=Path('docs/data/isar_20261018');out=root/'native-motion-review';out.mkdir(exist_ok=True)
    turn=root/'turn-control';seed=json.loads((turn/'seed-fit.json').read_text())
    plan=json.loads(Path('docs/data/injection_20261018/separated-500m/plan.json').read_text())
    camera=json.loads((turn/'camera.json').read_text());state=np.asarray(seed['coarse_state_enu']).copy()
    parameters=np.asarray(seed['fits']['constant_velocity_gls']['parameters'])
    state[:,:2]+=(parameters*SCALE).reshape(3,2)
    rotation=np.asarray(plan['ecef_to_enu']);rx=np.asarray(plan['receiver_ecef_m']);tx=np.asarray(plan['transmitters_ecef_m'])
    rate=json.loads((turn/'turn-hypothesis.json').read_text())['course_rate_hypothesis_degps']
    originals=[np.load(turn/n/'images.npz') for n in ['zero','course-rate','course-rate-static-control']]
    east,north=originals[0]['east_m'],originals[0]['north_m'];xx,yy=np.meshgrid(east,north)
    offsets=np.c_[xx.ravel()+camera['state_enu'][0][0]-state[0,0],yy.ravel()+camera['state_enu'][0][1]-state[0,1],np.zeros(xx.size)]
    pose=cartesian_state(np.array([seed['midpoint_s']]),state,seed['midpoint_s'],rotation,rx,rate)
    position=pose['position_ecef_m'][0]+offsets@pose['body_to_ecef'][0].T
    velocity=pose['velocity_ecef_mps'][0]+offsets@pose['frame_velocity'][0].T
    doppler=[];records=[];before,after=originals[1]['power'],originals[2]['power']
    for c in range(5):
        g=sum((position-a)/np.linalg.norm(position-a,axis=1)[:,None] for a in (tx[c],rx))
        fd=-plan['rf_hz'][c]/299792458.*np.sum(g*velocity,axis=1);doppler.append(fd.reshape(xx.shape))
        bright=before[c]>=np.quantile(before[c],.99);zero=abs(doppler[c])<1/30
        records.append(dict(channel=c,name=NAMES[c],near_zero_cell_fraction=float(zero.mean()),
            bright_near_zero_cell_fraction=float(zero[bright].mean()),
            energy_change_in_original_bright_cells_db=float(10*np.log10(after[c][bright].sum()/before[c][bright].sum()))))
    bright=before[5]>=np.quantile(before[5],.99)
    combined_change=float(10*np.log10(after[5][bright].sum()/before[5][bright].sum()))
    fig,axes=plt.subplots(1,3,figsize=(15,6),sharex=True,sharey=True,layout='constrained')
    titles=['Fixed physical orientation','Course-rate rotation hypothesis','Same hypothesis; stationary-mean control']
    for ax,z,title in zip(axes,originals,titles):
        im=ax.imshow(10*np.log10(np.maximum(z['power'][5],1e-30)),origin='lower',extent=[east[0],east[-1],north[0],north[-1]],
            cmap='magma',vmin=5,vmax=50,aspect='equal')
        if z is not originals[0]:ax.contour(xx,yy,doppler[2],levels=[0.],colors=['cyan'],linewidths=.7)
        ax.set(title=title,xlabel='East from AIS antenna (m)',ylabel='North (m)')
    fig.colorbar(im,ax=axes.tolist(),label='Coherent power / original information (dB)',shrink=.75)
    fig.suptitle('Identical original observations, physical seed, band gains and colour scale\nCyan: predicted zero-Doppler locus; mean subtraction is diagnostic, not an optimal-SNR image')
    fig.savefig(out/'stationary-locus-control.png',dpi=160);plt.close(fig)
    result=dict(complete=True,accepted_ship_motion=False,stationary_control=records,
        combined_bright_cell_energy_change_db=combined_change,
        source='Current uninjected production replay, native rows, 142–172 s',
        projection_information_correction_applied=False)
    rows=[]
    for name in ['fading-point-motion','fading-turn-motion']:
        d=json.loads((root/name/'analysis.json').read_text());result[name]=dict(complete=d['complete'],
            excluded_prediction_fraction=[v['test_complex_prediction_fraction'] for v in d.get('prediction',[])])
        if d['complete']:rows.append('<tr><td>'+name+'</td>'+''.join(f'<td>{v:.1%}</td>' for v in result[name]['excluded_prediction_fraction'])+'</tr>')
    scans=[]
    for name in ['native-main-branch-chirps-6s','native-main-branch-chirps-18s']:
        d=json.loads((root/name/'analysis.json').read_text())
        scans.append(f'<h2>{d["interval_s"][0]:g}–{d["interval_s"][1]:g} s, independently acquired strong branch</h2><table><tr><th>Band</th><th>Delay from AIS (m)</th><th>Residual rate (m/s)</th><th>Residual acceleration (m/s²)</th><th>Excluded prediction improvement</th></tr>'+''.join(
            f'<tr><td>{c["name"]}</td><td>{c["origin_m"]:.1f}</td><td>{c["rate_mps"]:+.3f}</td><td>{c["acceleration_mps2"]:+.3f}</td><td>{c["prediction"]["test_complex_prediction_fraction"]:.1%}</td></tr>' for c in d['channels'])+
            f'</table><img src="../{name}/independent-band-chirps.png"><p><a href="../{name}/analysis.json">Full fit details</a>. These are independent best local hypotheses; they do not certify that the same material point was selected in every band. Red phase samples were excluded from fitting.</p>')
    scene_sections=[]
    for name in ['native-main-branch-scene-6s','native-main-branch-scene-18s']:
        path=root/name/'analysis.json'
        if not path.exists():continue
        d=json.loads(path.read_text())
        result[name]=dict(complete=d['complete'],channels=[dict(name=c['name'],prediction=[v['prediction']['explained_fraction'] for v in c['fits']]) for c in d['channels']])
        scene_sections.append('<h2>'+name+'</h2><p>One to three quadratic histories, with one frozen complex coefficient per echo and band. Subsequent candidates are acquired from training residuals only. Component count is compared on excluded data; this is a deblending test, not geographic autofocus.</p><table><tr><th>Band</th><th>One echo</th><th>Two echoes</th><th>Three echoes</th></tr>'+''.join('<tr><td>'+c['name']+'</td>'+''.join(f'<td>{v["prediction"]["explained_fraction"]:.1%}</td>' for v in c['fits'])+'</tr>' for c in d['channels'])+'</table>')
    brightness=json.loads((root/'native-positive-brightness-6s/analysis.json').read_text())
    result['brightness_control']=brightness
    short_rows=''.join(f'<tr><td>{c["name"]}</td><td>{c["constant"]["prediction"]:.1%}</td><td>{c["fading"]["prediction"]:.1%}</td></tr>' for c in brightness['channels'])
    radial=json.loads((root/'native-shared-radial-point-6s/analysis.json').read_text())
    planar=json.loads((root/'native-shared-main-point-6s/analysis.json').read_text())
    result['shared_short_arc']=dict(radial_fit=radial['fit'],radial_prediction=[v['test_complex_prediction_fraction'] for v in radial['prediction']],
        planar_active_bounds=planar['fit']['active_bounds'],transverse_motion_fixed_to_ais=True,geographic_localisation_certified=False)
    calibration=json.loads((root/'native-shared-radial-point-6s/calibration-transfer.json').read_text())
    result['short_arc_calibration_transfer']=calibration
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for fold,style in zip(calibration['folds'],['o-','s--']):
        axes[0].plot(NAMES,fold['band_phase_error_deg'],style,label=fold['fold'])
    axes[0].set(ylabel='Matched phase relative to frozen calibration (degrees)',title='The fitted point; no excluded-data phase refit');axes[0].legend()
    axes[1].bar([v['fold'] for v in calibration['folds']],[v['coherent_to_independent_power'] for v in calibration['folds']])
    axes[1].set(ylim=(0,1.05),ylabel='Coherent / independent observed power',title='Descriptive transfer statistic, not an SNR bound')
    fig.savefig(out/'calibration-transfer.png',dpi=160);plt.close(fig)
    with np.load(root/'native-shared-radial-point-6s/image-course-rate/images.npz') as z:
        power=z['power'];east,north=z['east_m'],z['north_m']
    camera=json.loads((root/'native-shared-radial-point-6s/camera.json').read_text())
    high=float(10*np.log10(power.max()))+.1
    fig,axes=plt.subplots(2,3,figsize=(12,11),sharex=True,sharey=True,layout='constrained')
    for ax,name,power_c in zip(axes.ravel(),NAMES+['Coherent five-band sum'],power):
        im=ax.imshow(10*np.log10(np.maximum(power_c,1e-30)),origin='lower',extent=[east[0],east[-1],north[0],north[-1]],
            aspect='equal',cmap='magma',vmin=high-45,vmax=high)
        ax.plot(*camera['marker_east_north_m'],'+',color='cyan',ms=8)
        ax.set(title=name,xlabel='East from fitted point (m)',ylabel='North (m)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Conditional matched power (dB)',shrink=.7)
    fig.suptitle('Conditional shared-motion short-arc image · 158–164 s\nNorth up; cyan = AIS antenna prior; course-rate rotation is a hypothesis\nNo image registration or Doppler subtraction; geographic localisation remains unproved')
    fig.savefig(out/'short-arc-six-panels.png',dpi=150);plt.close(fig)
    extra='''<h2>Separate brightness from phase on the same excluded epochs</h2>
<p>The next control excludes quarter-second blocks throughout the six-second arc. Both constant and changing-brightness fits are acquired and estimated using exactly the same training epochs. One-second nonnegative real brightness knots allow scintillation without a complex per-time coefficient. This is a different validation fold from the two-second omissions above; improvements must only be compared within this table. Percentages describe whole-crop complex squared-error reduction, not target SNR efficiency.</p>
<table><tr><th>Band</th><th>Constant brightness</th><th>Positive brightness knots</th></tr>SHORTROWS</table>
<img src="../native-positive-brightness-6s/brightness-and-phase.png">
<p>The low-strength intervals retain large measured phase residuals. They are not hidden or labelled stable material points. The fit uses a constant phase and quadratic path per band; observed mixtures can still violate this model.</p>
<h2>Conditional shared physical short arc</h2><p>All five bands can be fitted together with a single radial physical position/velocity/acceleration correction, constant band phases and positive brightness knots. Transverse position and motion are explicitly conditioned on AIS. The radial fit has no active bounds, while the unconstrained planar fit hits three. This tests concentration in the observed motion directions; it does not prove geographic localisation or infer hull rotation.</p>
<img src="short-arc-six-panels.png"><p>All native C0/C1 rows contribute to these six-second images. The displayed course-rate rotation comes from sparse AIS COG as a hypothesis; true heading is unavailable. The complete observed scene is shown, including stationary clutter. A coherent sum is computed before power detection using band coefficients frozen from training. These are conditional experimental images, not a certified hull reconstruction or the final requested videos. <a href="../native-shared-radial-point-6s/image-fixed-ne/point-images.png">Fixed-orientation control</a> · <a href="../native-shared-radial-point-6s/analysis.json">Shared fit and excluded predictions</a>.</p>
<h2>Frozen interband calibration transfers at the fitted point</h2><img src="calibration-transfer.png">
<p>Across the excluded quarter-second blocks, the coherent observed matched power is 95.92% of the sum with independently fitted complex band amplitudes: a 0.181 dB difference. Band phase errors relative to frozen training calibration are −2.2°, +3.2°, +2.4°, +0.7° and +18.1° for Seven/SBS/Nine/Ten/ABC. Neither band phases nor stationary coefficients are refitted on excluded epochs. This is useful evidence of local interband coherence, not a target-SNR efficiency bound, an absolute instrument calibration, or proof that the same calibration transfers across other scatterers and the full recording. Transverse motion remains conditioned on AIS.</p>
'''.replace('SHORTROWS',short_rows)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Native ship-motion review</title><style>body{font:16px/1.55 system-ui;max-width:1250px;margin:30px auto;padding:0 22px}img{width:100%}table{border-collapse:collapse}th,td{padding:7px 12px;border-bottom:1px solid #ccc;text-align:left}</style>
<h1>Current native-row ship-motion review</h1>
<p><strong>A focused ship image remains unproved.</strong> Recent physical single-point fits were restricted to an AIS-centred ±300 m crop. The stronger branch approximately 450 m farther in excess range lies outside that crop. Those controls cannot reject a point model for the stronger branch. Earlier whole-capture work did include both branches; this is a limitation of the recent injection-matched controls, not a new claim about all earlier experiments.</p>
<p>The new acquisitions explicitly recenter the fixed range observation around that branch. They preserve all native symbol epochs, known waveform kernels and complex phases. The chirp bank proposes candidates; final fitting includes range migration, RF phase, C0/C1 moments and the identical stationary-nuisance projection on data and templates. No image alignment or fitted time-varying band phase is applied.</p>
<h2>Stationary structure can masquerade as a better image</h2><img src="stationary-locus-control.png">
<p>The rotation hypothesis creates a line on which pixel velocity has zero bistatic projection. Part of the bright structure follows that line. Removing only the information-weighted temporal mean lowers energy in the combined image’s original brightest 1% of cells by COMBINED dB. This is an energy comparison, not an SNR gain. Projection loss in target information has not been corrected in these diagnostic images; no broad Doppler notch or production change is adopted.</p>
<h2>Completed physical single-point controls on the AIS-centred crop</h2>
<p>Values below are reduction in squared complex prediction error versus the training-fitted stationary-only model. Negative values mean the predicted echo worsens excluded-data error. Smooth turning motion and positive brightness knots do not rescue these controls. They are rejected as focused-image evidence.</p>
<table><tr><th>Model</th><th>Seven</th><th>SBS</th><th>Nine</th><th>Ten</th><th>ABC</th></tr>ROWS</table>
EXTRA SCANS SCENES
<h2>Validation and interpretation</h2><p>The same inference/formation operator has passed the <a href="../../injection_20261018/constant-velocity-iq/index.html">paired channel-IQ point test</a>. Its stronger point loses 0.0019 dB in the recovered-motion versus oracle image, and its recovered position is 7.1 m from truth. The original-strength point has about 68 m error; strength and geometric conditioning remain relevant. These point results do not establish that the recorded ship is a single stable scatterer.</p>
<p>The acquisition/association, translational compensation, rotational compensation and final image tests remain distinct, as in <a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5233">Garry and Smith’s passive-ISAR framework</a>. The next acceptance requirement is a shared physical multi-scatterer model that predicts excluded complex data and focuses native-row images; a bright smear or lower training residual alone does not satisfy it.</p></html>'''
    text=text.replace('COMBINED',f'{-combined_change:.2f}').replace('ROWS',''.join(rows)).replace('SCANS',''.join(scans)).replace('SCENES',''.join(scene_sections)).replace('EXTRA',extra)
    (out/'index.html').write_text(text)
    print(json.dumps(serial(result)),flush=True)


if __name__=='__main__':main()
