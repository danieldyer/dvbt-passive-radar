"""Check short measured trajectories against every licensed site hypothesis.

The receiver leg cancels in the differences. Necessary geometric bounds are
evaluated over all declared correlated antenna boxes, a 500 m scene ball,
speed <= 8 m/s and material-point acceleration <= 1 m/s². A pass is not a
location solution. Conditional thermal errors are not model-error intervals.
"""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from antenna_motion import Antennas
from component_phase import component_measurements
from compare_ais_phase import COLOURS,NAMES
from fit_fixed_grid_phases import prepare_fixed
from flash_geometry import assignments
from phase_window import window_band
from relative_delay_geometry import correlated_pair_bounds
from report_measured_scatterer_phases import load_fields,serial
from shared_echo_cluster import SharedEchoCluster
from ship_site_signature import licensed_options


def main():
    root=Path('docs/data/isar_20261017');directory=root/'piecewise-fixed-cluster'
    source=directory/'fits.json';data=json.loads(source.read_text())
    if not data['complete']:raise ValueError('Wait for all declared starts')
    geometry=Path('docs/data/calibration_20261016_sites');geo=json.loads((geometry/'geometry-prior.json').read_text())
    bands,_,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,163.,167.,stationary=True)
    centre=bands[2].field.data['centre_ecef_m'][0];rotation=np.asarray(geo['ecef_to_enu'])
    bounds=[]
    for choice in assignments(licensed_options(geometry)):
        sites=[o['site_id'] for o in choice];tx=np.array([o['transmitter_ecef_m'] for o in choice])
        factor=Antennas(rotation,sites).factor[:15,:15]
        bound=correlated_pair_bounds(centre,tx,rotation,factor,scene_radius_m=500.,maximum_speed_mps=8.,maximum_acceleration_mps2=1.)
        bound['assignment']=''.join({'12013':'0','12015':'1','12014':'2'}[s] for s in sites)
        bounds.append(bound)
    maximum_rate=np.max([b['maximum_rate_difference_mps'][:,2] for b in bounds],axis=0)
    maximum_acceleration=np.max([b['maximum_acceleration_difference_mps2'][:,2] for b in bounds],axis=0)
    windows=[];errors=[]
    for window in data['windows']:
        first,last=window['interval_s'];entries=[]
        for c,entry in enumerate(window['independent']):
            band,part,_=window_band(bands[c],first,last)
            model=SharedEchoCluster([band],data['selected_origins_samples'],origin_radius=data['origin_radius_samples'])
            for fit in entry['trials']:
                r,_=model.evaluate(np.asarray(fit['parameters']))
                errors.append(abs(r@r*model.energy-fit['cost'])/fit['cost'])
            fit=entry['trials'][entry['best_index']];p=np.asarray(fit['parameters'])[model.indices[0]]
            cluster=model.clusters[0];value=cluster.evaluate(p);information=value['information']
            scale=np.sqrt(np.diag(information));normalized=information/scale[:,None]/scale[None,:]
            eigen=np.linalg.eigvalsh(normalized);covariance=None
            if eigen[0]>eigen[-1]*1e-10:
                covariance=np.linalg.solve(normalized,np.eye(len(scale)))/scale[:,None]/scale[None,:]
            measurements=component_measurements(cluster,p);components=[]
            rows=band.field.data['row_time']
            known_rate=np.gradient(band.known_path,rows)
            known_acceleration=np.gradient(known_rate,rows)
            for s,(q,record) in enumerate(zip(p.reshape(-1,4),measurements)):
                time=record['time_s'];path=record['physical_path_change_m']
                innovation=np.angle(record['complex_coefficient']*np.exp(1j*band.field.k*path))
                weight=np.sum(abs(record['frozen_component_prediction'])**2,axis=1)
                test=(part==3)&(weight>0)
                rms=np.degrees(np.sqrt(np.average(innovation[test]**2,weights=weight[test]))) if test.any() else None
                h=record['frozen_component_prediction'].ravel()[band.selected]
                retained=float(np.vdot(band.nuisance.remove(h),band.nuisance.remove(h)).real/np.vdot(h,h).real)
                rate=q[2]/band.scale+np.interp(band.epoch,rows,known_rate)
                acceleration=2*q[3]/band.scale**2+np.interp(band.epoch,rows,known_acceleration)
                uncertainty=None
                if covariance is not None:
                    derivative=np.diag([1/band.scale,2/band.scale**2])
                    uncertainty=derivative@covariance[4*s+2:4*s+4,4*s+2:4*s+4]@derivative
                selected=[2*s,2*s+1,8+2*s,8+2*s+1]
                components.append(dict(name=data['labels'][s],rate_mps=rate,acceleration_mps2=acceleration,
                    conditional_covariance=uncertainty,retained_signal_energy=retained,
                    unused_phase_rms_deg=rms,component_bound_indices=[i for i in fit['bound_indices'] if i in selected]))
            entries.append(dict(channel=c,name=NAMES[c],components=components,
                maximum_projected_gradient=fit['maximum_projected_gradient'],all_bounds=fit['bound_indices'],
                information_eigenvalues=eigen,training_cost=fit['cost'],unused_error=fit['prediction']['testing'][0]['heldout_error']))
        points=[]
        for s,name in enumerate(data['labels']):
            components=[e['components'][s] for e in entries];reference=components[2]
            rate=np.array([c['rate_mps']-reference['rate_mps'] for c in components])
            acceleration=np.array([c['acceleration_mps2']-reference['acceleration_mps2'] for c in components])
            sigma=[]
            for c in components:
                if c['conditional_covariance'] is None or reference['conditional_covariance'] is None:sigma.append([np.nan,np.nan])
                else:sigma.append(np.sqrt(np.diag(c['conditional_covariance']+reference['conditional_covariance'])))
            sigma=np.array(sigma);sigma[2]=0.
            rate_excess=np.maximum(0.,abs(rate)-maximum_rate-3*sigma[:,0])
            acceleration_excess=np.maximum(0.,abs(acceleration)-maximum_acceleration-3*sigma[:,1])
            points.append(dict(name=name,rate_difference_mps=rate,acceleration_difference_mps2=acceleration,
                conditional_sigma=sigma,rate_outside_union_plus_3sigma_mps=rate_excess,
                acceleration_outside_union_plus_3sigma_mps2=acceleration_excess,
                all_conditional_covariances_identifiable=bool(np.isfinite(sigma).all()),
                minimum_projected_signal_energy=min(c['retained_signal_energy'] for c in components),
                necessary_bound_violated=bool(np.any(rate_excess>0)|np.any(acceleration_excess>0)),
                any_component_bounds=any(c['component_bound_indices'] for c in components)))
        windows.append(dict(interval_s=[first,last],channels=entries,points=points))
        print(first,[(p['name'],p['necessary_bound_violated'],round(p['minimum_projected_signal_energy'],3),p['any_component_bounds']) for p in points],flush=True)
    if max(errors)>1e-9:raise ValueError('Likelihood reproduction failed')
    out=root/'piecewise-geometry-review';out.mkdir(exist_ok=True)
    times=np.array([np.mean(w['interval_s']) for w in windows])
    fig,axes=plt.subplots(2,4,figsize=(20,9),layout='constrained',sharex=True)
    for s,label in enumerate(data['labels']):
        for row,(key,bound) in enumerate((('rate_difference_mps',maximum_rate),('acceleration_difference_mps2',maximum_acceleration))):
            ax=axes[row,s]
            largest=float(max(bound));ax.axhspan(-largest,largest,color='#ddd',alpha=.65)
            for c,colour in enumerate(COLOURS):
                if c==2:continue
                values=np.array([w['points'][s][key][c] for w in windows])
                sigma=np.array([w['points'][s]['conditional_sigma'][c,row] for w in windows])
                ax.errorbar(times+(c-2)*.015,values,yerr=sigma,fmt='o-',color=colour,capsize=2,lw=1,ms=3,label=NAMES[c])
            ax.set(title=label+(' · rate' if row==0 else ' · acceleration'),xlabel='One-second interval midpoint (s)',
                ylabel='Band minus Nine ('+('m/s)' if row==0 else 'm/s²)'))
            ax.axhline(0,color='#888',lw=.6)
    axes[0,1].legend(ncol=2,fontsize=8)
    fig.suptitle('Can the measured local phase differences come from one point under any licensed site assignment?\n'
        'Grey: conservative union of geometric bounds, including all current correlated antenna boxes; speed ≤8 m/s, acceleration ≤1 m/s².\n'
        'Bars: conditional thermal 1σ only. Non-converged, bound-limited and near-static components remain in the tables; passing a bound is not a calibration.',fontsize=12)
    fig.savefig(out/'01-rate-acceleration-feasibility.png',dpi=160);plt.close(fig)
    result=dict(accepted_calibration=False,production_changed=False,scope=__doc__,windows=windows,
        geometric_bounds=bounds,union_maximum_rate_mps=maximum_rate,union_maximum_acceleration_mps2=maximum_acceleration,
        reproduced_objectives=len(errors),maximum_relative_cost_error=max(errors),
        policy='Three-sigma margins are conditional thermal diagnostics, not calibrated physical confidence or a multiple-comparison-corrected hypothesis test. The bounds apply to true simultaneous derivatives, not automatically to differently weighted quadratic approximations of a nonquadratic path. Model adequacy, convergence, support and identifiability must hold before interpreting coefficient contrasts geometrically. A common receiver cancels from true interband path differences. Overlapping windows must not be multiplied as independent likelihood terms.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source,Path(__file__),Path('support/isar/relative_delay_geometry.py'),Path('support/isar/phase_window.py'))})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    table=''
    for w in windows:
        for p in w['points']:
            table+=f'<tr><td>{w["interval_s"]}</td><td>{p["name"]}</td><td>{p["necessary_bound_violated"]}</td><td>{p["all_conditional_covariances_identifiable"]}</td><td>{100*p["minimum_projected_signal_energy"]:.1f}%</td><td>{p["any_component_bounds"]}</td></tr>'
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Short-window phase geometry feasibility</title>
<style>body{{font:16px/1.55 system-ui;max-width:1550px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{border:1px solid #ddd;padding:7px}}table{{border-collapse:collapse}}.note{{background:#eef4f8;padding:16px}}</style>
<h1>Short-window phase trajectories and receiver-independent geometry</h1>
<p class="note"><b>Interpretation limit:</b> the geometric inequalities apply to true simultaneous derivatives. A short quadratic fit is only a conditional approximation; different brightness histories can give different polynomial coefficients even for a shared nonquadratic trajectory. Several fits below also have weak static separation or do not converge. This diagnostic therefore does not establish that antenna refinement is impossible, or that scatterers differ by band. <a href="../flexible-flash-path/fits.json">The shared flexible-path control</a> directly tests temporal model mismatch on the original complex observations.</p>
<p class="note">The four-second flash averages over changing dominant returns. These seven overlapping one-second controls test whether shorter pieces have consistent material-point phases. Each channel and return is fitted independently. No transmitter refinement, receiver adjustment, multipath model, free phase screen or shared phase curvature is used to make them agree.</p>
<img src="01-rate-acceleration-feasibility.png" alt="All selected returns' interband path-rate and acceleration differences in seven one-second windows">
<p>For a common point, D(t)=|x(t)−Tₐ|−|x(t)−Tᵦ|. The receiver cancels exactly. Integrating the distance gradient and Hessian between transmitters gives |Ḋ|≤vB/r and |D̈|≤aB/r+3v²B/r². The code preserves correlated shared-site survey offsets through Fₐ−Fᵦ, then takes the conservative union over all 32 licensed primary/backup assignments. The declared 500 m scene, 8 m/s speed and 1 m/s² point acceleration bounds are generous ship hypotheses, not measurements.</p>
<p>A violation beyond the conditional thermal margin warns that the selected feature/model cannot be interpreted as one material point under those hypotheses. Passing is only a necessary condition; it does not determine antenna coordinates. Non-convergence, parameter bounds and static ambiguity are separate checks below. A near-static fit can have a large conditional amplitude that opposes the static profile and must not be called a precise moving scatterer.</p>
<table><tr><th>Interval (s)</th><th>Return</th><th>Necessary geometric bound violated beyond conditional 3σ?</th><th>Conditional covariance identified?</th><th>Minimum signal energy independent of static profile</th><th>Component bound contact?</th></tr>{table}</table>
<p>All {len(errors)} original-data start objectives reproduce (maximum relative discrepancy {max(errors):.3g}). The row epochs, information weights and original C0/C1 operator are retained. An exact reparameterization test proves that changing the local polynomial midpoint also changes its envelope and carrier intercept consistently. Overlapping windows are for association selection; later physical fitting must use disjoint original observations.</p>
<p><a href="analysis.json">Every measurement, covariance, convergence diagnostic and all 32 bound calculations</a> · <a href="tests.log">Numerical checks</a> · <a href="../multiple-scatterer-phase-plots/index.html">Measured multiple-return phase plots</a> · <a href="../index.html">Analysis archive</a></p></html>''')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
