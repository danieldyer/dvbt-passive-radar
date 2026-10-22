"""Review shared-antenna, separate-interval direct and HH sea-path fits."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES
from fit_joint_sea_arcs import prepare_problem, build_arcs
from joint_physical_arcs import JointPhysicalArcs
from spherical_sea import reflected_leg, horizontal_reflection
from terrain import RasterTerrain, geodetic


def heldout(record):
    return sum(c['test_error'] for p in record['prediction'] for c in p['channels'])


def footprints(case, record):
    """Check fitted specular centres, not just those at the nominal sites."""
    directory=Path('build/calibration-20261013')
    prior=json.loads((case['geometry_directory']/'geometry-prior.json').read_text())
    terrain=RasterTerrain(directory,prior,directory/'ga-footprint-dem.tif')
    centre=np.asarray(case['water']['centre_ecef_m']);radius=case['water']['radius_m']
    result=[]
    for description,g,prediction in zip(case['descriptions'],case['geometries'],record['prediction']):
        p,v,a,tx,rx=g.state(np.asarray(prediction['parameters'])[:-1])
        t=np.linspace(-1.5,1.5,31)
        position=g.receiver+(p+t[:,None]*v+.5*t[:,None]**2*a)@g.rotation
        velocity=(v+t[:,None]*a)@g.rotation
        for ch,antenna in enumerate(tx):
            direct=np.linalg.norm(position-(g.receiver+antenna@g.rotation),axis=1)
            ray=reflected_leg(position,velocity,g.receiver+antenna@g.rotation,centre,radius)
            lat,lon,_=geodetic(ray['specular_ecef_m']);ground=terrain.elevation(lat,lon)
            frequency=case['fields'][0][ch].metadata['rf_hz']
            gamma=horizontal_reflection(ray['sin_grazing'],frequency,relative_permittivity=80.,conductivity_s_per_m=4.)
            gain=gamma*ray['divergence']*direct/ray['length_m']
            result.append(dict(label=description['label'],channel=ch,latitude_deg=lat.tolist(),longitude_deg=lon.tolist(),
                dem_height_m=ground.tolist(),extra_path_m=(ray['length_m']-direct).tolist(),
                relative_reflected_power_db=(20*np.log10(abs(gain))).tolist(),
                fresnel_phase_deg=np.degrees(np.angle(gamma)).tolist(),
                scope='Specular centres only. A water centre does not certify an entirely water Fresnel footprint or a clear ray.'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/joint-ship-sea-review'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=args.output.parent
    paths=[root/'joint-ship-sea-arcs/fits.json',root/'joint-ship-sea-reference-phase/fits.json']
    inputs=[json.loads(p.read_text()) for p in paths]
    if not all(d['complete'] for d in inputs):raise ValueError('Both declared comparisons must finish')
    test_log=(args.output/'tests.log').read_text();tests=re.search(r'Ran (\d+) tests',test_log)
    if not tests or not test_log.rstrip().endswith('OK'):raise ValueError('Passing numerical test log required')
    case=prepare_problem();records=[]
    summary=dict(scope=__doc__,accepted_calibration=False,new_focused_videos=False,
        tests_passed=int(tests[1]),assignment='00002',intervals=case['descriptions'],trials=[],
        information_scope='Mean-response information in prior-standardised antenna coordinates, with no location priors or active bounds included. The scalar-noise model is conditional; this is not an unconditional CRLB in unmodelled interference.',
        numerical_provenance='The first fit process used the archived dense profiled residual derivative; the referenced-phase process used its algebraically equivalent adjoint. The input source hashes were taken at completion, not process launch. Every recorded objective is independently rebuilt here from the current, tested likelihood. The common phase coordinate is taken explicitly from each manifest; missing means physical common phase.',
        comparison_scope='Three disjoint but nearby intervals; they are not three widely separated viewing geometries. Only assignment 00002 and two initial carrier branches, not an exhaustive site or integer search.')
    for path,data in zip(paths,inputs):
        reference=data.get('common_phase_reference_band')
        for record in data['trials']:
            arcs=build_arcs(case,record['family'],reference)
            joint=JointPhysicalArcs(arcs);parameters=np.asarray(record['parameters'])
            rebuilt=joint.objective(parameters)[0]*joint.energy
            if abs(rebuilt-record['cost'])>1e-7*record['cost']:raise ValueError('Saved likelihood did not reproduce')
            information=joint.information(parameters)
            label=f"{'Physical phase' if reference is None else 'Midpoint phase'} / {record['family']} / start {record['seed_cycle']}"
            item=dict(label=label,source=str(path),family=record['family'],seed_cycle=record['seed_cycle'],
                common_phase_reference_band=reference,cost=record['cost'],rebuilt_cost=rebuilt,
                heldout_error=heldout(record),converged=record['audit']['converged'],
                stationarity=record['audit']['polishing']['final_kkt_residual'],
                bounds=record['audit']['bound_indices'],antenna_fit=record['antenna_fit'],height_extrema=record['height_extrema'],
                information=information)
            summary['trials'].append(item);records.append(record)
            print(label,'cost',round(rebuilt,4),'information',np.round(information['eigenvalues'],6),flush=True)
    rows=summary['trials'];labels=[r['label'] for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    for i,row in enumerate(rows):
        style=dict(marker='o' if row['converged'] else 'x',color='#24699b' if row['family']=='direct' else '#bd6b22')
        axes[0].plot(row['cost'],i,**style);axes[1].plot(row['heldout_error'],i,**style)
        axes[2].semilogx(row['stationarity'],i,**style)
    for ax in axes:ax.invert_yaxis();ax.set_yticks(range(len(rows)))
    axes[0].set_yticklabels(labels,fontsize=8)
    for ax in axes[1:]:ax.set_yticklabels([])
    axes[0].set(xlabel='Training squared error + declared priors',title='Same data and physical bounds')
    axes[1].set(xlabel='Held-out squared prediction error',title='Unused blocks; not an SNR estimate')
    axes[2].axvline(1e-6,color='#777',ls='--');axes[2].set(xlabel='Scaled KKT residual',title='Circle: converged; cross: incomplete')
    fig.suptitle('Joint antennas across Flash (163–166 s), F01 (159–162 s), F03 (151–154 s)\nBlue: direct only · orange: direct + transmit-side HH sea reflection',fontsize=12)
    fig.savefig(args.output/'01-fit-comparison.png',dpi=170);plt.close(fig)
    retained=[(r,d) for r,d in zip(rows,records) if r['converged']]
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for row,record in retained:
        offsets=np.r_[record['antenna_fit']['transmitter_offsets_enu_m'],[record['antenna_fit']['receiver_offset_enu_m']]]
        for j,ax in enumerate(axes):ax.plot(range(6),offsets[:,j],'o-',label=f"{row['family']} / start {row['seed_cycle']}")
    for ax,label in zip(axes,['East','North','Up']):
        ax.axhline(0,color='#aaa',lw=.7);ax.set_xticks(range(6),[*NAMES,'RX'],rotation=40)
        ax.set(ylabel=f'{label} displacement from user prior (m)',title=label)
    axes[0].legend(fontsize=9)
    fig.suptitle('Numerical convergence does not make these physically credible calibrations\nAll three minima contact antenna and motion bounds; the receiver is pushed to ground level',fontsize=12)
    fig.savefig(args.output/'02-location-displacements.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for row,_ in retained:ax.semilogy(np.arange(1,17),np.maximum(row['information']['eigenvalues'],1e-20),'o-',label=f"{row['family']} / start {row['seed_cycle']}")
    ax.axhline(1,color='#777',ls='--',label='Unit information in prior-standardised coordinates')
    ax.set(xlabel='Antenna mode (largest first)',ylabel='Conditional data information eigenvalue',xticks=np.arange(1,17),
        title='All target position, velocity, acceleration and phase directions removed\nColumn-scaled projection; no antenna priors or bound constraints counted as data')
    ax.legend();fig.savefig(args.output/'03-data-information.png',dpi=170);plt.close(fig)
    best={family:min([(r,d) for r,d in retained if r['family']==family],key=lambda x:x[0]['cost']) for family in ('direct','transmit')}
    direct,sea=[best[f][1] for f in ('direct','transmit')]
    comparison=[];fractions=np.empty((3,5));fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    delta=np.empty((3,5))
    for i,(first,second) in enumerate(zip(direct['prediction'],sea['prediction'])):
        for ch,(a,b) in enumerate(zip(first['channels'],second['channels'])):
            delta[i,ch]=100*(a['test_error']-b['test_error'])/a['test_error']
            fractions[i,ch]=100*b['test_complex_prediction_fraction']
            comparison.append(dict(label=first['label'],channel=ch,error_reduction_percent=delta[i,ch],sea_explained_energy_percent=fractions[i,ch]))
    for ax,values,title,cmap,limit in zip(axes,[delta,fractions],['Sea-model reduction in held-out error (%)','Sea model: explained held-out residual energy (%)'],['RdBu','viridis'],[max(abs(delta).max(),1),100.]):
        im=ax.imshow(values,cmap=cmap,vmin=-limit if cmap=='RdBu' else 0,vmax=limit,aspect='auto')
        for i in range(3):
            for ch in range(5):ax.text(ch,i,f'{values[i,ch]:+.2f}' if cmap=='RdBu' else f'{values[i,ch]:.1f}',ha='center',va='center',color='white' if (abs(values[i,ch])>limit*.6 if cmap=='RdBu' else values[i,ch]<50) else 'black')
        ax.set_xticks(range(5),NAMES);ax.set_yticks(range(3),[x['label'] for x in case['descriptions']]);ax.set_title(title,fontsize=11)
        fig.colorbar(im,ax=ax)
    fig.suptitle('Best converged direct and sea fits · identical held-out samples\nThese fractions measure prediction of residual data, not thermal SNR or imaging gain',fontsize=12)
    fig.savefig(args.output/'04-interval-prediction.png',dpi=170);plt.close(fig)
    summary['paired_prediction']=comparison
    summary['heldout_error_reduction_percent']=100*(heldout(direct)-heldout(sea))/heldout(direct)
    fig,axes=plt.subplots(5,3,figsize=(15,12),layout='constrained',sharey=True)
    for i,(first,second) in enumerate(zip(direct['prediction'],sea['prediction'])):
        for ch,(a,b) in enumerate(zip(first['channels'],second['channels'])):
            for values,colour,label in [(a,'#24699b','Direct'),(b,'#bd6b22','Direct + TX sea')]:
                use=np.asarray(values['testing'],bool);t=np.asarray(values['time_s'])[use]
                axes[ch,i].scatter(t,np.asarray(values['phase_residual_deg'])[use],s=5,alpha=.7,color=colour,label=label)
            axes[ch,i].axhline(0,color='#aaa',lw=.5);axes[ch,i].set_ylim(-180,180)
            if ch==0:axes[ch,i].set_title(first['label'])
            if i==0:axes[ch,i].set_ylabel(f'{NAMES[ch]}\nResidual phase (deg)')
            if ch==4:axes[ch,i].set_xlabel('Capture time (s)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Matched residual phase on unused samples, with training-fitted stationary terms removed\nLow-brightness samples are shown too; phase is not meaningful where target amplitude vanishes',fontsize=12)
    fig.savefig(args.output/'05-phase-prediction.png',dpi=170);plt.close(fig)
    summary['fitted_tx_sea_paths']=footprints(case,sea)
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for i,description in enumerate(case['descriptions']):
        data=[r for r in summary['fitted_tx_sea_paths'] if r['label']==description['label']]
        axes[0].plot(range(5),[np.mean(r['extra_path_m']) for r in data],'o-',label=description['label'])
        axes[1].plot(range(5),[np.mean(r['relative_reflected_power_db']) for r in data],'o-')
        axes[2].plot(range(5),[max(r['dem_height_m']) for r in data],'o-')
    for ax in axes:ax.set_xticks(range(5),NAMES,rotation=30)
    axes[0].set(ylabel='Additional transmit-leg path (m)',title='Sub-resolution multipath');axes[0].legend()
    axes[1].set(ylabel='Reflected / direct ray power (dB)',title='HH Fresnel × divergence × spreading')
    axes[2].set(ylabel='DEM elevation at specular centre (m)',title='Centre-only shoreline check')
    fig.suptitle('Fitted sea-model geometry · scalar point response held equal between rays\nClear paths and an entirely water Fresnel footprint are not certified by this DEM check',fontsize=12)
    fig.savefig(args.output/'06-fitted-sea-paths.png',dpi=170);plt.close(fig)
    table=''.join(f'<tr><td>{html.escape(r["label"])}</td><td>{r["cost"]:.3f}</td><td>{r["heldout_error"]:.3f}</td><td>{"yes" if r["converged"] else "no"}</td><td>{r["stationarity"]:.3g}</td><td>{len(r["bounds"])}</td></tr>' for r in rows)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Shared ship calibration and HH sea reflection</title>
<style>body{{font:16px/1.55 system-ui;max-width:1220px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border:1px solid #ccd4d9;text-align:left}}.note{{background:#fff1d3;padding:16px}}</style>
<h1>Shared ship calibration and horizontal-polarisation sea paths</h1>
<p class="note"><b>No calibration accepted.</b> The transmit-side sea model reaches two strict local minima and reduces held-out squared error by {summary['heldout_error_reduction_percent']:.2f}% relative to the best converged direct fit. All three converged solutions push antenna and motion parameters against bounds. In particular, the receiver is driven from 10 m above ground to ground level, and ABC is displaced upward by approximately 227 m. These are model-failure diagnostics, not credible updated antenna coordinates. The direct start-8 search is incomplete despite its lower cost; it remains visible.</p>
<h2>What horizontal polarisation implies</h2>
<p>Yes: a smooth sea can provide a strong additional illumination or reception path. At shallow grazing angles its horizontal-polarisation Fresnel reflection coefficient is close to −1, so a reflected ray can be comparable in amplitude to a direct ray, with nearly 180° reflection phase. For a directional hull or superstructure scatterer, a sea-assisted path can also couple more strongly than the unreflected one. The total depends on coherent addition, not a power sum. Strong cancellation and reinforcement are both possible.</p>
<p>This is the four-path mechanism used in <a href="https://www.jpier.org/ac_api/download.php?id=11101108">Chen, Zhang and Yin (2012)</a>. Their microwave simulations support the mechanism, not numerical predictions for this VHF capture. Here the coefficients and curved-water path lengths are calculated at each actual RF frequency. A sea-only reflection is a different hypothesis: under the nominal geometry its smooth-surface specular point lies inland, away from the ship; see the <a href="../../isar_20261016/flash-reflection/index.html">geometry and path-scale analysis</a>.</p>
<p>The added ship–sea path is typically less than a metre for a 20 m-high scatterer, far below the 37.47 m stored range sample spacing, yet large enough to change carrier phase markedly at VHF. It can therefore distort apparent phase and band-to-band coherence without producing a separately resolved range echo. This model does not explain a distant companion return solely by that short extra path.</p>
<h2>A controlled comparison across three intervals</h2>
<p>One antenna geometry is shared across the disjoint 151–154, 159–162 and 163–166 s intervals. Each has an AIS-anchored Cartesian position, velocity and acceleration, one common physical scattering phase, and independent nonnegative brightness histories per band. Antenna priors are counted once. The direct and sea models use identical observations, training/testing masks and physical bounds. They retain the original per-symbol response, range migration, Doppler response, averaging and whitening; reflected rays are summed before those operations.</p>
<p>Assignment 00002 places Seven/SBS/Nine/Ten at Ornata and ABC at Ridge. This is a two-start conditional experiment, not a new exhaustive 32-site or integer-ambiguity solution. The three intervals are close in time and cannot be treated as widely separated viewing geometries. The sea model assumes smooth water, relative permittivity 80 and conductivity 4 S/m, and the same scalar point-scattering coefficient toward direct and reflected rays. Shoreline truncation, ray blockage, directional facet response and actual sea roughness remain outside that hypothesis.</p>
<img src="01-fit-comparison.png" alt="All direct and sea-model starts, training and withheld errors, convergence">
<table><tr><th>Trial</th><th>Training objective</th><th>Held-out error</th><th>Converged</th><th>KKT residual</th><th>Bound contacts</th></tr>{table}</table>
<h2>Locations and information content</h2>
<img src="02-location-displacements.png" alt="Antenna east north up offsets">
<img src="03-data-information.png" alt="Data-only antenna information after all target nuisance directions">
<p>The information calculation removes the explicit common phases first, then all nine Cartesian position/velocity/acceleration nuisance directions per interval. AIS time shift is locally redundant with those nine state coordinates. Column scaling and a machine-precision rank criterion make the result invariant to the phase coordinate used by the optimiser. An earlier relative singular-value threshold could drop small motion directions when carrier phase dominated the scale; that diagnostic has been corrected and tested. Bounds and priors are excluded from the plotted data information. It is a local, conditional thermal-noise calculation, not proof of a globally unique solution or a model-mismatch-robust CRLB.</p>
<h2>Prediction on samples not used by the fit</h2>
<img src="04-interval-prediction.png" alt="Held-out prediction changes by interval and band">
<img src="05-phase-prediction.png" alt="Direct and sea-model phase residuals over time">
<p>The sea improvement is modest and does not remove the implausible antenna shifts. These blocks were withheld during each fit, but the recording has already been inspected; this is conditional predictive validation, not an untouched external test. No error reduction here is being called an SNR or coherent-imaging gain.</p>
<h2>Check the fitted reflection geometry</h2>
<img src="06-fitted-sea-paths.png" alt="Extra path, relative reflected power, and terrain elevation at fitted specular points">
<p>The reflection-centre check uses the approved one-arcsecond SRTM-derived DEM, not native DTED Level 2. Zero DEM height at the specular centre is necessary but does not certify the entire Fresnel footprint, propagation clearance, tide or sea state. The receive-side path is deliberately not used as the principal comparison because its reflection footprint can meet the shoreline.</p>
<h2>Numerical validation and reproducibility</h2>
<p>{summary['tests_passed']} tests passed. New checks cover moving spherical specular points, HH complex reflection and divergence derivatives; the direct limit of the sea operator; shared-prior counting; the exact original-row profiled-residual Jacobian; and invariance of both complex predictions and antenna information under a common-phase coordinate change. Conditioning the common phase on Ten's midpoint changes no physical freedom and enabled three strict local minima. The equivalent adjoint implementation reproduced the dense Jacobian to 7.6×10⁻¹⁶ relative error and reduced that measured residual/Jacobian kernel's runtime by 1.57×; this is an offline solver measurement, not a production speed claim.</p>
<p>Every stored objective was rebuilt by this report. The old process's executed dense solver is archived alongside its output; its completion-time source hashes alone are not a launch snapshot. See <a href="analysis.json">numerical results and provenance</a>, <a href="tests.log">test log</a>, <a href="../joint-ship-sea-arcs/adjoint-timing.json">adjoint equivalence/timing</a>, and <a href="../ship-phase-transfer-review/index.html">the preceding transfer controls</a>. Production processing and accepted recording metadata are unchanged.</p>
</html>'''
    (args.output/'index.html').write_text(page)
    sources=[*paths,Path(__file__),Path('support/isar/fit_joint_sea_arcs.py'),Path('support/isar/joint_physical_arcs.py'),
        Path('support/isar/profiled_arc_solver.py'),Path('support/isar/referenced_common_arc.py'),Path('support/isar/differentiable_sea.py'),
        Path('support/isar/sea_physical_arc.py'),args.output/'tests.log']
    summary['sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sources}
    (args.output/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',args.output/'index.html',flush=True)


if __name__=='__main__':main()
