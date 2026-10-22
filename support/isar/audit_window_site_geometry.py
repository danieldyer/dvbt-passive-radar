"""Test all licensed sites against a completed independent fixed-grid point fit.

This is a conditional geometry/association audit, not an accepted calibration.
Joint range/rate/curvature covariance comes from the original complex-data
likelihood after brightness and constant phases. Geometry priors stay separate.
Each hypothesis is fit from two starts, with fixed and free antennas. No fitted
phase slope, per-band range offset or reflected propagation is introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from compare_ais_phase import NAMES,COLOURS
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH
from report_fixed_grid_phases import reload_bands,phase_statistics
from report_measured_scatterer_phases import load_fields,serial
from report_shared_echo_cluster import conditional_observables
from shared_echo_cluster import SharedEchoCluster
from ship_site_signature import licensed_options


def code(assignment):
    return ''.join({'12013':'0','12015':'1','12014':'2'}[v['site_id']] for v in assignment)


def physical_observables(band,conditional):
    """Restore the exact fixed observer and known interband increment."""
    rows=band.field.data['row_time'];time=rows-band.epoch
    coefficient=np.polynomial.polynomial.polyfit(time,band.known_path,5)
    maximum_error=float(np.max(abs(np.polynomial.polynomial.polyval(time,coefficient)-band.known_path)))
    if maximum_error>1e-5:raise ValueError('Known interband path is not represented to 10 micrometres')
    increment=np.asarray(conditional['range_origin_m_rate_residual_mps_curvature_residual_mps2'])[0]
    physical=increment+np.array([band.field.data['centre_path_m'][0]+coefficient[0],coefficient[1],2*coefficient[2]])
    return physical,maximum_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    if out.exists():raise ValueError('Use a new audit directory')
    saved=json.loads(args.source.read_text())
    if not saved['complete']:raise ValueError('Wait for every original-data start')
    group=next(g for g in saved['groups'] if g['mode']=='stationary')
    geometry=Path('docs/data/calibration_20261016_sites');geo=json.loads((geometry/'geometry-prior.json').read_text())
    receiver=np.array(geo['receiver_ecef_m']);rotation=np.array(geo['ecef_to_enu'])
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    bands,parts,observations=reload_bands(args.source.parent,group,saved,fields,geometry)
    measured=[];covariance=[];statistics=[];reproduction=[];known_errors=[]
    for c,(band,entry) in enumerate(zip(bands,group['independent'])):
        model=SharedEchoCluster([band],[saved['selected_origin_samples']],origin_radius=1.5)
        for fit in entry['trials']:
            r,_=model.evaluate(np.asarray(fit['parameters']));reproduction.append(abs(r@r*model.energy-fit['cost']))
        best=entry['trials'][entry['best_index']];p=np.asarray(best['parameters'])
        conditional=conditional_observables(model,p)
        if not conditional['identifiable']:raise ValueError('Independent point information is singular')
        value,error=physical_observables(band,conditional)
        measured.append(value);known_errors.append(error);covariance.append(conditional['covariance'])
        statistics.append(phase_statistics(model,p,[parts[c]])[0])
    if max(reproduction)>1e-7:raise ValueError('Original observation likelihood changed')
    measured=np.array(measured);covariance=np.array(covariance)
    track_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(track_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    mid=float(np.mean(saved['interval_s']));trials=[]
    for assignment in assignments(licensed_options(geometry)):
        for free in (False,True):
            g=Flash(measured,covariance,track,EPOCH+mid,rotation,receiver,assignment,
                free_antennas=free,receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.) if free else None)
            starts=[np.zeros(g.count),np.zeros(g.count)]
            starts[1][g.nantenna:]=[.6,-.6,.1,.3,-.3,.1,.1,-.1,.1,.3]
            results=[]
            for start,initial in enumerate(starts):
                fit=g.fit(initial);record=g.describe(fit);record['start']=start;results.append(record)
            chosen=int(np.argmin([r['penalized_squared_residual'] for r in results]))
            trials.append(dict(assignment=code(assignment),free_antennas=free,trials=results,best_index=chosen))
        fixed,refined=trials[-2:]
        print(code(assignment),'fixed/refined costs',*[round(v['trials'][v['best_index']]['penalized_squared_residual'],3) for v in (fixed,refined)],flush=True)
    out.mkdir(parents=True)
    rankings={str(free):sorted([r for r in trials if r['free_antennas']==free],
        key=lambda r:r['trials'][r['best_index']]['penalized_squared_residual']) for free in (False,True)}
    winner=rankings['True'][0];best=winner['trials'][winner['best_index']]
    predicted=np.array(best['predicted_observables']);sd=np.sqrt(np.diagonal(covariance,axis1=1,axis2=2))
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    for i,label in enumerate(('Excess path (m)','Path rate (m/s)','Path acceleration (m/s²)')):
        axes[i].errorbar(np.arange(5),measured[:,i],yerr=sd[:,i],fmt='o',color='#0072b2',capsize=4,label='Independent measured point')
        axes[i].plot(np.arange(5),predicted[:,i],'s--',color='#d55e00',label='Best conditional site/point fit')
        axes[i].set_xticks(np.arange(5),NAMES,rotation=25);axes[i].set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Original {saved['interval_s'][0]:g}–{saved['interval_s'][1]:g} s point summaries versus all-32-site geometry audit\n"
        f"Displayed candidate {winner['assignment']}; conditional thermal 1σ only, no residual-noise rescaling or per-band delay corrections.",fontsize=12)
    fig.savefig(out/'01-range-rate-geometry.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,6.4),layout='constrained')
    for free,ax in zip((False,True),axes):
        ranked=rankings[str(free)];labels=[r['assignment'] for r in ranked]
        vals=[r['trials'][r['best_index']] for r in ranked]
        ax.barh(np.arange(len(vals)),[r['data_squared_residual'] for r in vals],label='Measured summaries',color='#0072b2')
        ax.barh(np.arange(len(vals)),[r['prior_squared_residual'] for r in vals],left=[r['data_squared_residual'] for r in vals],label='Declared priors',color='#ccc')
        ax.set_yticks(np.arange(len(vals)),labels,fontsize=7);ax.invert_yaxis();ax.set(xlabel='Conditional standardized squared residual',title='Antennas '+('free' if free else 'fixed'))
    axes[0].legend(fontsize=8);fig.savefig(out/'02-site-rankings.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(19,4.6),layout='constrained',sharey=True)
    for c,(s,ax) in enumerate(zip(statistics,axes)):
        t=np.array(s['time_s']);unused=np.array(s['testing']);path=np.array(s['path_m'])
        ax.plot(t,path,color=COLOURS[c],label='Independent quadratic')
        ax.plot(t[unused],np.array(s['measured_innovation_path_m'])[unused],'.',ms=3,color=COLOURS[c],alpha=.5,label='Unused phase measurement')
        ax.set(title=NAMES[c],xlabel='Capture time (s)')
    axes[0].set_ylabel('Physical path change relative to midpoint (m)');axes[0].legend(fontsize=8)
    fig.suptitle('Later-window phase measurements on the independent fitted branch\nDots are circular innovations on unused original observations; fitted curves do not replace those data.',fontsize=12)
    fig.savefig(out/'03-independent-phase.png',dpi=150);plt.close(fig)
    source_files=[args.source,Path(__file__),track_path,*observations,*[geometry/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')],
        *[Path('support/isar')/(n+'.py') for n in ('flash_geometry','antenna_motion','report_shared_echo_cluster','report_fixed_grid_phases',
            'shared_echo_cluster','fixed_range_field','amplitude_pair','field')]]
    result=dict(scope=__doc__,complete=True,accepted_calibration=False,production_changed=False,
        interval_s=saved['interval_s'],measured_observables=measured,conditional_covariance=covariance,
        original_objectives_reproduced=len(reproduction),maximum_absolute_cost_error=max(reproduction),
        maximum_known_increment_representation_error_m=max(known_errors),trials=trials,
        best_refined_assignment=winner['assignment'],best_refined_result=best,
        phase_statistics=statistics,
        limitations='Summary fit uses a single quadratic moving point plus one static complex coefficient per observed range cell. '
            'Conditional thermal covariance omits other moving returns and interference. AIS position/velocity/acceleration corrections '
            'and time shift are explicit nuisance; horizontal receiver position is fixed at the user coordinate. '
            'Site and receiver priors do not constitute measured location information. The summary ranking must be checked '
            'against original complex-data physical predictions and point association before any coordinates are adopted.',
        references=[dict(title='Zhao et al. 2019: calibration with uncertain known moving targets',url='https://doi.org/10.1109/ACCESS.2019.2932748',
            use='Separate target, sensor and measurement uncertainty; combine range and range rate. This summary audit retains their joint covariance and adds phase curvature. It does not reproduce the paper algorithm.'),
            dict(title='Xiong et al. 2019: multiband ISAR coherence',url='https://doi.org/10.1049/iet-rsn.2018.5400',
            use='Dominant scatterer association is required before estimating interband coherence. No per-band slope compensation is adopted here.')],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    sources=out/'sources';sources.mkdir()
    for p in source_files:
        if p.suffix=='.py':shutil.copyfile(p,sources/p.name)
    rows=''.join(f'<tr><td>{r["assignment"]}</td><td>{r["free_antennas"]}</td><td>{r["trials"][r["best_index"]]["data_squared_residual"]:.2f}</td><td>{r["trials"][r["best_index"]]["prior_squared_residual"]:.2f}</td><td>{r["trials"][r["best_index"]]["bound_parameter_indices"]}</td></tr>' for r in [*rankings['False'],*rankings['True']])
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Later-window physical site audit</title><style>body{{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px}}</style>
<h1>Does the later point constrain a consistent physical geometry?</h1>
<p>This completes 32 licensed site assignments, each with fixed/free antennas and two starts: 128 conditional geometry fits. Every one of the 15 independent original-data starting objectives reproduces. The selected source window is {saved['interval_s'][0]:g}–{saved['interval_s'][1]:g} s. A common nominal phase trajectory changes total unused prediction error only slightly; the absolute range and joint rate/curvature measurements must also be consistent before they can calibrate locations.</p>
<p><b>No calibration is accepted from this ranking.</b> Best conditional refined assignment: {winner['assignment']}; measured standardized squared residual {best['data_squared_residual']:.2f}; prior penalty {best['prior_squared_residual']:.2f}; bound indices {best['bound_parameter_indices']}. Point/scene mismatch is outside the displayed thermal error bars. Receiver horizontal coordinates remain fixed at the approved position, and its height is allowed ±10 m around the approved 10 m AGL.</p>
<img src="01-range-rate-geometry.png" alt="Measured range, rate and curvature versus the best physical point model">
<img src="03-independent-phase.png" alt="Measured unused phase and independent fitted trajectories">
<img src="02-site-rankings.png" alt="All licensed site assignments; measured residual and prior penalty separately">
<p>Codes are ordered Seven/SBS/Nine/Ten/ABC: 0 Ornata, 1 Eyre, 2 Ridge. Transmitters on one site share the declared location prior, with separate smaller antenna offsets. Those priors are not data. Target position, velocity, acceleration and AIS time shift are nuisance parameters. No independent per-band range offset or phase slope is added to make the physical fit work.</p>
<table><tr><th>Sites</th><th>Antennas free</th><th>Measured residual</th><th>Prior penalty</th><th>Bound indices</th></tr>{rows}</table>
<p>Prior art motivates the uncertainty separation and dominant-point association tests: <a href="https://doi.org/10.1109/ACCESS.2019.2932748">Zhao et al., calibration with uncertain moving targets</a>; <a href="https://doi.org/10.1049/iet-rsn.2018.5400">Xiong et al., multiband ISAR coherence</a>. Neither paper validates this recording's point association or makes a constrained minimum a conclusive physical survey.</p>
<p><a href="analysis.json">All fits, covariances, phase measurements and source hashes</a> · <a href="../w4-fixed-point-control/fits.json">All original-data starts</a> · <a href="../index.html">Analysis index</a></p></html>''')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
