"""Measure multi-band phase candidates around actual AIS report epochs.

This adds no physical motion or antenna fit. Candidate cells are selected by
the existing training-only discovery scores; phases are measured directly,
with the original carrier deramp restored and no fitted slope or curvature.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from ais import read_csv
from antenna_motion import Antennas
from flash_geometry import assignments
from flash_siting_report import EPOCH
from relative_delay_geometry import relative_rate_bound
from ship_site_signature import licensed_options
from report_measured_scatterer_phases import (load_fields,analyse_window,plot_selection,
    plot_histories,plot_double_differences,plot_increments,serial)


def main():
    root=Path('docs/data/isar_20261017');out=root/'ais-epoch-phase-candidates';out.mkdir(exist_ok=True)
    geometry=Path('docs/data/calibration_20261016_sites')
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    ais_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(ais_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    options=licensed_options(geometry);positions=np.unique([o['transmitter_ecef_m'] for o in options],axis=0)
    rotation=np.asarray(json.loads((geometry/'geometry-prior.json').read_text())['ecef_to_enu'])
    phase_centre_radius=0.
    for assignment in assignments(options):
        factor=Antennas(rotation,[o['site_id'] for o in assignment]).factor[:15,:15]
        box=3*np.sum(abs(factor),axis=1).reshape(5,3)
        phase_centre_radius=max(phase_centre_radius,float(np.linalg.norm(box,axis=1).max()))
    score_path=root/'full-capture-ship-arcs/scores.npz'
    with np.load(score_path) as score:
        minimum=np.min(score['per_band'],axis=0)
        useful=np.min(score['stationary_retained_fraction'],axis=0)>=.2
        minimum=np.where(useful[None],minimum,-np.inf)
        windows=[];selection=[]
        for number,utc in enumerate(track.seconds[[2,3,6]],1):
            epoch=float(utc-EPOCH);near=np.flatnonzero((score['seconds']<=epoch)&(score['seconds']+3>epoch))
            available=np.ones(minimum[:,near].shape,bool);choices=[];proposals=[]
            for label in ('B','C'):
                o,n,r=np.unravel_index(np.argmax(np.where(available,minimum[:,near],-np.inf)),available.shape)
                if not np.isfinite(minimum[o,near[n],r]):raise ValueError('Missing available discovery candidate')
                origin=int(score['origins'][o]);choices.append((label,origin))
                proposals.append(dict(label=label,origin_samples=origin,
                    training_minimum_band_score=minimum[o,near[n],r],discovery_first_s=score['seconds'][near[n]],
                    discovery_rate_mps=score['rates'][r]))
                available[abs(score['origins']-origin)<3]=False
            first,last=epoch-3,epoch+3;label=f'AIS{number}'
            w=analyse_window(fields,label,first,last,selections=choices)
            w['ais_report_epoch_s']=epoch;windows.append(w)
            selection.append(dict(label=label,ais_report_epoch_s=epoch,ais_report_utc_s=utc,candidates=proposals))
            print(label,round(epoch,3),'origins',choices,flush=True)
    plot_selection(windows,fields,out)
    for w in windows:plot_histories(w,out)
    differences=plot_double_differences(windows,out);increments=plot_increments(windows,out)
    table='';epoch_support=[];rate_checks=[]
    for w in windows:
        grid=np.asarray(w['time_s']);j=int(np.argmin(abs(grid-w['ais_report_epoch_s'])))
        centre=track.sample([EPOCH+w['ais_report_epoch_s']],max_gap_s=500.)['position_ecef_m'][0]
        bound=relative_rate_bound(centre,positions,scene_radius_m=500.,maximum_point_speed_mps=8.,phase_centre_radius_m=phase_centre_radius)
        for p in w['points']:
            flags=[];strength=[];rates=[]
            for ch in p['channels']:
                values=np.asarray(ch['grid_measured_path_m']);flags.append(bool(np.isfinite(values[j])))
                time=np.asarray(ch['effective_time_s']);i=int(np.argmin(abs(time-w['ais_report_epoch_s'])))
                strength.append(float(ch['power_to_noise'][i]))
                epoch=np.asarray(ch['effective_time_s']);dt=epoch[5:]-epoch[:-5]
                rate=np.asarray(ch['measured_increment_m'])/dt
                use=np.asarray(ch['increment_good'])&np.isfinite(rate)
                rates.append(float(np.median(rate[use])) if use.any() else None)
            rate_checks.append(dict(window=w['label'],point=p['name'],absolute_rate_median_mps=rates,
                geometric_point_rate_difference_bound=bound))
            epoch_support.append(dict(window=w['label'],point=p['name'],phase_connected_at_nominal_ais_epoch=flags,
                conditional_power_to_noise_at_epoch=strength))
            table+=f'<tr><td>{w["label"]}</td><td>{w["ais_report_epoch_s"]:.3f}</td><td>{p["name"]}</td><td>{p["offset_m"]:+.1f}</td><td>{flags}</td><td>'+', '.join(f'{10*np.log10(max(v,1e-300)):.1f}' for v in strength)+'</td></tr>'
        w.pop('maps')
    result=dict(scope=__doc__,accepted_calibration=False,production_changed=False,
        selection=selection,windows=windows,epoch_support=epoch_support,double_differences=differences,
        local_increment_comparisons=increments,
        rate_geometry_checks=rate_checks,
        rate_geometry_policy='Descriptive local-rate medians use only finite good endpoints and retain the local carrier-cycle ambiguity. They are not physical velocity estimates or confidence intervals. The analytic one-point bound uses an explicit speed limit of 8 m/s, a 500 m scene radius, every licensed site and an enclosing ball for all current standardized antenna-coordinate boxes. A common receiver contribution cancels exactly. Large measured differences warn against identifying mixed cells as one material point; they are not a calibrated statistical rejection.',
        policy='Discovery uses existing training-only minimum-band scores in three-second intervals containing the nominal AIS report epoch, at least 20% moving-template energy after its acquisition-only static-tone projection. The new measurements themselves apply no stationary projection. B is the strongest score and C the strongest range-separated alternative (at least three samples). Labels do not assert material points or identity across windows.',
        timing_policy='AIS report epochs are plotted in nominal capture time, before any fitted AIS delay. Flash time-shift estimates are not transferred. AIS interpolation is not assumed carrier-phase accurate; a connected phase segment at a report epoch is only a candidate for a later physical fit.',
        phase_policy='Original complex cells, 0.5 s measurements every 0.05 s, per-channel wavelength and original carrier-deramp restoration. Only constant origins removed; no slope/curvature fit or phase bridge through fades.',
        source_sha256={str(f['source']):f['source_sha256'] for f in fields})
    files=[Path(__file__),ais_path,score_path,Path('support/isar/report_measured_scatterer_phases.py'),
        Path('support/isar/measured_phase.py'),Path('support/isar/relative_delay_geometry.py'),
        *[(geometry/n) for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')]]
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for p in files:
        with p.open('rb') as stream:result['source_sha256'][str(p)]=hashlib.file_digest(stream,'sha256').hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshots/p.name)
    (out/'measurements.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    sections=''.join(f'<h2>{w["label"]}: AIS report at {w["ais_report_epoch_s"]:.3f} s</h2><a href="02-{w["label"]}-trajectories.png"><img src="02-{w["label"]}-trajectories.png" alt="Measured phases around an actual AIS report"></a>' for w in windows)
    rate_rows=''.join(f'<tr><td>{r["window"]}</td><td>{r["point"]}</td><td>'+', '.join('undefined' if v is None else f'{v:+.3f}' for v in r['absolute_rate_median_mps'])+f'</td><td>{r["geometric_point_rate_difference_bound"]["maximum_rate_difference_mps"]:.3f}</td></tr>' for r in rate_checks)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Measured phases near AIS report times</title>
<style>body{{font:16px/1.55 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:8px;border:1px solid #ddd}}table{{border-collapse:collapse}}.note{{padding:16px;background:#eef4f8}}</style>
<h1>Additional phase candidates near actual AIS reports</h1>
<p class="note">The flash occurs between AIS position reports. These additional six-second windows surround actual reports at 139.892, 195.892 and 334.892 s. They provide more widely separated candidates for testing physical geometry; no position fit is performed or accepted here.</p>
<p>B/C are local feature labels, not persistent point identities. Cells were selected using the existing training-only discovery scores in intervals containing each report. C is the strongest range-separated alternative, including weak alternatives so their phase failures remain visible. The measurements below add no stationary fit, notch, physical trajectory, fitted phase rate or curvature.</p>
<a href="01-selected-points.png"><img src="01-selected-points.png" alt="Selected cells in every channel near AIS position reports"></a>
{sections}
<p>Dashed curves are nominal AIS/geometry comparisons only. They do not align measured slopes. Black dash-dot vertical lines mark the nominal AIS report epoch; thin dotted lines mark each magnitude-selected constant phase origin. Phase lines stop at fades; connected support at the nominal report time is listed below. A global AIS time shift remains a nuisance to estimate; the ambiguous flash time shift has not been applied to these windows.</p>
<table><tr><th>Window</th><th>Nominal AIS report time (s)</th><th>Feature</th><th>Range offset (m)</th><th>Connected at report: Seven/SBS/Nine/Ten/ABC</th><th>Conditional power/noise at report (dB)</th></tr>{table}</table>
<h2>Do the selected cells represent the same moving component?</h2>
<p>At the third epoch, main-cell ABC has a nearly stationary restored path while several other channels follow the moving return. At the second epoch, the alternative cell has near-zero path rate in Ten and ABC. Being phase-connected is therefore insufficient: a cell can be dominated by stationary energy or a mixture. These candidates need component separation before physical calibration.</p>
<p>For one material point, the receiver leg cancels in an interband path-rate difference. Integrating the transmitter unit-direction derivative gives |ΔL̇| ≤ v_max B/r_min. The broad bound below permits all current antenna boxes, points within 500 m of AIS and a point speed up to 8 m/s. The measured medians are descriptive, correlated and carrier-cycle ambiguous; the comparison warns about association errors and is not a statistical rejection test.</p>
<table><tr><th>Window</th><th>Feature</th><th>Measured local-rate medians: Seven/SBS/Nine/Ten/ABC (m/s)</th><th>Maximum geometric interband difference under the declared hypotheses (m/s)</th></tr>{rate_rows}</table>
<a href="03-between-points.png"><img src="03-between-points.png" alt="Between-point phase-change differences across all five channels"></a>
<p><a href="04-local-increments.png">Local phase increments without long phase connections</a> · <a href="measurements.json">All complex measurements and discovery selections</a> · <a href="../physical-flash-review/index.html">Physical flash controls</a> · <a href="../physical-four-echo-tests.log">Numerical tests</a> · <a href="../index.html">Analysis index</a></p></html>''')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
