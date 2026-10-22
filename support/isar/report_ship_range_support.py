"""Check whether selected range separations can represent points on one hull.

Exact direct two-leg geometry only. Heading relative to AIS course is an
explicit hypothesis sweep, not a heading estimate. No multipath is modelled.
The AIS antenna location cancels from the within-hull range-span comparison.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from compare_ais_phase import NAMES
from flash_siting_report import EPOCH
from motion import C,body_basis,enu_basis,excess_path
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options


def main():
    root=Path('docs/data/isar_20261017');out=root/'selected-points-hull-support';out.mkdir(exist_ok=True)
    geometry=Path('docs/data/calibration_20261016_sites');prior_path=geometry/'geometry-prior.json'
    geo=json.loads(prior_path.read_text());rx=np.array(geo['receiver_ecef_m'])
    track_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(track_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    epoch=165.;state=track.sample(np.array([EPOCH+epoch]),max_gap_s=180.)
    lat,lon=state['latitude'][0],state['longitude'][0]
    local=enu_basis(lat,lon).T@state['velocity_ecef_mps'][0];course=float(np.arctan2(local[0],local[1]))
    vessel_path=Path('docs/data/isar_20261015/rio-delta-general-arrangement-source.json')
    vessel=json.loads(vessel_path.read_text());length=vessel['length_overall_m_in_report'];beam=vessel['beam_m_report_body']
    # 0–60 m is a declared generous vertical hull/superstructure support,
    # not an AIS altitude or a fitted reflector height.
    corners=np.array([[x,y,z] for x in (-length/2,length/2) for y in (-beam/2,beam/2) for z in (0.,60.)])
    headings=np.linspace(-90,90,721);basis=body_basis(np.full(len(headings),lat),np.full(len(headings),lon),course+np.radians(headings))
    displaced=state['position_ecef_m'][0]+np.einsum('nij,kj->nki',basis,corners)
    sites={o['site_id']:o for o in licensed_options(geometry)};spans={}
    for site,o in sites.items():
        path,_=excess_path(displaced,np.array(o['transmitter_ecef_m']),rx)
        spans[site]=np.ptp(path,axis=1)
    source=root/'longer-known-phases-6s/fits.json';data=json.loads(source.read_text())
    group=next(g for g in data['groups'] if g['jerk_std']==data['validation_selected_jerk_std'])
    measured=[]
    for entry in group['independent']:
        p=np.array(entry['trials'][entry['best_index']]['parameters'])
        measured.append([abs(p[2*s]-p[2])*C/8e6 for s in (0,2,3)])
    measured=np.array(measured)
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    for site,name in (('12013','Ornata'),('12015','Eyre'),('12014','Ridge')):
        ax.plot(headings,spans[site],label=name+' · exact whole-hull range span',lw=1.4)
    for j,(name,colour) in enumerate(zip(('A−B','C−B','D−B'),('#405e85','#b66a00','#8f4d86'))):
        low,high=measured[:,j].min(),measured[:,j].max()
        ax.axhspan(low,high,color=colour,alpha=.16,label=name+' · measured separation across bands')
    ax.axvline(0,color='#555',ls='--',lw=.8)
    ax.axvspan(-15,15,color='#999',alpha=.08)
    ax.set(xlabel='Heading offset from interpolated AIS course (degrees; heading is unmeasured)',
        ylabel='Total excess-path span / candidate separation (m)',xlim=(-90,90),ylim=(0,620))
    ax.legend(fontsize=9,ncol=2)
    fig.suptitle('Which selected returns could lie on one hull at 165 s?\n'
        f'{length:g} × {beam:g} m footprint, 0–60 m vertical support; exact geometry, no AIS antenna-offset assumption.',fontsize=12)
    fig.savefig(out/'01-range-support.png',dpi=150);plt.close(fig)
    central={site:float(np.interp(0,headings,span)) for site,span in spans.items()}
    within15={site:float(np.max(span[abs(headings)<=15])) for site,span in spans.items()}
    within30={site:float(np.max(span[abs(headings)<=30])) for site,span in spans.items()}
    dmin=measured[:,-1].min();compatible=np.any(np.array(list(spans.values()))>=dmin,axis=0)
    minimum_offset=float(np.min(abs(headings[compatible]))) if compatible.any() else None
    result=dict(scope=__doc__,complete=True,accepted_calibration=False,production_changed=False,
        time_s=epoch,ais_latitude_longitude_deg=[np.degrees(lat),np.degrees(lon)],
        interpolated_course_deg=float(np.degrees(course)%360),true_heading_available=False,
        hull_length_beam_height_m=[length,beam,60.],heading_offsets_deg=headings,exact_hull_spans_m=spans,
        band_order=NAMES,candidate_pairs=['A-B','C-B','D-B'],measured_candidate_separations_m=measured,
        course_aligned_span_m=central,maximum_span_within_15_deg_m=within15,
        maximum_span_within_30_deg_m=within30,minimum_heading_course_offset_for_D_minus_B_deg=minimum_offset,
        interpretation='The distant D candidate is a neighbouring return, not an established hull point. '
            'A common phase trajectory or shared intrinsic phase across bands does not put it on the same ship. '
            'This geometric support test is conditional on footprint and heading; it neither estimates true heading '
            'nor supplies a propagation explanation for other returns. Search distinct range/Doppler features within '
            'the physically supported main-return neighbourhood before using them as AIS-anchored hull points.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),source,prior_path,geometry/'acma-transmitters.json',track_path,vessel_path)})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Selected return range support</title><style>body{{font:16px/1.6 system-ui;max-width:1350px;margin:30px auto;padding:0 20px}}img{{width:100%}}</style>
<h1>Check physical hull support before treating a nearby return as a ship point</h1>
<p>At 165 s the interpolated AIS course is {np.degrees(course)%360:.2f}°. It is not a measured heading. Using the cached vessel dimensions ({length:g} × {beam:g} m) and a generous 0–60 m vertical support, the exact whole-hull excess-path span at course alignment is {min(central.values()):.1f}–{max(central.values()):.1f} m across the three sites. A ±15° heading allowance increases its largest possible span to {max(within15.values()):.1f} m; ±30° gives {max(within30.values()):.1f} m.</p>
<img src="01-range-support.png" alt="Exact hull range extent versus heading, compared with measured point separations">
<p>D−B is {measured[:,-1].min():.1f}–{measured[:,-1].max():.1f} m in the independent fits. It requires at least {minimum_offset:.1f}° heading/course disagreement under this footprint. Consequently D remains a neighbouring calibration candidate with independent motion, not an AIS-anchored point on the hull. The B/D constant-phase test does not assume common rigid motion and remains a valid separate association test.</p>
<p>Within-hull range span is independent of the unknown AIS antenna-to-hull origin. This plot uses the approved receiver coordinate and the three user-specified site priors. No heading, yaw, multipath or coordinates are fitted. A/C labels likewise denote local returns; their measured phase behaviour and physical support need checking before assigning them to the ship.</p>
<p>The next hull candidates should be distinct delay/Doppler features near the main return, including features separated mainly in Doppler. Requiring three full range cells between candidates can discard distinct ship features in this geometry. It is an acquisition convenience, not a physical-resolution criterion.</p>
<p><a href="analysis.json">Exact curves, measured separations and source identities</a> · <a href="https://taic.org.nz/inquiry/mo-2020-204">Vessel report underlying the cached dimensions</a> · <a href="../index.html">Analysis index</a></p></html>''')
    print(out/'index.html',central,within15,minimum_offset,flush=True)


if __name__=='__main__':main()
