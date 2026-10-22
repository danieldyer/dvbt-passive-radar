"""Compare local stationary-return phase curvature with the ship-arc mismatch.

This is a measured-data diagnostic, not a phase correction. The two fixed
range regions and disjoint reference interval are inherited from phase_control.
OLS curvature of a strong aggregate stationary return is not a CRLB estimate
and does not rule out target-specific or range-dependent propagation errors.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES
from geographic_io import read_profiles
from stationary_phase import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',type=Path,default=Path('build/calibration-20261013/static-with-psf.gprf'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/stationary-arc-phase-control'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    metadata,records=read_profiles(args.profiles)
    if metadata.get('stationary_subtraction') is not False:raise ValueError('Control requires unremoved stationary returns')
    intervals=[('F03',151.,154.),('F01',159.,162.),('Flash',163.,166.)]
    regions=[(600,5000),(5000,10000)];ranges=np.arange(1024)*299792458/8e6
    result=dict(scope=__doc__,reference_s=[30.,90.],regions_m=regions,channels=[],correction_applied=False,
        source_capture=metadata['source_capture'],source_product=metadata['product'],
        stationary_subtraction=metadata['stationary_subtraction'])
    fig,axes=plt.subplots(5,3,figsize=(15,12),layout='constrained')
    frequencies=[s['frequency_hz'] for s in metadata['signals']]
    result['frequency_hz']=frequencies
    if [s['name'] for s in metadata['signals']]!=list(NAMES):raise ValueError('Unexpected channel order')
    for ch,name in enumerate(NAMES):
        good=(records['channel']==ch)&(records['information']>0)
        reference_rows=records[good&(records['time']>=30)&(records['time']<90)]
        if len(reference_rows)<100:raise ValueError('Missing independent reference coverage')
        channel=dict(channel=ch,name=name,trials=[])
        for region in regions:
            keep=(ranges>=region[0])&(ranges<region[1])
            reference=np.average(reference_rows['values'][:,keep],axis=0,weights=reference_rows['information'])
            for i,(label,first,last) in enumerate(intervals):
                rows=records[good&(records['time']>=first)&(records['time']<last)]
                if len(rows)<80:raise ValueError('Insufficient local profile coverage')
                gain,coherence=compare(rows['values'][:,keep],reference)
                phase=np.unwrap(np.angle(gain));dt=rows['time']-(first+last)/2
                step=float(abs(np.diff(phase)).max())
                if step>np.pi/2:raise ValueError('Stationary phase has an unresolved jump')
                coefficient=np.polynomial.polynomial.polyfit(dt,phase,2)
                k=2*np.pi*frequencies[ch]/299792458
                channel['trials'].append(dict(interval=label,interval_s=[first,last],range_m=region,
                    phase_rate_radps=float(coefficient[1]),phase_curvature_radps2=float(2*coefficient[2]),
                    equivalent_path_rate_mps=float(-coefficient[1]/k),equivalent_path_curvature_mps2=float(-2*coefficient[2]/k),
                    phase_residual_rms_deg=float(np.degrees(np.sqrt(np.mean((phase-np.polynomial.polynomial.polyval(dt,coefficient))**2)))),
                    phase_excursion_deg=float(np.degrees(np.ptp(phase))),minimum_pattern_coherence_squared=float(coherence.min()),
                    maximum_adjacent_phase_step_deg=float(np.degrees(step))))
                axes[ch,i].plot(rows['time'],np.degrees(phase-coefficient[0]),lw=.8,label=f'{region[0]/1000:g}–{region[1]/1000:g} km')
                if ch==0:axes[ch,i].set_title(label)
                if i==0:axes[ch,i].set_ylabel(name+'\nStationary phase (deg)')
                if ch==4:axes[ch,i].set_xlabel('Capture time (s)')
        result['channels'].append(channel)
    axes[0,0].legend();fig.suptitle('Measured stationary returns during the same three ship intervals\nOnly a constant phase removed for display; no phase correction applied',fontsize=12)
    fig.savefig(args.output/'stationary-local-phase.png',dpi=170);plt.close(fig)
    largest=max(abs(r['equivalent_path_curvature_mps2']) for c in result['channels'] for r in c['trials'])
    result['maximum_absolute_equivalent_curvature_mps2']=largest
    sources=[args.profiles,Path(__file__),Path('support/calibration/stationary_phase.py')]
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sources}
    rows=''.join(f'<tr><td>{c["name"]}</td><td>{r["interval"]}</td><td>{r["range_m"]}</td><td>{r["equivalent_path_curvature_mps2"]:+.6f}</td><td>{r["phase_excursion_deg"]:.3f}</td><td>{r["minimum_pattern_coherence_squared"]:.5f}</td></tr>' for c in result['channels'] for r in c['trials'])
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Local stationary phase control</title>
<style>body{{font:16px/1.55 system-ui;max-width:1220px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:7px;border:1px solid #ccc}}</style>
<h1>Stationary phase during the candidate ship intervals</h1>
<p>The largest aggregate stationary-return phase curvature, expressed as equivalent path curvature, is {largest:.6f} m/s² across all five bands, three intervals and two range regions. Compare this with the roughly 0.42–0.52 m/s² difference between ABC and the other bands in the single-echo flash fits. This tests a channel-wide phase drift explanation using independent parts of the same recording; no correction is fitted or applied to the ship.</p>
<img src="stationary-local-phase.png" alt="Stationary-return phase over the three candidate intervals">
<p>Reference complex profiles are averaged over 30–90 s. Each later profile is matched to that fixed reference in the inherited 0.6–5 km and 5–10 km regions. Sampling is approximately 32.64 ms. The table gives ordinary quadratic phase fits over each three-second interval, plus actual phase excursions and minimum reference-pattern coherence. There are no fitted range weights or per-bin noise rescaling. A constant phase is removed only for plot readability.</p>
<table><tr><th>Band</th><th>Interval</th><th>Range (m)</th><th>Equivalent curvature (m/s²)</th><th>Phase excursion (deg)</th><th>Minimum coherence²</th></tr>{rows}</table>
<p>This is a conditional aggregate control: stable nearby clutter does not rule out ship-specific multipath, directional scattering or all range-dependent errors. It does test whether a comparably large common phase curvature is present in these stationary returns. It is not an antenna-location measurement or a statistical optimality certificate.</p>
<p><a href="analysis.json">All measurements and source hashes</a> · <a href="../joint-ship-band-phase-control/index.html">Ship motion comparison</a> · <a href="../ship-amplitude-pair-F01/index.html">Overlapping-echo test</a>.</p></html>'''
    (args.output/'index.html').write_text(page);(args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('maximum stationary equivalent curvature',largest,'report',args.output/'index.html',flush=True)


if __name__=='__main__':main()
