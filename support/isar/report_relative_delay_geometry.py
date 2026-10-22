"""Compare candidate range separations with the exact point-geometry bound."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from antenna_motion import Antennas
from compare_ais_phase import NAMES
from flash_geometry import assignments
from relative_delay_geometry import relative_delay_bound
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options
from motion import C


def main():
    root=Path('docs/data/isar_20261017');source=root/'shared-echo-cluster-split';out=root/'relative-delay-geometry';out.mkdir(exist_ok=True)
    fit=json.loads((source/'fits.json').read_text());analysis=json.loads((source/'analysis.json').read_text());mid=np.mean(fit['interval_s'])
    with np.load('build/isar-20261015/full-fields/channel-0.npz') as f:
        centre=np.array([np.interp(mid,f['row_time'],f['centre_ecef_m'][:,j]) for j in range(3)])
    geometry=Path('docs/data/calibration_20261016_sites');options=licensed_options(geometry)
    positions=np.unique([o['transmitter_ecef_m'] for o in options],axis=0)
    rotation=np.array(json.loads((geometry/'geometry-prior.json').read_text())['ecef_to_enu']);maximum_radius=0.
    for assignment in assignments(options):
        antenna=Antennas(rotation,[o['site_id'] for o in assignment])
        box=3*np.sum(abs(antenna.factor[:15,:15]),axis=1).reshape(5,3)
        maximum_radius=max(maximum_radius,float(np.linalg.norm(box,axis=1).max()))
    nominal=relative_delay_bound(centre,positions,scene_radius_m=500,point_separation_m=500)
    uncertain=relative_delay_bound(centre,positions,scene_radius_m=500,point_separation_m=500,phase_centre_radius_m=maximum_radius)
    matrix=[];variance=[];selections=[];n=len(fit['labels']);b=fit['labels'].index('B');d=fit['labels'].index('D');e=fit['labels'].index('E');c=fit['labels'].index('C')
    for record,stats in zip(fit['independent'],analysis['independent_conditional_observables']):
        if not stats['identifiable']:raise ValueError('Cannot interpret a singular conditional covariance')
        p=np.array(record['trials'][record['best_index']]['parameters']);origin=p[:2*n:2]
        near,far=sorted((d,e),key=lambda s:origin[s]);ids=[c,near,far];cov=np.array(stats['covariance'])
        row=[];var=[]
        for s in ids:
            contrast=np.zeros(3*n);contrast[3*s]=1;contrast[3*b]=-1
            row.append((origin[s]-origin[b])*C/8e6);var.append(float(contrast@cov@contrast))
        matrix.append(row);variance.append(var);selections.append(ids)
    matrix=np.array(matrix);variance=np.array(variance);delta=matrix-matrix[2]
    stderr=np.sqrt(variance+variance[2]);stderr[2]=0
    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained',sharey=True)
    for j,(ax,label) in enumerate(zip(axes,('C − B','Near delayed component − B','Far delayed component − B'))):
        ax.axhspan(-uncertain['maximum_range_difference_m'],uncertain['maximum_range_difference_m'],color='#ddd',label='All current antenna-prior bounds')
        ax.axhspan(-nominal['maximum_range_difference_m'],nominal['maximum_range_difference_m'],color='#b8d8e8',label='All three nominal sites')
        ax.errorbar(np.arange(5),delta[:,j],yerr=stderr[:,j],fmt='o',capsize=4,color='#222')
        ax.axhline(0,color='#777',lw=.7);ax.set_xticks(range(5),NAMES,rotation=25);ax.set_title(label)
        ax.set_ylabel('Inter-point delay difference relative to Nine (m)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Do the fitted envelope centres correspond to the same physical points?\n'
        'Exact point model: relative receiver path and per-band constant delay cancel. Error bars are conditional thermal 1σ only.',fontsize=13)
    fig.savefig(out/'01-relative-delay-bound.png',dpi=170);plt.close(fig)
    result=dict(scope=__doc__,accepted_calibration=False,nominal_bound=nominal,all_current_antenna_box_bound=uncertain,
        component_selection_by_band=selections,relative_delays_m=matrix,differences_from_nine_m=delta,conditional_standard_error_m=stderr,
        assumptions='Both points lie within 500 m of the AIS centre and are no more than 500 m apart; one common receiver phase centre; geometric point delays with a constant per-band instrumental delay. Nominal test permits every licensed site assignment; broad test also encloses every currently allowed standardized antenna-coordinate box. These are declared conservative ship hypotheses, not measured ship dimensions.',
        interpretation='A bound violation rejects the chosen point-envelope/association model under these hypotheses. It does not prove the data are invalid or identify a scattering mechanism. Distributed frequency-dependent scattering and biased unresolved envelope estimates remain possible.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'fits.json',source/'analysis.json',Path(__file__),Path('support/isar/relative_delay_geometry.py'),Path('support/isar/test_relative_delay_geometry.py'))})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Relative envelope-delay geometry</title>
<style>body{{font:17px/1.5 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}</style>
<h1>A receiver-independent check on candidate point correspondence</h1>
<p>For two physical points p and q, subtracting their excess paths in one band removes that transmitter’s direct-path baseline and any constant instrumental delay. Differencing that result between two transmitters also removes the receiver contribution exactly. The remaining expression is ||p−T₁||−||q−T₁||−||p−T₂||+||q−T₂||.</p>
<p>The distance Hessian has spectral norm 1/r. Integrating it over the point and transmitter segments bounds the absolute double difference by L B / r_min. This is an analytic bound, with no phase or motion fit. For points at most 500 m apart within 500 m of AIS, all three nominal licensed sites permit at most <b>{nominal['maximum_range_difference_m']:.2f} m</b> difference. Even enclosing every currently allowed antenna-coordinate box in a radius {maximum_radius:.1f} m ball permits at most <b>{uncertain['maximum_range_difference_m']:.2f} m</b>.</p>
<img src="01-relative-delay-bound.png" alt="Fitted inter-point range differences and exact geometric bounds">
<p>The plotted delays are from independent band fits, including full cross-component covariance. The delayed pair is sorted by fitted range separately in each band to remove the simple label-permutation ambiguity. The uncertainty bars are conditional on that fitted point model and thermal noise. They exclude unresolved-scene model error.</p>
<p>The two 500 m limits are deliberately broad hypotheses for points on the same vessel, not claims about its exact dimensions. Exceeding the bound means these envelope centres cannot yet be used as corresponding geometric points under that model. It does not invalidate the recording or prove a particular scattering mechanism. A common receiver-position adjustment cannot fix this double difference.</p>
<p><a href="analysis.json">Bound calculation and all measurements</a> · <a href="tests.log">Numerical controls</a> · <a href="../phase-fingerprint-search/index.html">Independent phase-fingerprint search</a> · <a href="../index.html">Analysis index</a></p></html>''')
    print(out/'index.html',nominal['maximum_range_difference_m'],uncertain['maximum_range_difference_m'],flush=True)


if __name__=='__main__':main()
