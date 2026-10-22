"""Test a weaker ABC range candidate with a trajectory fixed by other bands.

The four existing ABC returns remain free nuisance components. A fifth return
has rate/curvature fixed to the independent four-band guide. Its constant phase,
range and positive brightness are estimated using ABC training samples only.
"""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from fit_shared_echo_cluster import prepare,subset_prediction
from phase_fingerprint import measure
from report_measured_scatterer_phases import load_fields,serial
from shared_echo_cluster import SharedEchoCluster
from compare_ais_phase import NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel',type=int,choices=range(5),default=4)
    parser.add_argument('--guide-label',choices=['B','D'],default='D')
    parser.add_argument('--candidate-centre',type=float,default=15.)
    parser.add_argument('--candidate-radius',type=float,default=1.)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/guided-abc-return'))
    args=parser.parse_args()
    if args.candidate_radius<=0:raise ValueError('Positive candidate radius required')
    root=Path('docs/data/isar_20261017');out=args.output;out.mkdir(exist_ok=True);output=out/'fits.json'
    if output.exists():raise ValueError('Use a fresh output directory')
    source=root/'shared-echo-cluster-context/fits.json';fingerprint=root/'phase-fingerprint-search/search.json'
    previous=json.loads(source.read_text());guide=json.loads(fingerprint.read_text());first,last=previous['interval_s']
    geometry=Path('docs/data/calibration_20261016_sites');fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    bands,partition=prepare(fields,geometry,first,last,range_extent=16);band=bands[args.channel];part=partition[args.channel]
    oldn=len(previous['labels']);record=previous['independent'][args.channel];baseline=record['trials'][record['best_index']]
    old=np.asarray(baseline['parameters']);origins=list(previous['selected_origins_samples'])+[args.candidate_centre];n=len(origins)
    model=SharedEchoCluster([band],origins,origin_radius=2.)
    # The new hypothesis is the farther candidate singled out by the measured
    # fixed-trajectory search. No minimum component separation is imposed.
    model.lower[2*(n-1)]=args.candidate_centre-args.candidate_radius;model.upper[2*(n-1)]=args.candidate_centre+args.candidate_radius
    coefficients=np.asarray(guide['guide_path_coefficients_m'][args.guide_label]);seed=np.zeros(model.count)
    seed[:2*oldn]=old[:2*oldn];seed[2*(n-1)]=args.candidate_centre
    seed[2*n:2*n+2*oldn]=old[2*oldn:];seed[-2:]=coefficients
    baseline_model=SharedEchoCluster([band],previous['selected_origins_samples'],origin_radius=previous['origin_radius_samples'])
    r,_=baseline_model.evaluate(old)
    if abs(r@r*baseline_model.energy-baseline['cost'])>1e-7:raise ValueError('Baseline likelihood changed')
    # Use the residual after the four frozen nuisance returns only to acquire
    # the initial target phase. It never enters a test-time refit.
    cluster=baseline_model.clusters[0];value=cluster.evaluate(old[baseline_model.indices[0]])
    residual=band.field.y.ravel()-value['design']@value['amplitude']
    raw,_=band.response(np.r_[args.candidate_centre,0.,coefficients]);unit=band.reduced(raw[:,:,None])[:,0]
    seed[2*(n-1)+1]=np.angle(np.vdot(unit[band.selected],residual[band.selected]))
    free=np.arange(model.count-2);started=time.perf_counter()
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,interval_s=[first,last],channel=NAMES[args.channel],channel_index=args.channel,
        guide_label=args.guide_label,candidate_centre_samples=args.candidate_centre,candidate_radius_samples=args.candidate_radius,
        labels=[*previous['labels'],'guided_return'],selected_origins_samples=origins,
        fixed_external_path_coefficients_m=coefficients.tolist(),fixed_parameter_indices=list(range(model.count-2,model.count)),
        baseline=baseline,trials=[],baseline_source=str(source),guide_source=str(fingerprint),
        policy='Fixed guide rate/curvature; declared envelope-origin candidate interval; constant target phase and real nonnegative brightness fit on training only. Four pre-existing channel returns refit jointly. Final-quarter samples excluded from all fits and start selection. The guide excludes ABC, but includes training data of the other four channels.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,fingerprint,Path(__file__),Path('support/isar/shared_echo_cluster.py'),Path('support/isar/fit_shared_echo_cluster.py'))})
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for name in ('fit_guided_abc_return.py','shared_echo_cluster.py','fit_shared_echo_cluster.py'):
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    centre=args.candidate_centre;radius=args.candidate_radius
    for start,(origin,phase) in enumerate(((centre-radius*.5,0.),(centre,0.),(centre+radius*.5,0.),(centre,np.pi/2),(centre,-np.pi/2))):
        initial=seed.copy();initial[2*(n-1)]=origin;initial[2*(n-1)+1]+=phase
        fit=model.fit(initial,240,free_indices=free);p=np.asarray(fit['parameters'])
        if not np.array_equal(p[-2:],coefficients):raise ValueError('External trajectory was changed')
        fit['start']=start;fit['prediction']=subset_prediction(model,p,[part])
        result['trials'].append(fit);save()
        print(start,'cost',round(fit['cost'],2),'unused',round(fit['prediction']['testing'][0]['heldout_error'],2),
            'origin',round(p[2*(n-1)],4),'gradient',fit['maximum_projected_gradient'],flush=True)
    result['best_index']=int(np.argmin([r['cost'] for r in result['trials']]));result['complete']=True;save()
    print('complete',round(result['seconds'],1),'seconds',flush=True)


if __name__=='__main__':main()
