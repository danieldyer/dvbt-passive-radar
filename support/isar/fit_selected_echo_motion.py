"""Test each selected return's common trajectory with other returns independent.

Sharing every return at once confounds failures of different associations. This
control shares only the selected point and leaves all other band trajectories
free. The observation operator, positive brightness and data splits stay fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from fit_shared_echo_cluster import prepare, subset_prediction
from report_measured_scatterer_phases import load_fields, serial
from shared_echo_cluster import SharedEchoCluster


def local_parameters(parameters,components):
    p=np.asarray(parameters);n=components
    return np.column_stack((p[:2*n].reshape(n,2),p[2*n:].reshape(n,2))).ravel()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',default=['A','B','C'],
        help='Individual labels or comma-separated jointly shared groups, e.g. A,B,C')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    target=out/'fits.json'
    if target.exists():raise ValueError('Use a fresh output')
    source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Independent source fits are incomplete')
    geometry=Path('docs/data/calibration_20261016_sites')
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    first,last=source['interval_s'];origins=source['selected_origins_samples'];n=len(origins)
    bands,partition=prepare(fields,geometry,first,last,range_extent=source['range_extent_samples'])
    independent=[record['trials'][record['best_index']] for record in source['independent']]
    baseline_error=sum(f['prediction']['testing'][0]['heldout_error'] for f in independent)
    baseline=SharedEchoCluster(bands,origins,origin_radius=source['origin_radius_samples'],shared_components=[])
    ip=np.zeros(baseline.count)
    for c,fit in enumerate(independent):ip[baseline.indices[c]]=local_parameters(fit['parameters'],n)
    r,_=baseline.evaluate(ip)
    if abs(r@r*baseline.energy-sum(f['cost'] for f in independent))>1e-7:
        raise ValueError('Independent likelihood reproduction failed')
    started=time.perf_counter()
    result=dict(scope=__doc__,source=str(args.source),complete=False,accepted_calibration=False,
        interval_s=[first,last],labels=source['labels'],origins_samples=origins,
        radius_samples=source['origin_radius_samples'],range_extent_samples=source['range_extent_samples'],
        independent_unused_error=baseline_error,tests=[],
        policy='One selected component has a common residual quadratic path; all other components retain independent band paths. Each band/component has its own envelope origin, constant phase and real nonnegative brightness. Starts and all parameters use training only. Repeating 125 ms blocks 0/2 train; block 3 tests frozen prediction. No new stationary, multipath or time-dependent phase correction.',
        source_sha256={str(args.source):hashlib.sha256(args.source.read_bytes()).hexdigest()})
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for name in ('fit_selected_echo_motion.py','shared_echo_cluster.py','fit_shared_echo_cluster.py'):
        p=Path(__file__).with_name(name)
        result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        shutil.copyfile(p,snapshot/name)
    def save():
        result['seconds']=time.perf_counter()-started
        target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    for label in args.labels:
        selected=[source['labels'].index(v) for v in label.split(',')]
        model=SharedEchoCluster(bands,origins,origin_radius=source['origin_radius_samples'],shared_components=selected)
        seed=np.zeros(model.count);coefficients=[]
        for c,fit in enumerate(independent):
            p=np.asarray(fit['parameters']);seed[model.indices[c]]=local_parameters(p,n)
            coefficients.append(p[2*n:].reshape(n,2)[selected])
        seed[model.phase_indices[0,selected]]=np.average(coefficients,axis=0,weights=[b.energy for b in bands])
        record=dict(label=label,selected_components=selected,trials=[]);result['tests'].append(record);save()
        common=np.asarray(source['common'][source['best_common_index']]['parameters'])[-2*n:].reshape(n,2)[selected]
        for start,coefficient in enumerate((seed[model.phase_indices[0,selected]],common)):
            initial=seed.copy();initial[model.phase_indices[0,selected]]=coefficient
            fit=model.fit(initial,240)
            fit['start']=start;fit['prediction']=subset_prediction(model,fit['parameters'],partition)
            error=sum(v['heldout_error'] for v in fit['prediction']['testing'])
            fit['unused_error_ratio_to_independent']=error/baseline_error
            record['trials'].append(fit);save()
            print(first,label,start,'training',round(fit['cost'],2),'unused ratio',round(error/baseline_error,6),
                'gradient',fit['maximum_projected_gradient'],'bounds',fit['bound_indices'],flush=True)
        record['best_index']=int(np.argmin([f['cost'] for f in record['trials']]))
        save()
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
