"""Independent one-second multiscatterer fits inside the flash interval.

The overlapping controls locate short intervals worth a physical association
test. They are not independent observations to multiply in a later likelihood.
All re-referencing is algebraic and checked against the original row response.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from fit_fixed_grid_phases import prepare_fixed
from fit_shared_echo_cluster import subset_prediction
from phase_window import rebase_parameters,window_band
from report_measured_scatterer_phases import load_fields,serial
from shared_echo_cluster import SharedEchoCluster


def main():
    root=Path('docs/data/isar_20261017');out=root/'piecewise-fixed-cluster';out.mkdir(exist_ok=True)
    output=out/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    source=root/'fixed-grid-multiple-phases/fits.json';previous=json.loads(source.read_text())
    geometry=Path('docs/data/calibration_20261016_sites')
    bands,_,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,
        *previous['interval_s'],stationary=True)
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        labels=previous['labels'],selected_origins_samples=previous['selected_origins_samples'],
        origin_radius_samples=previous['origin_radius_samples'],windows=[],
        policy='Independent channel and component quadratics with a fixed-grid constant stationary profile and positive 250 ms brightness. Repeating 125 ms blocks 0/2 train, 1 validation, 3 unused prediction. Starts select only training cost. Overlapping windows are internally inspected controls, not independent calibration evidence.',
        source_sha256={})
    sources=[source,Path(__file__),Path('support/isar/phase_window.py'),
        Path('support/isar/shared_echo_cluster.py'),Path('support/isar/amplitude_pair.py'),
        Path('support/isar/fixed_range_field.py')]
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for p in sources:
        result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshots/p.name)
    started=time.perf_counter()
    def save():
        result['seconds']=time.perf_counter()-started
        output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for first in np.arange(163.,166.01,.5):
        last=first+1.;window=dict(interval_s=[first,last],independent=[]);result['windows'].append(window);save()
        for c,old in enumerate(bands):
            band,part,rows=window_band(old,first,last)
            model=SharedEchoCluster([band],previous['selected_origins_samples'],origin_radius=previous['origin_radius_samples'])
            entry=dict(channel=c,trials=[]);window['independent'].append(entry);save()
            for start,trial in enumerate(previous['independent'][c]['trials']):
                seed=rebase_parameters(trial['parameters'],model.components,old,band)
                fit=model.fit(seed,200);p=np.asarray(fit['parameters'])
                fit['start']=start;fit['prediction']=subset_prediction(model,p,[part])
                entry['trials'].append(fit);save()
            entry['best_index']=int(np.argmin([t['cost'] for t in entry['trials']]))
            best=entry['trials'][entry['best_index']]
            print(first,c,'gradient',best['maximum_projected_gradient'],'bounds',best['bound_indices'],flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
