"""Choose a kinematic continuity prior by prediction, not visual sharpness."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from cluster_phase import PolynomialPath,SplinePath
from fit_fixed_grid_phases import prepare_fixed
from fit_flexible_flash import diagnostics
from fit_shared_echo_cluster import subset_prediction
from flexible_echo_cluster import FlexibleEchoCluster
from report_measured_scatterer_phases import load_fields,serial
from smooth_echo_cluster import SmoothEchoCluster


def embed(source_model,parameters,model):
    p=np.zeros(model.count);n=model.components;count=2*n*len(model.bands)
    p[:count]=parameters[:count]
    for c,b in enumerate(model.bands):
        for s,index in enumerate(model.local_path_indices[c]):
            old=parameters[source_model.local_path_indices[c][s]]
            if s==model.selected:
                rows=b.field.data['row_time']
                p[index]=np.linalg.lstsq(model.curve(rows),source_model.curve(rows)@old,rcond=None)[0]
            else:p[index]=old
    for old,new,oi,ni in zip(source_model.clusters,model.clusters,source_model.indices,model.indices):
        for (expected,_),(actual,_) in zip(old.response_blocks(parameters[oi]),new.response_blocks(p[ni])):
            np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
    return p


def main():
    root=Path('docs/data/isar_20261017');out=root/'smooth-flash-path';out.mkdir(exist_ok=True)
    target=out/'fits.json'
    if target.exists():raise ValueError('Use a fresh output')
    source=root/'flexible-flash-path/fits.json';previous=json.loads(source.read_text())
    baseline_path=Path(previous['baseline_source']);baseline=json.loads(baseline_path.read_text())
    first,last=previous['interval_s'];geometry=Path('docs/data/calibration_20261016_sites')
    bands,partition,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    origins=baseline['selected_origins_samples'];curve=SplinePath(first,last,.5)
    source_models=[]
    for name in ('quadratic','spline-1s'):
        group=next(g for g in previous['groups'] if g['name']==name)
        source_curve=PolynomialPath(first,last,2) if name=='quadratic' else SplinePath(first,last,1.)
        model=FlexibleEchoCluster(bands,origins,1,source_curve)
        fit=group['common'][group['best_common_index']]
        source_models.append((model,np.asarray(fit['parameters'])))
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        interval_s=[first,last],basis_spacing_s=.5,groups=[],independent=[],
        prior_policy='Proper-complex data NLL plus 0.5 integral(path jerk squared)/jerk_std². Shared path prior is counted once across bands. White-jerk intensity has units m²/s⁵; quadratic motion is unpenalized. All data, brightness and stationary likelihood operators are unchanged.',
        selection_policy='Three prior scales compare frozen validation-block prediction. Minimum training cost selects two starts within each scale. The chosen scale alone then fits separate band paths as a diagnostic. Final unused blocks never select scale or start. Data have previously been inspected; this is internal validation.',
        source_sha256={})
    paths=[source,baseline_path,Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('smooth_echo_cluster','test_smooth_echo_cluster','flexible_echo_cluster','shared_echo_cluster','cluster_phase','amplitude_pair','field')]]
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for path in paths:
        result['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix=='.py':shutil.copyfile(path,snapshots/path.name)
    started=time.perf_counter()
    def save():
        result['seconds']=time.perf_counter()-started
        target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for std in (.001,.01,.1):
        model=SmoothEchoCluster(bands,origins,1,curve,jerk_std=std)
        group=dict(jerk_std=std,trials=[]);result['groups'].append(group);save()
        for start,(old,p) in enumerate(source_models):
            seed=embed(old,p,model)
            r,j=model.evaluate(seed);direction=np.cos(np.arange(model.count))*.002;step=1e-4
            finite=(model.evaluate(seed+step*direction)[0]-model.evaluate(seed-step*direction)[0])/(2*step)
            error=np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-15)
            if error>4e-4:raise ValueError(f'Real-data derivative mismatch {error}')
            fit=model.fit(seed,260);parameters=np.asarray(fit['parameters'])
            fit.update(start=start,initial_derivative_error=error,prior_cost=model.prior_cost(parameters),
                prediction=subset_prediction(model,parameters,partition),selected_diagnostics=diagnostics(model,parameters,partition))
            group['trials'].append(fit);save()
            d=fit['selected_diagnostics'][2]
            print(std,start,'cost',fit['cost'],'prior',fit['prior_cost'],'gradient',fit['maximum_projected_gradient'],
                'bounds',fit['bound_indices'],'rate range',np.ptp(d['reference_path_rate_mps']),
                'peak acceleration',max(abs(np.asarray(d['reference_path_acceleration_mps2']))),flush=True)
        group['best_index']=int(np.argmin([f['cost'] for f in group['trials']]))
        best=group['trials'][group['best_index']]
        group['validation_error']=sum(v['heldout_error'] for v in best['prediction']['validation']);save()
    chosen=min(result['groups'],key=lambda g:g['validation_error'])
    result['validation_selected_jerk_std']=chosen['jerk_std'];save()
    shared=SmoothEchoCluster(bands,origins,1,curve,jerk_std=chosen['jerk_std'])
    common=np.asarray(chosen['trials'][chosen['best_index']]['parameters'])
    for c,b in enumerate(bands):
        model=SmoothEchoCluster([b],origins,1,curve,shared=False,jerk_std=chosen['jerk_std'])
        local=common[shared.indices[c]];seed=np.zeros(model.count);seed[model.indices[0]]=local
        fit=model.fit(seed,240);parameters=np.asarray(fit['parameters'])
        fit.update(channel=c,prior_cost=model.prior_cost(parameters),prediction=subset_prediction(model,parameters,[partition[c]]),
            selected_diagnostics=diagnostics(model,parameters,[partition[c]]))
        result['independent'].append(fit);save()
        print('independent',c,'gradient',fit['maximum_projected_gradient'],'bounds',fit['bound_indices'],
            'phase',fit['selected_diagnostics'][0]['selected_unused_phase_rms_deg'],flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
