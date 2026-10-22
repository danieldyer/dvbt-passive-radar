"""Re-test an ABC companion with the corrected observed-grid static nuisance.

The earlier guided search used a superseded nuisance model. Estimate the D
trajectory from four separate channels, then hold it fixed while searching
ABC. Four pre-existing ABC returns remain independent nuisance components.
All original rows occur once; validation/test coefficients stay frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from cluster_phase import SplinePath
from fit_fixed_grid_phases import prepare_fixed
from fit_flexible_flash import seed_for, diagnostics
from fit_shared_echo_cluster import subset_prediction
from flexible_echo_cluster import FlexibleEchoCluster
from report_measured_scatterer_phases import load_fields, serial
from shared_echo_cluster import SharedEchoCluster
from smooth_echo_cluster import SmoothEchoCluster


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/fixed-grid-guided-abc-D'))
    parser.add_argument('--freeze-guide-path',action='append',default=[],metavar='CHANNEL:LABEL',
        help='Conditional control: keep a nuisance path at its independent source training estimate')
    args=parser.parse_args()
    root=Path('docs/data/isar_20261017');out=args.output
    out.mkdir(exist_ok=True);output=out/'fits.json'
    if output.exists():raise ValueError('Keep completed and partial fits; use a fresh destination')
    source=root/'fixed-grid-multiple-phases/fits.json';previous=json.loads(source.read_text())
    first,last=previous['interval_s'];origins=previous['selected_origins_samples']
    geometry=Path('docs/data/calibration_20261016_sites')
    bands,partition,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),
        geometry,first,last,stationary=True,observation_directory=out/'observations')
    curve=SplinePath(first,last,1.)
    guide=SmoothEchoCluster(bands[:4],origins,3,curve,jerk_std=.1)
    seed,seeds=seed_for(guide,dict(independent=previous['independent'][:4]),bands[:4])
    frozen=[];frozen_description=[]
    for specification in args.freeze_guide_path:
        channel,label=specification.split(':');channel=int(channel);component=previous['labels'].index(label)
        if channel not in range(4) or component==3:raise ValueError('Only a guide-channel nuisance path can be frozen')
        index=guide.local_path_indices[channel][component]
        e=previous['independent'][channel];p=np.asarray(e['trials'][e['best_index']]['parameters'])
        seed[index]=p[8+2*component:10+2*component]
        frozen.extend(index)
        frozen_description.append(dict(channel=channel,label=label,indices=index,coefficients=seed[index]))
    guide_free=np.setdiff1d(np.arange(guide.count),frozen)
    result=dict(scope=__doc__,complete=False,production_changed=False,accepted_calibration=False,
        interval_s=[first,last],guide_component='D',guide_channels=[0,1,2,3],guide_fits=[],trials=[],
        guide_jerk_std=.1,guide_knot_spacing_s=1.,source=str(source),
        conditional_frozen_nuisance_paths=frozen_description,
        nuisance_policy='One constant complex profile on actual measured fixed-grid cells; no unobserved delay columns. Positive 250 ms original-row brightness. Existing four ABC paths remain independent quadratics.',
        prediction_policy='125 ms blocks 0/2 train; block 1 validation and block 3 unused prediction. Guide excludes every ABC observation. ABC search freezes all external path coefficients. Best start uses training only. These recordings have previously been inspected; no blind significance claim.',
        source_sha256={})
    paths=[source,Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('fit_fixed_grid_phases','fixed_range_field','field','smooth_echo_cluster','flexible_echo_cluster',
         'shared_echo_cluster','amplitude_pair','fit_flexible_flash','report_measured_scatterer_phases')]]
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for path in paths:
        result['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix=='.py':shutil.copyfile(path,snapshots/path.name)
    started=time.monotonic()
    def save():
        result['seconds']=time.monotonic()-started
        output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for start,q in enumerate((np.median(seeds,axis=0),seeds[2])):
        initial=seed.copy();initial[guide.local_path_indices[0][3]]=q
        fit=guide.fit(initial,240,free_indices=guide_free);p=np.asarray(fit['parameters'])
        np.testing.assert_array_equal(p[frozen],seed[frozen])
        fit.update(start=start,prior_cost=guide.prior_cost(p),prediction=subset_prediction(guide,p,partition[:4]),
            selected_diagnostics=diagnostics(guide,p,partition[:4]))
        result['guide_fits'].append(fit);save()
        print('guide',start,'cost',fit['cost'],'gradient',fit['maximum_projected_gradient'],
            'bounds',fit['bound_indices'],flush=True)
    winner=int(np.argmin([f['cost'] for f in result['guide_fits']]))
    result['guide_best_index']=winner
    guided_path=np.asarray(result['guide_fits'][winner]['parameters'])[guide.local_path_indices[0][3]]
    result['fixed_guide_path_coefficients']=guided_path;save()
    abc=bands[4];entry=previous['independent'][4];old=entry['trials'][entry['best_index']]
    baseline=SharedEchoCluster([abc],origins,origin_radius=previous['origin_radius_samples'])
    before=np.asarray(old['parameters']);r,_=baseline.evaluate(before)
    if abs(r@r*baseline.energy-old['cost'])>1e-7:raise ValueError('Original baseline objective changed')
    result['baseline']=dict(cost=old['cost'],prediction=subset_prediction(baseline,before,[partition[4]]))
    model=FlexibleEchoCluster([abc],[*origins,14.],4,curve,origin_radius=2.)
    model.lower[8]=12.5;model.upper[8]=15.5
    initial=np.zeros(model.count);initial[:8]=before[:8];initial[8]=14.
    for s in range(4):initial[model.local_path_indices[0][s]]=before[8+2*s:10+2*s]
    fixed=model.local_path_indices[0][4];initial[fixed]=guided_path
    cluster=baseline.clusters[0];value=cluster.evaluate(before[baseline.indices[0]])
    prediction=value['design']@value['amplitude'];y=abc.field.y.ravel()
    beta=abc.nuisance.solve(y[abc.selected]-prediction[abc.selected])
    residual=y-prediction-abc.design@beta
    raw=model.clusters[0].response_blocks(initial[model.indices[0]])[4][0]
    unit=abc.reduced(raw[:,:,None])[:,0]
    initial[9]=np.angle(np.vdot(unit[abc.selected],residual[abc.selected]))
    free=np.setdiff1d(np.arange(model.count),fixed)
    result['fixed_indices']=fixed;result['candidate_origin_bounds_samples']=[12.5,15.5];save()
    for start,(origin,phase) in enumerate(((13.,0.),(14.,0.),(15.,0.),(14.,np.pi/2),(14.,-np.pi/2))):
        seed=initial.copy();seed[8]=origin;seed[9]+=phase
        r,j=model.evaluate(seed);direction=np.sin(np.arange(model.count))*.002;direction[fixed]=0;h=1e-4
        difference=(model.evaluate(seed+h*direction)[0]-model.evaluate(seed-h*direction)[0])/(2*h)
        error=np.linalg.norm(difference-j@direction)/max(np.linalg.norm(difference),1e-15)
        if error>4e-4:raise ValueError('ABC derivative mismatch')
        fit=model.fit(seed,260,free_indices=free);p=np.asarray(fit['parameters'])
        np.testing.assert_array_equal(p[fixed],guided_path)
        fit.update(start=start,directional_derivative_error=error,
            prediction=subset_prediction(model,p,[partition[4]]),selected_diagnostics=diagnostics(model,p,[partition[4]]))
        result['trials'].append(fit);save()
        print('ABC',start,'cost',fit['cost'],'unused',fit['prediction']['testing'][0]['heldout_error'],
            'origin',p[8],'gradient',fit['maximum_projected_gradient'],'bounds',fit['bound_indices'],flush=True)
    result['best_index']=int(np.argmin([f['cost'] for f in result['trials']]))
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
