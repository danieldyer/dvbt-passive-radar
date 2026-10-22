"""Release the acquired ABC path; test independent phase support and precision.

Fit the selected path independently with the same white-jerk prior, starting
from both observed branches. A small conditional phase residual is not treated
as independent trajectory precision: other return paths, phases, origins and
brightness are marginalized in the data-only information calculation.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from cluster_phase import SplinePath
from fit_fixed_grid_phases import prepare_fixed
from fit_flexible_flash import diagnostics
from fit_shared_echo_cluster import subset_prediction
from flexible_echo_cluster import FlexibleEchoCluster
from report_measured_scatterer_phases import load_fields, serial
from smooth_echo_cluster import SmoothEchoCluster


def path_information(model,parameters,*,component=None,channel=0):
    component=model.selected if component is None else component
    cluster=model.clusters[channel];local=parameters[model.indices[channel]]
    value=cluster.evaluate(local)
    matrix=value['mean_derivative']
    block=cluster.slices[component]
    selected=np.arange(block.start+2,block.stop)
    other=np.setdiff1d(np.arange(matrix.shape[1]),selected)
    nuisance=matrix[:,other];scale=np.linalg.norm(nuisance,axis=0);keep=scale>1e-15
    u,s,_=np.linalg.svd(nuisance[:,keep]/scale[keep],full_matrices=False)
    rank=int(np.count_nonzero(s>s[0]*1e-10));u=u[:,:rank]
    residual=matrix[:,selected]-u@(u.T@matrix[:,selected])
    information=2*residual.T@residual
    # Normalize with pre-projection column norms. Renormalizing a column that
    # was annihilated by the nuisance projection can turn roundoff into false
    # unit-scale information.
    original=np.linalg.norm(matrix[:,selected],axis=0)
    norm=np.sqrt(2)*np.where(original>0,original,1.)
    normalized=information/norm[:,None]/norm[None,:]
    eigen=np.linalg.eigvalsh(normalized)
    _,singular,right=np.linalg.svd(residual/np.where(original>0,original,1.),full_matrices=False)
    threshold=1e-5*max(1.,singular[0])
    full_rank=bool(np.all(original>0) and singular[-1]>threshold)
    result=dict(nuisance_rank=rank,normalized_eigenvalues=eigen,full_rank=bool(full_rank),
        normalized_singular_values=singular,numerical_rank_threshold=threshold,
        scope='Conditional data-only Fisher information after active real brightness and every other nonlinear phase/origin/path column; no motion prior included. Unmodelled scatterers and interference remain outside this bound.')
    # Some spline coefficients can be unobserved at a fade while earlier
    # phase contrasts remain identifiable. Report a functional only if it
    # annihilates every unresolved direction; never fill it from a prior.
    keep=singular>threshold
    arcs=getattr(cluster,'arcs',None)
    curve=model.curve if arcs is None else arcs[component].base.curve
    grid=np.linspace(curve.epoch-curve.scale,curve.epoch+curve.scale,201)
    basis=curve(grid);functional=basis/norm
    null=functional@right[~keep].T
    estimable=np.linalg.norm(null,axis=1)<=1e-8*np.linalg.norm(functional,axis=1)
    supported=(right[keep].T/singular[keep])/norm[:,None]
    deviation=np.sqrt(np.sum((basis@supported)**2,axis=1))
    deviation[~estimable]=np.nan
    result.update(time_s=grid,target_rank=int(keep.sum()),path_change_estimable=estimable,
        path_standard_deviation_m=deviation,
        phase_standard_deviation_deg=np.degrees(model.bands[channel].field.k*deviation))
    return result


def main():
    root=Path('docs/data/isar_20261017');source=root/'fixed-grid-guided-abc-D-fixed-static-C/fits.json'
    previous=json.loads(source.read_text());first,last=previous['interval_s']
    out=root/'fixed-grid-guided-abc-D-independent';out.mkdir(exist_ok=True);target=out/'fits.json'
    if target.exists():raise ValueError('Use a fresh output directory')
    baseline=json.loads(Path(previous['source']).read_text());geometry=Path('docs/data/calibration_20261016_sites')
    bands,part,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    curve=SplinePath(first,last,previous['guide_knot_spacing_s']);origins=[*baseline['selected_origins_samples'],14.]
    model=SmoothEchoCluster([bands[4]],origins,4,curve,jerk_std=previous['guide_jerk_std'])
    model.lower[8]=12.5;model.upper[8]=15.5
    fixed=FlexibleEchoCluster([bands[4]],origins,4,curve)
    starts=[previous['best_index'],1]
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        source=str(source),interval_s=[first,last],trials=[],
        policy='ABC path coefficients are now free and independent of the other four channels. Two starts from the training winner and the distinct farther-origin branch; same data, metric, support, brightness, stationary nuisance, 1 s basis and white-jerk intensity. All coefficients frozen on validation/test blocks.',source_sha256={})
    files=[source,Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('smooth_echo_cluster','flexible_echo_cluster','amplitude_pair','field','fit_fixed_grid_phases','component_phase')]]
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for p in files:
        result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshots/p.name)
    started=time.monotonic()
    def save():
        result['seconds']=time.monotonic()-started
        target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for start in starts:
        old=previous['trials'][start];seed=np.asarray(old['parameters'])
        residual,_=fixed.evaluate(seed)
        if abs(residual@residual*fixed.energy-old['cost'])>1e-7:raise ValueError('Guided objective changed')
        before=path_information(model,seed)
        fit=model.fit(seed,300);p=np.asarray(fit['parameters'])
        fit.update(source_start=start,prior_cost=model.prior_cost(p),
            prediction=subset_prediction(model,p,[part[4]]),selected_diagnostics=diagnostics(model,p,[part[4]]),
            information_at_guided_solution=before,information_at_independent_solution=path_information(model,p))
        result['trials'].append(fit);save()
        print('released',start,'cost',fit['cost'],'unused',fit['prediction']['testing'][0]['heldout_error'],
            'phase',fit['selected_diagnostics'][0]['selected_unused_phase_rms_deg'],
            'gradient',fit['maximum_projected_gradient'],'bounds',fit['bound_indices'],flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
