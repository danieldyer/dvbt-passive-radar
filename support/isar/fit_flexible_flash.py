"""Test shared nonquadratic flash motion before interpreting band differences.

Different scintillation weights can bias separate low-order approximations to
one nonquadratic path. Only B's temporal basis changes here; A/C/D remain
independent quadratic nuisance returns with the same observations and static
profile. No antenna coordinates or per-band time-varying phase corrections fit.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from cluster_phase import PolynomialPath,SplinePath
from field import average
from fit_fixed_grid_phases import prepare_fixed
from fit_shared_echo_cluster import subset_prediction
from flexible_echo_cluster import FlexibleEchoCluster
from report_measured_scatterer_phases import load_fields,serial


def seed_for(model,previous,bands):
    seeds=[];n=model.components
    for c,(record,b) in enumerate(zip(previous['independent'],bands)):
        old=np.asarray(record['trials'][record['best_index']]['parameters'])
        rows=b.field.data['row_time'];basis=model.curve(rows)
        q=old[2*n:].reshape(n,2)
        selected=np.linalg.lstsq(basis,b.base.basis@q[model.selected],rcond=None)[0]
        seeds.append(selected)
    p=np.zeros(model.count)
    for c,record in enumerate(previous['independent']):
        old=np.asarray(record['trials'][record['best_index']]['parameters'])
        p[2*n*c:2*n*(c+1)]=old[:2*n]
        for s,index in enumerate(model.local_path_indices[c]):
            p[index]=seeds[c] if s==model.selected else old[2*n+2*s:2*n+2*s+2]
    return p,seeds


def diagnostics(model,p,partitions):
    records=[]
    for cluster,index,part in zip(model.clusters,model.indices,partitions):
        b=cluster.base;local=p[index];value=cluster.evaluate(local);k=len(b.knots)
        parts=np.stack([value['design'][:,s*k:(s+1)*k]@value['amplitude'][s*k:(s+1)*k] for s in range(model.components)])
        y=b.field.y.ravel();prediction=parts.sum(axis=0)
        beta=b.nuisance.solve(y[b.selected]-prediction[b.selected]);static=b.design@beta
        clean=(y-static-prediction+parts[model.selected]).reshape(b.field.y.shape)
        raw,_=cluster.response_blocks(local)[model.selected]
        unit=b.reduced(raw[:,:,None])[:,0].reshape(b.field.y.shape)
        z=np.sum(unit.conj()*clean,axis=1);phase=np.angle(z)
        strength=np.sum(abs(parts[model.selected].reshape(b.field.y.shape))**2,axis=1)
        test=(part==3)&(strength>0)
        arc=cluster.arcs[model.selected];q=local[cluster.slices[model.selected]][2:]
        rows=b.field.data['row_time'];path=arc.known_path+model.curve(rows)@q
        h=parts[model.selected][b.selected]
        retained=np.vdot(b.nuisance.remove(h),b.nuisance.remove(h)).real/np.vdot(h,h).real
        grid=np.linspace(rows[0],rows[-1],301)
        records.append(dict(time_s=b.field.time,testing=part==3,phase_residual_deg=np.degrees(phase),
            selected_path_change_m=average(path,b.field.data['row_information'],b.field.metadata['block_rows'])[0],
            selected_unused_phase_rms_deg=np.degrees(np.sqrt(np.average(phase[test]**2,weights=strength[test]))),
            selected_unused_prediction_energy=strength[test].sum(),selected_projected_signal_energy_retained=retained,
            derivative_time_s=grid,reference_path_rate_mps=model.curve(grid,1)@q,
            reference_path_acceleration_mps2=model.curve(grid,2)@q))
    return records


def main():
    root=Path('docs/data/isar_20261017');out=root/'flexible-flash-path';out.mkdir(exist_ok=True)
    output=out/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    source=root/'fixed-grid-multiple-phases/fits.json';previous=json.loads(source.read_text())
    first,last=previous['interval_s'];geometry=Path('docs/data/calibration_20261016_sites')
    bands,partition,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,first,last,stationary=True)
    selected=previous['labels'].index('B');origins=previous['selected_origins_samples']
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,groups=[],
        interval_s=[first,last],selected_component='B',baseline_source=str(source),
        policy='Training uses 125 ms blocks 0/2; validation 1 and final unused prediction 3. Every original row occurs once in each fit. Degree/spacing comparisons use the same noise information, positive 250 ms brightness and fixed-grid stationary design. Scene nuisance is profiled jointly. Data have already been inspected; these are internal prediction controls.',source_sha256={})
    sources=[source,Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('flexible_echo_cluster','shared_echo_cluster','amplitude_pair','cluster_phase','field','fixed_range_field')]]
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for path in sources:
        result['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix=='.py':shutil.copyfile(path,snapshots/path.name)
    started=time.perf_counter()
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for label,curve in (('quadratic',PolynomialPath(first,last,2)),('spline-1s',SplinePath(first,last,1.)),('spline-0.5s',SplinePath(first,last,.5))):
        model=FlexibleEchoCluster(bands,origins,selected,curve)
        seed,seeds=seed_for(model,previous,bands)
        group=dict(name=label,degree=curve.degree,common=[],independent=[]);result['groups'].append(group);save()
        for start in range(2):
            p=seed.copy();p[model.local_path_indices[0][selected]]=np.median(seeds,axis=0) if start==0 else seeds[2]
            r,j=model.evaluate(p);direction=np.cos(np.arange(model.count))*.002;step=1e-4
            finite=(model.evaluate(p+step*direction)[0]-model.evaluate(p-step*direction)[0])/(2*step)
            error=np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-15)
            if error>4e-4:raise ValueError(f'Real-data derivative mismatch {error}')
            fit=model.fit(p,260);p=np.asarray(fit['parameters']);fit['start']=start
            fit['initial_directional_derivative_error']=error
            fit['prediction']=subset_prediction(model,p,partition);fit['selected_diagnostics']=diagnostics(model,p,partition)
            group['common'].append(fit);save()
            print(label,'common',start,'cost',fit['cost'],'gradient',fit['maximum_projected_gradient'],
                'bounds',fit['bound_indices'],'B unused phase',[round(d['selected_unused_phase_rms_deg'],2) for d in fit['selected_diagnostics']],flush=True)
        group['best_common_index']=int(np.argmin([f['cost'] for f in group['common']]))
        if label=='spline-1s':
            for c,b in enumerate(bands):
                separate=FlexibleEchoCluster([b],origins,selected,curve,shared=False)
                old=dict(independent=[previous['independent'][c]])
                initial,_=seed_for(separate,old,[b]);fit=separate.fit(initial,240)
                p=np.asarray(fit['parameters']);fit['channel']=c;fit['prediction']=subset_prediction(separate,p,[partition[c]])
                fit['selected_diagnostics']=diagnostics(separate,p,[partition[c]])
                group['independent'].append(fit);save()
                print(label,'independent',c,'cost',fit['cost'],'gradient',fit['maximum_projected_gradient'],
                    'bounds',fit['bound_indices'],'B unused phase',fit['selected_diagnostics'][0]['selected_unused_phase_rms_deg'],flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
