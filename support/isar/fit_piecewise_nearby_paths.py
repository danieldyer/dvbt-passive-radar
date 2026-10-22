"""Piecewise common-point controls on unchanged original flash observations.

Three predeclared two-second apertures compare independent B/E paths with two
common temporal paths. Independent A/C/D nuisance echoes remain. Long-fit
seeds are exactly reparameterized, not used as phase corrections. Neither
labels, smoothing nor good training residuals establish material-point identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from cluster_phase import SplinePath
from component_phase import component_measurements
from fit_shared_echo_cluster import subset_prediction
from joint_path_information import paths as joint_information
from nearby_echo_control import load,ROOT,shared_model
from phase_window import rebase_cluster,window_band
from release_fixed_grid_phase_guide import path_information
from report_measured_scatterer_phases import serial
from smooth_echo_cluster import SmoothEchoCluster


def guarded_model(bands,control,curve,common):
    model=SmoothEchoCluster(bands,control['origins'],(1,3,4),curve,
        shared_components=(1,4) if common else (),origin_radius=2.5,jerk_std=control['jerk_std'])
    for c in range(len(bands)):
        for s in (1,4):model.lower[10*c+2*s],model.upper[10*c+2*s]=-2.5,2.5
        for s in (0,2):
            ix=model.local_path_indices[c][s];limit=np.array([6*curve.scale,3*curve.scale**2])
            model.lower[ix],model.upper[ix]=-limit,limit
    return model


def restrict(source,parameters,target,channels=None):
    """Map every local response; shared coordinates must agree numerically."""
    parameters=np.asarray(parameters);result=np.full(target.count,np.nan);records=[]
    channels=range(len(source.clusters)) if channels is None else channels
    for c,cluster,index in zip(channels,target.clusters,target.indices):
        q,detail=rebase_cluster(source.clusters[c],parameters[source.indices[c]],cluster)
        present=np.isfinite(result[index])
        np.testing.assert_allclose(result[index][present],q[present],rtol=0,atol=1e-9)
        result[index]=q;records.append(detail)
    if not np.isfinite(result).all():raise ValueError('Incomplete global coordinate map')
    return result,records


def acquire_phases(model,parameters,independent,parameters_by_band):
    p=np.asarray(parameters).copy()
    for c,(cluster,index,old,q) in enumerate(zip(model.clusters,model.indices,independent,parameters_by_band)):
        b=cluster.base;value=old.clusters[0].evaluate(np.asarray(q)[old.indices[0]]);k=len(b.knots)
        others=sum((value['design'][:,s*k:(s+1)*k]@value['amplitude'][s*k:(s+1)*k]
            for s in (0,2,3)),start=np.zeros(b.field.y.size,complex))
        residual=b.clean-b.nuisance.remove(others[b.selected]);phase=[10*c+3,10*c+9];p[phase]=0.
        response=cluster.response_blocks(p[index])
        h=np.column_stack([b.nuisance.remove(b.reduced(response[s][0][:,:,None])[:,0][b.selected]) for s in (1,4)])
        p[phase]=np.angle(np.linalg.lstsq(h,residual,rcond=None)[0])
    return p


def measured(model,p):
    records=[];ratios=[]
    for c,(cluster,index) in enumerate(zip(model.clusters,model.indices)):
        rec=component_measurements(cluster,np.asarray(p)[index]);b=cluster.base
        pair=np.array([rec[s]['frozen_component_prediction'][b.training] for s in (1,4)])
        ratios.append(float(np.sum(abs(pair)**2)/max(np.sum(abs(pair.sum(axis=0))**2),1e-300)))
        records.append({str(s):dict(time_s=rec[s]['time_s'],physical_path_change_m=rec[s]['physical_path_change_m'],
            complex_coefficient_real_imag=np.c_[rec[s]['complex_coefficient'].real,rec[s]['complex_coefficient'].imag],
            conditional_information=rec[s]['conditional_information']) for s in (1,4)})
    return records,ratios


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    if out.exists():raise ValueError('Use a fresh output directory')
    control=load();source=ROOT/'joint-main-return-BE/fits.json';saved=json.loads(source.read_text())
    if not saved['complete']:raise ValueError('Wait for completed long common-path controls')
    long=shared_model(control);long_fit=saved['trials'][saved['best_index']];long_p=np.asarray(long_fit['parameters'])
    r,_=long.evaluate(long_p)
    if abs(r@r*long.energy-long_fit['cost'])>1e-7:raise ValueError('Long joint objective changed')
    out.mkdir(parents=True);started=time.monotonic()
    files=[source,*control['sources'],Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('phase_window','test_mixed_phase_window','smooth_echo_cluster','flexible_echo_cluster','shared_echo_cluster',
            'nearby_echo_control','component_phase','joint_path_information')]]
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        windows=[],jerk_std=control['jerk_std'],assignment='00002',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        policy='Only complete original bins wholly inside each requested aperture are included. Original row identities/known pulse kernels, C0/C1 reference, range covariance, brightness breakpoints and 125 ms train/validation/testing partition unchanged. Boundary bins remain in the original data, with no truncation or re-averaging. Independent fits use long independent and long common seeds. Common fits use long common and short ABC/Nine path seeds, with training-only constant-phase acquisition. Starts select training cost. Unused samples have been inspected previously and are exploratory prediction diagnostics.',
        guards='Both models use the same original observed candidate centres with 2.5 sample nuisance envelope guards; B/E stay inside ±2.5 samples of fixed range reference. Compared with long controls, A/C/D envelope guards expand by 0.5 samples to avoid inherited C guard contact masquerading as common-point mismatch. Private quadratic nuisance paths allow ±6 m/s and ±6 m/s²; no guard supplies data information. The two original Ten C coefficients remain fixed after exact time-coordinate transformation.',
        time_gauge='Known interband increments retain the old 165 s gauge. Each new path basis removes its own midpoint constant; envelope and carrier intercept both absorb that constant. Actual original-row complex responses are checked before fitting.')
    snapshot=out/'sources';snapshot.mkdir()
    for p in files:
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    def save():
        result['seconds']=time.monotonic()-started
        (out/'fits.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for first,last in ((162.,164.),(163.,165.),(164.,166.)):
        bands=[];parts=[];rows=[]
        for c,b in enumerate(control['bands']):
            short,part,row=window_band(b,first,last,complete_rows=True);keep=np.isin(b.field.time,short.field.time)
            np.testing.assert_array_equal(part,control['parts'][c][keep])
            np.testing.assert_array_equal(short.field.data['value'],b.field.data['value'][keep])
            np.testing.assert_allclose(short.field.y,b.field.y[keep],rtol=2e-14,atol=2e-12)
            bands.append(short);parts.append(part);rows.append(row)
        curve=SplinePath(first,last,1.);models=[guarded_model([b],control,curve,False) for b in bands]
        common=guarded_model(bands,control,curve,True)
        record=dict(interval_s=[first,last],knot_spacing_s=1.,curve_degree=curve.degree,independent=[],common=[],
            original_row_counts=[len(b.field.data['row_time']) for b in bands],response_reparameterization=[])
        result['windows'].append(record);save();chosen=[]
        for c,model in enumerate(models):
            old=control['independent_models'][c];d=control['saved'][c];old_p=np.asarray(d['trials'][d['best_index']]['parameters'])
            a,detail_a=restrict(old,old_p,model);b,detail_b=restrict(long,long_p,model,[c])
            record['response_reparameterization'].extend([detail_a,detail_b]);frozen=model.local_path_indices[0][2] if c==3 else []
            free=np.setdiff1d(np.arange(model.count),frozen);entry=dict(channel=c,trials=[]);record['independent'].append(entry)
            for label,seed in (('long independent',a),('long common',b)):
                r,j=model.evaluate(seed);direction=1e-4*np.cos(np.arange(model.count));direction[frozen]=0.;step=1e-4
                numerical=(model.evaluate(seed+step*direction)[0]-model.evaluate(seed-step*direction)[0])/(2*step)
                error=float(np.linalg.norm(numerical-j@direction)/max(np.linalg.norm(numerical),1e-15))
                if error>4e-4:raise ValueError('Short original-row derivative failed')
                fit=model.fit(seed,180,free_indices=free);fit.update(seed=label,derivative_error=error,
                    prior_cost=model.prior_cost(np.asarray(fit['parameters'])),prediction=subset_prediction(model,fit['parameters'],[parts[c]]))
                entry['trials'].append(fit);save()
                print(first,last,'independent',c,label,'cost',fit['cost'],'gradient',fit['maximum_projected_gradient'],flush=True)
            entry['best_index']=int(np.argmin([f['cost'] for f in entry['trials']]));p=np.asarray(entry['trials'][entry['best_index']]['parameters']);chosen.append(p)
            entry['measured'],entry['local_B_E_component_to_sum_energy_ratio']=measured(model,p)
            entry['information']={str(s):path_information(model,p,component=s) for s in (1,4)};save()
        direct,detail=restrict(long,long_p,common);record['response_reparameterization'].append(detail)
        seeds=[('long common',direct)]
        for reference in (4,2):
            q=np.zeros(common.count)
            for c,(m,p) in enumerate(zip(models,chosen)):
                q[10*c:10*(c+1)]=p[:10]
                for s in (0,2,3):q[common.local_path_indices[c][s]]=p[m.local_path_indices[0][s]]
            for s in (1,4):q[common.local_path_indices[0][s]]=chosen[reference][models[reference].local_path_indices[0][s]]
            seeds.append((f'short band {reference} paths; initialization only',acquire_phases(common,q,models,chosen)))
        frozen=common.local_path_indices[3][2];free=np.setdiff1d(np.arange(common.count),frozen)
        for label,seed in seeds:
            # Ten C's physical nuisance curve is the same in both seed families.
            np.testing.assert_allclose(seed[frozen],direct[frozen],rtol=0,atol=1e-9)
            fit=common.fit(seed,220,free_indices=free);p=np.asarray(fit['parameters'])
            fit.update(seed=label,prior_cost=common.prior_cost(p),prediction=subset_prediction(common,p,parts))
            record['common'].append(fit);save()
            print(first,last,'common',label,'cost',fit['cost'],'gradient',fit['maximum_projected_gradient'],flush=True)
        record['best_common_index']=int(np.argmin([f['cost'] for f in record['common']]))
        fit=record['common'][record['best_common_index']];p=np.asarray(fit['parameters'])
        record['common_measured'],record['common_local_B_E_component_to_sum_energy_ratio']=measured(common,p)
        record['common_information']=joint_information(common,p)
        baseline=sum(e['trials'][e['best_index']]['prediction']['testing'][0]['heldout_error'] for e in record['independent'])
        total=sum(v['heldout_error'] for v in fit['prediction']['testing'])
        record['total_unused_error_change_percent']=100*(total/baseline-1);record['complete']=True;save()
        print('window complete',first,last,'unused percent',record['total_unused_error_change_percent'],flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
