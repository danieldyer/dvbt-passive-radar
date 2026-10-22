"""Test cross-band point association on the unchanged piecewise observations.

B/E labels are only acquisition proposals. One Cartesian point drives primary
delay, phase and Doppler; every other echo keeps its independent trajectory.
No per-band envelope correction, moving phase screen or reflected ray enters.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import least_squares

from ais import read_csv
from cluster_phase import SplinePath
from component_phase import physical_primary_measurements
from fit_piecewise_nearby_paths import guarded_model
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH
from motion import C
from nearby_echo_control import load,ROOT
from phase_window import window_band
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_echo_cluster import PhysicalEchoCluster
from physical_pair import fit_parameters
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options


def assignment_code(assignment):
    return ''.join({'12013':'0','12015':'1','12014':'2'}[a['site_id']] for a in assignment)


def geometry_for(control,assignment,midpoint,track,*,ais_reference=None):
    geo=json.loads((control['geometry']/'geometry-prior.json').read_text())
    receiver=np.asarray(geo['receiver_ecef_m']);rotation=np.asarray(geo['ecef_to_enu'])
    # These scales condition acquisition summaries only. They are NOT data
    # covariance and never enter the original-complex-observation likelihood.
    covariance=np.tile(np.diag([(C/8e6)**2,1.,1.]),(5,1,1))
    g=Flash(np.zeros((5,3)),covariance,track,EPOCH+midpoint,rotation,receiver,assignment,
        free_antennas=True,receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.),ais_reference=ais_reference)
    # Generous scatterer position support; no unobserved AIS antenna offset is
    # asserted. The sphere is only a cached water-height datum, not multipath.
    g.lower[18:20],g.upper[18:20]=-350/150,350/150
    datum=json.loads(Path('docs/data/isar_20261016/flash-reflection/analysis.json').read_text())['water']
    position=receiver+g.state(np.zeros(g.count))[0]@rotation
    vector=position-np.asarray(datum['centre_ecef_m'])
    height=np.linalg.norm(vector)-datum['radius_m']
    sensitivity=20*np.dot(vector/np.linalg.norm(vector),rotation[2])
    g.lower[20]=max(g.lower[20],-height/sensitivity)
    g.upper[20]=min(g.upper[20],(65-height)/sensitivity)
    return g


def make_model(bands,source_models,control,g,midpoint):
    primary=[PhysicalAmplitudeArc(b.field,b.training,b.knots,g,c,midpoint,stationary_design=b.design)
        for c,b in enumerate(bands)]
    # Both B/E have exactly the same flexible arc. The fourth nuisance can
    # therefore start from either label without changing the model family.
    arcs=[tuple(m.clusters[0].arcs[s] for s in (0,2,3,4)) for m in source_models]
    model=PhysicalEchoCluster(primary,bands,[control['origins'][0],control['origins'][2],control['origins'][3],0.],
        origin_radius=2.5,nuisance_arcs=arcs,nuisance_jerk_std=control['jerk_std'])
    for cluster,index in zip(model.clusters,model.indices):
        for s in (0,1):
            source=cluster.nuisance_arcs[s];curve=source.base.curve
            if curve.degree!=2:raise ValueError('Expected existing quadratic A/C nuisance')
    return model


def source_observables(source,p,midpoint):
    """Absolute excess path, rate and acceleration, for proposal generation."""
    cluster=source.clusters[0];b=cluster.base;f=b.field;rows=f.data['row_time']
    local=np.asarray(p)[source.indices[0]];values=[]
    for s in (1,4):
        arc=cluster.arcs[s];q=local[cluster.slices[s]];curve=arc.base.curve
        path=arc.known_path+curve(rows)@q[2:]
        rate=arc.known_rate+curve(rows,1)@q[2:]-f.data['centre_omega']/(f.k*.001024)
        acceleration=np.gradient(arc.known_rate,rows)+curve(rows,2)@q[2:]
        values.append([np.interp(midpoint,rows,f.data['centre_path_m']+path)+q[0]*C/8e6,
            np.interp(midpoint,rows,rate),np.interp(midpoint,rows,acceleration)])
    return np.asarray(values)


def geometry_proposal(g,observed):
    g.measured=np.asarray(observed)
    free=np.arange(18,27)  # AIS lag is redundant with free Cartesian p/v/a.
    def residual(x):
        p=np.zeros(g.count,dtype=np.asarray(x).dtype);p[free]=x
        return np.r_[g.data_residual(p),x/np.sqrt(2.)]
    fit=least_squares(residual,np.clip(np.zeros(len(free)),g.lower[free]+1e-9,g.upper[free]-1e-9),
        jac='cs',bounds=(g.lower[free],g.upper[free]),max_nfev=100,
        ftol=1e-11,xtol=1e-11,gtol=1e-9)
    p=np.zeros(g.count);p[free]=fit.x
    return p,float(fit.fun@fit.fun)


def association_seed(model,source_models,source_parameters,choices,geometry_parameters):
    """Preserve each nuisance response exactly under a label permutation."""
    p=np.zeros(model.count);p[:model.geometry.count]=geometry_parameters;errors=[]
    for c,(cluster,index,source,sp,chosen) in enumerate(zip(model.clusters,model.indices,source_models,source_parameters,choices)):
        old=source.clusters[0];local=np.asarray(sp)[source.indices[0]]
        primary=1 if chosen==0 else 4;other=4 if primary==1 else 1
        p[model.geometry.count+c]=local[old.slices[primary]][1]
        for nuisance,(block,s) in enumerate(zip(cluster.nuisance_slices,(0,2,3,other))):
            q=local[old.slices[s]];p[index[block]]=q
            before,_=old.arcs[s].response(q)
            after,_=cluster.nuisance_arcs[nuisance].response(q)
            error=float(np.max(abs(before-after))/max(np.max(abs(before)),1e-30));errors.append(error)
    if max(errors)>1e-12:raise ValueError('Nuisance response changed during association initialization')
    if np.any(p<model.lower) or np.any(p>model.upper):raise ValueError('Initializer outside declared model guards')
    return p,max(errors)


def acquire_constant_phases(model,parameters):
    """Cold joint complex acquisition on training rows; no fitted amplitude used."""
    p=np.asarray(parameters).copy();records=[]
    for c,(cluster,index) in enumerate(zip(model.clusters,model.indices)):
        b=cluster.base;phase=np.r_[model.geometry.count+c,[index[s][1] for s in cluster.nuisance_slices]]
        p[phase]=0.
        matrix=np.column_stack([b.nuisance.remove(b.reduced(raw[:,:,None])[:,0][b.selected])
            for raw,_ in cluster.response_blocks(p[index])])
        coefficient=np.linalg.lstsq(matrix,b.clean,rcond=None)[0]
        p[phase]=np.angle(coefficient)
        records.append(dict(constant_complex_coefficients_real_imag=np.c_[coefficient.real,coefficient.imag],
            acquired_training_squared_error=float(np.sum(abs(b.clean-matrix@coefficient)**2))))
    return p,records


def diagnostics(model,p):
    records=[]
    for cluster,index in zip(model.clusters,model.indices):
        value=cluster.evaluate(p[index]);b=cluster.base;k=len(b.knots)
        h=b.nuisance.remove(value['design'][b.selected])[:,value['amplitude']>0]
        singular=np.linalg.svd(np.r_[h.real,h.imag],compute_uv=False)
        parts=np.stack([value['design'][:,s*k:(s+1)*k]@value['amplitude'][s*k:(s+1)*k]
            for s in range(cluster.components)])[:,b.selected]
        records.append(dict(active_brightness_condition=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>0 else None,
            component_to_sum_energy_ratio=float(np.sum(abs(parts)**2)/max(np.sum(abs(parts.sum(axis=0))**2),1e-300)),
            primary_training_energy=float(np.sum(abs(parts[0])**2)),
            primary_amplitudes=value['amplitude'][:k]))
    return records


def measured(model,p):
    result=[]
    for cluster,index in zip(model.clusters,model.indices):
        rec=physical_primary_measurements(cluster,p[index]);z=rec['complex_coefficient']
        result.append(dict(time_s=rec['time_s'],physical_path_change_m=rec['physical_path_change_m'],
            complex_coefficient_real_imag=np.c_[z.real,z.imag],conditional_information=rec['conditional_information']))
    return result


def prepare_window(control,window):
    first,last=window['interval_s'];curve=SplinePath(first,last,1.);bands=[];parts=[];models=[];parameters=[];fits=[]
    for c,(b,entry) in enumerate(zip(control['bands'],window['independent'])):
        short,part,_=window_band(b,first,last,complete_rows=True);bands.append(short);parts.append(part)
        model=guarded_model([short],control,curve,False);models.append(model)
        fit=entry['trials'][entry['best_index']];p=np.asarray(fit['parameters']);parameters.append(p);fits.append(fit)
        r,_=model.evaluate(p)
        if abs(r@r*model.energy-fit['cost'])>1e-7:raise ValueError('Source objective changed')
    return bands,parts,models,parameters,fits


def free_parameters(model,source_models,*,nuisance_free):
    n=model.geometry.count
    free=np.r_[np.arange(18,27),np.arange(n,model.primary_count)]
    if nuisance_free:
        free=np.r_[free,np.arange(model.primary_count,model.count)]
        # Preserve the exact same formerly fixed Ten C trajectory control.
        frozen=model.indices[3][model.clusters[3].nuisance_slices[1]][2:]
        free=np.setdiff1d(free,frozen)
    return free


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'piecewise-nearby-common-paths/fits.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--windows',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--assignments',nargs='+',default=['00002'])
    parser.add_argument('--finalists',type=int,default=2)
    parser.add_argument('--max-evaluations',type=int,default=180)
    args=parser.parse_args();out=args.output
    if out.exists():raise ValueError('Use a fresh output directory')
    source=json.loads(args.source.read_text());control=load()
    track_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(track_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    options=assignments(licensed_options(control['geometry']));out.mkdir(parents=True);snapshot=out/'sources';snapshot.mkdir()
    files=[args.source,track_path,Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('physical_echo_cluster','physical_amplitude_arc','physical_pair','component_phase','flash_geometry',
         'nearby_echo_control','fit_piecewise_nearby_paths','amplitude_pair','shared_echo_cluster','phase_window')],
        *[control['geometry']/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')]]
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,groups=[],
        source=str(args.source),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        assignment_policy='Only declared licensed assignments run here. No nominal assignment is accepted as calibrated.',
        association_policy='All 32 B/E label patterns initialize the SAME physical-primary model. Other A/C/D/nearby returns are independently moving nuisance components. Weak or absent primary brightness is allowed in every band.',
        prior_policy='One half standardized real geometry prior squared, counted once; independently moving spline nuisances receive the same white-jerk prior as the empirical controls. Fixed receiver horizontal coordinates. Antennas fixed during this association test. Primary position within ±350 m E/N of AIS and height 0–65 m above the cached local water-height datum. No heading, reflected ray or fitted band envelope offset.',
        gauge_policy='AIS delay fixed to zero while p/v/a are all free. Each band initially retains an independent constant phase. No time-dependent phase correction is applied to observations.',
        validation_policy='Original rows and bins, known waveform kernels, range noise metric, stationary observed-cell design, brightness knots and 125 ms train/test partition preserved. Source baselines and all association screening use training; unused predictions are exploratory controls on previously inspected data.')
    for p in files:
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    started=time.monotonic()
    def save():
        result['seconds']=time.monotonic()-started
        (out/'fits.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for window_index in args.windows:
        window=source['windows'][window_index];bands,parts,models,parameters,baseline=prepare_window(control,window)
        midpoint=np.mean(window['interval_s']);summary=np.array([source_observables(m,p,midpoint) for m,p in zip(models,parameters)])
        for code in args.assignments:
            assignment=next(a for a in options if assignment_code(a)==code);g=geometry_for(control,assignment,midpoint,track)
            model=make_model(bands,models,control,g,midpoint)
            group=dict(window_index=window_index,interval_s=window['interval_s'],assignment=code,
                independent_baseline_cost=sum(f['cost'] for f in baseline),
                independent_baseline_unused_error=sum(f['prediction']['testing'][0]['heldout_error'] for f in baseline),
                independent_baseline_projected_gradients=[f['maximum_projected_gradient'] for f in baseline],
                proposal_observables=summary,proposals=[],trials=[])
            result['groups'].append(group);save()
            for choices in itertools.product(range(2),repeat=5):
                geometry,summary_cost=geometry_proposal(g,summary[np.arange(5),choices])
                seed,remap_error=association_seed(model,models,parameters,choices,geometry)
                seed,acquisition=acquire_constant_phases(model,seed)
                r,_=model.evaluate(seed)
                group['proposals'].append(dict(labels=['B' if s==0 else 'E' for s in choices],parameters=seed,
                    training_cost=float(r@r*model.energy),acquisition=acquisition,
                    initialization_summary_cost=summary_cost,nuisance_response_remapping_error=remap_error))
            order=np.argsort([p['training_cost'] for p in group['proposals']]);group['initialization_order']=order;save()
            print(window['interval_s'],code,'proposals complete',[(group['proposals'][i]['labels'],round(group['proposals'][i]['training_cost'],3)) for i in order[:args.finalists]],flush=True)
            for index in order[:args.finalists]:
                proposal=group['proposals'][index];seed=np.asarray(proposal['parameters']);free=free_parameters(model,models,nuisance_free=True)
                r,j=model.evaluate(seed);direction=.0001*np.cos(np.arange(model.count));direction[np.setdiff1d(np.arange(model.count),free)]=0.;step=.001
                finite=(model.evaluate(seed+step*direction)[0]-model.evaluate(seed-step*direction)[0])/(2*step)
                error=float(np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-15))
                if error>4e-4:raise ValueError(f'Original-row derivative failed: {error}')
                trial=dict(proposal_index=int(index),initialization_labels=proposal['labels'],directional_derivative_error=error,
                    steps=[]);group['trials'].append(trial);save()
                for release in (False,True):
                    free=free_parameters(model,models,nuisance_free=release)
                    p,audit=fit_parameters(model,seed,free,max_evaluations=args.max_evaluations)
                    prediction=model.prediction(p,[part==3 for part in parts]);cost=model.objective(p)[0]*model.energy
                    trial['steps'].append(dict(nuisance_paths_free=release,parameters=p,audit=audit,cost=cost,
                        prior_cost=model.prior_cost(p),prediction=prediction,diagnostics=diagnostics(model,p),
                        primary_observables=g.prediction(p[:g.count]),measured_primary=measured(model,p),
                        unused_change_from_independent_percent=100*(sum(v['heldout_error'] for v in prediction)/group['independent_baseline_unused_error']-1)))
                    save();seed=p
                    print(window['interval_s'],code,proposal['labels'],'nuisance free',release,'cost',round(cost,4),
                        'gradient',audit['projected_gradient'],'unused%',trial['steps'][-1]['unused_change_from_independent_percent'],flush=True)
            group['best_index']=int(np.argmin([t['steps'][-1]['cost'] for t in group['trials']]))
            group['complete']=True;save()
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
