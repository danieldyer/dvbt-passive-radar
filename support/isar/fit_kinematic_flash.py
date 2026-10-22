"""Physical multiband flash fit with kinematic E/N motion corrections.

This is an explicit fixed-height control: transmitter coordinates/heights use
the user priors; the refined case frees transmitter E/N and receiver height.
All envelope delay, RF phase and Doppler use the same physical point trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from ais import read_csv
from ais_point_arc import AisPointArc
from compare_ais_phase import path_geometry
from fit_fixed_grid_phases import prepare_fixed
from flash_geometry import assignments
from flash_siting_report import EPOCH
from kinematic_point_arc import KinematicPointGeometry
from physical_echo_cluster import PhysicalEchoCluster
from physical_pair import fit_parameters
from report_measured_scatterer_phases import load_fields,serial
from ship_site_signature import licensed_options
from cluster_phase import SplinePath


def code_for(assignment):return ''.join({'12013':'0','12015':'1','12014':'2'}[a['site_id']] for a in assignment)


def target_diagnostics(model,p,parts):
    records=[];g=model.geometry
    for c,(cluster,index,part) in enumerate(zip(model.clusters,model.indices,parts)):
        b=cluster.base;local=p[index];value=cluster.evaluate(local);k=len(b.knots)
        signals=np.stack([value['design'][:,s*k:(s+1)*k]@value['amplitude'][s*k:(s+1)*k] for s in range(4)])
        y=b.field.y.ravel();prediction=signals.sum(axis=0)
        beta=b.nuisance.solve(y[b.selected]-prediction[b.selected]);static=b.design@beta
        clean=(y-static-prediction+signals[0]).reshape(b.field.y.shape)
        raw,_=cluster.response_blocks(local)[0]
        unit=b.reduced(raw[:,:,None])[:,0].reshape(b.field.y.shape)
        phase=np.angle(np.sum(unit.conj()*clean,axis=1));weight=np.sum(abs(signals[0].reshape(b.field.y.shape))**2,axis=1)
        test=(part==3)&(weight>0);h=signals[0][b.selected]
        t=b.field.time;position,velocity,acceleration=g.trajectory(p[:g.count],t-165.)
        tx,rx=g.antennas_at(p[:g.count]);path,rate=path_geometry(position,velocity,tx[c],rx)
        records.append(dict(time_s=t,testing=part==3,phase_residual_deg=np.degrees(phase),
            unused_phase_rms_deg=np.degrees(np.sqrt(np.average(phase[test]**2,weights=weight[test]))),
            selected_signal_retained_energy=np.vdot(b.nuisance.remove(h),b.nuisance.remove(h)).real/np.vdot(h,h).real,
            physical_path_change_m=path-np.interp(165.,t,path),physical_path_rate_mps=rate,
            position_enu_m=position,velocity_enu_mps=velocity,acceleration_enu_mps2=acceleration))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignments',nargs='+',default=['00002','10002','00000'])
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/kinematic-physical-flash'))
    parser.add_argument('--proposal-file',type=Path,help='Frozen completed assignment proposals from the local likelihood screen')
    args=parser.parse_args()
    root=Path('docs/data/isar_20261017');out=args.output;out.mkdir(exist_ok=True)
    output=out/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    source=root/'smooth-flash-path/fits.json';smooth=json.loads(source.read_text())
    if not smooth['complete']:raise ValueError('Wait for completed continuity comparison')
    empirical_path=root/'fixed-grid-multiple-phases/fits.json';empirical=json.loads(empirical_path.read_text())
    old_physical_path=root/'ais-course-four-echo-flash/fits.json';old_physical=json.loads(old_physical_path.read_text())
    proposals=json.loads(args.proposal_file.read_text()) if args.proposal_file else None
    geometry=Path('docs/data/calibration_20261016_sites');geo=json.loads((geometry/'geometry-prior.json').read_text())
    rotation=np.asarray(geo['ecef_to_enu']);receiver=np.asarray(geo['receiver_ecef_m'])
    bands,partition,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,163.,167.,stationary=True)
    ais_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(ais_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    group=next(g for g in smooth['groups'] if g['jerk_std']==smooth['validation_selected_jerk_std'])
    shared=group['trials'][group['best_index']]
    common_curve=SplinePath(163.,167.,.5)
    degree=common_curve.degree
    common_coefficients=np.asarray(shared['parameters'])[40:40+degree]
    choices=assignments(licensed_options(geometry));water_path=Path('docs/data/isar_20261016/flash-reflection/analysis.json')
    water=json.loads(water_path.read_text())['water']
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        interval_s=[163.,167.],groups=[],
        geometry_policy='Approved receiver horizontal coordinates fixed; receiver height 0–20 m AGL. TX heights fixed to the user-supplied priors in this conditional control; refined trials free only their E/N coordinates, and receiver height. Explicitly requested site assignments are conditional controls, not an exhaustive site identification.',
        motion_policy='A physical E/N spline correction to coarse AIS, with fixed E/N lever arm, constant height and AIS time offset. Both horizontal components are free; no fitted heading/yaw, roll/pitch, phase screen or envelope bias. Gaussian velocity sigma 0.5 m/s, acceleration sigma 0.1 m/s² and white-jerk position scale 0.05 m/s^(5/2), counted once with the geometry prior.',
        nuisance_policy='A/C/D keep independent measured quadratic trajectories and positive brightness. A jointly fitted constant stationary profile uses actual fixed-grid cells with condition one; data and all templates share its exact projection. Every original row enters once, and unused 125 ms blocks predict with all coefficients frozen.',
        source_sha256={})
    files=[source,empirical_path,old_physical_path,ais_path,water_path,Path(__file__),
        *[geometry/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')],
        *[Path('support/isar')/(n+'.py') for n in ('kinematic_point_arc','ais_point_arc','physical_echo_cluster',
            'physical_pair','amplitude_pair','field','shared_echo_cluster','smooth_echo_cluster')]]
    if args.proposal_file:files.append(args.proposal_file)
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for path in files:
        result['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix=='.py':shutil.copyfile(path,snapshots/path.name)
    started=time.perf_counter()
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for code in args.assignments:
        assignment=next(a for a in choices if code_for(a)==code)
        g=KinematicPointGeometry(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,EPOCH+165.,rotation,receiver,assignment)
        point=receiver+g.state(np.zeros(g.count))[0]@rotation
        vector=point-np.asarray(water['centre_ecef_m']);height=np.linalg.norm(vector)-water['radius_m']
        sensitivity=20*np.dot(vector/np.linalg.norm(vector),rotation[2])
        g.lower[20]=max(g.lower[20],(8-height)/sensitivity);g.upper[20]=min(g.upper[20],(57-height)/sensitivity)
        primary=[AisPointArc(b.field,b.training,b.knots,g,c,165.,stationary_design=b.design) for c,b in enumerate(bands)]
        model=PhysicalEchoCluster(primary,bands,[-7.,6.,13.])
        old=next((a for a in old_physical['groups'] if a['assignment']==code),old_physical['groups'][0])['trials'][0]
        seed=np.zeros(model.count);seed[:22]=np.asarray(old['parameters'])[:22];seed[:18]=0
        # Initialize the physical radial component using the measured shared
        # path; the tangential spline remains free in the final likelihood.
        rows=bands[2].field.data['row_time'];dt=rows-165.
        pos,vel,_=g.trajectory(seed[:g.count],dt);tx,rx=g.antennas_at(seed[:g.count])
        path,_=path_geometry(pos,vel,tx[2],rx);midpath=g.prediction(seed[:g.count])[2,0]
        wanted=common_curve(rows)@common_coefficients-(path-midpath)
        centre=g.state(seed[:g.count])[0];gradient=(centre-tx[2])/np.linalg.norm(centre-tx[2])+(centre-rx)/np.linalg.norm(centre-rx)
        horizontal=gradient[:2];displacement=wanted[:,None]*horizontal/(horizontal@horizontal)
        coefficient=np.linalg.lstsq(g.curve(dt),displacement,rcond=None)[0]
        seed[22:g.count]=np.linalg.solve(g.correction_factor,coefficient).T.ravel()
        if proposals:
            proposal=next(p for p in proposals['groups'] if p['assignment']==code)
            seed[:g.count]=np.asarray(proposal['trials'][0]['parameters'])
        for c,entry in enumerate(empirical['independent']):
            fit=entry['trials'][entry['best_index']];p=np.asarray(fit['parameters'])
            local=np.column_stack([p[:8].reshape(4,2),p[8:].reshape(4,2)])
            seed[g.count+c]=local[1,1]
            seed[model.indices[c][model.primary_count:]]=local[[0,2,3]].ravel()
        record=dict(assignment=code,motion_prior=g.motion_prior,trials=[]);result['groups'].append(record);save()
        for free in (False,True):
            fixed=np.array([2,5,8,11,14,15,16]) if free else np.arange(18)
            indices=np.setdiff1d(np.arange(model.count),fixed)
            seed=np.clip(seed,model.lower+1e-10,model.upper-1e-10)
            # Exact zero for fixed coordinates; no Gaussian prior is confused
            # with a measured height or a finite-difference observation.
            seed[fixed]=0
            r,j=model.evaluate(seed);direction=np.cos(np.arange(model.count))*.002;direction[fixed]=0;step=.003
            finite=(model.evaluate(seed+step*direction)[0]-model.evaluate(seed-step*direction)[0])/(2*step)
            error=np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-15)
            if error>4e-4:raise ValueError(f'Original-data derivative mismatch {error}')
            print(code,'starting',free,'cost',r@r*model.energy,'derivative',error,flush=True)
            p,audit=fit_parameters(model,seed,indices,max_evaluations=180)
            value=dict(refine_horizontal_antennas=free,parameters=p,audit=audit,
                cost=model.objective(p)[0]*model.energy,prior_cost=.5*p[:g.count]@p[:g.count],
                prediction=model.prediction(p,[part==3 for part in partition]),target=target_diagnostics(model,p,partition),
                directional_derivative_error=error,antenna_fit=g.antennas.describe(p[:18]),
                midpoint_observables=g.prediction(p[:g.count]))
            record['trials'].append(value);save();seed=p
            print(code,'finished',free,'cost',value['cost'],'unused',sum(v['heldout_error'] for v in value['prediction']),
                'gradient',audit['projected_gradient'],'bounds',audit['bound_indices'],flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
