"""Fit the flash primary physically with A/C/D as independent nuisance echoes.

Use each original row once over the complete interval. Licensed assignments
are screened only for initialization; declared finalists fit the original
complex data under fixed and refined antenna coordinates. No multipath or
additional stationary model is introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import least_squares

from ais import read_csv
from ais_point_arc import AisPointGeometry,AisPointArc
from fit_shared_echo_cluster import prepare
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_echo_cluster import PhysicalEchoCluster,EnvelopeOffsetArc
from report_measured_scatterer_phases import load_fields,serial
from report_shared_echo_cluster import conditional_observables
from shared_echo_cluster import SharedEchoCluster
from ship_site_signature import licensed_options
from motion import C


def assignment_code(assignment):
    return ''.join({'12013':'0','12015':'1','12014':'2'}[a['site_id']] for a in assignment)


def geometry_for(assignment,track,midpoint,receiver,rotation,measured,covariance,motion_model='quadratic'):
    if motion_model=='quadratic':
        g=Flash(measured,covariance,track,EPOCH+midpoint,rotation,receiver,assignment,
            free_antennas=True,receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.))
    else:
        g=AisPointGeometry(measured,covariance,track,EPOCH+midpoint,rotation,receiver,assignment,
            follow_course_change=motion_model=='ais-course')
    # This cached datum sets a height interval only. No reflected ray enters.
    datum=json.loads(Path('docs/data/isar_20261016/flash-reflection/analysis.json').read_text())['water']
    position=receiver+g.state(np.zeros(g.count))[0]@rotation
    vector=position-np.asarray(datum['centre_ecef_m']);height=np.linalg.norm(vector)-datum['radius_m']
    sensitivity=20*np.dot(vector/np.linalg.norm(vector),rotation[2])
    g.lower[20]=max(g.lower[20],(8-height)/sensitivity)
    g.upper[20]=min(g.upper[20],(57-height)/sensitivity)
    return g


def make_model(bands,g,midpoint,origins,envelope_bias=False):
    arc=AisPointArc if isinstance(g,AisPointGeometry) else PhysicalAmplitudeArc
    primary=[arc(b.field,b.training,b.knots,g,c,midpoint,stationary_paths=[])
        for c,b in enumerate(bands)]
    if envelope_bias:primary=[EnvelopeOffsetArc(b) for b in primary]
    return PhysicalEchoCluster(primary,bands,origins)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/shared-echo-cluster-flash-four-seconds/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/physical-four-echo-flash'))
    parser.add_argument('--assignments',nargs='+',default=['00002','10002','00000'])
    parser.add_argument('--max-evaluations',type=int,default=240)
    parser.add_argument('--motion-model',choices=['quadratic','ais-fixed','ais-course'],default='quadratic')
    parser.add_argument('--profile-envelope',action='store_true',help='Sensitivity test only: fit fixed per-band envelope offsets while preserving geometric carrier phase')
    args=parser.parse_args();source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Wait for completed empirical reference fits')
    out=args.output;out.mkdir(parents=True,exist_ok=True);output=out/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    geometry=Path('docs/data/calibration_20261016_sites')
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    first,last=source['interval_s'];midpoint=(first+last)/2
    bands,partition=prepare(fields,geometry,first,last,range_extent=source['range_extent_samples'])
    geo=json.loads((geometry/'geometry-prior.json').read_text())
    receiver=np.asarray(geo['receiver_ecef_m']);rotation=np.asarray(geo['ecef_to_enu'])
    track_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(track_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    choices=assignments(licensed_options(geometry));nominal=next(a for a in choices if assignment_code(a)=='00002')
    reference=Flash(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,EPOCH+midpoint,
        rotation,receiver,nominal,free_antennas=False)
    predicted=reference.prediction(np.zeros(reference.count));selected=source['labels'].index('B')
    nuisance=[s for s in range(len(source['labels'])) if s!=selected]
    origins=np.asarray(source['selected_origins_samples'])[nuisance]
    observed=[];covariance=[];independent=[]
    for c,entry in enumerate(source['independent']):
        best=entry['trials'][entry['best_index']];p=np.asarray(best['parameters']);independent.append(best)
        model=SharedEchoCluster([bands[c]],source['selected_origins_samples'],origin_radius=source['origin_radius_samples'])
        data=conditional_observables(model,p)
        if not data['identifiable']:raise ValueError('Independent source covariance is singular')
        increment=np.asarray(data['range_origin_m_rate_residual_mps_curvature_residual_mps2'])[selected]
        old=np.interp(midpoint,bands[c].field.data['row_time'],bands[c].field.data['centre_path_m'])
        observed.append([old+increment[0],predicted[c,1]+increment[1],predicted[c,2]+increment[2]])
        indices=np.arange(3*selected,3*selected+3)
        covariance.append(np.asarray(data['covariance'])[np.ix_(indices,indices)])
    observed=np.asarray(observed);covariance=np.asarray(covariance)
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,interval_s=[first,last],
        motion_model=args.motion_model,
        envelope_bias=args.profile_envelope,
        source=str(args.source),primary_label='B',nuisance_labels=[source['labels'][s] for s in nuisance],
        nuisance_origins_samples=origins,reference_independent_unused_error=sum(f['prediction']['testing'][0]['heldout_error'] for f in independent),
        measured_initialization_observables=observed,conditional_initialization_covariance=covariance,
        prior_policy='Proper-complex squared data residual plus half the real standardized geometry prior squared, once across all five bands. Approved receiver horizontal coordinates fixed; height varies within ±10 m. Primary height restricted to 8–57 m above the existing local height datum. No reflected propagation.',
        initialization_policy='All 32 licensed assignments get conditional summary proposals with antennas fixed. Those summaries provide starts only; final evidence uses original complex observations and frozen unused-block predictions.',
        validation_policy='125 ms blocks 0/2 train and 3 test. Each original retained bin and all its original rows occur once in the four-second likelihood. These previously inspected data provide internal controls, not a blind survey.',
        proposals=[],groups=[],source_sha256={str(args.source):hashlib.sha256(args.source.read_bytes()).hexdigest()})
    files=[Path(__file__),track_path,*[geometry/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')],
        *[Path('support/isar')/(n+'.py') for n in ('physical_echo_cluster','physical_amplitude_arc','ais_point_arc','piecewise_ais','amplitude_pair','amplitude_arc','field','flash_geometry','fit_shared_echo_cluster','shared_echo_cluster')]]
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for p in files:
        result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    started=time.perf_counter()
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for assignment in choices:
        g=geometry_for(assignment,track,midpoint,receiver,rotation,observed,covariance,args.motion_model)
        def residual(q):return np.r_[g.data_residual(np.r_[np.zeros(18),q]),q]
        fit=least_squares(residual,np.clip(np.zeros(g.count-18),g.lower[18:]+1e-9,g.upper[18:]-1e-9),
            jac='cs',bounds=(g.lower[18:],g.upper[18:]),max_nfev=180,ftol=1e-12,xtol=1e-12,gtol=1e-9)
        result['proposals'].append(dict(assignment=assignment_code(assignment),parameters=np.r_[np.zeros(18),fit.x],
            conditional_summary_cost=fit.fun@fit.fun,optimality=fit.optimality))
    save()
    for code in args.assignments:
        assignment=next(a for a in choices if assignment_code(a)==code)
        g=geometry_for(assignment,track,midpoint,receiver,rotation,observed,covariance,args.motion_model)
        model=make_model(bands,g,midpoint,origins,args.profile_envelope)
        proposal=next(p for p in result['proposals'] if p['assignment']==code)
        seed=np.r_[proposal['parameters'],np.zeros(model.primary_count-g.count),np.zeros(model.count-model.primary_count)]
        for c,(best,index) in enumerate(zip(independent,model.indices)):
            p=np.asarray(best['parameters']);n=len(source['labels'])
            local=np.column_stack((p[:2*n].reshape(n,2),p[2*n:].reshape(n,2)))
            seed[g.count+c]=local[selected,1]
            seed[index[model.primary_count:]]=local[nuisance].ravel()
            if args.profile_envelope:
                old=np.interp(midpoint,bands[c].field.data['row_time'],bands[c].field.data['centre_path_m'])
                geometric_origin=(g.prediction(seed[:g.count])[c,0]-old)*8e6/C
                seed[g.count+5+c]=local[selected,0]-geometric_origin
        group=dict(assignment=code,trials=[]);result['groups'].append(group);save()
        for free in (False,True):
            r,j=model.evaluate(seed);direction=np.cos(np.arange(model.count))*.01
            if not free:direction[:18]=0
            step=5e-4
            finite=(model.evaluate(seed+step*direction)[0]-model.evaluate(seed-step*direction)[0])/(2*step)
            error=np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-12)
            if error>4e-4:raise ValueError(f'Original-data derivative error {error}')
            print(code,'starting',free,'cost',r@r*model.energy,'derivative',error,flush=True)
            parameters,audit=model.fit(seed,free_antennas=free,max_evaluations=args.max_evaluations)
            prediction=model.prediction(parameters,[p==3 for p in partition])
            record=dict(free_antennas=free,parameters=parameters,audit=audit,cost=model.objective(parameters)[0]*model.energy,
                data_energy=model.energy,prior_cost=.5*parameters[:g.count]@parameters[:g.count],
                directional_derivative_error=error,prediction=prediction,
                primary_observables=g.prediction(parameters[:g.count]),antenna_fit=g.antennas.describe(parameters[:18]),
                primary_range_origin_m=g.prediction(parameters[:g.count])[:,0]-np.array([
                    np.interp(midpoint,b.field.data['row_time'],b.field.data['centre_path_m']) for b in bands]))
            if args.profile_envelope:record['envelope_offset_m']=parameters[g.count+5:g.count+10]*C/8e6
            group['trials'].append(record);save();seed=parameters
            print(code,'finished',free,'cost',record['cost'],'unused',sum(p['heldout_error'] for p in prediction),
                'gradient',audit['projected_gradient'],'bounds',audit['bound_indices'],flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
