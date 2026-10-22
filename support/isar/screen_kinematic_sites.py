"""All 32 licensed site proposals from a local original-data likelihood.

Unlike an instantaneous quadratic-rate summary, this retains the measured
flexible temporal basis, its score and cross-coordinate coupling. It is an
initialization approximation only; candidates still require original-data fits.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from ais import read_csv
from cluster_phase import SplinePath
from fit_fixed_grid_phases import prepare_fixed
from flash_geometry import assignments
from flash_siting_report import EPOCH
from kinematic_point_arc import KinematicPointGeometry
from local_likelihood import compress
from report_measured_scatterer_phases import load_fields,serial
from ship_site_signature import licensed_options
from smooth_echo_cluster import SmoothEchoCluster


def code_for(assignment):return ''.join({'12013':'0','12015':'1','12014':'2'}[a['site_id']] for a in assignment)


def predictions(g,p,records):
    tx,rx=g.antennas_at(p);mid=g.prediction(p)[:,0];values=[];errors=[]
    for c,record in enumerate(records):
        position,_,_=g.trajectory(p,record['row_time']-165.)
        path=np.sqrt(np.sum((position-tx[c])**2,axis=1))+np.sqrt(np.sum((position-rx)**2,axis=1))-np.sqrt(np.sum((tx[c]-rx)**2))
        increment=path-mid[c]-record['known_path']
        coefficients=record['projection']@increment
        values.append(np.r_[mid[c]-record['reference_path_m'],coefficients])
        errors.append(np.max(abs(record['basis']@coefficients-increment)))
    return values,errors


def main():
    root=Path('docs/data/isar_20261017');out=root/'kinematic-sites-32';out.mkdir(exist_ok=True)
    output=out/'proposals.json'
    if output.exists():raise ValueError('Use a fresh output')
    source=root/'smooth-flash-path/fits.json';smooth=json.loads(source.read_text())
    empirical_path=root/'fixed-grid-multiple-phases/fits.json';old=json.loads(empirical_path.read_text())
    geometry=Path('docs/data/calibration_20261016_sites');geo=json.loads((geometry/'geometry-prior.json').read_text())
    rotation=np.asarray(geo['ecef_to_enu']);receiver=np.asarray(geo['receiver_ecef_m'])
    bands,_,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,163.,167.,stationary=True)
    curve=SplinePath(163.,167.,.5);std=smooth['validation_selected_jerk_std'];records=[]
    for c,b in enumerate(bands):
        model=SmoothEchoCluster([b],old['selected_origins_samples'],1,curve,shared=False,jerk_std=std)
        fit=smooth['independent'][c];p=np.asarray(fit['parameters']);cluster=model.clusters[0];local=p[model.indices[0]]
        value=cluster.evaluate(local);residual=value['residual']*np.sqrt(model.energy)
        jac=value['jacobian']*np.sqrt(model.energy)
        block=cluster.slices[1];selected=np.r_[block.start,np.arange(block.start+2,block.stop)]
        jac[:,selected[0]]*=8e6/299792458
        compact=compress(residual,jac,selected)
        measured=np.r_[local[block.start]*299792458/8e6,local[block.start+2:block.stop]]
        rows=b.field.data['row_time'];basis=curve(rows);q,r=np.linalg.qr(basis,mode='reduced')
        records.append(dict(channel=c,measured=measured,**compact,row_time=rows,known_path=b.known_path,
            basis=basis,projection=np.linalg.solve(r,q.T),reference_path_m=b.field.data['centre_path_m'][0]))
    ais_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(ais_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    water_path=Path('docs/data/isar_20261016/flash-reflection/analysis.json');water=json.loads(water_path.read_text())['water']
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        phase_summary='Flexible paths on original row epochs; original-data residual score retained. Exact profiled-residual Jacobians eliminate selected band phase and every neighbouring return parameter before local QR/SVD compression. No smoother prior enters the compressed data quadratic.',
        interpretation='Local Gauss-Newton likelihood for initialization only, not exact site evidence or a physical confidence bound. Original-data refits and transfer checks remain necessary. All 32 licensed assignments are screened under the same bounded kinematic model; transmitter heights are fixed to user priors.',
        compressed_records=[{k:v for k,v in r.items() if k not in ('basis','projection')} for r in records],
        groups=[],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source,empirical_path,ais_path,water_path,Path(__file__),Path('support/isar/local_likelihood.py'),Path('support/isar/kinematic_point_arc.py'))})
    started=time.perf_counter()
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for assignment in assignments(licensed_options(geometry)):
        g=KinematicPointGeometry(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,EPOCH+165.,rotation,receiver,assignment)
        point=receiver+g.state(np.zeros(g.count))[0]@rotation;vector=point-np.asarray(water['centre_ecef_m'])
        height=np.linalg.norm(vector)-water['radius_m'];sensitivity=20*np.dot(vector/np.linalg.norm(vector),rotation[2])
        g.lower[20]=max(g.lower[20],(8-height)/sensitivity);g.upper[20]=min(g.upper[20],(57-height)/sensitivity)
        group=dict(assignment=code_for(assignment),trials=[]);result['groups'].append(group);save()
        seed=np.zeros(g.count)
        for refine in (False,True):
            fixed=np.array([2,5,8,11,14,15,16]) if refine else np.arange(18)
            free=np.setdiff1d(np.arange(g.count),fixed);seed=np.clip(seed,g.lower+1e-10,g.upper-1e-10);seed[fixed]=0
            def unpack(x):
                p=seed.astype(np.result_type(x,float));p[free]=x;return p
            def residual(x):
                p=unpack(x);values,_=predictions(g,p,records)
                return np.r_[np.concatenate([r['offset']+r['matrix']@(v-r['measured']) for r,v in zip(records,values)]),p/np.sqrt(2.)]
            fitted=least_squares(residual,seed[free],jac='cs',bounds=(g.lower[free],g.upper[free]),
                x_scale='jac',ftol=1e-11,xtol=1e-11,gtol=1e-8,max_nfev=160)
            p=unpack(fitted.x);_,error=predictions(g,p,records)
            group['trials'].append(dict(refine_horizontal_antennas=refine,parameters=p,cost=fitted.fun@fitted.fun,
                optimality=fitted.optimality,evaluations=fitted.nfev,termination=str(fitted.message),
                bound_indices=free[(p[free]<g.lower[free]+1e-5)|(p[free]>g.upper[free]-1e-5)],
                maximum_path_basis_error_m=max(error),antenna_fit=g.antennas.describe(p[:18])))
            seed=p;save()
        print(group['assignment'],[round(t['cost'],2) for t in group['trials']],flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
