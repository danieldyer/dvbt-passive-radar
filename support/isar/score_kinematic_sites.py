"""Original-data scoring of every local site proposal with frozen paths.

Physical geometry and neighbouring trajectories are held fixed per proposal.
Target constant phase, all real positive brightness and the stationary profile
fit the original training rows; unused blocks predict without any refitting.
This validates the local screen, but does not replace full geometry refinement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import nnls

from ais import read_csv
from ais_point_arc import AisPointArc
from cluster_phase import SplinePath
from fit_fixed_grid_phases import prepare_fixed
from flash_geometry import assignments
from flash_siting_report import EPOCH
from kinematic_point_arc import KinematicPointGeometry
from phase_gram import fit as fit_phase
from report_measured_scatterer_phases import load_fields,serial
from ship_site_signature import licensed_options
from smooth_echo_cluster import SmoothEchoCluster


def code_for(assignment):return ''.join({'12013':'0','12015':'1','12014':'2'}[a['site_id']] for a in assignment)


def target_template(b,g,p,c):
    f=b.field;rows=f.data['row_time'];local,velocity,_=g.trajectory(p,rows-165.)
    reference=(f.data['centre_ecef_m']-g.receiver)@g.rotation.T
    offset=(local-reference)@g.rotation;position=f.data['centre_ecef_m']+offset
    tx,rx=g.antennas_at(p)
    raw=f.raw_trajectory(position,velocity@g.rotation,transmitter=g.receiver+tx[c]@g.rotation,
        receiver=g.receiver+rx@g.rotation,position_offset=offset)
    midpoint=g.prediction(p)[c,0];old=np.interp(165.,rows,f.data['centre_path_m'])
    return raw*np.exp(1j*f.k*(midpoint-old))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=Path('docs/data/isar_20261017');out=root/'kinematic-sites-exact-score';out.mkdir(exist_ok=True)
    target=out/'scores.json'
    if target.exists() and not args.resume:raise ValueError('Use a fresh output or explicit --resume')
    source=root/'kinematic-sites-32/proposals.json';proposals=json.loads(source.read_text())
    if not proposals['complete']:raise ValueError('Wait for every assignment proposal')
    smooth_path=root/'smooth-flash-path/fits.json';smooth=json.loads(smooth_path.read_text())
    old_path=root/'fixed-grid-multiple-phases/fits.json';old=json.loads(old_path.read_text())
    geometry=Path('docs/data/calibration_20261016_sites');geo=json.loads((geometry/'geometry-prior.json').read_text())
    rotation=np.asarray(geo['ecef_to_enu']);receiver=np.asarray(geo['receiver_ecef_m'])
    bands,partition,_=prepare_fixed(load_fields(Path('build/isar-20261015/full-fields'),geometry),geometry,163.,167.,stationary=True)
    curve=SplinePath(163.,167.,.5);records=[]
    for c,b in enumerate(bands):
        model=SmoothEchoCluster([b],old['selected_origins_samples'],1,curve,shared=False,jerk_std=smooth['validation_selected_jerk_std'])
        p=np.asarray(smooth['independent'][c]['parameters']);cluster=model.clusters[0]
        value=cluster.evaluate(p[model.indices[0]]);k=len(b.knots)
        nuisance=np.concatenate([value['design'][:,s*k:(s+1)*k] for s in (0,2,3)],axis=1)
        records.append(dict(nuisance=nuisance,initial_phase=p[3]))
    ais_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(ais_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    choices=assignments(licensed_options(geometry))
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,groups=[],
        policy='Every proposed geometry uses the exact original-row ray, C0/C1 Doppler, averaging and range covariance. A/C/D trajectories/phase remain frozen from the independent smooth-path controls. Target constant phase and all positive brightness knots refit jointly on training rows after the shared stationary projection. Both stationary and brightness coefficients are frozen on validation/unused blocks. A 32-point circular phase acquisition refines every detected grid minimum; this is a numerical acquisition, not a proof of a global geometric optimum.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source,smooth_path,old_path,Path(__file__),Path('support/isar/phase_gram.py'),Path('support/isar/kinematic_point_arc.py'))})
    if target.exists():
        previous=json.loads(target.read_text());current=result['source_sha256']
        for name,want in previous['source_sha256'].items():
            if name!=str(Path(__file__)) and current[name]!=want:raise ValueError('Resumption changes an input or numerical model')
        previous.setdefault('runner_history',[]).append(dict(sha256=previous['source_sha256'][str(Path(__file__))],
            snapshot=str(out/'sources/score_kinematic_sites.py'),change='Correct a runner variable shadowing ENU rotation after the first completed assignment; completed numerical work unchanged.'))
        previous['source_sha256']=current;result=previous
    started=time.perf_counter()
    def save():
        result['seconds']=time.perf_counter()-started;target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for number,proposal in enumerate(proposals['groups']):
        choice=next(a for a in choices if code_for(a)==proposal['assignment'])
        g=KinematicPointGeometry(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,EPOCH+165.,rotation,receiver,choice)
        group=next((g for g in result['groups'] if g['assignment']==proposal['assignment']),None)
        if group is None:group=dict(assignment=proposal['assignment'],trials=[]);result['groups'].append(group);save()
        for candidate_index,candidate in enumerate(proposal['trials']):
            if candidate_index<len(group['trials']):continue
            p=np.asarray(candidate['parameters']);channels=[]
            for c,(b,record,part) in enumerate(zip(bands,records,partition)):
                raw=target_template(b,g,p,c)
                if number==0 and not candidate['refine_horizontal_antennas']:
                    arc=AisPointArc(b.field,b.training,b.knots,g,c,165.,stationary_design=b.design)
                    expected,_=arc.response(np.r_[p,np.zeros(5)])
                    np.testing.assert_allclose(raw,expected,atol=2e-12,rtol=2e-12)
                primary=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])
                design=np.concatenate([primary,record['nuisance']],axis=1)
                projected=b.nuisance.remove(design[b.selected]);data=b.nuisance.remove(b.field.y.ravel()[b.selected])
                fitted=fit_phase(projected,data,len(b.knots),record['initial_phase'])
                phase_rotation=np.ones(design.shape[1],complex);phase_rotation[:len(b.knots)]=np.exp(1j*fitted['phase_rad'])
                design*=phase_rotation[None,:];amplitude=np.asarray(fitted['amplitude']);prediction=design@amplitude
                y=b.field.y.ravel();beta=b.nuisance.solve(y[b.selected]-prediction[b.selected]);static=b.design@beta
                training_error=float(np.sum(abs(y[b.selected]-static[b.selected]-prediction[b.selected])**2))
                if abs(training_error-fitted['cost'])>1e-7*max(training_error,1.):raise ValueError('Gram objective differs from original residual')
                if number==0:
                    matrix=projected*phase_rotation[None,:]
                    direct,_=nnls(np.r_[matrix.real,matrix.imag],np.r_[data.real,data.imag],maxiter=40*len(amplitude))
                    direct_error=np.linalg.norm(data-matrix@direct)**2
                    if abs(training_error-direct_error)>1e-7*max(direct_error,1.):raise ValueError('Gram NNLS differs from direct original-data NNLS')
                scores={}
                for label,value in (('validation',1),('testing',3)):
                    selected=np.repeat(part==value,len(b.field.offsets))
                    scores[label]=dict(error=float(np.sum(abs(y[selected]-static[selected]-prediction[selected])**2)))
                channels.append(dict(channel=c,phase_rad=fitted['phase_rad'],training_error=training_error,
                    numerical_rank=fitted['numerical_rank'],phase_starts=fitted['phase_starts'],
                    brightness_coefficients=amplitude,stationary_real=beta.real,stationary_imag=beta.imag,**scores))
            record=dict(refine_horizontal_antennas=candidate['refine_horizontal_antennas'],parameters=p,
                original_training_error=sum(c['training_error'] for c in channels),prior_cost=.5*p@p,
                validation_error=sum(c['validation']['error'] for c in channels),unused_error=sum(c['testing']['error'] for c in channels),channels=channels)
            group['trials'].append(record);save()
        print(group['assignment'],[(round(t['original_training_error'],2),round(t['unused_error'],2)) for t in group['trials']],flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
