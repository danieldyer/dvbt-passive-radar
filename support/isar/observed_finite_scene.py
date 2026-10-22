"""Compare finite-symbol and pulse templates on original recorded profile rows.

Motion and scene support stay fixed. Alternate sparse blocks fit and predict
complex amplitudes; this is conditional forward-model validation, not a new
independent validation of the already fitted motion or target discovery.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_triangular

from captured_response import captured_rows, observations
from finite_reference import FS,NFFT,coefficient_planes,covariance,echo,probes
from finite_scene_information import probe_groups
from finite_scene_report import load
from geographic_movie import state_at
from imaging import corrected_sites,sites
from motion import C,excess_path
from range_fit import Pulse
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from symbol_moments import moments


def predict(y,target,stationary,cancellation,selected,*,finite_covariance):
    """GLS of fixed scene amplitudes, with Gaussian conditioning across time."""
    train,test=np.flatnonzero(selected),np.flatnonzero(~selected)
    b=cancellation[train]
    lower=np.linalg.cholesky(np.eye(len(train))+b@b.conj().T) if finite_covariance else np.eye(len(train))
    ytrain=solve_triangular(lower,y[train],lower=True)
    a=solve_triangular(lower,target[train],lower=True)
    s=solve_triangular(lower,stationary[train],lower=True)
    if s.shape[1]:
        u,singular,vh=np.linalg.svd(s,full_matrices=False)
        keep=singular>singular[0]*1e-10
        q=u[:,keep];inverse=vh[keep].conj().T/singular[keep]
        projected_a=a-q@(q.conj().T@a);projected_y=ytrain-q@(q.conj().T@ytrain)
    else:
        q=np.zeros((len(train),0),complex);inverse=np.zeros((0,0),complex)
        projected_a,projected_y=a,ytrain
    u,sv,vh=np.linalg.svd(projected_a,full_matrices=False)
    if sv[-1]<=sv[0]*1e-12:
        raise ValueError('Moving-scene inverse is unresolved')
    amplitude=(vh.conj().T/sv)@(u.conj().T@projected_y)
    beta=inverse@(q.conj().T@(ytrain-a@amplitude))
    residual=y[train]-target[train]@amplitude-stationary[train]@beta
    posterior=np.zeros(cancellation.shape[1],complex)
    if finite_covariance:
        white=solve_triangular(lower,residual,lower=True)
        solved=solve_triangular(lower.conj().T,white,lower=False)
        posterior=b.conj().T@solved
    predicted=target[test]@amplitude+stationary[test]@beta+cancellation[test]@posterior
    return {'prediction_error':float(np.linalg.norm(y[test]-predicted)**2),
        'withheld_energy':float(np.linalg.norm(y[test])**2),'stationary_rank':q.shape[1],
        'moving_condition':float(sv[0]/sv[-1]),'training_residual':float(np.linalg.norm(residual)**2)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','index','motion','fields','cancellation','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--channels',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--implementation-archive',type=Path,nargs='*',default=[])
    args=p.parse_args();started=time.monotonic()
    if args.output.exists():raise ValueError('Use a new experiment directory')
    original,_=load(args.cancellation/'information.json',args.implementation_archive)
    if original['source_sha256'].get(str(args.motion))!=fingerprint(args.motion):
        raise ValueError('Keep the same motion as the exact cancellation audit')
    model=json.loads(args.motion.read_text());curve=SplineMotion(model)
    nominal_tx,nominal_rx,_=sites(Path(model['geometry_directory']))
    tx,rx=corrected_sites(nominal_tx,nominal_rx,model.get('antenna_fit'))
    grouped=defaultdict(list)
    for row in probes(args.input):
        if row['channel'] in args.channels and model['interval_s'][0]<=row['time_s']<model['interval_s'][1]:
            grouped[row['channel']].append({k:v for k,v in row.items() if k not in ('measured','zero','one')})
    files=[args.input,args.index,args.motion,args.cancellation/'information.json',Path(__file__),
           *[Path(__file__).with_name(n) for n in ('captured_response.py','finite_reference.py','field.py','symbol_moments.py')]]
    result={'complete':False,'scope':__doc__,'source_sha256':{str(f):fingerprint(f) for f in files},
            'records':[],'motion_refitted':False,'phase_centre_calibration_accepted':False}
    args.output.mkdir(parents=True);offsets=np.arange(-8,9)
    for c in sorted(grouped):
        groups,rejected=probe_groups(grouped[c],16);rows=[r for g in groups for r in g]
        times=np.array([r['time_s'] for r in rows])
        saved=captured_rows(args.index,c,times,NFFT/FS*(1+1/rows[0]['guard']))
        field_path=args.fields/f'channel-{c}.npz';result['source_sha256'][str(field_path)]=fingerprint(field_path)
        with np.load(field_path) as field:
            epochs=field['row_time'];idx=np.clip(np.searchsorted(epochs,times),1,len(epochs)-1)
            idx-=abs(epochs[idx-1]-times)<abs(epochs[idx]-times)
            if np.max(abs(epochs[idx]-times))>2e-8:raise ValueError('Original observation epochs changed')
            centre_path=field['centre_path_m'][idx];centre_omega=field['centre_omega'][idx]
            mean_pulse=Pulse(field['pulse_samples'])
        coordinate=centre_path*FS/C
        raw_y=observations(saved,coordinate,centre_omega,offsets)
        state=state_at(None,model,curve,nominal_tx,nominal_rx,times)
        points=np.asarray(model['selected_points_body_m'][c])
        position=state['position_ecef_m'][:,None,:]+np.einsum('nij,pj->npi',state['body_to_ecef'],points)
        velocity=state['velocity_ecef_mps'][:,None,:]+np.einsum('nij,pj->npi',state['frame_velocity'],points)
        path,gradient=excess_path(position.reshape(-1,3),tx[c],rx);path=path.reshape(len(rows),len(points))
        rate=np.sum(gradient.reshape(position.shape)*velocity,axis=-1)
        delay=(path+model['range_bias_m'][c])*FS/C;rf=rows[0]['rf_hz'];k=2*np.pi*rf/C
        omega=-k*rate*NFFT/FS;m0,m1,_,_=moments(omega)
        moving_factor=(m0-1j*centre_omega[:,None]*m1)*np.exp(-1j*k*path)
        static=np.arange(np.floor(coordinate.min())+offsets.min()-5,np.ceil(coordinate.max())+offsets.max()+6)
        data=[];stationary={name:[] for name in ('mean','known','finite')};moving={name:[] for name in ('mean','known')}
        for block in range(len(groups)):
            sl=slice(block*16,(block+1)*16);info=np.array([s['information'] for s in saved[sl]])
            coefficient=info/info.sum()*np.exp(1j*k*centre_path[sl])
            spectral=sum(abs(s['response'])**2*abs(r['current'])**2/s['variance'] for r,s in zip(rows[sl],saved[sl]))/info.sum()
            packed=np.zeros(NFFT,complex);packed[np.arange(-3408,3409)%NFFT]=spectral
            known_pulse=Pulse(np.fft.ifft(packed)[np.arange(-64,65)%NFFT])
            noise=[];exact_static=[]
            for i in range(sl.start,sl.stop):
                r,s=rows[i],saved[i];origin=s['reference_origin']-s['source_offset']
                previous=s['reference_origin']-s['previous_source_offset']
                planes=coefficient_planes(r['current'],r['previous'],s['response'],s['variance'],origin,
                    coordinate[i],offsets,guard=r['guard'],previous_fraction=previous-np.floor(previous))
                a=planes[0]-1j*centre_omega[i]*planes[1];noise.append(covariance(a,s['variance']))
                q=np.column_stack([echo(r['current'],r['previous'],s['response'],origin,d,0,guard=r['guard']) for d in static])
                exact_static.append(a@q)
            lower=np.linalg.cholesky(np.einsum('n,nij->ij',(info/info.sum())**2,noise))
            def observed(values):return solve_triangular(lower,np.einsum('n,n...->...',coefficient,values),lower=True)
            data.append(observed(raw_y[sl]));stationary['finite'].append(observed(exact_static))
            for name,pulse in [('mean',mean_pulse),('known',known_pulse)]:
                target=pulse(coordinate[sl,None,None]+offsets[None,:,None]-delay[sl,None,:])*moving_factor[sl,None,:]
                nuisance=pulse(coordinate[sl,None,None]+offsets[None,:,None]-static[None,None,:])
                moving[name].append(observed(target));stationary[name].append(observed(nuisance))
        file=args.cancellation/f'channel-{c}.npz'
        with np.load(file) as cache:
            np.testing.assert_array_equal(cache['times_s'],times)
            moving['finite']=cache['design'].copy();b=cache['cancellation_factor'].copy()
        result['source_sha256'][str(file)]=fingerprint(file)
        moving={n:np.concatenate(a) if isinstance(a,list) else a for n,a in moving.items()}
        stationary={n:np.concatenate(a) for n,a in stationary.items()};y=np.concatenate(data)
        checks=[]
        for name in ('mean','known','finite'):
            for remove in (False,True):
                nuisance=stationary[name] if remove else np.zeros((len(y),0),complex)
                for finite in (False,True):
                    for parity in (0,1):
                        selected=np.repeat(np.arange(len(groups))%2==parity,len(offsets))
                        score=predict(y,moving[name],nuisance,b,selected,finite_covariance=finite)
                        checks.append({'template':name,'unknown_stationary':remove,'finite_covariance':finite,'parity':parity,**score})
        destination=args.output/f'channel-{c}.npz'
        np.savez(destination,y=y,**{'moving_'+n:a for n,a in moving.items()},**{'stationary_'+n:a for n,a in stationary.items()},
                 times_s=times,static_delay_samples=static)
        result['records'].append({'channel':c,'rows':len(rows),'blocks':len(groups),'checks':checks,
            'arrays':str(destination),'arrays_sha256':fingerprint(destination)})
        (args.output/'experiment.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'channel':c,'checks':checks,'seconds':time.monotonic()-started}),flush=True)
    result.update(complete=True,seconds=time.monotonic()-started)
    (args.output/'experiment.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
