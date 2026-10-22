"""Shared physical short-arc point in an independently acquired range crop.

This conditional control keeps one physical planar position/velocity/
acceleration for all five bands, one constant phase per band, positive real
brightness and fixed-profile stationary nuisance. AIS supplies only the broad
Cartesian prior. Independent training-only chirps initialise range/rate; no
independent band phase history is applied to the observations.
"""
import argparse,json
from pathlib import Path
import numpy as np
from ais import read_csv
from flash_siting_report import EPOCH
from piecewise_ais import LocalAisReference
from recorded_injection_fields import load_fields
from fixed_range_field import stationary_profile_design
from fading_point_motion import PlanarGeometry,RadialGeometry,FadingPointArc,FadingPointFit
from recovery_experiment import SCALE
from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brightness',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--radial',action='store_true',help='Condition transverse position/velocity on AIS and fit radial corrections only')
    parser.add_argument('--radial-path-bound-m',type=float,default=900.,help='Maximum radial position correction measured in excess-path metres')
    args=parser.parse_args();source=json.loads((args.brightness/'analysis.json').read_text())
    first,last=source['interval_s'];mid=(first+last)/2
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    profiles=Path(source.get('profiles','build/injection-20261018/deterministic-baseline/profiles.bin'))
    observed=source.get('observed_range_samples',[0,24])
    fields,rotation,rx,tx,_,_=load_fields(profiles,
        offsets=np.arange(observed[0],observed[1]+1),fixed_epoch_s=mid,
        interval_s=(first,last) if 'profiles' in source else None)
    fields=[f.windowed(first,last) for f in fields]
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    reference=LocalAisReference(track,EPOCH+mid,rotation,rx)
    state=np.asarray([reference(0.,j) for j in range(3)])
    geometry=(RadialGeometry(rx,rotation,tx,state,position_bound_m=args.radial_path_bound_m) if args.radial else
        PlanarGeometry(rx,rotation,tx,state,sigma=[200.,200.,.5,.5,.05,.05]))
    keep=[np.floor((f.time-first)/.25).astype(int)%4!=1 for f in fields]
    arcs=[]
    for c,(f,train) in enumerate(zip(fields,keep)):
        knots=np.r_[f.data['row_time'][0],np.arange(first+source['knot_s'],last,source['knot_s']),f.data['row_time'][-1]]
        arcs.append(FadingPointArc(f,train,knots,geometry,c,mid,stationary_profile_design(f)))
    physical=geometry.coarse.copy();local_tx=(tx[2]-rx)@rotation.T
    g=sum((physical[0]-a)/np.linalg.norm(physical[0]-a) for a in (local_tx,np.zeros(3)))
    direction=np.r_[g[:2]/np.dot(g[:2],g[:2]),0.]
    values=np.array([v['fading']['parameters'] for v in source['channels']]);scale=(fields[0].time[-1]-fields[0].time[0])/2
    delay=float(np.median(values[:,0])*299792458./8e6)
    rate=float(np.median(values[:,2])/scale);acceleration=float(np.median(2*values[:,3])/scale**2)
    physical[0]+=direction*delay;physical[1]+=direction*(rate-g@physical[1]);physical[2]+=direction*acceleration
    if args.radial:
        initial=np.einsum('ij,j->i',physical-geometry.coarse,g)/geometry.sigma
    else:initial=(physical-geometry.coarse)[:,:2].ravel()/geometry.sigma
    lower,upper=geometry.bounds();initial=np.clip(initial,lower+.001,upper-.001)
    result=dict(scope=__doc__,complete=False,interval_s=[first,last],midpoint_s=mid,initial_parameters=initial,
        priors_sigma_enu=geometry.sigma,prior_mean_state_enu=geometry.coarse,geographic_localisation_certified=False,
        training=source['training'],kinematics='quadratic',transverse_motion_fixed_to_ais=args.radial,
        stationary_model='Constant vector on fixed observed range grid',profiles=str(profiles),
        radial_path_bound_m=args.radial_path_bound_m if args.radial else None,observed_range_samples=observed)
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    save();model=FadingPointFit(arcs);fit,_=model.fit([initial])
    result['fit']=dict(parameters=fit.x,objective=fit.fun,success=fit.success,message=str(fit.message),gradient=fit.jac,iterations=fit.nit,
        active_bounds=np.flatnonzero(np.isclose(fit.x[:geometry.count],lower,atol=1e-5)|np.isclose(fit.x[:geometry.count],upper,atol=1e-5)))
    result['prediction']=model.prediction(fit.x,[~v for v in keep])
    predicted=[b.original_prediction(fit.x) for b in arcs]
    physical=np.asarray(geometry.state(fit.x)[:3]);correction=(physical-geometry.coarse)[:,:2].ravel()
    phases=fit.x[geometry.count:]+np.array([f.k*(geometry.prediction(fit.x)[c,0]-b.reference_path) for c,(f,b) in enumerate(zip(fields,arcs))])
    gains=np.array([np.mean(a) for _,a,_ in predicted])*np.exp(1j*phases)
    render=dict(midpoint_s=mid,coarse_state_enu=geometry.coarse,interval_s=[first,last],motion_model='quadratic',
        fits=dict(constant_velocity_gls=dict(parameters=correction/SCALE,coefficients_real_imag=[[[z.real,z.imag]] for z in gains])),
        note='Shared physical short arc; training-only constant band phases and positive brightness; position remains prior constrained.')
    (out/'render-fit.json').write_text(json.dumps(serial(render),indent=2)+'\n')
    (out/'camera.json').write_text(json.dumps(dict(state_enu=physical.tolist(),
        marker_east_north_m=(state[0,:2]-physical[0,:2]).tolist(),marker_label='AIS antenna prior',
        viewport_label='fitted main-branch point'),indent=2)+'\n')
    result['physical_state_enu']=physical;result['complete']=True;save()
    print(json.dumps(serial({k:result[k] for k in ('fit','physical_state_enu')})),flush=True)
    print('Prediction',[v['test_complex_prediction_fraction'] for v in result['prediction']],flush=True)


if __name__=='__main__':main()
