"""Fit one physically moving fading scatterer to the current uninjected replay."""
import argparse,json
from pathlib import Path
import numpy as np

from correlated_likelihood import crossvalidated_metric
from fading_point_motion import PlanarGeometry,FadingPointArc,FadingPointFit,TurningGeometry,TurningPointArc
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment,SCALE
from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--knot-seconds',type=float,default=3.)
    parser.add_argument('--kinematics',choices=['quadratic','coordinated-turn'],default='quadratic')
    parser.add_argument('--starts',type=int,default=3)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    fields,rotation,rx,tx,state,mid=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    # Every eighth second pair is omitted, including part of the bright flash.
    train=[np.floor((f.time-142.)/2).astype(int)%4!=3 for f in fields]
    base=Experiment(fields,rotation,rx,tx,state,mid,training=train)
    geometry=(TurningGeometry if args.kinematics=='coordinated-turn' else PlanarGeometry)(rx,rotation,tx,state)
    knots=np.r_[fields[0].data['row_time'][0],np.arange(142.+args.knot_seconds,172.,args.knot_seconds),max(f.data['row_time'][-1] for f in fields)]
    knots[0]=min(f.data['row_time'][0] for f in fields)
    result=dict(scope=__doc__,complete=False,midpoint_s=mid,interval_s=[142.,172.],
        training='Omit 2 s blocks whose ordinal modulo four is three',
        prior_sigma_enu=geometry.sigma,prior_mean_state_enu=geometry.coarse,
        amplitude_knots_s=knots,motion_model=args.kinematics,phase_model='One constant phase per band; shared physical planar motion; positive real brightness knots',
        trials=[])
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    proposals=base.acquire([f.y.ravel() for f in fields],np.zeros(6),count=args.starts,path_rate_halfwidth_mps=.8,path_acceleration_halfwidth_mps2=.08)
    seeds=[np.zeros(6)]
    for q in proposals:
        physical=np.asarray(state).copy();physical[:,:2]+=(np.asarray(q['parameters'])*SCALE).reshape(3,2)
        seeds.append(((physical-geometry.coarse)[:,:2].ravel()/geometry.sigma))
    result['acquisition']=proposals;save()
    def model(metrics=None):
        arc=TurningPointArc if args.kinematics=='coordinated-turn' else FadingPointArc
        return FadingPointFit([arc(f,keep,knots,geometry,c,mid,base.designs[c],None if metrics is None else metrics[c])
            for c,(f,keep) in enumerate(zip(fields,train))])
    thermal=model();trials=[]
    for seed in seeds:
        candidate,_=thermal.fit([seed]);trials.append(candidate)
        result['trials'].append(dict(parameters=candidate.x,objective=candidate.fun,success=candidate.success,iterations=candidate.nit))
        save();print('Fading trial',serial(result['trials'][-1]),flush=True)
    best=min(trials,key=lambda f:f.fun)
    result['thermal_fit']=dict(parameters=best.x,objective=best.fun,gradient=best.jac,success=best.success,message=str(best.message),iterations=best.nit)
    save();print('Thermal fading fit',serial(result['thermal_fit']),flush=True)
    predictions=[b.original_prediction(best.x)[0] for b in thermal.bands]
    metric=[crossvalidated_metric((f.y.ravel()-p).reshape(f.y.shape)[keep],np.median(np.diff(f.time)))
        for f,p,keep in zip(fields,predictions,train)]
    gls=model(metric)
    refined,starts=gls.fit([best.x[:6]])
    result['fit']=dict(parameters=refined.x,objective=refined.fun,gradient=refined.jac,success=refined.success,message=str(refined.message),iterations=refined.nit)
    result['covariance']=[m.description for m in metric]
    result['prediction']=gls.prediction(refined.x,[~keep for keep in train])
    result['amplitudes']=[b.original_prediction(refined.x)[1] for b in gls.bands]
    physical=np.array(geometry.state(refined.x)[:3]);correction=(physical-geometry.coarse)[:,:2].ravel()
    phases=refined.x[6:]+np.array([f.k*(geometry.prediction(refined.x)[c,0]-b.reference_path) for c,(f,b) in enumerate(zip(fields,gls.bands))])
    gains=np.array([np.mean(a) for a in result['amplitudes']])*np.exp(1j*phases)
    render=dict(midpoint_s=mid,coarse_state_enu=geometry.coarse,interval_s=[142.,172.],motion_model=args.kinematics,
        fits=dict(constant_velocity_gls=dict(parameters=correction/SCALE,coefficients_real_imag=[[[z.real,z.imag]] for z in gains])),
        note='Gaussian AIS/motion priors; time-varying positive brightness used in estimation; renderer uses original data and constant band coefficients.')
    (out/'render-fit.json').write_text(json.dumps(serial(render),indent=2)+'\n')
    result['physical_state_enu']=physical;result['complete']=True;save()
    print('Excluded fading prediction',serial([dict(channel=c,explained=v['test_complex_prediction_fraction']) for c,v in enumerate(result['prediction'])]),flush=True)


if __name__=='__main__':main()
