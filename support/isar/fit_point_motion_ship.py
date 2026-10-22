"""Broad quadratic-chirp acquisition followed by exact geometric ship fitting.

The saved IQ is reprocessed through current production DSP by the injection
baseline. This tool uses that uninjected observation only. Native symbols,
waveform kernels and the physical position/velocity/acceleration model are
identical to the validated point experiment; no image-derived phase is fitted.
"""
import argparse,json
from pathlib import Path
import numpy as np

from correlated_likelihood import crossvalidated_metric,MetricExperiment
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment
from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--starts',type=int,default=5)
    parser.add_argument('--first',type=float,default=142.)
    parser.add_argument('--last',type=float,default=172.)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    fields,rotation,rx,tx,state,mid=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    fields=[f.windowed(args.first,args.last) for f in fields]
    split=args.first+.65*(args.last-args.first);train=[f.time<split for f in fields]
    ex=Experiment(fields,rotation,rx,tx,state,mid,training=train);data=[f.y.ravel() for f in fields]
    result=dict(scope=__doc__,complete=False,interval_s=[args.first,args.last],training_end_s=split,
        midpoint_s=mid,coarse_state_enu=state,ecef_to_enu=rotation,receiver_ecef_m=rx,transmitters_ecef_m=tx,
        yaw_fitted=False,trials=[])
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    proposals=ex.acquire(data,np.zeros(6),count=args.starts,path_rate_halfwidth_mps=.8,path_acceleration_halfwidth_mps2=.08)
    result['acquisition']=proposals;save();print('Broad acquisition',serial(proposals),flush=True)
    for candidate in proposals:
        fit,_=ex.fit(data,np.asarray(candidate['parameters']),120,position_halfwidth_m=1000.,velocity_halfwidth_mps=3.)
        result['trials'].append(fit);save();print('Physical fit',serial(fit),flush=True)
    best=min(result['trials'],key=lambda f:f['training_rss']);result['thermal_fit']=best
    _,_,_,prediction=ex.profile(np.asarray(best['parameters']),data,derivative=False)
    covariance=[crossvalidated_metric((y-p).reshape(f.y.shape)[keep],np.median(np.diff(f.time)))
        for y,p,f,keep in zip(data,prediction,fields,train)]
    gls=MetricExperiment(ex,covariance)
    fit,prediction=gls.fit(data,np.asarray(best['parameters']),120,position_halfwidth_m=1000.,velocity_halfwidth_mps=3.)
    result['covariance_fit']=fit;result['covariance']=[m.description for m in covariance]
    result['prediction']=[]
    for c,(f,keep) in enumerate(zip(fields,train)):
        use=np.repeat(keep,len(f.offsets))
        alpha=np.linalg.lstsq(ex.designs[c][use],data[c][use],rcond=None)[0]
        baseline=ex.designs[c]@alpha
        static_error=np.sum(abs((data[c]-baseline).reshape(f.y.shape)[~keep])**2)
        error=np.sum(abs((data[c]-prediction[c]).reshape(f.y.shape)[~keep])**2)
        result['prediction'].append(dict(channel=c,static_only_error=static_error,point_plus_static_error=error,
            explained_fraction=1-error/static_error))
    result['complete']=True;save();print('Excluded data',serial(result['prediction']),flush=True)


if __name__=='__main__':main()
