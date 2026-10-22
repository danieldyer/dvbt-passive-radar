"""Acquire and fit a CV ship-return hypothesis under the injection-test operator.

This is a falsifiable point-motion hypothesis for the recorded echo, not an
assumption that a distributed ship has one constant complex response per band.
The final 10.5 seconds are excluded from acquisition, calibration and fitting.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from correlated_likelihood import crossvalidated_metric,MetricExperiment
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment,SCALE
from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    fields,rotation,rx,tx,state,mid=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    state[2]=np.zeros(3)
    ex=Experiment(fields,rotation,rx,tx,state,mid,training=[f.time<161.5 for f in fields])
    data=[f.y.ravel() for f in fields]
    result=dict(scope=__doc__,interval_s=[142.,172.],training_end_s=161.5,midpoint_s=mid,
        coarse_state_enu=state,receiver_ecef_m=rx,transmitters_ecef_m=tx,ecef_to_enu=rotation,
        yaw_fitted=False,complete=False,trials=[])
    def save(): (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    result['acquisition']=ex.acquire(data,np.zeros(6),count=3,constant_velocity=True);save()
    for proposal in result['acquisition']:
        fit,_=ex.fit(data,np.asarray(proposal['parameters']),120,free=[0,1,2,3],position_halfwidth_m=1000.)
        result['trials'].append(fit);save();print('CV ship',serial(fit),flush=True)
    cv=min(result['trials'],key=lambda f:f['training_rss']);result['constant_velocity']=cv
    _,_,_,prediction=ex.profile(np.asarray(cv['parameters']),data,derivative=False)
    covariance=[crossvalidated_metric((y-p).reshape(f.y.shape)[keep],np.median(np.diff(f.time)))
        for y,p,f,keep in zip(data,prediction,fields,ex.training)]
    gls=MetricExperiment(ex,covariance)
    fit,prediction=gls.fit(data,np.asarray(cv['parameters']),120,free=[0,1,2,3],position_halfwidth_m=1000.)
    result['constant_velocity_gls']=fit;result['covariance']=[m.description for m in covariance]
    result['frozen_prediction_checks']=[]
    for c,(f,keep) in enumerate(zip(fields,ex.training)):
        h=ex.templates(np.asarray(fit['parameters']),c)[:,0].reshape(f.y.shape)
        alpha=np.asarray(fit['coefficients_real_imag'][c])[0];alpha=complex(*alpha)
        # A static-only fit is the comparator; keep the final time block excluded.
        use=np.repeat(keep,len(f.offsets));coef=np.linalg.lstsq(ex.designs[c][use],data[c][use],rcond=None)[0]
        static=ex.designs[c]@coef
        baseline=np.sum(abs((data[c]-static).reshape(f.y.shape)[~keep])**2)
        error=np.sum(abs((data[c]-prediction[c]).reshape(f.y.shape)[~keep])**2)
        result['frozen_prediction_checks'].append(dict(channel=c,static_only_rss=baseline,
            point_plus_static_rss=error,explained_fraction=1-error/baseline,
            predicted_point_energy=np.sum(abs(h[~keep]*alpha)**2)))
    result['complete']=True;save()
    print('CV ship excluded predictions',serial(result['frozen_prediction_checks']),flush=True)


if __name__=='__main__':main()
