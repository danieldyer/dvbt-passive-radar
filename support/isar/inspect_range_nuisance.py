"""Evaluate a frozen range-supported nuisance fitted from the observed scene.

Neighbours are acquired/fitted in the summed data by joint_point_echoes.py.
No injected trajectory determines their location. Truth enters only the
information/retention diagnostics; final physical fits start at blind fits.
"""
import argparse,json
from pathlib import Path
import numpy as np

from correlated_likelihood import MetricExperiment
from joint_point_echoes import JointPointEchoes,PATH_SCALE
from range_nuisance import RangeNuisance
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment,SCALE
from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('docs/data/injection_20261018/separated-500m'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fit',action='store_true')
    args=parser.parse_args();root=args.root;out=args.output or root/'range-nuisance'
    out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((root/'analytic-profile-fit/analysis.json').read_text())
    neighbour=json.loads((root/'joint-neighbour/analysis.json').read_text())
    fields,rotation,rx,tx,state,mid=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    train=[f.time<prior['training_end_s'] for f in fields]
    base=Experiment(fields,rotation,rx,tx,state,mid,training=train);paths=JointPointEchoes(base)
    with np.load(root/'analytic-profile-fit/observations.npz') as saved:
        data=[saved[f'data_{c}'] for c in range(5)];clean=[saved[f'injected_{c}'] for c in range(5)]
    params=np.asarray(neighbour['fit']['parameters']);truth=np.asarray(prior['truth_parameters'])
    result=dict(scope=__doc__,cases=[],complete=False)
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    for count in [1,3,5]:
        metric=[]
        for c,(f,keep) in enumerate(zip(fields,train)):
            designs=[]
            for offset in np.arange(count)-(count-1)/2:
                p=params[paths.indices(c,0)].copy();p[0]+=offset*299792458./8e6/PATH_SCALE[0]
                designs.append(paths.path_template(p,c).reshape(f.y.shape)[keep])
            metric.append(RangeNuisance(np.stack(designs,axis=-1)))
        ex=MetricExperiment(base,metric)
        r,j,_,_=ex.profile(truth,clean)
        rb,_,_,_=ex.profile(truth,[y-s for y,s in zip(data,clean)],derivative=False)
        prediction=-np.linalg.lstsq(j,rb,rcond=None)[0]*SCALE
        retention=[]
        for s,m,f,keep in zip(clean,metric,fields,train):
            v=s.reshape(f.y.shape)[keep].ravel()
            retention.append(-10*np.log10(np.vdot(m.apply(v),m.apply(v)).real/np.vdot(v,v).real))
        record=dict(responses_per_bin=count,predicted_physical_error=prediction,
            predicted_horizontal_error_m=np.linalg.norm(prediction[:2]),point_information_loss_db=retention,
            maximum_range_condition=[m.condition.max() for m in metric])
        if args.fit:
            fit,_=ex.fit(data,np.asarray(prior['selected']['parameters']),100,position_halfwidth_m=1000.)
            error=(np.asarray(fit['parameters'])-truth)*SCALE
            record.update(fit=fit,physical_error=error,horizontal_error_m=np.linalg.norm(error[:2]))
        result['cases'].append(record);save();print(serial(record),flush=True)
    result['complete']=True;save()


if __name__=='__main__':main()
