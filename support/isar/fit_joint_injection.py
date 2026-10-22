"""Acquire recorded neighbours from residuals, then fit them with the point."""
import argparse
import json
from pathlib import Path

import numpy as np

from joint_point_echoes import JointPointEchoes
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment, SCALE
from report_measured_scatterer_phases import serial
from validate_recorded_injection import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('docs/data/injection_20261018/separated-500m'))
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(); root = args.root; out = args.output or root/'joint-neighbour'
    out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((root/'analytic-profile-fit/analysis.json').read_text())
    fields,rotation,rx,tx,state,mid=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    train=[f.time<prior['training_end_s'] for f in fields]
    base=Experiment(fields,rotation,rx,tx,state,mid,training=train)
    ex=JointPointEchoes(base)
    with np.load(root/'analytic-profile-fit/observations.npz') as saved:
        data=[saved[f'data_{c}'] for c in range(5)]
        clean=[saved[f'injected_{c}'] for c in range(5)]
    seed=np.r_[prior['selected']['parameters'],np.zeros(ex.parameter_count-6)]
    result=dict(scope=__doc__,acquisition=[],complete=False)
    def save(): (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    for c in range(5):
        proposal=ex.acquire_neighbour(data[c],seed,c)
        seed[ex.indices(c,0)]=proposal['parameters'];result['acquisition'].append(proposal)
        save(); print('Neighbour',c,serial(proposal),flush=True)
    neighbour_fit,_=ex.fit(data,seed,80,free=np.arange(6,ex.parameter_count))
    result['neighbour_initialisation']=neighbour_fit;save()
    print('Neighbour refinement complete',flush=True)
    fit,_=ex.fit(data,np.asarray(neighbour_fit['parameters']),100)
    result['fit']=fit
    truth=np.asarray(prior['truth_parameters'])
    error=(np.asarray(fit['parameters'])[:6]-truth)*SCALE
    result['physical_error']=error;result['horizontal_error_m']=np.linalg.norm(error[:2])
    result['point_template_loss']=[]
    for c in range(5):
        alpha=np.asarray(fit['coefficients_real_imag'][c])[0]; alpha=complex(*alpha)
        result['point_template_loss'].append(metrics(base.templates(np.asarray(fit['parameters'])[:6],c)[:,0]*alpha,clean[c]))
    result['complete']=True;save()
    print(json.dumps(serial({k:result[k] for k in ['horizontal_error_m','physical_error','point_template_loss']})),flush=True)


if __name__=='__main__': main()
