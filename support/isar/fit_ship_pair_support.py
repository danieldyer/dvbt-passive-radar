"""Independent-band control with the same wider secondary range support.

Starts include each original pair solution and the common physical solution.
All choices use training data only. This avoids attributing a gain from a
wider search to a shared physical motion model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from amplitude_pair import AmplitudePair
from fit_joint_sea_arcs import prepare_problem
from fit_ship_amplitude_pair import phase_initialise
from fit_ship_physical_pair import build_model
from flash_geometry import observables
from motion import C
from ship_arc_coherence import archived_observables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-amplitude-pair-F01-wider'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);output=args.output/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    root=Path('docs/data/isar_20261017')
    original_path=root/'ship-amplitude-pair-F01/fits.json';physical_path=root/'ship-physical-pair-F01-wider/fits.json'
    original=json.loads(original_path.read_text());physical=json.loads(physical_path.read_text())
    if not original['complete'] or not physical['complete']:raise ValueError('Source fits must be complete')
    best=min(physical['trials'],key=lambda r:r['cost']);params=np.asarray(best['parameters'])
    case=prepare_problem();shared,i=build_model(case,(-2.8,2.8));g=shared.geometry
    p,v,a,tx,rx=g.state(params[:g.count]);result=dict(scope=__doc__,complete=False,channels=[],
        origin_bounds_samples=[-2.8,2.8],shared_seed_start=best['start_rank'],accepted_calibration=False)
    paths=[original_path,physical_path,Path(__file__),Path(__file__).with_name('amplitude_pair.py'),
        Path(__file__).with_name('fit_ship_amplitude_pair.py')]
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    for ch,(pair,index,old,test) in enumerate(zip(shared.pairs,shared.indices,original['channels'],case['testing'][i])):
        base=pair.secondary;dt=base.epoch-pair.base.midpoint
        point=observables(p+dt*v+.5*dt**2*a,v+dt*a,a,tx,rx)[ch]
        difference=point-archived_observables(base.field.data,base.epoch)
        primary=np.array([difference[0]/(C/8e6),0.,base.scale*difference[1],.5*base.scale**2*difference[2]])
        seeds=[(f'original-{j}',np.asarray(row['parameters'])) for j,row in enumerate(old['trials'])]
        seeds.append(('shared-physical',phase_initialise(base,np.r_[primary,params[index[-4:]]])))
        model=AmplitudePair(base);trials=[]
        for label,seed in seeds:
            fit=model.fit(seed,(-2.8,2.8));fit['seed_label']=label
            pp=np.asarray(fit['parameters']);fit['prediction']=model.prediction(pp,test)
            u=(pair.base.midpoint-base.epoch)/base.scale
            transform=np.array([[C/8e6,0,u,u*u],[0,0,1/base.scale,2*u/base.scale],[0,0,0,2/base.scale**2]])
            fit['observables']=(archived_observables(base.field.data,pair.base.midpoint)[None]+pp.reshape(2,4)@transform.T).tolist()
            fit['numerically_regular']=bool(fit['information_rank']==8 and fit['projected_gradient']<1e-6 and
                not fit['bound_indices'] and min(fit['curvature_eigenvalues'])>0)
            trials.append(fit)
            print('channel',ch,label,'cost',fit['objective']*base.energy,'heldout',fit['prediction']['heldout_error'],
                'bounds',fit['bound_indices'],'regular',fit['numerically_regular'],flush=True)
        result['channels'].append(dict(channel=ch,data_energy=base.energy,trials=trials,
            best_index=int(np.argmin([r['objective'] for r in trials]))))
        output.write_text(json.dumps(result,indent=2,allow_nan=False))
    result['complete']=True;output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
