"""Held-out complex-data checks for the permitted transmitter configurations.

Initial branches come from the full-data search, so the validation is conditional
on association and branch selection. No held-out complex amplitudes enter a fit.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from complex_likelihood import projected_profile
from fit_geometry import observations
from geographic_io import read_profiles
from phase_fit import load_histories
from polar_model import PolarTrajectory
from range_fit import Field,RangeProblem,load_fields
from site_hypotheses import licensed_geometry,AboveGroundProblem
from terrain import Terrain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['histories','patches','templates','terrain','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--labels',nargs='+',default=['A','B','C','E'])
    p.add_argument('--above-ground',action='store_true')
    args=p.parse_args();_,records=read_profiles(args.templates);results=[]
    for name,assignment in [('main',[0,0,0,0,2]),('ozdigitaltv',[0,2,1,2,0])]:
        suffix='-physical' if args.above_ground else ''
        saved=json.loads((args.output/f'sites-{name}{suffix}.json').read_text())
        sites,offsets,_=licensed_geometry(args.output,args.terrain,assignment)
        for label,coarse in zip(args.labels,observations(args.histories,args.labels)):
            channels,desc=load_histories(args.histories,label,1.)
            original=load_fields(args.patches,label,desc,records)
            split=[]
            for training in [True,False]:
                fields=[]
                for field in original:
                    selected=(np.floor(field.time-desc['start']).astype(int)%2==0)==training
                    fields.append(Field(field.data,field.pulse,field.seed,selected=selected))
                if args.above_ground:
                    terrain=Terrain(args.terrain,json.loads((args.output/'geometry-prior.json').read_text()))
                    pr=AboveGroundProblem(channels,coarse,sites,fields,terrain=terrain)
                else:
                    pr=RangeProblem(channels,coarse,sites,fields)
                pr.model=PolarTrajectory(channels[1]['seed'],channel_sites=assignment,channel_offsets=offsets)
                split.append(pr)
            train,test=split
            initial=np.array(next(t['state'] for t in saved['targets'] if t['label']==label))
            fit=train.fit(initial,max_nfev=450)
            metrics=[]
            for channel in range(5):
                u,_=train.channel_model(fit.x,channel)
                _,amplitude=projected_profile(train.channel_data(channel),u)
                v=test.channel_data(channel);u,_=test.channel_model(fit.x,channel)
                _,best=projected_profile(v,u)
                predicted=amplitude*u
                metrics.append({'channel':channel,'complex_prediction_error':float(np.vdot(v-predicted,v-predicted).real),
                    'heldout_focused_energy':float(abs(np.vdot(u,v))**2/np.vdot(u,u).real),
                    'heldout_phase_change_deg':float(np.rad2deg(np.angle(best/amplitude))),
                    'heldout_amplitude_ratio':float(abs(best/amplitude))})
            result={'hypothesis':name,'target':label,'converged':bool(fit.success),
                    'state':fit.x.tolist(),'training_cost':float(fit.cost),'channels':metrics,
                    'scope':'conditional on full-data branch selection; same per-channel empirical covariance in both hypotheses'}
            results.append(result)
            (args.output/f'site-heldout-validation{suffix}.json').write_text(json.dumps(results,indent=2))
            print(name,label,'heldout energy',[round(x['heldout_focused_energy'],2) for x in metrics],flush=True)


if __name__=='__main__':
    main()
