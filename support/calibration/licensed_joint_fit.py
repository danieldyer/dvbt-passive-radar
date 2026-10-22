"""Joint physical calibration trial using explicitly licensed site assignments."""
import argparse
import json
from pathlib import Path

import numpy as np

from coherent_fit import CoherentProblem,information
from fit_geometry import observations
from geographic_io import read_profiles
from phase_fit import load_histories
from polar_model import PolarTrajectory
from range_fit import load_fields
from site_hypotheses import AboveGroundProblem,licensed_geometry
from terrain import Terrain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['histories','patches','templates','terrain','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--hypothesis',choices=['main','ozdigitaltv'],default='main')
    p.add_argument('--labels',nargs='+',default=['A','B','C','E'])
    p.add_argument('--range-bias',action='store_true')
    args=p.parse_args()
    assignment={'main':[0,0,0,0,2],'ozdigitaltv':[0,2,1,2,0]}[args.hypothesis]
    sites,offsets,_=licensed_geometry(args.output,args.terrain,assignment)
    terrain=Terrain(args.terrain,json.loads((args.output/'geometry-prior.json').read_text()))
    _,records=read_profiles(args.templates)
    saved=json.loads((args.output/f'sites-{args.hypothesis}-physical.json').read_text())
    problems=[];states=[]
    for label,coarse in zip(args.labels,observations(args.histories,args.labels)):
        channels,desc=load_histories(args.histories,label,1.)
        pr=AboveGroundProblem(channels,coarse,sites,load_fields(args.patches,label,desc,records),terrain=terrain)
        pr.model=PolarTrajectory(channels[1]['seed'],channel_sites=assignment,channel_offsets=offsets)
        problems.append(pr);states.append(np.array(next(t['state'] for t in saved['targets'] if t['label']==label)))
    problem=CoherentProblem(problems,args.range_bias,site_prior_sigma=[10,10,10])
    initial=problem.initial(states)
    seeds=problem.constant_phase_starts(initial,24)
    trials=[]
    destination=args.output/f'licensed-joint-{args.hypothesis}{"-bias" if args.range_bias else ""}.json'
    for _,start in seeds[:3]:
        fit=problem.fit(start,max_nfev=600)
        item={'state':fit.x.tolist(),'cost':float(fit.cost),'converged':bool(fit.success),
              'evaluations':fit.nfev,'transmitter_corrections_enu_m':(fit.x[:9].reshape(3,3)*[30,30,75]).tolist()}
        trials.append(item);print(args.hypothesis,fit.cost,fit.success,item['transmitter_corrections_enu_m'],flush=True)
        destination.write_text(json.dumps({'trials':trials},indent=2))
    best=min(trials,key=lambda r:r['cost']);state=np.array(best['state'])
    result={'hypothesis':args.hypothesis,'labels':args.labels,'trials':trials,'best':best,
        'information':information(problem,state),'site_prior_sigma_enu_m':[10,10,10],
        'normalization_scale_enu_m':[30,30,75],'channel_sites':assignment,
        'site_positions_enu_m':sites.tolist(),'channel_offsets_enu_m':offsets.tolist(),
        'status':'physical hypothesis fit; no accepted calibration or metadata update',
        'targets':[],'range_bias_m':state[problem.bias_offset:].tolist() if args.range_bias else None}
    for i,(label,pr) in enumerate(zip(args.labels,problems)):
        local=np.r_[state[9+i*problem.n:9+(i+1)*problem.n],state[:9]]
        curve=pr.describe(local,True)
        result['targets'].append({'label':label,'curve':curve,
            'minimum_agl_m':float(np.min(terrain.clearance(np.array(curve['position_enu_m']))))})
    destination.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
