"""Alternate discrete road associations and the shared complex calibration.

Independent-band road scores supply candidates, not the final association.
Every replacement must improve the same complete coherent likelihood with
unchanged measured samples and physical priors.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from calibration_solver import solve
from coherent_fit import CoherentProblem,information
from road_fit import load_trial
from road_geometry import RoadCurve
from road_model import RoadTrajectory
from profiled_bands import ProfiledBandProblem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','histories','patches','templates','routes','trials','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=180)
    parser.add_argument('--labels',nargs='+',help='Reconsider these road associations while retaining every target in the joint fit')
    args=parser.parse_args();saved=json.loads(args.fit.read_text())
    if 'antenna_state' in saved or 'receiver_prior_sigma_m' in saved:
        raise ValueError('Road reassociation currently uses the fixed-receiver, three-site likelihood')
    if args.labels and set(args.labels)-set(saved['labels']):
        raise ValueError('Road reassociation requested an unknown target')
    routes=json.loads(args.routes.read_text())['routes']
    joint=load_trial(saved,args.routes,args.histories,args.patches,args.templates)
    state=np.array(saved['state']);clock=time.monotonic();history=[]
    code=''.join(map(str,saved['assignment']))
    for round_index in range(args.rounds):
        replacements=0;results=[]
        for index,label in enumerate(saved['labels']):
            if args.labels and label not in args.labels:continue
            source=joint.problems[index]
            choices=json.loads((args.trials/f'{code}-{label}.json').read_text())['trials']
            if saved.get('road_association_names'):
                choices=[choice for choice in choices if choice['name'] in saved['road_association_names']]
            previous=saved['selected_roads'][index]
            one=CoherentProblem([source],range_bias=joint.range_bias,site_prior_sigma=np.array(saved['site_prior_sigma_m']))
            current=one.initial([state[9+index*joint.n:9+(index+1)*joint.n]],state[:9])
            current[one.beta_offset:one.magnitude_offset]=state[joint.beta_offset:joint.magnitude_offset]
            current[one.magnitude_offset:one.bias_offset]=state[joint.magnitude_offset+4*index:joint.magnitude_offset+4*(index+1)]
            if joint.range_bias:current[one.bias_offset:]=state[joint.bias_offset:]
            r=one.residual(current);best=(float(r@r/2),previous,source,current)
            trials=[]
            for choice in choices:
                if choice['route']==previous['route']:continue
                problem=copy.copy(source)
                problem.model=RoadTrajectory(source.model.seed,RoadCurve(routes[choice['route']]),
                    choice['coarse_state'],saved['assignment'],np.array(saved['channel_offsets_enu_m']))
                problem.model.receiver_enu_m=source.model.receiver_enu_m.copy()
                single=CoherentProblem([problem],range_bias=joint.range_bias,site_prior_sigma=np.array(saved['site_prior_sigma_m']))
                initial=single.initial([np.array(choice['state'])],state[:9])
                initial[single.beta_offset:single.magnitude_offset]=state[joint.beta_offset:joint.magnitude_offset]
                if joint.range_bias:initial[single.bias_offset:]=state[joint.bias_offset:]
                lo,hi=single.bounds();initial=np.clip(initial,lo+1e-7,hi-1e-7)
                active=np.r_[np.arange(9,single.beta_offset),np.arange(single.magnitude_offset,single.bias_offset)]
                starts=single.fixed_phase_starts(initial,np.linspace(-20,20,41))
                if saved.get('profiled_band_gains'):
                    reduced=ProfiledBandProblem(single)
                    starts=sorted(starts,key=lambda q:np.linalg.norm(reduced.residual(reduced.initial(q[1]))))
                    fits=[]
                    for _,p in starts[:2]:
                        fit,q=reduced.fit_columns(reduced.initial(p),np.arange(9,single.beta_offset),args.iterations)
                        expanded,_=reduced.expanded_state(q)
                        fits.append((fit,expanded))
                else:
                    fits=[single.fit_columns(p,active,args.iterations) for _,p in starts[:2]]
                fit,trial=min(fits,key=lambda item:item[0].cost)
                trials.append({'route':choice['route'],'name':choice['name'],'cost':float(fit.cost),
                    'converged':bool(fit.success),'evaluations':fit.nfev})
                if fit.cost<best[0]-1e-6:best=(fit.cost,choice,problem,trial)
            changed=best[1]['route']!=previous['route']
            if changed:
                joint.problems[index]=best[2];saved['selected_roads'][index]=best[1]
                state[9+index*joint.n:9+(index+1)*joint.n]=best[3][9:one.beta_offset]
                state[joint.magnitude_offset+4*index:joint.magnitude_offset+4*(index+1)]=best[3][one.magnitude_offset:one.bias_offset]
                replacements+=1
            results.append({'label':label,'old_road':previous['name'],'new_road':best[1]['name'],
                'cost':best[0],'changed':changed,'trials':trials})
            print('round',round_index,label,previous['name'],'->',best[1]['name'],'cost',round(best[0],2),
                'elapsed',round(time.monotonic()-clock,2),flush=True)
        if saved.get('profiled_band_gains'):
            reduced=ProfiledBandProblem(joint)
            fit=solve(reduced.residual,reduced.initial(state),reduced.bounds(),args.iterations*2)
            state,absent=reduced.expanded_state(fit.x)
            saved['zero_amplitude_channels_by_target']=absent
            saved['reduced_state']=fit.x.tolist()
        else:
            fit=solve(joint.residual,state,joint.bounds(),args.iterations*2)
            state=fit.x
        history.append({'round':round_index,'replacements':replacements,'targets':results,
            'joint_cost':float(fit.cost),'converged':bool(fit.success)})
        result={**saved,'state':state.tolist(),'cost':float(fit.cost),'converged':bool(fit.success),
            'evaluations':fit.nfev,'optimality':float(fit.optimality),'termination':str(fit.message),
            'site_corrections_enu_m':(state[:9].reshape(3,3)*[30,30,75]).tolist(),
            'range_bias_m':state[joint.bias_offset:].tolist(),
            'source_fit':str(args.fit),'association_history':history,
            'information':information(joint,state),
            'status':'conditional alternating road/calibration trial; validation still required'}
        args.output.write_text(json.dumps(result,indent=2))
        print('round',round_index,'replacements',replacements,'joint cost',fit.cost,flush=True)
        if replacements==0:break


if __name__=='__main__':main()
