"""Road-constrained range/rate screening for all licensed site combinations.

The screening likelihood is not used as additional data in the later complex
fit. Coarse peak positions are correlated with the retained complex samples.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from fit_geometry import observations
from geometry import unit
from road_geometry import RoadCurve
from site_hypotheses import licensed_geometry
from terrain import RasterTerrain
from transmitter_config import candidate_indices, nominal_indices

OPTIONS=candidate_indices()


class RoadScreen:
    def __init__(self,road,obs,transmitters):
        self.road,self.obs,self.tx=road,obs,transmitters[obs['channel']]
        self.half=(obs['description']['end']-obs['description']['start'])/2
        u=obs['t']/self.half
        self.b=np.polynomial.polynomial.polyvander(u,3)
        self.d=np.c_[np.zeros(len(u)),np.ones(len(u)),2*u,3*u*u]/self.half
        self.probe=np.polynomial.polynomial.polyvander(np.linspace(-1,1,31),3)
        u=np.linspace(-1,1,31)
        self.vprobe=np.c_[np.zeros(len(u)),np.ones(len(u)),2*u,3*u*u]/self.half
        self.aprobe=np.c_[np.zeros(len(u)),np.zeros(len(u)),np.full(len(u),2),6*u]/self.half**2
        # One-second coarse peaks have interpolation/association uncertainty
        # beyond the ideal isolated-target CRLB. This is screening only.
        self.sigma_l=np.maximum(obs['sigma_l'],3.)
        self.sigma_v=np.maximum(obs['sigma_v'],.15)

    def evaluate(self,p,jacobian=False):
        s=self.b@p;clipped=np.clip(s,0,self.road.length)
        x=self.road.centre(clipped).copy();x[:,2]+=2
        tangent=self.road.centre(clipped,1);second=self.road.centre(clipped,2)
        a=x-self.tx;b=x;ua,ub=unit(a),unit(b)
        grad=ua+ub
        slope=np.sum(grad*tangent,axis=1)
        curvature=np.sum(grad*second,axis=1)
        for q,u in ((a,ua),(b,ub)):
            curvature+=(np.sum(tangent*tangent,axis=1)-np.sum(u*tangent,axis=1)**2)/np.linalg.norm(q,axis=1)
        rate=slope*(self.d@p)
        length=np.linalg.norm(a,axis=1)+np.linalg.norm(b,axis=1)-np.linalg.norm(self.tx,axis=1)
        within=(s>=0)&(s<=self.road.length)
        if jacobian:
            jl=slope[:,None]*self.b*within[:,None]/self.sigma_l[:,None]
            jv=(curvature[:,None]*(self.d@p)[:,None]*self.b*within[:,None]+slope[:,None]*self.d)/self.sigma_v[:,None]
            return length,rate,np.r_[jl,jv]
        return length,rate,None

    def residual(self,p,jacobian=False):
        length,rate,j=self.evaluate(p,jacobian)
        pieces=[j] if jacobian else [(length-self.obs['length'])/self.sigma_l,(rate-self.obs['rate'])/self.sigma_v]
        s=self.probe@p;v=self.vprobe@p;a=self.aprobe@p
        boundary=s-np.clip(s,0,self.road.length)
        directions=self.road.description['directions']
        maxspeed=self.road.description['posted_speed_mps']
        maxspeed=40. if maxspeed is None else max(maxspeed*1.2,maxspeed+3.)
        vl=-maxspeed if -1 in directions else 0.
        vh=maxspeed if 1 in directions else 0.
        for value,matrix,low,high,scale in [(s,self.probe,0,self.road.length,1.),
                 (v,self.vprobe,vl,vh,1.),(a,self.aprobe,-4,4,.5)]:
            active=(value<low)|(value>high)
            pieces.append(matrix*active[:,None]/scale if jacobian else (value-np.clip(value,low,high))/scale)
        return np.concatenate(pieces)

    def fit(self,start,nfev=100):
        return least_squares(self.residual,start,jac=lambda x:self.residual(x,True),
            loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=nfev,ftol=2e-6,xtol=1e-7,gtol=1e-5)

    def describe(self,fit,route):
        length,rate,_=self.evaluate(fit.x)
        s=self.probe@fit.x;v=self.vprobe@fit.x
        return {'route':route,'name':self.road.description['name'],'state':fit.x.tolist(),'cost':float(fit.cost),
                'converged':bool(fit.success),'range_rms_m':float(np.sqrt(np.mean((length-self.obs['length'])**2))),
                'rate_rms_mps':float(np.sqrt(np.mean((rate-self.obs['rate'])**2))),
                'road_position_minmax_m':[float(s.min()),float(s.max())],
                'speed_minmax_mps':[float(v.min()),float(v.max())],
                'boundary_violation_m':float(np.max(abs(s-np.clip(s,0,self.road.length)))),
                'midpoint_enu_m':self.road.positions(np.array([np.clip(fit.x[0],0,self.road.length)]))[0].tolist()}


def reference_starts(roads,obs,transmitters):
    channel=obs['description'].get('reference_channel',1)
    seed=obs['description']['channels'][channel]
    length=seed['path'][0];rate=seed['path'][1]/seed['scale']
    reference=transmitters[channel]
    half=(obs['description']['end']-obs['description']['start'])/2
    for index,road in enumerate(roads):
        if road.description['elevation_uncertain']:
            continue
        lo,hi=road.description['excess_range_minmax_m']
        if not lo-150<length<hi+150:
            continue
        for s in road.crossings(length,reference):
            x=road.positions(np.array([s]))[0];d=road.centre(s,1)
            slope=np.dot(unit(x-reference)+unit(x),d)
            if abs(slope)<.03:continue
            speed=rate/slope
            maximum=road.description['posted_speed_mps']
            maximum=40. if maximum is None else max(maximum*1.25,maximum+5.)
            if abs(speed)>maximum or int(np.sign(speed)) not in road.description['directions']:
                continue
            # A complete interval needs a connected road. Allow 40 m for
            # acceleration while seeding; the final likelihood checks all times.
            if min(s-speed*half,s+speed*half)<-40 or max(s-speed*half,s+speed*half)>road.length+40:
                continue
            yield index,np.array([s,speed*half,0.,0.])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--routes',type=Path,required=True);p.add_argument('--histories',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prior-directory',type=Path,required=True)
    p.add_argument('--terrain',type=Path,required=True);p.add_argument('--dem',type=Path,required=True)
    p.add_argument('--labels',nargs='+',required=True);p.add_argument('--retain',type=int,default=24)
    p.add_argument('--refine',type=int,default=6);p.add_argument('--reference-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    geometry=json.loads((args.prior_directory/'geometry-prior.json').read_text())
    terrain=RasterTerrain(args.terrain,geometry,args.dem)
    routes=json.loads(args.routes.read_text())['routes'];roads=[RoadCurve(r) for r in routes]
    main=nominal_indices(geometry)
    sites,offsets,_=licensed_geometry(args.prior_directory,args.terrain,main,terrain)
    transmitter=sites[np.array(main)]+offsets
    data=observations(args.histories,args.labels)
    starts={};clock=time.monotonic()
    for obs in data:
        path=args.output/(obs['label']+'-screen.json')
        if path.exists():
            starts[obs['label']]=json.loads(path.read_text())['trials'];continue
        trials=[]
        for index,start in reference_starts(roads,obs,transmitter):
            problem=RoadScreen(roads[index],obs,transmitter)
            fit=problem.fit(start)
            trial=problem.describe(fit,index)
            if trial['boundary_violation_m']<1:
                trials.append(trial)
        trials.sort(key=lambda q:q['cost'])
        starts[obs['label']]=trials[:args.retain]
        result={'label':obs['label'],'hypothesis':'main sites, coarse road screening only',
                'tested_starts':len(trials),'trials':trials[:args.retain]}
        path.write_text(json.dumps(result,indent=2))
        print(obs['label'],'roads',len(trials),'best',[(q['name'],round(q['cost'],1),round(q['rate_rms_mps'],2)) for q in trials[:4]],'elapsed',round(time.monotonic()-clock,1),flush=True)
    if args.reference_only:return
    all_trials=[]
    for assignment in itertools.product(*candidate_indices(geometry)):
        code=''.join(map(str,assignment));path=args.output/f'sites-{code}.json'
        if path.exists():
            all_trials.append(json.loads(path.read_text()));continue
        sites,offsets,channels=licensed_geometry(args.prior_directory,args.terrain,assignment,terrain)
        transmitter=sites[np.array(assignment)]+offsets
        results=[]
        for obs in data:
            trials=[]
            for start in starts[obs['label']][:args.refine]:
                index=start['route'];problem=RoadScreen(roads[index],obs,transmitter)
                trials.append(problem.describe(problem.fit(np.array(start['state'])),index))
            trials.sort(key=lambda q:q['cost'])
            results.append({'label':obs['label'],'trials':trials})
        result={'assignment':assignment,'channels':channels,'targets':results,
                'sum_screening_cost':sum(t['trials'][0]['cost'] for t in results),
                'status':'coarse association screening, not a phase-centre calibration'}
        path.write_text(json.dumps(result,indent=2));all_trials.append(result)
        print(code,'screen cost',round(result['sum_screening_cost'],1),'elapsed',round(time.monotonic()-clock,1),flush=True)
    all_trials.sort(key=lambda q:q['sum_screening_cost'])
    (args.output/'site-ranking.json').write_text(json.dumps([{'assignment':r['assignment'],'sum_screening_cost':r['sum_screening_cost']} for r in all_trials],indent=2))


if __name__=='__main__':main()
