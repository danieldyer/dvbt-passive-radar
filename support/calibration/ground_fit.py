"""Compare a ground-target hypothesis with the free-altitude point-target fit."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fit_geometry import observations
from geographic_io import read_profiles
from ground_model import GroundTrajectory
from phase_fit import load_histories
from range_fit import RangeProblem,load_fields
from terrain import Terrain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--histories',type=Path,required=True);p.add_argument('--patches',type=Path,required=True)
    p.add_argument('--templates',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--labels',nargs='+',default=['A','B','C','E'])
    args=p.parse_args()
    geometry=json.loads((args.output/'geometry-prior.json').read_text());sites=np.array(geometry['enu_transmitters_m'])
    terrain=Terrain(args.terrain,geometry);_,records=read_profiles(args.templates)
    for label,coarse in zip(args.labels,observations(args.histories,args.labels)):
        channels,description=load_histories(args.histories,label,1.)
        problem=RangeProblem(channels,coarse,sites,load_fields(args.patches,label,description,records))
        old=json.loads((args.output/f'{label}-range.json').read_text());state=np.array(old['fixed_state'])
        model=GroundTrajectory(channels[1]['seed'],terrain);problem.model=model
        initial=np.r_[state[:model.count-1],.15]
        trials=[];columns=np.arange(model.radial_count,model.count)
        low,high=model.bounds()
        for azimuth in np.r_[initial[model.radial_count],np.linspace(-np.pi,np.pi,17)[:-1]]:
            start=initial.copy();start[model.radial_count]=azimuth
            def fun(angle,jacobian=False):
                q=start.copy();q[columns]=angle
                return problem.residual(q,jacobian=jacobian,columns=columns)
            fit=least_squares(fun,start[columns],jac=lambda x:fun(x,True),
                bounds=(low[columns],high[columns]),x_scale='jac',max_nfev=80,ftol=1e-7)
            start[columns]=fit.x;trials.append((fit.cost,start))
        trials.sort(key=lambda a:a[0]);fits=[problem.fit(start,max_nfev=300) for _,start in trials[:3]]
        fit=min(fits,key=lambda f:f.cost)
        result={'label':label,'state':fit.x.tolist(),'cost':fit.cost,'converged':bool(fit.success),
                'free_altitude_cost':old['fixed_cost'],'curve':problem.describe(fit.x),
                'scattering_height_agl_m':fit.x[model.count-1]*10,
                'status':'ground hypothesis test; not an accepted target classification or calibration'}
        (args.output/f'{label}-ground.json').write_text(json.dumps(result,indent=2))
        print(label,'ground/free cost',fit.cost,old['fixed_cost'],'height AGL',result['scattering_height_agl_m'],
              'speed',result['curve']['max_speed_mps'],flush=True)


if __name__=='__main__':
    main()
