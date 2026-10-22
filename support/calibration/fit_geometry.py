"""Coarse multistatic calibration and identifiability diagnostics; no production edits."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from geometry import prior, observables, SITES, RF, C, unit, central_jacobian


def observations(directory, labels):
    result = []
    for label in labels:
        description = json.loads((directory / f'{label}.json').read_text())
        channel, times, lengths, rates, sigma_l, sigma_v = [], [], [], [], [], []
        for ch, seed in enumerate(description['channels']):
            for t, length, rate, snr_db in seed['coarse_observations']:
                if description['start'] <= t <= description['end']:
                    snr = 10**(snr_db/10)
                    channel.append(ch);times.append(t-description['channels'][0]['epoch'])
                    lengths.append(length);rates.append(rate)
                    # Conditional flat-band rectangular-CPI Fisher scales.
                    # Selection noise is only a proxy: validate model residuals.
                    sigma_l.append(max(.5, .3898484*C/(6.65625e6*np.sqrt(snr))))
                    sigma_v.append(max(.01, .3898484*C/(RF[ch]*np.sqrt(snr))))
        result.append({'label':label, 'description':description, 'channel':np.array(channel),
            't':np.array(times), 'length':np.array(lengths), 'rate':np.array(rates),
            'sigma_l':np.array(sigma_l), 'sigma_v':np.array(sigma_v)})
    return result


def predict(state, obs, sites):
    t = obs['t']
    x = state[:3]*1e4 + t[:,None]*state[3:6]*100
    v = np.broadcast_to(state[3:6]*100, x.shape)
    if len(state) == 9:
        x = x + .5*t[:,None]**2*state[6:9]
        v = v + t[:,None]*state[6:9]
    return observables(x, v, sites[SITES[obs['channel']]], np.zeros(3))


def residual(state, obs, sites):
    length, rate = predict(state, obs, sites)
    return np.r_[(length-obs['length'])/obs['sigma_l'], (rate-obs['rate'])/obs['sigma_v']]


def starts(obs, sites, n):
    rng = np.random.default_rng(413)
    transmitter = sites.mean(axis=0);baseline = np.linalg.norm(transmitter)
    e = unit(transmitter);across=unit(np.cross([0,0,1],e));up=np.cross(e,across)
    length = np.median(obs['length']);a=(baseline+length)/2;b=np.sqrt(a*a-baseline*baseline/4)
    for i in range(n):
        theta = 2*np.pi*(i+.5)/n
        elevation = rng.uniform(.04, 1.3)
        x=transmitter/2+a*np.cos(theta)*e+b*np.sin(theta)*(np.cos(elevation)*across+np.sin(elevation)*up)
        x[2]=abs(x[2])+50
        g=unit(x-transmitter)+unit(x)
        v=np.median(obs['rate'])*g/np.dot(g,g)
        v+=rng.normal(0,30,3);v[2]=0
        yield np.r_[x/1e4,v/100]


def bounds(order):
    lo=np.r_[[-3,-4,-.03],[-3,-3,-.5]]
    hi=np.r_[[6,4,1.2],[3,3,.5]]
    if order==9:lo=np.r_[lo,[-5,-5,-2]];hi=np.r_[hi,[5,5,2]]
    return lo,hi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--histories',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',nargs='+',default=['A','B','C'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sites,description=prior(args.terrain,json.loads(args.metadata.read_text()))
    (args.output/'geometry-prior.json').write_text(json.dumps(description,indent=2))
    data=observations(args.histories,args.labels);solutions=[]
    for obs in data:
        trials=[]
        for start in starts(obs,sites,32):
            low,high=bounds(6);start=np.clip(start,low+1e-7,high-1e-7)
            fit=least_squares(residual,start,args=(obs,sites),bounds=bounds(6),loss='soft_l1',f_scale=2,
                              max_nfev=350,ftol=1e-8,xtol=1e-9,gtol=1e-7)
            trials.append(fit)
        trials.sort(key=lambda f:f.cost)
        best=trials[0]
        affine=least_squares(residual,best.x,args=(obs,sites),bounds=bounds(6),max_nfev=500)
        accelerated=least_squares(residual,np.r_[best.x,[0,0,0]],args=(obs,sites),bounds=bounds(9),
                                   loss='soft_l1',f_scale=2,max_nfev=500)
        solutions.append(accelerated.x)
        predicted=predict(accelerated.x,obs,sites)
        result={'label':obs['label'],'observations':len(obs['t']),
            'constant_velocity_robust_state':best.x.tolist(),'constant_velocity_gaussian_state':affine.x.tolist(),
            'constant_acceleration_robust_state':accelerated.x.tolist(),
            'state_units':'position / 10000 m, velocity / 100 m/s, acceleration / m/s^2',
            'constant_velocity_robust_cost':best.cost,'constant_velocity_gaussian_cost':affine.cost,
            'constant_acceleration_robust_cost':accelerated.cost,
            'range_rms_m':float(np.sqrt(np.mean((predicted[0]-obs['length'])**2))),
            'rate_rms_mps':float(np.sqrt(np.mean((predicted[1]-obs['rate'])**2))),
            'data_singular_values':np.linalg.svd(accelerated.jac,compute_uv=False).tolist(),
            'multistarts':[{'cost':f.cost,'state':f.x.tolist()} for f in trials]}
        (args.output/f'{obs["label"]}-coarse.json').write_text(json.dumps(result,indent=2))
        print(obs['label'],json.dumps({k:result[k] for k in ('range_rms_m','rate_rms_mps','constant_acceleration_robust_cost')}),flush=True)
    # Joint adjustment: data and prior residuals stay separately inspectable.
    base=np.r_[np.zeros(9),np.concatenate(solutions)]
    sigma=np.array([30,30,75])
    def joint(p,with_prior=True):
        tx=sites+p[:9].reshape(3,3)*sigma
        pieces=[residual(p[9+9*i:18+9*i],obs,tx) for i,obs in enumerate(data)]
        if with_prior:pieces.append(p[:9])
        return np.concatenate(pieces)
    low,high=bounds(9);low=np.r_[np.full(9,-3),np.tile(low,len(data))];high=np.r_[np.full(9,3),np.tile(high,len(data))]
    fit=least_squares(joint,base,bounds=(low,high),loss='soft_l1',f_scale=2,max_nfev=700)
    # Validate the exact rotation gauge without priors. Both trajectories and
    # transmitter positions are rotated about the fixed receive phase centre.
    tx=sites+fit.x[:9].reshape(3,3)*sigma
    rotation=Rotation.from_rotvec([.001,-.0007,.0013]).as_matrix()
    worst_l=worst_v=0.
    for i,obs in enumerate(data):
        state=fit.x[9+9*i:18+9*i];rotated=state.reshape(3,3)@rotation.T
        a,b=predict(state,obs,tx);c,d=predict(rotated.ravel(),obs,tx@rotation.T)
        worst_l=max(worst_l,float(np.max(abs(a-c))));worst_v=max(worst_v,float(np.max(abs(b-d))))
    j=central_jacobian(lambda x:joint(x,False),fit.x)
    sv=np.linalg.svd(j,compute_uv=False)
    # Eliminate target-state directions before inspecting calibration information.
    jt=j[:,9:];q,s,_=np.linalg.svd(jt,full_matrices=False);q=q[:,s>s[0]*1e-10]
    projected=j[:,:9]-q@(q.T@j[:,:9])
    result={'state':fit.x.tolist(),'transmitter_corrections_enu_m':(fit.x[:9].reshape(3,3)*sigma).tolist(),
        'robust_cost':fit.cost,'data_singular_values':sv.tolist(),
        'calibration_singular_values_after_target_elimination':np.linalg.svd(projected,compute_uv=False).tolist(),
        'exact_rotation_gauge_max_path_error_m':worst_l,'exact_rotation_gauge_max_rate_error_mps':worst_v,
        'status':'diagnostic bounded fit; not accepted calibration',
        'receiver_uncertainty_included_in_fit':False,
        'warning':'Conditional on approximate receiver, unverified mast assignments and point-target motion model.'}
    name='joint-coarse.json' if args.labels==['A','B','C'] else 'joint-'+''.join(args.labels)+'-coarse.json'
    (args.output/name).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
