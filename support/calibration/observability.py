"""Controlled local-information study: candidate count versus look diversity.

This is a conditional ideal point-target calculation, not a real-data CRLB.
Constant-velocity trajectories deliberately give kinematic continuity its
strongest reasonable case; compare the flexible real-track fits separately.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from geometry import C, RF
from phase_fit import path_model, physical_problem
from joint_refinement import projected_information


def synthetic(sites, positions, velocities, degree=1):
    data, states = [], []
    for index, (position, velocity) in enumerate(zip(positions, velocities)):
        state = np.r_[position/1e4, velocity*15/100, np.zeros(3*(degree-1))]
        states.append(state)
        channels = []
        for ch in range(5):
            t = np.arange(300)*.1+.05+ch*.00013
            u = (t-15)/15
            length, _ = path_model(state, u, np.full(len(u), ch), sites, degree, False)
            seed_delta = length-length[len(u)//2]
            k = 2*np.pi*RF[ch]/C
            channels.append({'u':u, 'time':t, 'seed_delta':seed_delta, 'k':k,
                'variance':np.full(len(u),.1), 'value':np.exp(-1j*k*(length-seed_delta)),
                'seed':{'epoch':15.}})
        ct = np.tile(np.arange(-14.5,15,1.),5); cc = np.repeat(np.arange(5),30)
        length, _ = path_model(state,ct/15,cc,sites,degree,False)
        coarse = {'t':ct,'channel':cc,'length':length,'sigma_l':np.full(len(ct),5.)}
        data.append((channels,coarse))
    return data,np.r_[np.zeros(9),np.concatenate(states)]


def report(sites, positions, velocities, degree):
    data,state = synthetic(sites,np.asarray(positions),np.asarray(velocities),degree)
    result = projected_information(data,sites,state,degree)
    singular = np.array(result['prior_scaled_data_singular_values'])
    vectors = np.array(result['calibration_directions'])
    result['positions_enu_m'] = positions
    result['velocities_mps'] = velocities
    result['trajectory_degree'] = degree
    result['prior_to_posterior_std_ratio_per_mode'] = (1/np.sqrt(1+singular**2)).tolist()
    # Pairwise tower distances are invariant under the three common rotations.
    covariance = (vectors[:6].T/singular[:6]**2)@vectors[:6]
    distance_sigma = []
    for a,b in [(0,1),(0,2),(1,2)]:
        v=(sites[a]-sites[b])/np.linalg.norm(sites[a]-sites[b])
        gradient=np.zeros((3,3));gradient[a]=v;gradient[b]=-v
        gradient=(gradient*[30,30,75]).ravel()
        distance_sigma.append(float(np.sqrt(gradient@covariance@gradient)))
    result['conditional_pairwise_distance_sigma_m'] = distance_sigma
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sites=np.array(json.loads((args.output/'geometry-prior.json').read_text())['enu_transmitters_m'])
    near=[[-800.,-140+70*i,30.] for i in range(5)]
    near_v=[[-14.,float(i-2),0.] for i in range(5)]
    diverse=[[-5000.,-8000.,30.],[6000.,-8000.,1200.],[18000.,9000.,3000.],
             [35000.,2000.,2000.],[12000.,11000.,2500.]]
    diverse_v=[[15.,4.,0.],[40.,25.,2.],[-45.,30.,-2.],[-20.,-60.,3.],[55.,-35.,1.]]
    result={'assumptions':{'duration_s':30,'per_100ms_sample_snr_db':10,'range_peak_sigma_m':5,
        'scatterer':'one ideal nondispersive point per target with unknown constant per-band complex gain',
        'noise':'independent complex Gaussian; stronger than what has been established for the capture',
        'scope':'local information only, assumes correct association and geometric branch; not global ambiguity resolution'},
        'one_near_cv':report(sites,near[:1],near_v[:1],1),
        'five_near_cv':report(sites,near,near_v,1),
        'five_diverse_cv':report(sites,diverse,diverse_v,1),
        'five_diverse_flexible':report(sites,diverse,diverse_v,4)}
    (args.output/'observability.json').write_text(json.dumps(result,indent=2))
    for name,item in result.items():
        if name!='assumptions':
            print(name,'singular',item['prior_scaled_data_singular_values'],
                  'distance sigma',item['conditional_pairwise_distance_sigma_m'],flush=True)


if __name__=='__main__':
    main()
