"""Conditional information with unknown versus independently known tracks.

Known-track cases are idealized design studies at the current fitted positions,
not added observations. They help determine whether an external anchor would
resolve the present degeneracy. No physical coordinates are adopted here.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from coherent_fit import CoherentProblem, OTHER, information
from complex_likelihood import projected_profile
from fit_geometry import observations
from geographic_io import read_profiles
from geometry import unit
from phase_fit import load_histories
from polar_model import PolarTrajectory
from range_fit import load_fields
from site_hypotheses import AboveGroundProblem
from terrain import Terrain


def fixed_world_jacobian(problem, p, i):
    pr = problem.problems[i]
    state = np.r_[p[9+i*problem.n:9+(i+1)*problem.n], p[:9]]
    magnitude = np.ones(5)
    magnitude[OTHER] = np.exp(p[problem.magnitude_offset+4*i:problem.magnitude_offset+4*(i+1)])
    beta = np.zeros(5); beta[OTHER] = p[problem.beta_offset:problem.magnitude_offset]
    bias = np.zeros(5)
    if problem.range_bias:
        bias[OTHER] = p[problem.bias_offset:]
    templates, derivatives = [], []
    for ch, field in enumerate(pr.fields):
        x, _, tx = pr.model.positions(state, field.time, pr.sites, True)
        site = pr.model.channel_sites[ch]
        transmitter = tx[site]+pr.model.channel_offsets[ch]
        path = np.linalg.norm(x-transmitter,axis=1)+np.linalg.norm(x,axis=1)-np.linalg.norm(transmitter)
        derivative = np.zeros((len(x), 9))
        derivative[:,3*site:3*site+3] = (-unit(x-transmitter)-unit(transmitter))*[30.,30.,75.]
        base, local = field.template(path, derivative, bias[ch], problem.range_bias)
        gain = magnitude[ch]*np.exp(1j*beta[ch])
        u = base*gain
        du = np.zeros((len(u), problem.count), complex)
        du[:,:9] = local[:,:9]*gain
        if ch != 1:
            c = int(np.flatnonzero(OTHER==ch)[0])
            du[:,problem.beta_offset+c] = 1j*u
            du[:,problem.magnitude_offset+4*i+c] = u
            if problem.range_bias:
                du[:,problem.bias_offset+c] = gain*local[:,-1]
        templates.append(u); derivatives.append(du)
    u, du = np.concatenate(templates), np.concatenate(derivatives)
    _, amplitude = projected_profile(problem.data[i], u)
    _, j, _ = projected_profile(amplitude*u, u, du)
    return np.r_[j.real, j.imag]


def covariance(j, prior_weights):
    u, s, _ = np.linalg.svd(j[:,9:], full_matrices=False)
    u = u[:,s>s[0]*1e-10]
    projected = j[:,:9]-u@(u.T@j[:,:9])
    return np.linalg.inv(projected.T@projected+np.diag(prior_weights**2)), np.linalg.svd(projected,compute_uv=False)


def calibration_covariance(j, problem):
    indices = np.r_[np.arange(9), np.arange(problem.beta_offset,problem.magnitude_offset)]
    other = np.setdiff1d(np.arange(problem.count),indices)
    u, s, _ = np.linalg.svd(j[:,other],full_matrices=False)
    u = u[:,s>s[0]*1e-10]
    projected = j[:,indices]-u@(u.T@j[:,indices])
    precision = projected.T@projected+np.diag(np.r_[problem.prior_weights**2,np.zeros(4)])
    eigen = np.linalg.eigvalsh(precision)
    if eigen[0] <= 0:
        raise ValueError('Calibration posterior contains an unanchored direction')
    return {'parameter_indices':indices.tolist(),'posterior_covariance':np.linalg.inv(precision).tolist(),
            'precision_eigenvalues':eigen.tolist(),
            'parameters':'Three site ENU corrections divided by 30/30/75 m, then Seven/Nine/Ten/ABC phase offsets in radians; SBS fixes the phase gauge.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['histories','patches','templates','terrain','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--range-bias',action='store_true')
    args = parser.parse_args()
    suffix = '-bias' if args.range_bias else ''
    source = args.output/f'licensed-joint-main{suffix}.json'
    saved = json.loads(source.read_text())
    state = np.array(saved['best']['state'])
    sites, offsets = np.array(saved['site_positions_enu_m']), np.array(saved['channel_offsets_enu_m'])
    terrain = Terrain(args.terrain,json.loads((args.output/'geometry-prior.json').read_text()))
    _, records = read_profiles(args.templates)
    problems = []
    for label, coarse in zip(saved['labels'], observations(args.histories,saved['labels'])):
        channels, description = load_histories(args.histories,label,1.)
        pr = AboveGroundProblem(channels,coarse,sites,
            load_fields(args.patches,label,description,records),terrain=terrain)
        pr.model = PolarTrajectory(channels[1]['seed'], channel_sites=saved['channel_sites'], channel_offsets=offsets)
        problems.append(pr)
    problem = CoherentProblem(problems,args.range_bias,site_prior_sigma=[10,10,10])
    corrected = information(problem,state)
    (args.output/f'licensed-joint-main{suffix}-information.json').write_text(json.dumps(corrected,indent=2))
    expected = copy.copy(problem); expected.data = problem.fitted_observations(state)
    j = expected.residual(state,True,False,False)
    predictive = calibration_covariance(j,problem)
    predictive.update({'site_positions_enu_m':sites.tolist(),
        'transmitter_corrections_enu_m':(state[:9].reshape(3,3)*[30,30,75]).tolist(),
        'channel_sites':saved['channel_sites'],'channel_offsets_enu_m':offsets.tolist(),
        'status':'Local single-branch conditional covariance, empirical noise and fixed receiver; no accepted calibration'})
    (args.output/f'calibration-predictive{suffix}.json').write_text(json.dumps(predictive,indent=2))
    rows = np.r_[0, np.cumsum([2*len(v) for v in problem.data])]
    fixed = [fixed_world_jacobian(problem,state,i) for i in range(problem.m)]
    fitted_sites = sites+state[:9].reshape(3,3)*[30,30,75]
    difference = unit(fitted_sites[0]-fitted_sites[2])
    gradient = np.r_[difference,np.zeros(3),-difference]*np.tile([30,30,75],3)
    results = []
    for known in [[], *[[i] for i in range(problem.m)], list(range(problem.m))]:
        trial = j.copy()
        for i in known:
            trial[rows[i]:rows[i+1]] = fixed[i]
        cov, singular = covariance(trial,problem.prior_weights)
        item = {'exact_known_tracks':[saved['labels'][i] for i in known],
            'ornata_ridge_separation_posterior_sigma_m':float(np.sqrt(gradient@cov@gradient)),
            'site_coordinate_posterior_sigma_m':(np.sqrt(np.diag(cov))*np.tile([30,30,75],3)).tolist(),
            'data_normalized_singular_values':singular.tolist()}
        results.append(item)
        print(item['exact_known_tracks'],'baseline sigma',item['ornata_ridge_separation_posterior_sigma_m'],flush=True)
    result = {'cases':results,'baseline_prior_sigma_m':float(np.sqrt(200.)),
        'construction':'Expected signal Jacobian; fitted amplitudes and empirical covariance; independently known world trajectories held fixed. Unknown channel phases and magnitudes are still eliminated.',
        'scope':'Hypothetical zero-error track anchors at current trial positions, not measurements or a prediction for an arbitrary real anchor.',
        'source_fit':str(source),'motion_prior_used':False,'site_prior_sigma_m':10.}
    (args.output/f'anchor-information{suffix}.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
