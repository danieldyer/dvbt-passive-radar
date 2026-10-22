"""Joint coherent geometry trials with explicit kinematics and held-out checks.

All trials remain diagnostic until they pass model/target stability checks. This
tool never writes production transmitter metadata. The motion and site priors
are exposed in the output instead of being confused with measured information.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize, LinearConstraint, Bounds

from geometry import SITES, observables, unit
from fit_geometry import observations
from phase_fit import (load_histories, physical_problem, physical_bounds,
                       fit_physical, evaluate_physical, path_model, trajectory, motion_penalty)


def kinematic_constraints(targets, degree):
    """Conservative component bounds checked at 61 common time nodes.

    These are screening limits, not a claim about the object's class: horizontal
    speed 150 m/s, vertical 20 m/s; acceleration 5/2 and jerk 2/1 in SI units.
    Constant-velocity, acceleration and jerk all derive from the same curve.
    """
    n = 3*(degree+1)
    derivative = motion_penalty(degree, [1.,1.,1.])*np.sqrt(61)
    limits = np.concatenate([np.tile(x,61) for x in ([150.,150.,20.],[5.,5.,2.],[2.,2.,1.])])
    local = derivative/limits[:,None]
    matrix = np.zeros((len(local)*targets,9+n*targets))
    for i in range(targets):
        matrix[i*len(local):(i+1)*len(local),9+i*n:9+(i+1)*n] = local
    return matrix


def fit_joint(data, sites, starts, degree, motion, max_nfev=1200, hard_motion=False):
    n = 3*(degree+1)
    problems = [physical_problem(ch, coarse, sites, degree, True, motion=motion)
                for ch, coarse in data]
    p = np.r_[np.zeros(9), np.concatenate(starts)]
    def fun(p, jacobian=False):
        pieces = []
        for i, problem in enumerate(problems):
            q = np.r_[p[9+n*i:9+n*(i+1)], p[:9]]
            if jacobian:
                local = problem(q, True)[:-9]
                j = np.zeros((len(local), len(p)))
                j[:, :9] = local[:, n:]
                j[:, 9+n*i:9+n*(i+1)] = local[:, :n]
                pieces.append(j)
            else:
                pieces.append(problem(q)[:-9])
        pieces.append(np.c_[np.eye(9), np.zeros((9, n*len(data)))] if jacobian else p[:9])
        return np.concatenate(pieces)
    low, high = physical_bounds(degree, False)
    low = np.r_[np.full(9, -3.), np.tile(low, len(data))]
    high = np.r_[np.full(9, 3.), np.tile(high, len(data))]
    if not hard_motion:
        return least_squares(fun, p, jac=lambda p:fun(p, True), bounds=(low, high),
                             x_scale='jac', max_nfev=max_nfev, ftol=1e-8, xtol=1e-9, gtol=1e-5)
    # A smoothness penalty alone is insufficient: large observation residuals
    # can overwhelm it. This trial enforces explicit physical derivative bounds.
    scale = np.r_[np.ones(9),np.tile(np.r_[np.full(3,.01),np.ones(n-3)],len(data))]
    constraint = kinematic_constraints(len(data),degree)*scale[None,:]
    norm = len(fun(p))
    def cost(x):
        r = fun(x*scale)
        return .5*r@r/norm
    def gradient(x):
        r = fun(x*scale);j = fun(x*scale,True)
        return (j.T@r)*scale/norm
    fit = minimize(cost,p/scale,jac=gradient,method='SLSQP',bounds=Bounds(low/scale,high/scale),
        constraints=[LinearConstraint(constraint,-1.,1.)],
        options={'maxiter':max_nfev,'ftol':1e-9})
    fit.x *= scale
    fit.cost = fit.fun*norm
    fit.optimality = None
    fit.max_kinematic_constraint = float(np.max(abs(kinematic_constraints(len(data),degree)@fit.x)))
    return fit


def projected_information(data, sites, state, degree):
    n = 3*(degree+1)
    parts = []
    for i, (ch, coarse) in enumerate(data):
        p = np.r_[state[9+n*i:9+n*(i+1)], state[:9]]
        problem = physical_problem(ch, coarse, sites, degree, True)
        local = problem(p, True)[:-9]
        j = np.zeros((len(local), len(state)))
        j[:, :9] = local[:, n:]; j[:, 9+n*i:9+n*(i+1)] = local[:, :n]
        parts.append(j)
    j = np.concatenate(parts)
    q, s, _ = np.linalg.svd(j[:, 9:], full_matrices=False)
    q = q[:, s>s[0]*1e-10]
    projected = j[:, :9]-q@(q.T@j[:, :9])
    _, singular, directions = np.linalg.svd(projected, full_matrices=False)
    return {'prior_scaled_data_singular_values':singular.tolist(),
            'calibration_directions':directions.tolist(),
            'definition':'noise-weighted local Jacobian, target curves eliminated; no kinematic or site prior rows',
            'limitation':'conditional empirical weights, point scattering and selected local branch; not a certified CRLB'}


def describe_curve(state, sites, degree, channels, coarse):
    u = np.linspace(-1, 1, 151)
    from numpy.polynomial import legendre as leg
    coefficient = state[:3*(degree+1)].reshape(degree+1, 3)*np.r_[1e4, np.full(degree, 100.)][:, None]
    derivatives = [leg.legval(u, leg.legder(coefficient, m=order, axis=0)).T/15**order
                   for order in range(4)]
    x, v, a, jerk = derivatives
    tx = sites+state[-9:].reshape(3,3)*[30,30,75] if len(state)>3*(degree+1) else sites
    look = unit(x-tx.mean(axis=0))
    curve = {'position_enu_m':x.tolist(), 'time':(u*15+channels[0]['seed']['epoch']).tolist(),
        'max_speed_mps':float(np.max(np.linalg.norm(v, axis=1))),
        'rms_acceleration_mps2':float(np.sqrt(np.mean(np.sum(a*a, axis=1)))),
        'rms_jerk_mps3':float(np.sqrt(np.mean(np.sum(jerk*jerk, axis=1)))),
        'midpoint_look_from_transmitter':look[len(u)//2].tolist(),
        'look_sweep_deg':float(np.rad2deg(np.arccos(np.clip(look[0]@look[-1], -1, 1))))}
    cu = coarse['t']/15
    cx = leg.legval(cu, coefficient).T
    cv = leg.legval(cu, leg.legder(coefficient, axis=0)).T/15
    length, rate = observables(cx, cv, tx[SITES[coarse['channel']]], np.zeros(3))
    curve['range_rms_m'] = float(np.sqrt(np.mean((length-coarse['length'])**2)))
    curve['path_rate_rms_mps'] = float(np.sqrt(np.mean((rate-coarse['rate'])**2)))
    curve['measured_time'] = (coarse['t']+channels[0]['seed']['epoch']).tolist()
    curve['measured_channel'] = coarse['channel'].tolist()
    curve['measured_length_m'] = coarse['length'].tolist()
    curve['predicted_length_m'] = length.tolist()
    curve['measured_path_rate_mps'] = coarse['rate'].tolist()
    curve['predicted_path_rate_mps'] = rate.tolist()
    return curve


def rank_one(z, w, train, iterations=100):
    """Weighted complex rank-one least squares with free per-time amplitude."""
    beta = np.ones(z.shape[0], dtype=complex)
    _, _, v = np.linalg.svd((z*np.sqrt(w))[:, train].T, full_matrices=False)
    beta = v[0].copy()
    for _ in range(iterations):
        a = np.sum(w[:, train]*np.conj(beta[:, None])*z[:, train], axis=0)/np.sum(w[:, train]*abs(beta[:, None])**2, axis=0)
        beta = np.sum(w[:, train]*np.conj(a)*z[:, train], axis=1)/np.sum(w[:, train]*abs(a)**2, axis=1)
        beta /= np.linalg.norm(beta)
    return beta


def coherent_response_check(channels, sites, state, degree):
    z, variance = [], []
    for ch, d in enumerate(channels):
        path, _ = path_model(state, d['u'], np.full(len(d['u']), ch), sites, degree,
                              len(state)>3*(degree+1))
        signal = d['value']*np.exp(1j*d['k']*(path-d['seed_delta']))
        bins = np.floor(d['time']-(d['seed']['epoch']-15)).astype(int)
        w = 1/d['variance']
        norm = np.bincount(bins, weights=w, minlength=30)[:30]
        values = (np.bincount(bins, weights=w*signal.real, minlength=30)[:30]+
                  1j*np.bincount(bins, weights=w*signal.imag, minlength=30)[:30])/norm
        z.append(values); variance.append(1/norm)
    z, variance = np.array(z), np.array(variance); w = 1/variance
    train = np.arange(30)%2==0
    beta = rank_one(z, w, train)
    test = ~train
    coherent = abs(np.sum(w[:, test]*np.conj(beta[:, None])*z[:, test], axis=0))**2/np.sum(w[:, test]*abs(beta[:, None])**2, axis=0)
    independent = np.sum(w[:, test]*abs(z[:, test])**2, axis=0)
    # Subtract the conditional noise expectation, not a display normalisation.
    pc = np.sum(coherent)-np.count_nonzero(test)
    pi = np.sum(independent)-len(channels)*np.count_nonzero(test)
    return {'beta_real':beta.real.tolist(), 'beta_imag':beta.imag.tolist(),
        'heldout_common_response_loss_db':float(10*np.log10(pi/pc)) if min(pi,pc)>0 else None,
        'coherent_score':float(pc), 'independent_score':float(pi),
        'time_bins':30, 'training':'even 1-second bins; odd bins evaluate frozen channel gains',
        'limitation':'conditional on fitted geometry and empirical noise; target association and geometry used the full interval',
        'values_real':z.real.tolist(), 'values_imag':z.imag.tolist(), 'variance':variance.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', default=list('ABCDE'))
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--motion', nargs=3, type=float, default=[100.,2.,.5], metavar=('SPEED','ACCEL','JERK'))
    parser.add_argument('--name', default='joint-kinematic')
    parser.add_argument('--bandwidth-hz', type=float)
    parser.add_argument('--hard-motion', action='store_true')
    args = parser.parse_args()
    sites = np.array(json.loads((args.output/'geometry-prior.json').read_text())['enu_transmitters_m'])
    coarse = observations(args.histories, args.labels)
    data = [(load_histories(args.histories, label, args.bandwidth_hz)[0], obs) for label, obs in zip(args.labels, coarse)]
    starts = []
    fixed_reports = []
    for label, (ch, obs) in zip(args.labels, data):
        prefix = label+'-gated' if args.bandwidth_hz else label
        models = json.loads((args.output/f'{prefix}-phase-geometry.json').read_text())
        model = next(m for m in models if m['degree']==args.degree)
        fixed = fit_physical(ch, obs, sites, np.array(model['fixed_state']), args.degree, motion=args.motion)
        starts.append(fixed.x)
        fixed_reports.append(evaluate_physical(ch, obs, sites, fixed.x, args.degree, False))
    fit = fit_joint(data, sites, starts, args.degree, args.motion, hard_motion=args.hard_motion)
    n = 3*(args.degree+1)
    result = {'labels':args.labels, 'degree':args.degree, 'motion_rms_penalty_scales':args.motion,
        'bandwidth_hz':args.bandwidth_hz,
        'hard_motion':args.hard_motion,
        'maximum_normalized_kinematic_constraint':getattr(fit,'max_kinematic_constraint',None),
        'state':fit.x.tolist(), 'cost_including_priors':fit.cost, 'converged':bool(fit.success),
        'termination':fit.message, 'evaluations':fit.nfev, 'optimality':fit.optimality,
        'transmitter_corrections_enu_m':(fit.x[:9].reshape(3,3)*[30,30,75]).tolist(),
        'status':'diagnostic candidate, never automatically applied', 'fixed':fixed_reports,
        'targets':[], 'information':projected_information(data, sites, fit.x, args.degree)}
    for i, (label, (ch, obs)) in enumerate(zip(args.labels, data)):
        state = np.r_[fit.x[9+n*i:9+n*(i+1)], fit.x[:9]]
        result['targets'].append({'label':label,
            'focus':evaluate_physical(ch, obs, sites, state, args.degree, True),
            'curve':describe_curve(state, sites, args.degree, ch, obs),
            'common_response':coherent_response_check(ch, sites, state, args.degree)})
    (args.output/(args.name+'.json')).write_text(json.dumps(result, indent=2))
    print(args.name, 'converged', fit.success, 'cost', fit.cost,
          'TX', result['transmitter_corrections_enu_m'], flush=True)
    for target in result['targets']:
        curve = target['curve']
        print(target['label'], 'range/rate RMS', curve['range_rms_m'], curve['path_rate_rms_mps'],
              'max speed', curve['max_speed_mps'], 'RMS a/j', curve['rms_acceleration_mps2'],curve['rms_jerk_mps3'],
              'shared loss', target['common_response']['heldout_common_response_loss_db'], flush=True)


if __name__=='__main__':
    main()
