"""Complex, variable-projection fits for the one-off calibration analysis.

The phase seed is reversible and is included in every forward model. Channel
amplitudes are nuisance parameters, never assumed equal across RF frequencies.
Empirical bin variances supply weights, not a claim of independent thermal noise.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre as leg
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from geometry import C, RF, SITES, unit, central_jacobian
from fit_geometry import observations
from complex_likelihood import gate, profile, operator_for, rebin


def load_histories(directory, label, bandwidth_hz=None):
    description = json.loads((directory / f'{label}.json').read_text())
    channels = []
    for i, seed in enumerate(description['channels']):
        seed['duration_s'] = description['end']-description['start']
        with np.load(directory / f'{label}-{i}.npz') as data:
            if bandwidth_hz is not None:
                d = rebin(data, description['start'], description['end'])
            else:
                good = (data['rows'] > 1) & (data['variance'] > 0)
                d = {key: data[key][good] for key in ('time', 'value', 'variance')}
        d['u'] = (d['time']-seed['epoch']) / seed['scale']
        d['seed_delta'] = np.polynomial.polynomial.polyval(d['u'], seed['path'])-seed['path'][0]
        d['k'] = 2*np.pi*seed['rf_hz']/C
        d['seed'] = seed
        d['duration_s'] = seed['duration_s']
        if bandwidth_hz is not None:
            d['bandwidth_hz'] = bandwidth_hz
            d['operator'], d['basis'] = gate(d['time'], d['variance'], bandwidth_hz, seed['epoch'], d['duration_s'])
        channels.append(d)
    return channels, description


def projection(y, variance, phase, derivative=None, train=None):
    """Profile out a constant complex amplitude; exact real residual/Jacobian."""
    w = 1/variance
    if train is None:
        train = np.ones(len(y), dtype=bool)
    w = w*train
    total = w.sum()
    u = np.exp(1j*np.remainder(phase, 2*np.pi))
    a = np.sum(w*np.conj(u)*y)/total
    error = (a*u-y)*np.sqrt(2/variance)
    if derivative is None:
        return error, a, u
    du = 1j*u[:, None]*derivative
    da = np.sum(w[:, None]*np.conj(du)*y[:, None], axis=0)/total
    jacobian = (u[:, None]*da + a*du)*np.sqrt(2/variance[:, None])
    return error, jacobian, a, u


def phase_initializer(d, order, train):
    # Only used to enter the basin of the complex likelihood. Faint-bin phases
    # are never treated as Gaussian measurements in the final phase fit.
    w = train/d['variance']
    if 'operator' in d:
        q = operator_for(d, train)
        smooth = d['basis']@np.linalg.lstsq(q@d['basis'], q@d['value'], rcond=None)[0]
    else:
        smooth = (gaussian_filter1d((d['value']*w).real, 3) +
                  1j*gaussian_filter1d((d['value']*w).imag, 3))
        smooth /= np.maximum(gaussian_filter1d(w, 3), 1e-30)
    basis = leg.legvander(d['u'], order)
    phase = np.unwrap(np.angle(smooth))
    sw = np.sqrt(w*abs(smooth)**2)
    return np.linalg.lstsq(basis*sw[:, None], phase*sw, rcond=None)[0][1:]


def polynomial_fit(d, order, train=None):
    if train is None:
        train = np.ones(len(d['time']), dtype=bool)
    basis = leg.legvander(d['u'], order)[:, 1:]
    start = phase_initializer(d, order, train)
    q = operator_for(d, train) if 'operator' in d else None
    def fun(p, jacobian=False):
        if q is not None:
            r, j, _, _ = profile(d['value'], q, basis@p, basis)
            return np.r_[j.real, j.imag] if jacobian else np.r_[r.real, r.imag]
        r, j, _, _ = projection(d['value'], d['variance'], basis@p, basis, train)
        return np.r_[j[train].real, j[train].imag] if jacobian else np.r_[r[train].real, r[train].imag]
    fit = least_squares(fun, start, jac=lambda p:fun(p, True), max_nfev=250,
                        ftol=1e-10, xtol=1e-10, gtol=1e-7)
    if q is not None:
        r, a, u = profile(d['value'], q, basis@fit.x)
    else:
        r, a, u = projection(d['value'], d['variance'], basis@fit.x, train=train)
    return fit.x, a, u, r


def trajectory(state, u, degree):
    coefficients = state[:3*(degree+1)].reshape(degree+1, 3)
    scales = np.r_[1e4, np.full(degree, 100.)]
    basis = leg.legvander(u, degree)*scales
    return basis@coefficients, basis


def path_model(state, u, channel, sites, degree, free_sites):
    x, basis = trajectory(state, u, degree)
    n = 3*(degree+1)
    sigma = np.array([30., 30., 75.])
    tx = sites[SITES[channel]]
    if free_sites:
        tx = tx+state[n:].reshape(3, 3)[SITES[channel]]*sigma
    a, b = x-tx, x
    path = np.linalg.norm(a, axis=1)+np.linalg.norm(b, axis=1)-np.linalg.norm(tx, axis=-1)
    grad = unit(a)+unit(b)
    j = (basis[:, :, None]*grad[:, None, :]).reshape(len(u), n)
    if free_sites:
        jt = np.zeros((len(u), 3, 3))
        rows = np.arange(len(u))
        jt[rows, SITES[channel]] = (-unit(a)-unit(tx))*sigma
        j = np.c_[j, jt.reshape(len(u), 9)]
    return path, j


def motion_penalty(degree, scales, times=None):
    """Integrated squared speed, acceleration and jerk of one physical curve.

    Scales are RMS regularisation strengths, not measured manoeuvre statistics.
    Their sensitivity must be reported. Derivatives use seconds, not the
    dimensionless Legendre coordinate; a scale of 15 s is part of this study.
    """
    u = np.linspace(-1, 1, 61) if times is None else np.asarray(times)
    normalise = np.r_[1e4, np.full(degree, 100.)]
    blocks = []
    for order, sigma in enumerate(scales, 1):
        coefficients = leg.legder(np.eye(degree+1), m=order, axis=0)
        basis = leg.legval(u, coefficients).T*normalise/15**order
        blocks.append(np.kron(basis, np.eye(3))/(sigma*np.sqrt(len(u))))
    return np.concatenate(blocks)


def physical_problem(channels, coarse, sites, degree, free_sites=False, train=None, motion=None):
    n = 3*(degree+1)
    if train is None:
        train = [np.ones(len(d['time']), dtype=bool) for d in channels]
    cu = coarse['t']/15
    # Independent 1-s peak observations were used to select the candidates.
    # Their weights are conditional scales, so no chi-square p-value is claimed.
    cl = coarse['channel']
    range_train = np.ones(len(cu), dtype=bool)
    if not all(np.all(t) for t in train):
        range_train = np.array([np.interp(t, channels[c]['u'], train[c].astype(float)) > .5
                                for t, c in zip(cu, cl)])
    regularise = None if motion is None else motion_penalty(degree, motion)
    operators = [operator_for(d, t) if 'operator' in d else None for d, t in zip(channels, train)]
    def fun(p, jacobian=False):
        pieces = []
        for c, d in enumerate(channels):
            length, derivative = path_model(p, d['u'], np.full(len(d['u']), c), sites, degree, free_sites)
            phase = -d['k']*(length-d['seed_delta'])
            if operators[c] is not None:
                r, j, _, _ = profile(d['value'], operators[c], phase, -d['k']*derivative)
                pieces.append(np.r_[j.real, j.imag] if jacobian else np.r_[r.real, r.imag])
            else:
                r, j, _, _ = projection(d['value'], d['variance'], phase, -d['k']*derivative, train[c])
                pieces.append(np.r_[j[train[c]].real, j[train[c]].imag] if jacobian else
                              np.r_[r[train[c]].real, r[train[c]].imag])
        length, j = path_model(p, cu, cl, sites, degree, free_sites)
        pieces.append((j/coarse['sigma_l'][:, None])[range_train] if jacobian else
                      ((length-coarse['length'])/coarse['sigma_l'])[range_train])
        if regularise is not None:
            matrix = np.c_[regularise, np.zeros((len(regularise), 9))] if free_sites else regularise
            pieces.append(matrix if jacobian else matrix@p)
        if free_sites:
            pieces.append(np.c_[np.zeros((9, n)), np.eye(9)] if jacobian else p[n:])
        return np.concatenate(pieces)
    return fun


def physical_bounds(degree, free_sites):
    n = 3*(degree+1); total = n+9*free_sites
    low, high = np.full(total, -np.inf), np.full(total, np.inf)
    low[:3], high[:3] = [-3., -4., -.03], [6., 4., 1.2]
    if free_sites:
        low[n:], high[n:] = -3., 3.
    return low, high


def fit_physical(channels, coarse, sites, start, degree, free_sites=False, train=None, max_nfev=1500, motion=None):
    n = 3*(degree+1)
    state = np.r_[start[:n], np.zeros(9)] if free_sites and len(start)==n else start.copy()
    fun = physical_problem(channels, coarse, sites, degree, free_sites, train, motion)
    low, high = physical_bounds(degree, free_sites)
    fit = least_squares(fun, state, jac=lambda p:fun(p, True), bounds=(low, high), x_scale='jac',
                        max_nfev=max_nfev, ftol=1e-9, xtol=1e-10, gtol=1e-6)
    return fit


def unwrapped_start(channels, coarse, sites, seed, degree, polys):
    n = 3*(degree+1)
    # Coarse range/rate trajectory supplies the spatial branch. The unwrapped
    # initializer is followed by the circular complex likelihood.
    t = np.linspace(-15, 15, 101)
    x = seed[:3]*1e4+t[:, None]*seed[3:6]*100
    if len(seed)==9:
        x += .5*t[:, None]**2*seed[6:9]
    coefficients = np.linalg.lstsq(leg.legvander(t/15, degree), x, rcond=None)[0]
    scales = np.r_[1e4, np.full(degree, 100.)]
    start = (coefficients/scales[:, None]).ravel()
    paths = []
    weights = []
    for d, poly in zip(channels, polys):
        path = d['seed_delta']-leg.legvander(d['u'], len(poly))[:, 1:]@poly/d['k']
        paths.append(path)
        # Constant fitted amplitude SNR avoids weighting individual noise spikes.
        _, amp, _ = projection(d['value'], d['variance'],
                               leg.legvander(d['u'], len(poly))[:, 1:]@poly)
        weights.append(d['k']*np.sqrt(2*abs(amp)**2/d['variance']))
    def fun(p, jacobian=False):
        pieces = []
        for c, d in enumerate(channels):
            length, j = path_model(p, d['u'], np.full(len(d['u']), c), sites, degree, False)
            w = weights[c]
            delta = length-paths[c]
            if jacobian:
                pieces.append((j-np.average(j, axis=0, weights=w*w))*w[:, None])
            else:
                pieces.append((delta-np.average(delta, weights=w*w))*w)
        length, j = path_model(p, coarse['t']/15, coarse['channel'], sites, degree, False)
        pieces.append(j/coarse['sigma_l'][:, None] if jacobian else (length-coarse['length'])/coarse['sigma_l'])
        return np.concatenate(pieces)
    low, high = np.full(n, -np.inf), np.full(n, np.inf)
    low[:3], high[:3] = [-3., -4., -.03], [6., 4., 1.2]
    fit = least_squares(fun, start, jac=lambda p:fun(p, True), bounds=(low, high), x_scale='jac',
                        max_nfev=500, ftol=1e-9, xtol=1e-10, gtol=1e-6)
    return fit


def evaluate_physical(channels, coarse, sites, state, degree, free_sites, train=None):
    report = []
    for c, d in enumerate(channels):
        path, _ = path_model(state, d['u'], np.full(len(d['u']), c), sites, degree, free_sites)
        if 'operator' in d:
            q = operator_for(d, None if train is None else train[c])
            r, amp, u = profile(d['value'], q, -d['k']*(path-d['seed_delta']))
            test_q = q if train is None else operator_for(d, ~train[c])
            error = test_q@(amp*u-d['value'])
            coherent = abs(np.vdot(test_q@u, test_q@d['value']))**2/np.linalg.norm(test_q@u)**2
            total_power = np.linalg.norm(test_q@d['value'])**2
            report.append({'channel':c, 'amplitude':[amp.real, amp.imag],
                'weighted_error':float(np.sum(abs(error)**2)),
                'coherent_snr_db':float(10*np.log10(max(coherent, 1e-30))),
                'coherent_fraction':float(coherent/total_power)})
            continue
        r, amp, u = projection(d['value'], d['variance'], -d['k']*(path-d['seed_delta']),
                               train=None if train is None else train[c])
        w = 1/d['variance']
        coherent = abs(np.sum(w*np.conj(u)*d['value']))**2/w.sum()
        test = np.ones(len(w), dtype=bool) if train is None else ~train[c]
        report.append({'channel':c, 'amplitude':[amp.real, amp.imag],
            'weighted_error':float(np.sum(abs(r[test])**2)/2),
            'coherent_snr_db':float(10*np.log10(max(coherent, 1e-30))),
            'coherent_fraction':float(coherent/np.sum(w*abs(d['value'])**2))})
    length, _ = path_model(state, coarse['t']/15, coarse['channel'], sites, degree, free_sites)
    return {'channels':report,'range_rms_m':float(np.sqrt(np.mean((length-coarse['length'])**2)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--histories', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--labels', nargs='+', default=['A', 'B', 'C'])
    p.add_argument('--bandwidth-hz', type=float)
    args = p.parse_args()
    sites = np.array(json.loads((args.output/'geometry-prior.json').read_text())['enu_transmitters_m'])
    all_coarse = observations(args.histories, args.labels)
    for label, coarse in zip(args.labels, all_coarse):
        channels, description = load_histories(args.histories, label, args.bandwidth_hz)
        prefix = label+'-gated' if args.bandwidth_hz else label
        polys = []; diagnostics = []
        for d in channels:
            fits = []
            # 1 s alternating blocks expose overfitting without demanding that
            # an unknown manoeuvre extrapolate from one half of a pass to another.
            train = np.floor(d['time']-description['start']).astype(int)%2==0
            for order in [3, 5, 7, 9]:
                coef, amp, u, r = polynomial_fit(d, order)
                cv, ca, cu, cr = polynomial_fit(d, order, train)
                if 'operator' in d:
                    test_q = operator_for(d, ~train)
                    test_error = np.linalg.norm(test_q@(ca*cu-d['value']))**2
                    snr = abs(amp)**2*np.linalg.norm(d['operator']@u)**2
                else:
                    test_error = np.sum(abs(cr[~train])**2)/2
                    snr = abs(amp)**2*np.sum(1/d['variance'])
                fits.append({'order':order, 'coefficients':coef.tolist(),
                    'amplitude':[amp.real, amp.imag], 'weighted_error':float(np.sum(abs(r)**2)/2),
                    'heldout_error':float(test_error),
                    'coherent_snr_db':float(10*np.log10(snr))})
            polys.append(np.array(fits[-1]['coefficients']))
            diagnostics.append(fits)
        (args.output/f'{prefix}-phase-polynomials.json').write_text(json.dumps(diagnostics, indent=2))
        seed = np.array(json.loads((args.output/f'{label}-coarse.json').read_text())['constant_acceleration_robust_state'])
        models = []
        for degree in [3, 4, 5]:
            initializer = unwrapped_start(channels, coarse, sites, seed, degree, polys)
            fixed = fit_physical(channels, coarse, sites, initializer.x, degree)
            free = fit_physical(channels, coarse, sites, fixed.x, degree, True)
            model = {'degree':degree, 'fixed_state':fixed.x.tolist(), 'free_state':free.x.tolist(),
                'fixed_cost':fixed.cost, 'free_cost_including_prior':free.cost,
                'fixed':evaluate_physical(channels, coarse, sites, fixed.x, degree, False),
                'free':evaluate_physical(channels, coarse, sites, free.x, degree, True),
                'transmitter_corrections_enu_m':(free.x[-9:].reshape(3,3)*[30,30,75]).tolist(),
                'fixed_converged':bool(fixed.success), 'free_converged':bool(free.success),
                'fixed_optimality':fixed.optimality, 'free_optimality':free.optimality,
                'fixed_evaluations':fixed.nfev, 'free_evaluations':free.nfev}
            models.append(model)
            print(label, degree, 'cost', fixed.cost, free.cost, 'TX', model['transmitter_corrections_enu_m'], flush=True)
            (args.output/f'{prefix}-phase-geometry.json').write_text(json.dumps(models, indent=2))


if __name__=='__main__':
    main()
