"""Radar-derived path/yaw initialization over the complete ship recording.

Local exact-path phase likelihoods use measured complex BP histories. AIS only
defines the starting position/course frame. This is an imaging initialization,
not a certified six-degree-of-freedom trajectory: the full range response and
cross-range observability must still be checked separately.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import LinearConstraint, minimize

from ais import read_csv, utc_seconds
from heading_seed import Band
from imaging import physical_state, sites
from motion import excess_path
from spline_motion import basis, canonical_model


def choose_points(values, weight, time, pixels, selected, count=8):
    """Noncoherent 1 s subaperture power, using training observations only."""
    group = np.floor(time-time[0]).astype(int)
    power = np.zeros(values.shape[1])
    for g in np.unique(group):
        mask = (group == g)&selected&(weight > 0)
        if mask.any():
            power += abs(np.sum(values[mask]*weight[mask, None], axis=0))**2/weight[mask].sum()
    allowed, chosen = np.ones(len(pixels), bool), []
    for _ in range(count):
        if not allowed.any():
            break
        i = int(np.argmax(np.where(allowed, power, -1)))
        chosen.append(i)
        allowed &= np.sum((pixels[:, :2]-pixels[i, :2])**2, axis=1) >= 48**2
    return np.asarray(chosen)


def local_fit(bands, *, iterations=250):
    npath, size = bands[0].npath, bands[0].count
    active = np.r_[np.arange(npath-1), np.arange(npath, size)]
    energy = sum(np.sum(b.weight[b.selected, None]*abs(b.value[b.selected])**2) for b in bands)
    def expand(x):
        p = np.zeros(size); p[active] = x
        return p
    def objective(x):
        results = [b.evaluate(expand(x)) for b in bands]
        return sum(r[0] for r in results)/energy, sum(r[1] for r in results)[active]/energy
    # The coarse delay-rate seed is also radar-derived. Adjacent phase products
    # estimate a rate only; they are not applied to the observations or accepted
    # as the final likelihood solution.
    t = bands[0].time
    rate = np.zeros(len(t)-1); precision = np.zeros(len(t)-1)
    for b in bands:
        pair = np.sum(b.value[1:]*b.value[:-1].conj(), axis=1)
        dt = np.diff(b.time)
        w = abs(pair)*(b.selected[1:] & b.selected[:-1])
        measured = -np.angle(pair)/(b.k*dt)
        rate += w*measured; precision += w
    rate = np.divide(rate, precision, out=np.zeros_like(rate), where=precision > 0)
    # Initializer only; remove out-of-band sample noise with a 1 s mean.
    step = float(np.median(np.diff(t)))
    kernel = np.ones(max(1, round(1/step)))
    valid = precision > 0
    total = np.convolve(valid.astype(float), kernel, 'same')
    rate = np.divide(np.convolve(rate*valid, kernel, 'same'), total,
        out=np.zeros_like(rate), where=total > 0)
    path = cumulative_trapezoid(np.r_[rate[0], rate], t, initial=0)
    path -= np.interp(np.mean(bands[0].time), t, path)
    seed = np.linalg.lstsq(bands[0].path_basis[:, :npath-1], path, rcond=None)[0]
    trials = []
    bounds = [(-80., 80.)]*(npath-1)+[(-25., 25.)]*(size-npath)
    for seeded, yaw in ((False, 0.), (True, 0.), (True, -5.), (True, 5.)):
        initial = np.r_[np.clip(seed, -79, 79) if seeded else np.zeros(npath-1), np.full(size-npath, yaw)]
        r = minimize(objective, initial, jac=True, bounds=bounds, method='L-BFGS-B',
            options={'maxiter': iterations, 'maxcor': 30, 'maxls': 30, 'ftol': 1e-13, 'gtol': 1e-7})
        trials.append(r)
    best = min(trials, key=lambda r: r.fun)
    full = expand(best.x)
    return full, {'objective': float(best.fun), 'converged': bool(best.success and abs(best.jac).max() < 2e-6),
        'maximum_gradient': float(abs(best.jac).max()), 'iterations': best.nit,
        'bound_hit': bool(np.any(abs(best.x[:npath-1]) > 79.99) or np.any(abs(best.x[npath-1:]) > 24.99)),
        'trials': [{'objective': float(r.fun), 'success': bool(r.success), 'maximum_gradient': float(abs(r.jac).max())} for r in trials],
        'validation': [b.validation(full) for b in bands]}


def constrained_curve(design, values, weight, constraint, limit):
    """Convex weighted fit with explicit derivative bounds, including endpoints."""
    scale = weight.sum()
    root = np.sqrt(weight/scale)
    u, s, vh = np.linalg.svd(design*root[:, None], full_matrices=False)
    if s[-1] < s[0]*1e-12:
        raise ValueError('Curve joining has unresolved coefficient directions')
    transform = vh.T/s
    b = u.T@(values*root)
    def objective(x):
        return float(.5*x@x-b@x), x-b
    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method='SLSQP',
        constraints=[LinearConstraint(constraint@transform, -limit, limit)],
        options={'maxiter': 1000, 'ftol': 1e-12})
    coefficients = transform@result.x
    violation = float(max(0., np.max(abs(constraint@coefficients)-limit)))
    if not result.success or violation > 1e-7:
        raise ValueError(f'Constrained motion joining failed: {result.message}; {violation=}')
    return coefficients


def join_curves(locals, interval, path_spacing=3., heading_spacing=8.):
    """Join derivative observations without inventing unobservable path constants.

    Each local phase likelihood has a path-origin gauge. Only its path rate is
    transferred. One global constant is fixed at the capture midpoint. Yaw
    retains the declared coarse body-frame convention.
    """
    path_design, rate_values, yaw_design, yaw_values, weights = [], [], [], [], []
    for local in locals:
        local = canonical_model(local)
        span = local['interval_s']
        t = np.linspace(*span, int(np.ceil((span[1]-span[0])*4))+1)
        p = np.asarray(local['parameters'])
        n = basis([np.mean(span)], span, local['path_spacing_seconds']).shape[1]
        rate = basis(t, span, local['path_spacing_seconds'], derivative=1)@p[:n]
        yaw = basis(t, span, local['heading_spacing_seconds'])@p[n:]
        w = np.ones(len(t))
        path_design.append(basis(t, interval, path_spacing, fixed_midpoint=True, derivative=1))
        yaw_design.append(basis(t, interval, heading_spacing))
        rate_values.append(rate); yaw_values.append(yaw); weights.append(w)
    root = np.sqrt(np.concatenate(weights))
    a = np.concatenate(path_design); y = np.concatenate(rate_values)
    epochs = np.linspace(*interval, int(np.ceil((interval[1]-interval[0])*4))+1)
    d2 = basis(epochs, interval, path_spacing, derivative=2)[:, :-1]
    path = np.r_[constrained_curve(a[:, :-1], y, root**2, d2, .15), 0.]
    b = np.concatenate(yaw_design)
    dy = np.concatenate([basis(epochs, interval, heading_spacing, derivative=n) for n in (1, 2)])
    limits = np.r_[np.full(len(epochs), .5), np.full(len(epochs), .02)]
    yaw = constrained_curve(b, np.concatenate(yaw_values), root**2, dy, limits)
    return np.r_[path, yaw], {'path_rate_overlap_rms_mps': float(np.sqrt(np.average((a@path-y)**2, weights=root**2))),
        'yaw_overlap_rms_degrees': float(np.sqrt(np.average((b@yaw-np.concatenate(yaw_values))**2, weights=root**2))),
        'correction_bounds': {'path_acceleration_mps2': .15, 'yaw_rate_deg_ps': .5, 'yaw_acceleration_deg_ps2': .02},
        'bounds_scope': 'Bounds on corrections, not absolute motion; initialization only.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('histories', 'ais', 'geometry-directory', 'registration', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=250)
    p.add_argument('--window-s', type=float, default=45.)
    p.add_argument('--step-s', type=float, default=20.)
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    track = read_csv(args.ais, reference_height_ellipsoid_m=reg['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    epoch = utc_seconds(reg['nominal_start_utc'])
    sources = []
    for c in range(5):
        directory = args.histories/f'channel-{c}'
        with np.load(directory/'coordinates.npz') as f:
            d = dict(f)
        d['value'] = np.load(directory/'values.npy', mmap_mode='r')
        d['meta'] = json.loads((directory/'metadata.json').read_text())
        sources.append(d)
    interval = [max(d['time'][0] for d in sources), min(d['time'][-1] for d in sources)]
    starts = np.r_[np.arange(interval[0], interval[1]-args.window_s, args.step_s), interval[1]-args.window_s]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    local_dir = args.output.with_suffix(''); local_dir.mkdir(exist_ok=True)
    locals = []
    started = time.perf_counter()
    for i, start in enumerate(starts):
        destination = local_dir/f'window-{i:03d}.json'
        if destination.exists():
            locals.append(json.loads(destination.read_text())); continue
        span = [float(start), float(start+args.window_s)]
        bands, selections = [], []
        # Use common epochs only for the initializer; exact per-channel geometry
        # and epochs are retained by the subsequent likelihood.
        count = min(np.count_nonzero((d['time'] >= span[0])&(d['time'] < span[1])) for d in sources)
        for c, d in enumerate(sources):
            keep = np.flatnonzero((d['time'] >= span[0])&(d['time'] < span[1]))[:count]
            t, w, z = d['time'][keep], d['information'][keep], d['value'][keep]
            selected = np.floor((t-span[0])/.5).astype(int)%2 == 0
            chosen = choose_points(z, w, t, d['pixels'], selected)
            state = physical_state(track, epoch+t, reg['attitude_prior'], max_gap_s=400.)
            _, gradient = excess_path(state['position_ecef_m'], tx[3], rx)
            direction = gradient/np.sum(gradient**2, axis=1)[:, None]
            bands.append(Band(t, z[:, chosen], w, d['pixels'][chosen, :3], state['position_ecef_m'],
                state['body_to_ecef'], tx[c], rx, direction, d['meta']['rf_hz'], span,
                path_spacing=3., heading_spacing=8., selected=selected))
            selections.append(d['pixels'][chosen, :3].tolist())
        parameters, diagnostics = local_fit(bands, iterations=args.iterations)
        local = {'interval_s': span, 'parameters': parameters.tolist(), 'path_spacing_seconds': 3.,
            'heading_spacing_seconds': 8., 'selected_points_body_m': selections, **diagnostics}
        destination.write_text(json.dumps(local, indent=2)+'\n'); locals.append(local)
        print(json.dumps({'window': i, 'interval_s': span, 'objective': diagnostics['objective'],
            'converged': diagnostics['converged'], 'gain_db': [v['heldout_power_gain_db'] for v in diagnostics['validation']],
            'seconds': time.perf_counter()-started}), flush=True)
    parameters, continuity = join_curves(locals, interval)
    model = {key: reg[key] for key in ('nominal_start_utc', 'attitude_prior', 'reference_height_ellipsoid_m',
        'reference_channel', 'yaw_convention', 'geometry_directory', 'channel_gains_real_imag', 'range_bias_m')}
    model.update(interval_s=interval, path_spacing_seconds=3., heading_spacing_seconds=8., parameters=parameters.tolist(),
        path_reference_time_s=float(np.mean(reg['interval_s'])),
        scope=__doc__, local_fits=str(local_dir), complete=True, roll_pitch_fitted=False,
        channel_calibration_source=str(args.registration), channel_calibration_validated=False,
        max_ais_gap_s=400., continuity=continuity,
        limitation='Radar phase initializer with independent band reflectors. Full-response radar motion validation remains separate; overlap joining is not the maximum-likelihood solution of the full recording.')
    args.output.write_text(json.dumps(model, indent=2)+'\n')


if __name__ == '__main__':
    main()
