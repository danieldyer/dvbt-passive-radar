"""Fit revised physical geometry directly to short multiband vehicle arcs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from cluster_phase import crop
from coherent_arc import CoherentArc
from geographic_io import read_profiles
from road_arc import observation, observables as arc_observables
from road_arc_geometry import RoadGeometry, road_state
from road_arc_raw import RawRoadGeometry
from flash_geometry import observables
from site_hypotheses import licensed_geometry
from terrain import RasterTerrain


def load_problem(summary, assignment, labels, *, free_antennas=False, free_origins=False,
                 arc_index=0, fixed_calibration=None, span_arcs=False):
    """Rebuild the likelihood from immutable samples, with explicit hypotheses."""
    summary = Path(summary); saved = json.loads(summary.read_text())
    assignment = list(map(int, assignment)); code = ''.join(map(str, assignment))
    if not saved['complete']:
        raise ValueError('Use completed summary controls for declared road associations')
    reference = next((t for t in saved['trials'] if t['assignment'] == assignment), None)
    targets = {t['label']: t for source in saved['source_arcs']
               for t in json.loads(Path(source).read_text())['targets']}
    arcs_by_label = {label: target['arcs'][arc_index] for label, target in targets.items()}
    route_path = Path('build/calibration-20261017/road-routes.json')
    routes = json.loads(route_path.read_text())['routes']
    screen_path = Path(f'build/calibration-20261017/road-screen/sites-{code}.json')
    screens = {t['label']: t['trials'] for t in json.loads(screen_path.read_text())['targets']}
    if reference is not None:
        associations = {a['label']: a for a in reference['associations']}
    else:
        associations = {}
        for label, choices in screens.items():
            preferred = [s for s in choices if s['name'] in ('Albert Street', 'Victoria Parade')]
            selected = min(preferred or choices, key=lambda s: s['cost'])
            associations[label] = {'label': label, 'road': selected['name'], 'route': selected['route']}
    geometry = Path(saved['geometry']); prior_path = geometry/'geometry-prior.json'
    prior = json.loads(prior_path.read_text())
    terrain = RasterTerrain(Path('build/calibration-20261013'), prior, Path('build/calibration-20261013/ga-footprint-dem.tif'))
    sites, offsets, descriptions = licensed_geometry(geometry, Path('build/calibration-20261013'), assignment, terrain)
    base = Path('build/calibration-20261013')
    template_path = base/'static-with-psf.gprf'; _, records = read_profiles(template_path)
    chosen, arcs, likelihoods = [], [], []
    sources = [summary, route_path, screen_path, prior_path, template_path, *map(Path, saved['source_arcs'])]
    for label in labels:
        arc = dict(arcs_by_label[label]); selected = associations[label]
        if span_arcs:
            selected_arcs = targets[label]['arcs']
            arc['first_s'] = min(a['first_s'] for a in selected_arcs)
            arc['last_s'] = max(a['last_s'] for a in selected_arcs)
            arc['label'] = f"{label} · {arc['first_s']:g}–{arc['last_s']:g} s continuous"
            arc['channels'] = [dict(r) for r in arc['channels']]
        screen = next(s for s in screens[label] if s['route'] == selected['route'])
        source = base/'shared-seed-histories'/f'{label}.json'; sources.append(source)
        description = json.loads(source.read_text())
        seed = description['channels'][0]; half = (description['end']-description['start'])/2
        midpoint = (arc['first_s']+arc['last_s'])/2; u = (midpoint-seed['epoch'])/half
        if span_arcs:
            for row, seed in zip(arc['channels'], description['channels']):
                measured, covariance = arc_observables(row, seed, midpoint)
                row['absolute_observables'] = measured.tolist()
                row['absolute_observable_covariance'] = covariance.tolist()
        state = [np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyder(screen['state'], j))/half**j for j in range(3)]
        arc['initial_road_state'] = [*state, 0., 2.]
        arcs.append(arc); chosen.append(routes[selected['route']]); bands = []
        for ch, seed in enumerate(description['channels']):
            keep = ((records['channel'] == ch)&(records['time'] >= description['start'])&
                    (records['time'] <= description['end'])&(records['information'] > 0))
            pulse = np.average(records[keep]['psf'], axis=0, weights=records[keep]['information'])
            path = base/'shared-seed-patches'/f'{label}-{ch}.npz'; sources.append(path)
            with np.load(path) as raw:
                data = observation(raw, seed, pulse)
            centre = arc['centre_sample']; f = crop(data, arc['first_s'], arc['last_s'], np.arange(centre-2, centre+3))
            training = ((f.time-arc['first_s'])//.125).astype(int)%2 == 0
            bands.append(CoherentArc(f, training))
        likelihoods.append(bands)
    transmitter = sites[assignment]+offsets
    receiver, origins = np.zeros(3), np.zeros(5)
    if fixed_calibration is not None:
        if free_antennas or free_origins or fixed_calibration['assignment'] != assignment:
            raise ValueError('Transferred calibration must be fixed and use the same site assignment')
        fit = fixed_calibration['antenna_fit']
        if fit is not None:
            transmitter = transmitter+np.asarray(fit['transmitter_offsets_enu_m'])
            receiver = np.asarray(fit['receiver_offset_enu_m'])
        origins = np.asarray(fixed_calibration['range_origins_m'])
    g = RoadGeometry(arcs, chosen, transmitter, [d['site_id'] for d in descriptions],
        np.array(prior['ecef_to_enu']), free_origins=free_origins, free_antennas=free_antennas,
        receiver=receiver, range_origins=origins)
    associations = [dict(associations[label], arc=arcs[i]['label']) for i, label in enumerate(labels)]
    return RawRoadGeometry(g, likelihoods), associations, sources


def initialise_motion(model, initial):
    """Use every independently fitted band only to seed the raw likelihood.

    Coarse long-arc fits can put a two-second coherent solve on a Doppler
    sidelobe. Each band's training-only range/rate/curvature provides another
    physical road-state start. Selection uses the full raw training likelihood,
    never held-out prediction or an unreliable weak-band summary as a datum.
    """
    if type(model) is not RawRoadGeometry:
        raise TypeError('Initialise the independent geometry before testing shared phase')
    g = model.geometry; result = np.array(initial, copy=True)
    tx, rx, origins, states = g.split(result)
    trials = []
    for i, (arc, road, state, bands) in enumerate(zip(g.arcs, g.roads, states, model.likelihoods)):
        local = RoadGeometry([arc], [road.description], tx, g.antennas.channel_sites,
            g.antennas.rotation, free_antennas=False, receiver=rx, range_origins=origins)
        problem = RawRoadGeometry(local, [bands])
        seed = np.clip(state/g.scale, problem.lower+1e-7, problem.upper-1e-7)
        starts = [seed]
        for channel, measured in enumerate(arc['channels']):
            target = np.array(measured['absolute_observables']); target[0] -= origins[channel]
            def residual(x):
                full = seed.copy(); full[:3] = x
                return (observables(*road_state(road, full*g.scale), tx, rx)[channel]-target)/np.array([10., 1., 1.])
            mapped = least_squares(residual, seed[:3], bounds=(problem.lower[:3], problem.upper[:3]),
                                   max_nfev=60, ftol=1e-9, xtol=1e-9, gtol=1e-8)
            candidate = seed.copy(); candidate[:3] = mapped.x
            starts.append(candidate)
        fitted = [solve(problem, start) for start in starts]
        best = int(np.argmin([f.cost for f in fitted]))
        result[g.start+5*i:g.start+5*(i+1)] = fitted[best].x
        trials.append({'label': arc['label'], 'selected_start': best,
            'start_order': 'Current physical state, then Seven/SBS/Nine/Ten/ABC training-only summaries',
            'training_costs': [float(f.cost) for f in fitted],
            'selected_road_state': (fitted[best].x*g.scale).tolist()})
    return result, trials


def solve(model, initial=None, *, iterations=120, callback=None, search_motion=False):
    initial = model.initial if initial is None else initial
    initial = np.clip(initial, model.lower+1e-7, model.upper-1e-7)
    if search_motion:
        initial, model.motion_starts = initialise_motion(model, initial)
    return least_squares(lambda p: model.evaluate(p)[0], initial, jac=lambda p: model.evaluate(p)[1],
        bounds=(model.lower, model.upper), x_scale='jac', max_nfev=iterations,
        ftol=1e-10, xtol=1e-10, gtol=1e-6, callback=callback)


def describe(model, fit):
    g = model.geometry
    tx, rx, origins, states = g.split(fit.x[:len(g.initial)])
    return dict(parameters=fit.x.tolist(), road_states=states.tolist(), range_origins_m=origins.tolist(),
        antenna_fit=g.antennas.describe(fit.x[:18]) if g.free_antennas else None,
        converged=bool(fit.success), termination=fit.message, optimality=float(fit.optimality), evaluations=int(fit.nfev),
        cost=float(fit.cost), bound_indices=np.flatnonzero((fit.x<model.lower+1e-4)|(fit.x>model.upper-1e-4)).tolist(),
        prediction=model.prediction(fit.x), diagnostics=model.diagnostics(fit.x),
        motion_initialisation=getattr(model, 'motion_starts', None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, default=Path('docs/data/isar_20261017/road-arc-geometry/fits.json'))
    p.add_argument('--assignment', default='02002')
    p.add_argument('--labels', nargs='+', default=['A', 'C', 'R'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--free-antennas', action='store_true')
    p.add_argument('--free-origins', action='store_true')
    p.add_argument('--initial', type=Path)
    p.add_argument('--arc-index', type=int, default=0)
    p.add_argument('--span-arcs', action='store_true', help='Test one continuous trajectory and complex amplitude over the complete span of both selected arcs')
    p.add_argument('--iterations', type=int, default=120)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'fit.json').exists():
        raise ValueError('Use a fresh output for a controlled raw-data fit')
    started = time.monotonic(); assignment = list(map(int, args.assignment))
    model, associations, sources = load_problem(args.summary, assignment, args.labels,
        free_antennas=args.free_antennas, free_origins=args.free_origins, arc_index=args.arc_index, span_arcs=args.span_arcs)
    g = model.geometry; initial = g.initial
    if args.initial:
        old = json.loads(args.initial.read_text())
        if (old['assignment'] != assignment or old['labels'] != args.labels
                or old.get('span_arcs', False) != args.span_arcs or old.get('arc_index', 0) != args.arc_index):
            raise ValueError('Initial fit must use the same target order, intervals and site assignment')
        initial = g.initial.copy(); initial[g.start:] = np.array(old['road_states']).ravel()/np.tile(g.scale, len(args.labels))
        if g.free_antennas and old['antenna_fit']:
            initial[:18] = old['antenna_fit']['standardised_parameters']
        if g.free_origins:
            initial[g.antenna_count:g.start] = np.array(old['range_origins_m'])/30
        sources.append(args.initial)
    initial = np.clip(initial, model.lower+1e-7, model.upper-1e-7)
    result = {'scope': __doc__, 'complete': False, 'accepted_calibration': False,
        'assignment': assignment, 'labels': args.labels, 'associations': associations, 'arc_index': args.arc_index, 'span_arcs': args.span_arcs,
        'free_antennas': args.free_antennas, 'free_origins': args.free_origins,
        'initial_parameters': initial.tolist(), 'parameters': initial.tolist()}
    result['initial_prediction'] = model.prediction(initial)
    def save(path):
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
    def progress(intermediate_result):
        if intermediate_result.nfev%10 == 0:
            result.update(parameters=intermediate_result.x.tolist(), evaluations=int(intermediate_result.nfev),
                          cost=float(intermediate_result.cost), seconds=time.monotonic()-started)
            save(args.output/'checkpoint.json')
            print('raw road fit', intermediate_result.nfev, round(intermediate_result.cost, 3),
                  'seconds', round(time.monotonic()-started, 1), flush=True)
    print('loaded', args.labels, 'parameters', len(initial), 'seconds', round(time.monotonic()-started, 2), flush=True)
    fit = solve(model, initial, iterations=args.iterations, callback=progress, search_motion=True)
    sources.extend(Path(__file__).with_name(name) for name in
        ['fit_road_arc_raw.py', 'road_arc_raw.py', 'road_arc_geometry.py', 'road_arc.py', 'field.py', 'coherent_arc.py'])
    result.update(complete=True, **describe(model, fit), seconds=time.monotonic()-started,
        source_sha256={str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in sources})
    save(args.output/'fit.json')
    print('completed', 'cost', fit.cost, 'optimality', fit.optimality, 'origins', result['range_origins_m'], 'seconds', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
