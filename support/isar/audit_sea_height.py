"""Test declared water-path families and point heights using held-out blocks.

All variants share a fixed motion, horizontal support and antenna geometry.
Height and phase selection uses training observations only. Independent band
responses are reported at the exact same selected heights. Sea eligibility
checks specular centres; it does not validate their complete Fresnel footprints.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from linear_observation import LinearObservation
from phase_coupling import PhaseCoupling
from scene_dictionary import fingerprint
from stationary import Stationary
from variant_phase import VariantPhase


def augment(observation):
    return (np.vstack([observation.design, np.zeros(observation.design.shape[1])]),
        np.r_[observation.data, np.sqrt(observation.orthogonal_energy)])


def observations(directory, family, interval, cache):
    cache.mkdir(parents=True, exist_ok=True)
    result, eligibility = [], []
    for c in range(5):
        with np.load(directory/f'channel-{c}-observation.npz') as data:
            y, t, background, water = data['data'], data['time'], data['background'], data['water_valid']
        allowed = np.ones(water.shape[1:], bool)
        if family in ('transmit', 'both'):
            allowed &= water[0]
        if family in ('receive', 'both'):
            allowed &= water[1]
        eligibility.append(allowed)
        path = cache/f'{family}-{interval[0]:g}-{interval[1]:g}-channel-{c}.npz'
        if path.exists():
            with np.load(path) as f:
                result.append(tuple(LinearObservation(f[name+'_design'], f[name+'_data'],
                    float(f[name+'_outside']), int(f[name+'_count'])) for name in ('train', 'test')))
            continue
        keep = (t >= interval[0])&(t < interval[1])
        rows = np.repeat(keep, y.shape[1])
        y, t, background = y[keep], t[keep], background[rows]
        block = np.floor(t-interval[0]).astype(int)
        selected = np.repeat((block%2 == 0)|(block == block.max()), y.shape[1])
        design = np.load(directory/f'channel-{c}-{family}.npy', mmap_mode='r')[rows]
        pair, saved = [], {}
        for name, mask in [('train', selected), ('test', ~selected)]:
            nuisance = Stationary(background, mask)
            observation = LinearObservation.compress(nuisance.remove(design[mask]),
                nuisance.remove(y.ravel()[mask]), nuisance.rank)
            pair.append(observation)
            saved.update({name+'_design': observation.design, name+'_data': observation.data,
                name+'_outside': observation.orthogonal_energy, name+'_count': observation.observations})
        np.savez(path, **saved)
        result.append(tuple(pair))
        print(json.dumps({'family': family, 'compressed_channel': c, 'interval': interval}), flush=True)
    return result, np.logical_and.reduce(eligibility)


def score(observations, columns, coefficients):
    errors = []
    for (_, test), coefficient in zip(observations, coefficients):
        residual = test.data-test.design[:, columns]@coefficient
        errors.append(float(np.vdot(residual, residual).real+test.orthogonal_energy))
    return {'channels': errors, 'total': sum(errors)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dictionary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--families', nargs='+', choices=['direct', 'transmit', 'receive', 'both'], default=['direct', 'transmit', 'receive', 'both'])
    p.add_argument('--interval', nargs=2, type=float, default=[142., 172.])
    p.add_argument('--maximum-cycles', type=int, default=20)
    args = p.parse_args()
    source = json.loads((args.dictionary/'dictionary.json').read_text())
    if not source['complete']:
        raise ValueError('Complete the physical dictionary first')
    identity = {'dictionary_sha256': fingerprint(args.dictionary/'dictionary.json'),
        'script_sha256': fingerprint(Path(__file__)),
        'fitter_sha256': fingerprint(Path(__file__).with_name('variant_phase.py')),
        'interval': args.interval, 'maximum_cycles': args.maximum_cycles}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'identity': identity, 'scope': __doc__, 'complete': False, 'correction_adopted': False,
        'families': {}, 'water': source['water']}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['identity'] != identity:
            raise ValueError('Changed model inputs; use a fresh output path')
    cache = args.dictionary/'compressed'
    cache_manifest = cache/'identity.json'
    cache.mkdir(exist_ok=True)
    if cache_manifest.exists():
        if json.loads(cache_manifest.read_text()) != identity['dictionary_sha256']:
            raise ValueError('Changed dictionary with stale compression cache')
    else:
        # Independent aperture jobs share a read-only dictionary. Publish the
        # identity atomically and never truncate an existing cache manifest.
        with tempfile.NamedTemporaryFile(mode='w', dir=cache, delete=False) as temporary:
            temporary.write(json.dumps(identity['dictionary_sha256'])+'\n')
            temporary_path = Path(temporary.name)
        temporary_path.replace(cache_manifest)
    heights = np.asarray(source['identity']['heights_body_m'])
    default = int(np.argmin(abs(heights-20.)))
    started = time.perf_counter()
    for family in args.families:
        if family in result['families']:
            continue
        measured, allowed = observations(args.dictionary, family, args.interval, cache)
        pairs = [augment(train) for train, _ in measured]
        designs, data = [v[0] for v in pairs], [v[1] for v in pairs]
        model = VariantPhase(designs, data, allowed, reference=3)
        uniform = lambda h: np.array([np.flatnonzero(row)[np.argmin(abs(heights[row]-heights[h]))] for row in allowed])
        baseline_variants = uniform(default)
        baseline = model.model(baseline_variants)
        baseline_coefficients, baseline_fit = baseline.fit(coordinate_sweeps=8)
        baseline_columns = model.columns(baseline_variants)
        base_scores = {name: score(measured, baseline_columns, a) for name, a in
            [('coupled', baseline_coefficients), ('independent', baseline.independent)]}
        print(json.dumps({'family': family, 'baseline': base_scores, 'baseline_phase_gradient': baseline_fit['maximum_gradient']}), flush=True)
        starts = [(baseline_variants, 3, baseline_fit['parameters'])]
        starts += [(uniform(h), 3) for h in range(len(heights)) if allowed[:, h].any()]
        starts += [(baseline_variants, c) for c in (0, 1, 2, 4)]
        def progress(trial):
            print(json.dumps({'family': family, 'start': trial['start'], 'objective': trial['objective'],
                'converged': trial['converged'], 'cycles': len(trial['cycles']),
                'seconds': time.perf_counter()-started}), flush=True)
        coefficients, fit = model.fit(starts, maximum_cycles=args.maximum_cycles, callback=progress)
        variants = np.asarray(fit['best']['variants'])
        columns = model.columns(variants)
        selected_model = model.model(variants)
        scores = {name: score(measured, columns, a) for name, a in
            [('coupled', coefficients), ('independent', selected_model.independent)]}
        record = {'baseline_fixed_height_m': float(heights[default]), 'baseline_fit': baseline_fit,
            'baseline_scores': base_scores, 'fit': fit, 'scores': scores,
            'selected_heights_m': heights[variants].tolist(),
            'eligible_height_count_per_point': allowed.sum(axis=1).tolist(),
            'coupled_prediction_penalty_percent': 100*(scores['coupled']['total']/scores['independent']['total']-1),
            'coefficient_real_imag': np.stack([coefficients.real, coefficients.imag], axis=-1).tolist()}
        result['families'][family] = record
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'family': family, 'scores': scores,
            'penalty_percent': record['coupled_prediction_penalty_percent']}), flush=True)
    result['complete'] = all(f in result['families'] for f in args.families)
    result['seconds_this_run'] = time.perf_counter()-started
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
