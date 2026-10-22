"""Re-evaluate all licensed site assignments with the corrected extraction.

Historical roads and phase branches are initial guesses only. Every likelihood
is recomputed on the same declared target set and revised observations. These
local fits do not replace independent road-association or held-out validation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

from road_search import OPTIONS


def select_targets(saved, labels):
    """Subset the serialized constant-gain state without changing any geometry."""
    if saved.get('range_bias') or saved.get('amplitude_block_seconds') or 'receiver_prior_sigma_m' in saved:
        raise ValueError('The site comparison requires the same fixed-receiver constant-response model')
    original = saved['labels']
    indices = [original.index(label) for label in labels]
    if len(set(indices)) != len(indices):
        raise ValueError('Repeated target label')
    state = saved['state']
    quotient, remainder = divmod(len(state)-13, len(original))
    count = quotient-4
    if remainder or count != 15:
        raise ValueError('Unsupported serialized road-state shape')
    phase = 9+len(original)*count
    selected = state[:9]+sum((state[9+i*count:9+(i+1)*count] for i in indices), [])
    selected += state[phase:phase+4]
    selected += sum((state[phase+4+4*i:phase+8+4*i] for i in indices), [])
    return {**{key: saved[key] for key in ('assignment', 'channels', 'site_prior_sigma_m',
                                           'site_positions_enu_m', 'channel_offsets_enu_m') if key in saved},
            'labels': labels, 'state': selected,
            'selected_roads': [saved['selected_roads'][i] for i in indices],
            'status': 'Historical geometric initializer only; no objective or uncertainty reused'}


def source_for(directory, code):
    for suffix in ('joint-associated', 'joint-exact-path', 'joint-road10',
                   'joint-roads6-all12', 'joint-all12'):
        path = directory/f'{code}-{suffix}.json'
        if path.exists():
            return path
    raise ValueError(f'No historical road/phase initializer for {code}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('previous', 'histories', 'patches', 'templates', 'routes', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--labels', nargs='+', default=['A', 'B', 'C', 'E', 'K', 'O', 'P', 'Q', 'R', 'S'])
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('Use one to four bounded CPU workers')
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = args.output/'initializers'
    seeds.mkdir(exist_ok=True)
    jobs = []
    nominal = None
    for assignment in itertools.product(*OPTIONS):
        code = ''.join(map(str, assignment))
        source = source_for(args.previous, code)
        saved = json.loads(source.read_text())
        seed = select_targets(saved, args.labels)
        if seed['assignment'] != list(assignment):
            raise ValueError('Initializer assignment disagrees with its filename')
        if nominal is None:
            nominal = seed['site_positions_enu_m']
        if nominal != seed['site_positions_enu_m']:
            raise ValueError('Site assignments use different nominal geometry: normalize explicitly first')
        seed['historical_source'] = str(source)
        path = seeds/f'{code}.json'
        if path.exists() and json.loads(path.read_text()) != seed:
            raise ValueError(f'Initializer changed: use a new output directory ({path})')
        path.write_text(json.dumps(seed, indent=2)+'\n')
        jobs.append((code, path))
    manifest = {'assignments': len(jobs), 'labels': args.labels, 'iterations': args.iterations,
                'workers': args.workers,
                'inputs': {key: str(getattr(args, key)) for key in ('histories', 'patches', 'templates', 'routes')},
                'scope': 'Same ten targets and deterministic likelihood in all assignments; historical road/phase branches are starting hypotheses. Fixed receiver, three mast corrections, shared band phases, independently profiled positive band gains. No revised global road search or phase-centre acceptance.',
                'results': [], 'failures': []}
    clock = time.monotonic()

    def run(job):
        code, source = job
        output = args.output/f'{code}.json'
        command = [sys.executable, str(Path(__file__).with_name('refit_trial.py')),
                   '--fit', str(source), '--iterations', str(args.iterations), '--output', str(output)]
        for key in ('histories', 'patches', 'templates', 'routes'):
            command += ['--'+key, str(getattr(args, key))]
        if not output.exists():
            with (args.output/f'{code}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        if result['labels'] != args.labels or any(result['input_'+key] != str(getattr(args,key))
                                                  for key in ('histories','patches','templates','routes')):
            raise ValueError(f'Cached result uses different observations: {output}')
        return {'assignment': result['assignment'], 'code': code, 'source': str(source),
                'result': str(output), 'cost': result['cost'], 'converged': result['converged'],
                'evaluations': result['evaluations'], 'optimality': result['optimality'],
                'expanded_state_error': result.get('expanded_state_error'),
                'zero_amplitude_channels_by_target': result['zero_amplitude_channels_by_target']}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run, job): job[0] for job in jobs}
        for job in as_completed(pending):
            try:
                result = job.result()
            except Exception as error:
                manifest['failures'].append({'code': pending[job], 'error': str(error)})
                print(pending[job], 'failed', str(error), flush=True)
                result = None
            if result is not None:
                manifest['results'].append(result)
            manifest['results'].sort(key=lambda entry: entry['cost'])
            manifest['seconds'] = time.monotonic()-clock
            (args.output/'progress.json').write_text(json.dumps(manifest, indent=2)+'\n')
            if result is not None:
                print(result['code'], 'cost', round(result['cost'], 3), 'converged', result['converged'],
                      'completed', len(manifest['results']), 'seconds', round(manifest['seconds'], 1), flush=True)
    if manifest['failures']:
        raise SystemExit('Some site fits failed; see progress.json and individual logs')


if __name__ == '__main__':
    main()
