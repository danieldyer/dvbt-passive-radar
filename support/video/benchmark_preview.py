#!/usr/bin/env python3
"""Compare full-map CPU previews with compact Vulkan previews, without encoding.

Includes initialization and acquisition. Run without competing GPU jobs for
useful elapsed times; this is not the I/O-excluded production stage benchmark.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import render_recording as recording
import render_video as video


def measure(args, backend, run, channels):
    destination = args.output / f'{backend}-{run}'
    destination.mkdir()
    processor = video.ROOT / 'build/release' / ('dvbt_detect' if backend == 'cpu' else 'dvbt_preview')
    command = [str(processor), str(args.input.resolve()), '--metadata', str(args.metadata.resolve()),
               '--output', str(destination), '--start', str(args.start), '--seconds', str(args.seconds),
               '--integration-seconds', str(args.integration), '--update-seconds', str(1 / args.map_rate),
               '--doppler-hz', str(args.doppler_hz), '--range-metres', '30000', '--map-stream', '-', '--stream-only']
    count = [0] * channels
    first, last, peak = [None] * channels, [None] * channels, [-math.inf] * channels
    processor_job, reducer = None, None
    started = time.monotonic()
    try:
        with (destination / 'pipeline.log').open('w') as log, (destination / 'preview.log').open('w') as preview_log:
            processor_job = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, cwd=video.ROOT)
            source = processor_job.stdout
            if backend == 'cpu':
                reducer = subprocess.Popen([str(video.ROOT / 'build/release/rdm_preview')], stdin=source,
                                           stdout=subprocess.PIPE, stderr=preview_log)
                source.close()
                source = reducer.stdout
            for meta, db in recording.read_pooled(source):
                c = meta['channel']
                count[c] += 1
                last[c] = meta['time_end_s']
                peak[c] = max(peak[c], float(db.max()))
                if first[c] is None:
                    first[c] = meta['time_end_s']
            if reducer is not None and reducer.wait() != 0:
                raise RuntimeError(f'CPU reducer failed: {destination / "preview.log"}')
            if processor_job.wait() != 0:
                raise RuntimeError(f'Processor failed: {destination / "pipeline.log"}')
        return dict(backend=backend, run=run, signal_seconds=args.seconds, elapsed_s=time.monotonic()-started,
                    maps=count, first_s=first, last_s=last, peaks_db=peak, command=command)
    finally:
        for job in (reducer, processor_job):
            if job is not None and job.poll() is None:
                job.terminate()
                try:
                    job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    job.kill()
                    job.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=18)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--integration', type=float, default=3)
    parser.add_argument('--doppler-hz', type=float, default=100)
    parser.add_argument('--map-rate', type=float, default=3125/102)
    args = parser.parse_args()
    if args.runs < 1 or args.seconds <= 0 or args.start < 0 or args.map_rate <= 0:
        parser.error('Require positive runs, duration and map rate, and nonnegative start')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    channels = len(json.loads(args.metadata.read_text())['signals'])
    results = []
    for run in range(args.runs):
        for backend in ('cpu', 'gpu'):
            result = measure(args, backend, run, channels)
            results.append(result)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            if backend == 'gpu':
                reference = results[-2]
                if any(result[key] != reference[key] for key in ('maps', 'first_s', 'last_s')):
                    raise RuntimeError('CPU and GPU paths produced different maps or coordinates')
                if any(abs(a-b) > .00002 for a, b in zip(result['peaks_db'], reference['peaks_db'])):
                    raise RuntimeError('CPU and GPU preview peaks disagree')
    cpu = sum(r['elapsed_s'] for r in results if r['backend'] == 'cpu') / args.runs
    gpu = sum(r['elapsed_s'] for r in results if r['backend'] == 'gpu') / args.runs
    print(f'CPU path {cpu:.3f} s; GPU path {gpu:.3f} s; speedup {cpu/gpu:.3f}x')


if __name__ == '__main__':
    main()
