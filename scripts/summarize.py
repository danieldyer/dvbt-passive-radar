#!/usr/bin/env python3
"""Summarize measured residual stability; this is not absolute clock calibration."""
import argparse
import cmath
import csv
import itertools
import json
import math
import pathlib
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=pathlib.Path)
parser.add_argument('--settling-seconds', type=float, default=1.0)
args = parser.parse_args()
metadata = json.loads((args.directory / 'metadata.json').read_text())
groups = {}
with (args.directory / 'timing.csv').open() as tf, (args.directory / 'coherence.csv').open() as cf:
    for time, coherence in itertools.zip_longest(csv.DictReader(tf), csv.DictReader(cf)):
        if time is None or coherence is None or time['row'] != coherence['row']:
            raise RuntimeError('Timing and coherence rows disagree')
        elapsed = (float(time['input_sample']) - metadata['input_origin']) / metadata['input_sample_rate']
        if time['all_carriers_valid'] != '1' or elapsed < args.settling_seconds:
            continue
        groups.setdefault(time['segment'], []).append((time, coherence))

def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.pstdev(values), min=min(values), max=max(values))

result = dict(metadata=metadata, settling_seconds=args.settling_seconds, segments={})
for segment, rows in groups.items():
    phase = [float(c['phase_error_deg']) for t, c in rows]
    timing = [float(c['timing_error_samples']) for t, c in rows]
    rate = [float(t['sample_rate_error_ppm']) for t, c in rows]
    amplitude = abs(sum(cmath.exp(1j * math.radians(p)) for p in phase)) / len(phase)
    result['segments'][segment] = dict(valid_rows=len(rows), phase_error_deg=stats(phase),
        timing_error_samples=stats(timing), clock_ppm=stats(rate),
        common_phase_coherent_amplitude_ratio=amplitude,
        common_phase_coherent_loss_db=20 * math.log10(max(amplitude, 1e-300)))
if not groups:
    raise RuntimeError('No fully valid rows remain after settling')
text = json.dumps(result, indent=2)
(args.directory / 'validation_summary.json').write_text(text + '\n')
print(text)
