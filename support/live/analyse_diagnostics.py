#!/usr/bin/env python3
"""Summarise capture health and test clock innovations without reading IQ payloads.

NIS and lag correlation are model diagnostics, not a certification of oscillator
stability or achieved radar sensitivity. A mismatched model must remain visible.
"""
import argparse
import json
import math
import gzip
from pathlib import Path


def summarize(path):
    config, latest, anchors = {}, {}, []
    channels, statuses = {}, []
    complete = False
    source_path = Path(path)
    with (gzip.open(source_path, 'rt') if source_path.suffix == '.gz' else source_path.open()) as source:
        for line in source:
            row = json.loads(line)
            kind = row['type']
            if kind == 'configuration':
                config = row
            elif kind == 'status':
                latest = row
                statuses.append(row)
            elif kind == 'utc_anchor':
                anchors.append(row)
            elif kind == 'complete':
                complete = True
            elif kind == 'timing_observations':
                for values in row['rows']:
                    kind, channel, epoch, _, variance, _, _, _, innovation, prediction, nis, inlier, late = values[:13]
                    weight = values[13] if len(values) > 13 else (1. if inlier else 0.)
                    state = channels.setdefault(str(channel), dict(offsets=0, rejected=0, downweighted=0, late=0, count=0,
                        nis_sum=0., previous=None, pairs=0, xy=0., xx=0., yy=0., x=0., y=0.))
                    if kind == 2:
                        state['previous'] = None
                    if kind != 1:
                        continue
                    state['offsets'] += 1
                    state['rejected'] += weight == 0 and not inlier
                    state['downweighted'] += 0 < weight < 1
                    state['late'] += late
                    # Acquisition priors deliberately start broad. Do not call
                    # their initial settling a stationary covariance test.
                    if late or variance is None or not prediction or epoch/config['corrected_adc_rate_hz'] < 5:
                        continue
                    z = innovation/math.sqrt(prediction)
                    state['count'] += 1
                    state['nis_sum'] += nis
                    previous = state['previous']
                    if previous is not None:
                        state['pairs'] += 1
                        state['xy'] += previous*z
                        state['xx'] += previous*previous
                        state['yy'] += z*z
                        state['x'] += previous
                        state['y'] += z
                    state['previous'] = z
    for state in channels.values():
        n, pairs = state['count'], state['pairs']
        state['mean_nis'] = state['nis_sum']/n if n else None
        numerator = state['xy']-state['x']*state['y']/pairs if pairs else 0
        denominator = ((state['xx']-state['x']**2/pairs)*(state['yy']-state['y']**2/pairs)) if pairs else 0
        state['lag_one_correlation'] = numerator/math.sqrt(denominator) if denominator > 0 else None
        state['nis_approx_95_half_width'] = 1.96*math.sqrt(2/n) if n else None
        for key in ['previous', 'nis_sum', 'pairs', 'xy', 'xx', 'yy', 'x', 'y']:
            del state[key]
    return dict(complete=complete, configuration=config, latest_status=latest, clock_innovations=channels,
                utc_anchors=anchors, status_samples=statuses,
                scope='Conditional clock-model diagnostics; host UTC accuracy and RF analogue linearity are not certified.')


def plot(report, output):
    import matplotlib.pyplot as plt
    rows = report['status_samples']
    if not rows:
        raise ValueError('No status history to plot')
    time = [r['elapsed_s'] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    axes[0].plot(time, [r['recent_headroom'] for r in rows], label='Recent DSP headroom')
    axes[0].axhline(1, color='red', linestyle='--', label='Realtime')
    axes[0].set_ylabel('Speed / realtime'); axes[0].legend()
    axes[1].plot(time, [r['queue_ms'] for r in rows], label='Receive backlog')
    axes[1].plot(time, [r['capacity_ms'] for r in rows], linestyle='--', label='Buffer capacity')
    axes[1].set_ylabel('Milliseconds'); axes[1].legend()
    for channel in range(len(rows[-1]['channels'])):
        axes[2].plot(time, [r['channels'][channel]['normalized_innovation_square'] for r in rows],
                     label=rows[-1]['channels'][channel]['id'])
    axes[2].axhline(1, color='black', linestyle='--')
    axes[2].set_ylabel('Latest clock NIS'); axes[2].set_xlabel('Capture wall time (s)'); axes[2].legend(ncol=5)
    fig.savefig(output, dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('diagnostics', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--plot', type=Path)
    args = p.parse_args()
    report = summarize(args.diagnostics)
    if args.plot:
        plot(report, args.plot)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    else:
        report.pop('status_samples')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
