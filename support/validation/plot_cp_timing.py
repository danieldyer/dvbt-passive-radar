#!/usr/bin/env python3
"""Plot fixed-clock CP measurements; jumps are modulo one OFDM symbol."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def analyse(path, output, period=8704, rate=8e6):
    rows = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf8')
    channels = list(dict.fromkeys(rows['channel']))
    tracks = [rows[rows['channel'] == c] for c in channels]
    if len({len(t) for t in tracks}) != 1:
        raise ValueError('Incomplete or differently sized channel timelines')
    times = np.stack([t['time_s'] for t in tracks])
    phases = np.stack([t['cp_phase_samples'] for t in tracks])
    quality = np.stack([t['quality'] for t in tracks])
    delta = np.remainder(np.diff(phases, axis=1) + period / 2, period) - period / 2
    good = (quality[:, 1:] > .8) & (quality[:, :-1] > .8)
    usable = good.any(axis=0)
    consensus = np.full(delta.shape[1], np.nan)
    consensus[usable] = np.nanmedian(np.where(good[:, usable], delta[:, usable], np.nan), axis=0)
    agreeing = good & (np.abs(delta - consensus) <= 8)
    detected = (agreeing.sum(axis=0) >= 3) & (np.abs(consensus) > 32)
    event_times = np.median(times[:, 1:], axis=0)[detected]
    jumps = consensus[detected]
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'timing_discontinuities.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'modulo_jump_channel_samples', 'modulo_jump_us', 'agreeing_channels', 'minimum_cp_quality'])
        for n in np.flatnonzero(detected):
            writer.writerow([float(np.median(times[:, n + 1])), float(consensus[n]),
                             float(consensus[n] / rate * 1e6), int(agreeing[:, n].sum()),
                             float(quality[agreeing[:, n], n + 1].min())])
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    for c, track in zip(channels, tracks):
        offset = track['unwrapped_phase_samples'] - track['unwrapped_phase_samples'][0]
        axes[0].plot(track['time_s'], offset / rate * 1e3, lw=.85, label=c.title())
    axes[0].set_ylabel('Accumulated nearest-period\nCP offset (ms)')
    axes[0].legend(ncols=5, loc='lower left')
    axes[0].set_title('VHF recording: common timing discontinuities across five DVB-T channels')
    axes[1].vlines(event_times, 0, jumps / rate * 1e6, color='#ba3c3c', lw=.8)
    axes[1].set_ylabel('Consensus abrupt jump\n(µs, modulo 1088 µs)')
    for c, track in zip(channels, tracks):
        axes[2].plot(track['time_s'][::8], track['quality'][::8], lw=.65, alpha=.8)
    axes[2].set_ylabel('CP correlation quality')
    axes[2].set_ylim(.8, 1.005)
    axes[2].set_xlabel('Nominal time from start of WAV (s)')
    for axis in axes:
        axis.grid(alpha=.18)
        axis.set_xlim(0, float(times.max()))
    fig.savefig(output / 'timing_discontinuities.png', dpi=160)
    fig.savefig(output / 'timing_discontinuities.pdf')
    summary = dict(channels=channels, measurements_per_channel=len(tracks[0]),
                   measured_start_s=float(times.min()), measured_end_s=float(times.max()),
                   cadence_s=float(np.median(np.diff(times, axis=1))),
                   detected_consensus_jumps=len(jumps), median_cp_quality=float(np.median(quality)),
                   cp_quality_p01=float(np.quantile(quality, .01)), symbol_period_us=period/rate*1e6,
                   search_resolution_us=4/rate*1e6,
                   threshold_us=32/rate*1e6, agreement_us=8/rate*1e6,
                   interpretation='Nearest-period CP changes, not absolute lost-sample counts. Full-period losses are invisible; multiple jumps between measurements can merge. Accumulated offsets include oscillator drift and modulo ambiguity. Time uses nominal WAV indices, not a repaired physical capture timeline.')
    (output / 'timing_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    analyse(args.csv, args.output)
