"""Select persistent isolated returns for offline calibration, not production tracking."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

C = 299792458.0
NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']


def load(path):
    channels = [[] for _ in NAMES]
    with open(path) as source:
        header = json.loads(next(source))
        for line in source:
            record = json.loads(line)
            if record['type'] == 'map':
                channels[record['channel']].append(record)
    return header, channels


def tracks(maps, minimum_detections=15, minimum_duration=20.):
    result = []
    for frame in maps:
        t, rf = frame['time'], frame['rf_hz']
        points = np.array([[t, p['range_m'], -C * p['doppler_hz'] / rf,
                            p['selection_snr_db']] for p in frame['peaks']])
        if not len(points):
            continue
        active = [i for i, tr in enumerate(result) if t - tr[-1][0] < 3.1]
        assigned = set()
        if active:
            cost = np.full((len(active), len(points)), 1e8)
            for a, index in enumerate(active):
                tr = np.array(result[index][-6:])
                last = tr[-1]
                acceleration = np.polyfit(tr[:, 0] - last[0], tr[:, 2], 1)[0] if len(tr) >= 3 else 0
                acceleration = np.clip(acceleration, -8, 8)
                dt = t - last[0]
                dr = points[:, 1] - (last[1] + last[2] * dt + .5 * acceleration * dt * dt)
                dv = points[:, 2] - (last[2] + acceleration * dt)
                cost[a] = (dr / 40) ** 2 + (dv / 1.5) ** 2
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 4:
                    result[active[a]].append(points[b].tolist())
                    assigned.add(b)
        for b, point in enumerate(points):
            if b not in assigned:
                result.append([point.tolist()])
    return [np.array(tr) for tr in result if len(tr) >= minimum_detections
            and tr[-1][0] - tr[0][0] >= minimum_duration]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-duration', type=float, default=20.)
    parser.add_argument('--minimum-detections', type=int, default=15)
    parser.add_argument('--maximum-range', type=float, default=30000.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    header, maps = load(args.candidates)
    if args.minimum_duration<=0 or args.minimum_detections<4 or args.maximum_range<=0:
        parser.error('Positive duration/range and at least four detections are required')
    maps=[[dict(m,peaks=[p for p in m['peaks'] if p['range_m']<=args.maximum_range]) for m in c] for c in maps]
    found = [tracks(c,args.minimum_detections,args.minimum_duration) for c in maps]
    report = {'columns': ['seconds', 'excess_path_m', 'path_rate_mps', 'selection_snr_db'],
              'tracks': [[t.tolist() for t in ch] for ch in found],
              'minimum_duration_s':args.minimum_duration,'minimum_detections':args.minimum_detections,
              'maximum_range_m':args.maximum_range}
    (args.output / 'tracks.json').write_text(json.dumps(report))
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True, layout='constrained')
    for ch, ax in enumerate(axes):
        for i, tr in enumerate(found[ch]):
            ax.scatter(tr[:, 0], tr[:, 1] / 1000, c=tr[:, 2], vmin=-160, vmax=160,
                       cmap='coolwarm', s=7)
            mid = tr[len(tr) // 2]
            ax.text(mid[0], mid[1] / 1000, str(i), fontsize=8, color='black')
        ax.set(ylabel=NAMES[ch] + '\nExcess path (km)', ylim=(0,args.maximum_range/1000))
    axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('Persistent moving-return candidates; labels identify per-channel tracks\n'
                 'Screening only: colours show excess-path rate, not calibration confidence')
    fig.savefig(args.output / 'candidate_tracks.png', dpi=150)
    plt.close(fig)
    for ch, items in enumerate(found):
        print(NAMES[ch], len(items))
        for i, tr in enumerate(items):
            if len(tr) >= 25:
                print(i, 'time', np.round(tr[[0, -1], 0], 1), 'range', np.round(tr[[0, -1], 1]),
                      'rate', np.round(tr[[0, -1], 2], 2), 'SNR', round(np.median(tr[:, 3]), 1))


if __name__ == '__main__':
    main()
