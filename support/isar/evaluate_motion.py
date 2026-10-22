"""Compare full-range motion compensation at fixed physical body points.

The same body coordinates are evaluated with both trajectories. Points are
selected on the full fitted image, so this is a conditional diagnostic rather
than an independent validation of point selection or physical heading.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fit_autofocus import anchors, extract
from fit_gpga import binned_metrics
from gpga import phase_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    args = parser.parse_args()
    before_path, after_path = args.output/'before', args.output/'after'
    before_path.mkdir(parents=True, exist_ok=True)
    after_path.mkdir(parents=True, exist_ok=True)
    description = json.loads((args.after/'image-description.json').read_text())
    result = {'scope': __doc__, 'channels': [], 'interval_s': description['interval_s'],
              'before': str(args.before), 'after': str(args.after)}
    for channel in range(5):
        with np.load(args.after/f'channel-{channel}-image.npz') as image:
            pixels = anchors(image['complex_image'], image['along_m'], image['port_m'])
        stages = []
        for source, destination in ((args.before, before_path), (args.after, after_path)):
            data = extract(source, destination, channel, args.executable,
                           description['height_offset_m'], pixels=pixels.copy())
            t, z, w = data['time'], data['value'], data['information']
            train = np.arange(len(t))%2 == 0
            reference = np.sum(w[train, None]*z[train], axis=0)/w[train].sum()
            stages.append({'unbinned': phase_metrics(t, z, w, ~train, reference),
                           'one_second_bins': binned_metrics(t, z, w, ~train, reference, 1.)})
        before, after = [s['unbinned'] for s in stages]
        record = {'channel': channel, 'points_body_m': pixels[:, :3].tolist(),
            'before': stages[0], 'after': stages[1],
            'heldout_power_gain_db': float(10*np.log10(sum(after['coherent_energy_by_point'])/
                                                      sum(before['coherent_energy_by_point']))),
            'prediction_error_comparison': 'Not comparable across reimaging: the two histories sample different physical ranges. Use validate_motion.py for a likelihood on unchanged observations.'}
        result['channels'].append(record)
        print(json.dumps({k: record[k] for k in ('channel', 'heldout_power_gain_db')}), flush=True)
        (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
