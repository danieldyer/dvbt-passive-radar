"""Test a constant per-channel excess-path-rate residual with exact phase bookkeeping."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from channel_alignment import Alignment, AlignmentBand
from fit_channel_alignment import load_trial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'scene', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=150)
    args = p.parse_args()
    source, curve, previous = load_trial(args.alignment, args.ais, args.geometry_directory)
    scene = json.loads(args.scene.read_text())
    if scene['input_alignment'] != str(args.alignment):
        raise ValueError('Subpixel scene belongs to a different experiment')
    points = scene['fit']['points_body_m']
    bands = [AlignmentBand(b.field, b.state, points, b.selected_time) for b in previous.bands]
    model = Alignment(bands, path_rate_scale_mps=.01, phase_epoch_s=float(np.mean(curve.interval)))
    initial = np.r_[scene['parameters'][:3*model.count], np.zeros(model.count)]
    before = model.validation(initial, curve.interval)
    started = time.perf_counter()
    fitted, result = model.fit(initial, iterations=args.iterations)
    output = {'input_alignment': str(args.alignment), 'input_scene': str(args.scene),
        'parameters': fitted.tolist(), 'fit': result, 'before': before,
        'after': model.validation(fitted, curve.interval), 'seconds': time.perf_counter()-started,
        'information': model.information(fitted),
        'scope': 'Frozen subpixel scene and shared physical ship motion. Per-channel residual dL=v*(t-157s) changes compressed range, RF phase -k*dL and symbol Doppler -k*v*T together. Ten fixes the relative path-rate gauge. A local effective path model, not an accepted antenna-location or transmitter-clock estimate.',
        'calibration_accepted': False}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('objective', 'converged', 'maximum_projected_gradient', 'iterations', 'path_rate_bias_mps', 'range_bias_m')}), flush=True)
    print(json.dumps({'seconds': output['seconds'], 'conditional_loss_db': output['after']['conditional_coherent_loss_db'],
        'error_change_percent': 100*(output['after']['prediction_error']/output['before']['prediction_error']-1)}), flush=True)


if __name__ == '__main__':
    main()
