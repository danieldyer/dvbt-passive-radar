"""Refine a validated alignment operator with shared subpixel scene positions."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from coherent_scene import CoherentScene
from fit_channel_alignment import load_trial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=180)
    p.add_argument('--radius-m', type=float, default=2.)
    args = p.parse_args()
    source, curve, alignment = load_trial(args.alignment, args.ais, args.geometry_directory)
    scene = CoherentScene(alignment)
    initial = source['stages'][-1]['parameters']
    started = time.perf_counter()
    parameters, fit = scene.fit(initial, iterations=args.iterations, position_radius_m=args.radius_m)
    output = {'input_alignment': str(args.alignment), 'parameters': parameters.tolist(),
        'fit': fit, 'before': source['stages'][-1]['validation'],
        'after': scene.validation(parameters, curve.interval), 'seconds': time.perf_counter()-started,
        'scope': 'Shared fixed-height scattering coordinates refined within one half grid step, plus per-band envelope range origin and complex gains. First point fixed to anchor coordinates; actual heading and hull centroid remain uncalibrated.',
        'source_holdout_limitation': source['limitation'], 'calibration_accepted': False}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:fit[k] for k in ('objective', 'converged', 'maximum_projected_gradient', 'iterations', 'range_bias_m', 'gain_phase_degrees')}), flush=True)
    print(json.dumps({'seconds': output['seconds'], 'conditional_loss_db': output['after']['conditional_coherent_loss_db'],
                      'error_change_percent': 100*(output['after']['prediction_error']/output['before']['prediction_error']-1)}), flush=True)


if __name__ == '__main__':
    main()
