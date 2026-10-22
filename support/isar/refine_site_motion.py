"""Refine a licensed-site motion trial against original rows for a movie.

Optional frozen envelope origins test transfer of the shared 30-second timing
reference to the longer aperture. Yaw and antenna coordinates stay fixed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from audit_joint_sites import load_joint
from export_shared_motion import exported_model
from fit_physical_profile import solve
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationarity import check
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--initial-motion', type=Path)
    p.add_argument('--freeze-origins', action='store_true')
    p.add_argument('--iterations', type=int, default=180)
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve existing original-row refinements')
    source = json.loads(args.source.read_text())
    joint = load_joint(source, args.fields, args.geometry)
    initial = np.array(source['joint_parameters'])
    if args.initial_motion:
        seed = json.loads(args.initial_motion.read_text()); curve = SplineMotion(seed)
        if seed['interval_s'] != joint.bands[0].model['interval_s']:
            raise ValueError('Initial motion must describe this aperture')
        full, active = curve.gauge_fixed_parameters()
        np.testing.assert_array_equal(active, joint.active)
        initial[:joint.nmotion] = full[active]
        initial[joint.range_slice] = seed['range_bias_m']
    joint.initial = initial
    model = joint.bands[0].model; curve = SplineMotion(model)
    if not model['fixed_ne_frame'] or np.any(joint.expand(initial)[curve.sections[1]]):
        raise ValueError('Keep exactly zero yaw in fixed N/E')
    _, matrix, lo, hi, lower, upper = constraints(model, common_path=True)
    origins = np.array(source['initial_range_bias_m'])
    lower = np.r_[lower, np.zeros(joint.antennas.count), origins-75.]
    upper = np.r_[upper, np.zeros(joint.antennas.count), origins+75.]
    yaw = (joint.active >= curve.sections[1].start)&(joint.active < curve.sections[1].stop)
    lower[np.flatnonzero(yaw)] = upper[np.flatnonzero(yaw)] = 0.
    if args.freeze_origins:
        lower[joint.range_slice] = upper[joint.range_slice] = initial[joint.range_slice]
    matrix = np.pad(matrix, ((0, 0), (0, len(initial)-matrix.shape[1])))
    files = [args.source, *([args.initial_motion] if args.initial_motion else []), Path(__file__),
        *[args.fields/f'channel-{c}.npz' for c in range(5)], Path('support/isar/geometry_profile.py'),
        Path('support/isar/physical_profile.py'), Path('support/isar/fit_physical_profile.py')]
    identity = {'source_sha256': {str(path): fingerprint(path) for path in files},
        'original_rows': True, 'frozen_antennas': True, 'frozen_origins': args.freeze_origins}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(parameters, trace):
        path = args.output.with_suffix('.checkpoint.json'); temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'identity': identity, 'parameters': parameters.tolist(), 'trace': trace}, indent=2)+'\n')
        temporary.replace(path)
        print(json.dumps({'stage': 'fit', **trace[-1]}), flush=True)
    print(json.dumps({'stage': 'loaded', 'free_parameters': int(np.sum(lower < upper)),
        'frozen_origins': args.freeze_origins, 'seconds': time.monotonic()-started}), flush=True)
    parameters, fit = solve(joint, matrix, lo, hi, lower, upper,
        iterations=args.iterations, checkpoint=checkpoint, polish_stationarity=False)
    value = joint.evaluate(parameters)
    kkt, violation = check(parameters, value['gradient'], matrix, lo, hi, lower, upper)
    validation = joint.validation(parameters)
    output = exported_model(joint, parameters)
    output.update(scope=__doc__, identity=identity, source_fit=str(args.source), fit=fit,
        joint_parameters=parameters.tolist(), original_row_kkt=kkt, constraint_violation=violation,
        original_row_validation_after=validation, after_summary=summarize(validation),
        frozen_origins=args.freeze_origins, seconds=time.monotonic()-started, complete=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', 'converged': fit['converged'], 'original_kkt': kkt,
        'validation': output['after_summary'], 'seconds': output['seconds']}), flush=True)


if __name__ == '__main__':
    main()
