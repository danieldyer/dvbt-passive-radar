"""Refit saved geometric starting hypotheses against explicitly revised inputs.

Old objectives, band amplitudes and uncertainty estimates are not reused as
evidence. This holds the road association and site assignment fixed; held-out
validation and alternative-association checks are still required.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calibration_solver import solve
from coherent_fit import information
from profiled_bands import ProfiledBandProblem, MissingReferenceBand
from road_fit import load_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'histories', 'patches', 'templates', 'routes', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    saved = json.loads(args.fit.read_text())
    joint = load_trial(saved, args.routes, args.histories, args.patches, args.templates)
    for label in saved['labels']:
        description = json.loads((args.histories/f'{label}.json').read_text())
        if description.get('seed_policy') != 'one selected source track shared across bands; no independent peak reselection':
            raise ValueError('Revised fit requires the audited common source seed')
    problem = ProfiledBandProblem(joint)
    initial = problem.initial(np.array(saved['state']))
    r = problem.residual(initial)
    initial_cost = float(r@r/2)
    print('Revised-input initial cost', initial_cost, flush=True)

    def progress(result):
        if result.nfev%10 == 0:
            print('evaluation', result.nfev, 'cost', result.cost, flush=True)

    fitted = solve(problem.residual, initial, problem.bounds(), args.iterations, progress)
    serialization_error = None
    try:
        state, absent = problem.expanded_state(fitted.x)
    except MissingReferenceBand as error:
        # The old state format uses SBS as the amplitude reference. A valid
        # nonnegative optimum may discard it; retain that result and the exact
        # reduced state instead of inventing a positive SBS response.
        serialization_error = str(error)
        state = None
        absent = [np.flatnonzero(abs(problem.target(fitted.x, index)[1]) == 0).tolist()
                  for index in range(joint.m)]
    result = {key: saved[key] for key in (
        'assignment', 'labels', 'site_prior_sigma_m', 'site_positions_enu_m',
        'channel_offsets_enu_m')}
    for key in ('channels', 'range_bias', 'road_association_names', 'receiver_correction_enu_m'):
        if key in saved:
            result[key] = saved[key]
    result.update(
        source_fit=str(args.fit), input_histories=str(args.histories), input_patches=str(args.patches),
        input_templates=str(args.templates), input_routes=str(args.routes),
        selected_roads=[{key: road[key] for key in ('route', 'name', 'coarse_state', 'state')}
                        for road in saved['selected_roads']],
        initializer_scope='Selected-road states were discovered under the previous extraction and are used only as starting hypotheses, not as revised-input objective values.',
        state=None if state is None else state.tolist(), reduced_state=fitted.x.tolist(),
        expanded_state_error=serialization_error,
        cost=float(fitted.cost), revised_input_initial_cost=initial_cost,
        converged=bool(fitted.success), evaluations=fitted.nfev,
        optimality=float(fitted.optimality), termination=str(fitted.message),
        profiled_band_gains=True, zero_amplitude_channels_by_target=absent,
        site_corrections_enu_m=(fitted.x[:9].reshape(3, 3)*[30, 30, 75]).tolist(),
        range_bias_m=fitted.x[problem.bias_offset:].tolist(),
        information=None if state is None else information(joint, state),
        status='Conditional refit with revised common-source extraction. Fixed road/site hypotheses; no accepted calibration or metadata update.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Final', result['cost'], result['converged'], 'absent bands', absent, flush=True)


if __name__ == '__main__':
    main()
