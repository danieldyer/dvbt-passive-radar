"""Refit physical motion and antenna offsets for every licensed site assignment.

Unlike the fixed-motion dictionary screen, each hypothesis gets a full joint
likelihood fit. Yaw remains fixed at zero for the requested geographic-frame
control. Independent complex band reflectivities and free envelope origins
avoid treating unknown scattering phase or range origin as surveyed geometry.
The fixed reflector support and coarse transverse anchor remain conditional.
"""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from ais import utc_seconds
from antenna_motion import Antennas
from field import Field
from fit_physical_profile import solve
from full_motion import Band
from geometry_profile import GeometryProfile
from imaging import motion_state, sites
from midpoint_model import discrepancy, representative
from motion import excess_path
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from ship_site_signature import licensed_options
from spline_motion import SplineMotion
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'motion', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=180)
    p.add_argument('--limit', type=int, help='Stop after this many completed assignments; rerun to resume')
    p.add_argument('--combinations', type=int, nargs='+', help='Run only these enumerated assignments')
    p.add_argument('--fix-antenna-locations', action='store_true', help='Matched control: refit motion and origins only')
    p.add_argument('--initial-results', type=Path, help='Continue matching completed fits from another sweep')
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    if not model.get('fixed_ne_frame') or model.get('rock_spacing_seconds'):
        raise ValueError('Use the fixed-N/E model with no fitted roll or pitch')
    curve = SplineMotion(model)
    if np.any(curve.yaw_coefficients):
        raise ValueError('This control keeps yaw exactly zero')
    tx, rx, _ = sites(args.geometry_directory)
    geometry = json.loads((args.geometry_directory/'geometry-prior.json').read_text())
    options = licensed_options(args.geometry_directory)
    groups = [sorted((o for o in options if o['channel'] == c),
        key=lambda o: (o['configured_role'] != 'primary', o['site_id'])) for c in range(5)]
    assignments = list(product(*groups))
    chosen = list(range(len(assignments))) if args.combinations is None else args.combinations
    if any(i not in range(len(assignments)) for i in chosen) or len(set(chosen)) != len(chosen):
        raise ValueError('Requested assignment index is invalid or repeated')
    code = [Path(__file__), Path(__file__).with_name('geometry_profile.py'),
        Path(__file__).with_name('physical_profile.py'), Path(__file__).with_name('fit_physical_profile.py'),
        Path(__file__).with_name('stationarity.py'), Path(__file__).with_name('ship_site_signature.py')]
    identity = {'motion': str(args.motion), 'motion_sha256': fingerprint(args.motion), 'fields': str(args.fields),
        'code_sha256': {str(path): fingerprint(path) for path in code},
        'geometry_sha256': {str(path): fingerprint(path) for path in
            [args.geometry_directory/'geometry-prior.json', args.geometry_directory/'acma-transmitters.json',
             args.geometry_directory/'sites-main.json']},
        'fixed_antenna_locations': args.fix_antenna_locations, 'requested_assignments': chosen,
        'candidate_assignments':[[o['site_id'] for o in a] for a in assignments],
        'iteration_limit': args.iterations}
    if args.initial_results:
        identity['initial_results'] = {str(p): fingerprint(p) for p in sorted(args.initial_results.glob('assignment-??.json'))}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'experiment.json'
    result = {'scope': __doc__, 'identity': identity, 'total_licensed_combinations': len(assignments),
        'trials': [], 'complete': False, 'accepted_calibration': False,
        'ranking_scope': 'Penalized profile likelihood; not marginal Bayesian model evidence or posterior site probabilities.'}
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume after changing inputs, model or numerical implementation')
    fields, fast_fields = [], []
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        fields.append(field); fast_fields.append(representative(field))
    epoch = utc_seconds(model['nominal_start_utc'])
    priors = [[motion_state(None, epoch+f.data['row_time'], model) for f in family]
              for family in (fields, fast_fields)]
    # Data, whitening and stationary response subspaces do not depend on the
    # licensed-site hypothesis. Reuse their exact factorizations across fits.
    band_families = []
    for family, states in zip((fields, fast_fields), priors):
        bands = []
        for c, (field, prior) in enumerate(zip(family, states)):
            selected = np.floor((field.time-curve.interval[0])/.5).astype(int)%2 == 0
            bands.append(Band(field, prior, tx[model['reference_channel']], rx,
                model['selected_points_body_m'][c], model, selected, range_bias_m=model['range_bias_m'][c]))
        band_families.append(bands)
    _, motion_matrix, lo, hi, motion_lower, motion_upper = constraints(model, common_path=True)
    for index in chosen:
        if any(t['index'] == index for t in result['trials']):
            continue
        if args.limit is not None and len(result['trials']) >= args.limit:
            break
        assignment = assignments[index]
        antennas = Antennas(geometry['ecef_to_enu'], [o['site_id'] for o in assignment],
            receiver_sigma=geometry['receiver']['prior_sigma_enu_m'])
        base, origins = [], []
        for c, (field, option) in enumerate(zip(fields, assignment)):
            alternate = np.asarray(option['transmitter_ecef_m'])
            base.append([alternate-field.data['transmitter_ecef_m'], np.zeros(3)])
            state = curve.apply(field.data['row_time'], priors[0][c], tx[model['reference_channel']], rx)
            centre = state['position_ecef_m']
            delta = excess_path(centre, alternate, rx)[0]-excess_path(centre, field.data['transmitter_ecef_m'], rx)[0]
            origins.append(model['range_bias_m'][c]-float(np.average(delta, weights=field.data['row_information'])))
        exact = GeometryProfile(band_families[0], curve.parameters, antennas, base_offsets=base, range_origins=origins)
        joint = GeometryProfile(band_families[1], curve.parameters, antennas, base_offsets=base, range_origins=origins)
        initial_source = None
        if args.initial_results:
            candidate = args.initial_results/f'assignment-{index:02d}.json'
            if candidate.exists():
                previous = json.loads(candidate.read_text())
                if (previous['site_assignment'] != list(assignment) or previous['source_motion'] != str(args.motion)
                        or previous['initial_range_bias_m'] != origins):
                    raise ValueError('Initial fit uses a different geometry or motion reference')
                joint.initial = np.asarray(previous['joint_parameters'])
                initial_source = str(candidate)
        lower = np.r_[motion_lower, np.full(antennas.count, -3.), np.array(origins)-75.]
        upper = np.r_[motion_upper, np.full(antennas.count, 3.), np.array(origins)+75.]
        yaw = (joint.active >= curve.sections[1].start)&(joint.active < curve.sections[1].stop)
        frozen = np.flatnonzero(yaw)
        lower[frozen] = upper[frozen] = 0.
        if args.fix_antenna_locations:
            lower[joint.antenna_slice] = upper[joint.antenna_slice] = 0.
        matrix = np.pad(motion_matrix, ((0, 0), (0, len(joint.initial)-joint.nmotion)))
        def checkpoint(parameters, trace):
            (args.output/f'assignment-{index:02d}.checkpoint.json').write_text(json.dumps({
                'parameters': parameters.tolist(), 'trace': trace, 'identity': identity}, indent=2)+'\n')
            if len(trace)%10 == 0:
                print(json.dumps({'assignment': index, **trace[-1]}), flush=True)
        parameters, fit = solve(joint, matrix, lo, hi, lower, upper,
            iterations=args.iterations, checkpoint=checkpoint)
        validation = exact.validation(parameters)
        summary = summarize(validation)
        motion = joint.expand(parameters)
        approximation = []
        for a, b in zip(exact.responses(parameters), joint.responses(parameters)):
            da, db = a['target'], b['target']
            approximation.append([discrepancy(da[:, j], db[:, j]) for j in range(da.shape[1])])
        fitted = antennas.describe(parameters[joint.antenna_slice])
        total_offset = np.array([joint.offsets(parameters, c) for c in range(5)])
        render_fit = {**fitted,
            'transmitter_offsets_enu_m': (total_offset[:, 0]@np.array(geometry['ecef_to_enu']).T).tolist(),
            'receiver_offset_enu_m': (total_offset[0, 1]@np.array(geometry['ecef_to_enu']).T).tolist(),
            'offset_reference': 'Original nominal sites used by the field, including the selected licensed-site change.'}
        output = {**model, 'parameters': motion.tolist(), 'range_bias_m': parameters[joint.range_slice].tolist(),
            'antenna_fit': render_fit, 'site_local_fit': fitted, 'site_assignment': assignment,
            'joint_parameters': parameters.tolist(), 'source_motion': str(args.motion),
            'initial_result': initial_source,
            'initial_range_bias_m': origins, 'fit': fit, 'original_row_validation_after': validation,
            'after_summary': summary, 'midpoint_approximation': approximation,
            'antenna_bound_hits': int(np.sum(abs(parameters[joint.antenna_slice]) > 3-1e-5)),
            'range_bound_hits': int(np.sum(abs(parameters[joint.range_slice]-origins) > 75-1e-5)),
            'location_prior_penalty': float(np.sum(parameters[joint.antenna_slice]**2)/2),
            'scope': __doc__, 'absolute_geometry_validated': False, 'fitted_real_parameters': int(np.sum(lower < upper))}
        path = args.output/f'assignment-{index:02d}.json'
        path.write_text(json.dumps(output, indent=2)+'\n')
        trial = {'index': index, 'sites': [o['site_id'] for o in assignment], 'path': str(path),
            'objective': fit['objective'], 'converged': fit['converged'],
            'maximum_kkt_residual': fit['maximum_kkt_residual'], 'validation': summary,
            'antenna_bound_hits': output['antenna_bound_hits'], 'range_bound_hits': output['range_bound_hits']}
        result['trials'].append(trial)
        result['complete'] = len(result['trials']) == len(chosen)
        result['training_selected_index'] = min(result['trials'], key=lambda t:t['objective'])['index']
        manifest.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(trial), flush=True)


if __name__ == '__main__':
    main()
