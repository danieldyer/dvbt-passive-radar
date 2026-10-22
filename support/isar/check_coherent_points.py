"""Check phase escape and nuisance-profiled information at a joint point fit.

Expected Gaussian information assumes the declared thermal covariance and
current active scattering model. All point coordinates are nuisance variables,
including coordinates at search bounds. Motion/sites/origins remain fixed, so
the fractions are optimistic conditional diagnostics, not calibration errors.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from ais import read_csv, utc_seconds
from channel_alignment import AlignmentBand
from coherent_points import CoherentPoints
from field import Field
from imaging import physical_state, sites
from phase_coupling import PhaseCoupling
from profile_information import profile
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import validate, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'alignment', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, help='Inspect an unfinished training iterate; never labels it converged')
    args = p.parse_args()
    fitted, alignment = [json.loads(path.read_text()) for path in (args.fit, args.alignment)]
    if args.checkpoint:
        checkpoint = json.loads(args.checkpoint.read_text())
        parameters = np.asarray(checkpoint['parameters'])
        reference_objective = checkpoint['trace'][-1]['objective']
    elif not fitted['complete'] or not fitted['fit']['converged']:
        raise ValueError('Reach the joint fit stationarity gate first')
    else:
        parameters = np.asarray(fitted['parameters'])
        reference_objective = fitted['fit']['objective']
    motion = json.loads(Path(fitted['source_motion']).read_text())
    curve = SplineMotion(motion)
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=motion['reference_height_ellipsoid_m'], mmsi=563052200)
    points = np.asarray(fitted['cell_centres_body_m'])
    bands = []
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        state = physical_state(track, utc_seconds(motion['nominal_start_utc'])+field.data['row_time'], motion['attitude_prior'])
        state = curve.apply(field.data['row_time'], state, tx[3], rx)
        block = np.floor(field.time-curve.interval[0]).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        bands.append(AlignmentBand(field, state, points, selected))
    model = CoherentPoints(bands, points, motion['range_bias_m'], reference=3)
    final = model.evaluate(parameters, jacobian=True)
    if abs(final['value']-reference_objective) > 1e-9:
        raise ValueError('Recreated objective disagrees with the completed fit')
    information = final['expected_information']
    n = model.count
    interest = np.arange(3*n, 3*n+4)
    checks = []
    for cutoff in (1e-8, 1e-10, 1e-12):
        fixed, fixed_details = profile(information, interest, np.arange(2*n, 3*n), cutoff=cutoff)
        free, free_details = profile(information, interest, np.arange(3*n), cutoff=cutoff)
        fraction, axes = eigh(free, fixed)
        checks.append({'cutoff': cutoff, 'fixed_coordinate_information': fixed.tolist(),
            'free_coordinate_information': free.tolist(), 'retained_information_fractions': fraction.tolist(),
            'generalized_eigenvectors': axes.tolist(), 'fixed': fixed_details, 'free': free_details})
    coupling = PhaseCoupling([b.nuisance.remove(o['target'][b.selected]) for b, o in zip(bands, final['bands'])],
        [b.clean for b in bands], reference=3)
    phase_start = np.r_[final['point_phase_radians'], np.array(final['band_phase_radians'])[model.other]]
    updated, escape = coupling.coordinate_refine(phase_start, sweeps=16)
    coefficients, phase_fit = coupling.fit(coordinate_sweeps=8)
    # The existing fit and its coordinate refinement are candidates too; a
    # new set of initializations must not discard a better training solution.
    candidates = [(phase_start, 'joint_fit'), (updated, 'coordinate'),
        (np.asarray(phase_fit['parameters']), 'expanded_phase_starts')]
    selected, label = min(candidates, key=lambda item: coupling.evaluate(item[0])['value'])
    best = coupling.evaluate(selected)
    coefficients = best['coefficients']
    _, band_phase = coupling.phases(selected)
    validation = [validate(b.field, o['target'], b.nuisance, b.selected_time, motion['interval_s'],
        coefficients=a, stationary_mode='project') for b, o, a in zip(bands, final['bands'], coefficients)]
    restart = model.initial(model.coordinates(parameters), selected[:n], band_phase)
    result = {'scope': __doc__, 'source_fit': str(args.fit), 'source_sha256': fingerprint(args.fit),
        'source_checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'source_converged': not bool(args.checkpoint),
        'information': checks, 'phase_escape': escape, 'expanded_phase_fit': phase_fit,
        'selected_phase_candidate': label, 'before_objective': final['value'], 'after_objective': best['value'],
        'phase_refit_maximum_gradient': float(abs(best['gradient']).max()),
        'relative_training_improvement': (final['value']-best['value'])/final['value'],
        'parameters': restart.tolist(), 'channel_phase_radians': band_phase.tolist(),
        'validation': {'channels': validation, 'summary': summarize(validation)},
        'restart_needed': final['value']-best['value'] > 1e-9, 'correction_adopted': False, 'complete': True}
    gradient = final['gradient'].copy()
    projected = parameters[:2*n]-np.clip(parameters[:2*n]-gradient[:2*n], -1., 1.)
    result['source_gradient'] = {'unconstrained_coordinate': float(abs(gradient[:2*n]).max()),
        'projected_coordinate': float(abs(projected).max()),
        'point_phase': float(abs(gradient[2*n:3*n]).max()), 'band_phase': float(abs(gradient[3*n:]).max())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('selected_phase_candidate', 'before_objective', 'after_objective', 'restart_needed')}))
    print(json.dumps([{'cutoff': row['cutoff'], 'fractions': row['retained_information_fractions']} for row in checks]))


if __name__ == '__main__':
    main()
