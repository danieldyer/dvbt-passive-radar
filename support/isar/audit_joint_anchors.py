"""Test shared-phase anchors on the refined support, profiling all other points.

The strongest common points are selected on full-aperture training data alone.
The same indices and physical coordinates are retained in both aperture halves.
All other point responses are independent complex nuisance coefficients per
band. Motion/support discovery used this aperture, so transfer is conditional.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr

from ais import read_csv, utc_seconds
from channel_alignment import AlignmentBand
from field import Field
from imaging import physical_state, sites
from partial_phase import common_visibility, fit
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import validate, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('support', 'alignment', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--counts', type=int, nargs='+', default=[2, 4, 8, 16, 32])
    p.add_argument('--leave-one-out-count', type=int,
        help='Sensitivity control: fit this training-ranked set, then omit each of its anchors in turn')
    args = p.parse_args()
    support, alignment = [json.loads(path.read_text()) for path in (args.support, args.alignment)]
    if not support['complete'] or not support['fit']['converged']:
        raise ValueError('Use completed joint physical point refinement')
    points = np.asarray(support['points_body_m'])
    if min(args.counts) < 2 or max(args.counts) > len(points):
        raise ValueError('Use at least two shared anchors and no more than the available support')
    if args.leave_one_out_count is not None and not 3 <= args.leave_one_out_count <= len(points):
        raise ValueError('Leave-one-out control requires at least three available anchors')
    motion_path = Path(support['source_motion'])
    motion = json.loads(motion_path.read_text())
    curve = SplineMotion(motion)
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=motion['reference_height_ellipsoid_m'], mmsi=563052200)
    middle = float(np.mean(curve.interval))
    intervals = [motion['interval_s'], [float(curve.interval[0]), middle], [middle, float(curve.interval[1])]]
    result = {'scope': __doc__, 'source_support': str(args.support), 'source_motion': str(motion_path),
        'source_sha256': {str(path): fingerprint(path) for path in
            (args.support, args.alignment, args.ais, motion_path, Path(__file__), Path(__file__).with_name('partial_phase.py'))},
        'complete': False, 'correction_adopted': False, 'trials': []}
    result['leave_one_out_count'] = args.leave_one_out_count
    ranking = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for interval in intervals:
        bands, targets, matrices, rhs = [], [], [], []
        for c in range(5):
            with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as data:
                field = Field(dict(data), data['pulse_samples']).windowed(interval[0]+.02, interval[1]-.02)
            prior = physical_state(track, utc_seconds(motion['nominal_start_utc'])+field.data['row_time'], motion['attitude_prior'])
            state = curve.apply(field.data['row_time'], prior, tx[3], rx)
            block = np.floor(field.time-interval[0]).astype(int)
            selected = (block%2 == 0)|(block == block.max())
            band = AlignmentBand(field, state, points, selected)
            target, _ = band.design(motion['range_bias_m'][c])
            design = band.nuisance.remove(target[band.selected])
            q, r = qr(design, mode='economic', check_finite=False)
            matrices.append(r); rhs.append(q.conj().T@band.clean)
            bands.append(band); targets.append(target)
        if ranking is None:
            ranking, visibility = common_visibility(matrices, rhs)
            result.update(anchor_ranking=ranking.tolist(), conditional_information_by_band=visibility.tolist())
        independent = [np.linalg.solve(a, y) for a, y in zip(matrices, rhs)]
        checks = [validate(b.field, d, b.nuisance, b.selected_time, interval,
            coefficients=a, stationary_mode='project') for b, d, a in zip(bands, targets, independent)]
        independent_score = {'channels': checks, 'summary': summarize(checks)}
        if args.leave_one_out_count:
            base = ranking[:args.leave_one_out_count]
            selections = [(base, None)]+[(base[base != omitted], int(omitted)) for omitted in base]
        else:
            selections = [(ranking[:count], None) for count in args.counts]
        for anchors, omitted in selections:
            count = len(anchors)
            coefficients, details = fit(matrices, rhs, anchors, reference=3, coordinate_sweeps=8)
            checks = [validate(b.field, d, b.nuisance, b.selected_time, interval,
                coefficients=a, stationary_mode='project') for b, d, a in zip(bands, targets, coefficients)]
            coupled_score = {'channels': checks, 'summary': summarize(checks)}
            penalty = 100*(coupled_score['summary']['prediction_error']/independent_score['summary']['prediction_error']-1)
            result['trials'].append({'interval_s': interval, 'count': count, 'fit': details,
                'omitted_anchor': omitted,
                'validation': {'independent': independent_score, 'coupled': coupled_score},
                'coupled_prediction_penalty_percent': penalty})
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'interval_s': interval, 'count': count, 'converged': details['converged'],
                'omitted_anchor': omitted,
                'gradient': details['maximum_gradient'], 'phase_degrees': np.degrees(details['channel_phase_radians']).tolist(),
                'penalty_percent': penalty}), flush=True)
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
