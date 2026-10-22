"""Recreate joint range/phase fits and check real-data derivatives and phase escape."""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from channel_alignment import AlignmentBand
from field import Field
from imaging import physical_state, sites
from range_phase import RangePhase
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'alignment', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    source, alignment = [json.loads(path.read_text()) for path in (args.fit, args.alignment)]
    if not source['complete'] or not all(t['fit']['converged'] for t in source['trials']):
        raise ValueError('Finish all range/phase trials first')
    support = json.loads(Path(source['source_support']).read_text())
    motion = json.loads(Path(source['source_motion']).read_text())
    curve = SplineMotion(motion)
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=motion['reference_height_ellipsoid_m'], mmsi=563052200)
    points = np.asarray(support['points_body_m'])
    output = {'scope': __doc__, 'source_fit': str(args.fit), 'source_sha256': fingerprint(args.fit),
              'complete': False, 'trials': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for trial in source['trials']:
        span, bands = trial['interval_s'], []
        for c in range(5):
            with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as data:
                field = Field(dict(data), data['pulse_samples']).windowed(span[0]+.02, span[1]-.02)
            prior = physical_state(track, utc_seconds(motion['nominal_start_utc'])+field.data['row_time'], motion['attitude_prior'])
            state = curve.apply(field.data['row_time'], prior, tx[3], rx)
            block = np.floor(field.time-span[0]).astype(int)
            selected = (block%2 == 0)|(block == block.max())
            bands.append(AlignmentBand(field, state, points, selected))
        phase = np.r_[trial['point_phase_radians'], np.asarray(trial['band_phase_radians'])[[0, 1, 2, 4]]]
        seeds = [phase, *source['phase_seeds_radians']]
        model = RangePhase(bands, trial['range_origins_m'], seeds, reference=3, joint=True)
        parameters = model.initial.copy()
        result = model.evaluate(parameters)
        if abs(result['value']-trial['fit']['objective']) > 1e-9:
            raise ValueError('Recreated physical likelihood does not match the recorded fit')
        # Check away from stationarity so the comparison tests a nonzero
        # derivative, not just cancellation of two nearly identical costs.
        rng = np.random.default_rng(71247)
        probe = parameters+rng.normal(size=len(parameters))*.01
        value = model.evaluate(probe)
        directions = np.r_[np.eye(len(parameters))[:5],
                           rng.normal(size=(3, len(parameters))) / np.sqrt(len(parameters))]
        checks = []
        for direction in directions:
            step = 1e-4
            numerical = (model.objective(probe+step*direction)[0]-model.objective(probe-step*direction)[0])/(2*step)
            analytic = float(value['gradient']@direction)
            checks.append({'analytic': analytic, 'numerical': float(numerical), 'absolute_error': float(abs(analytic-numerical))})
        maximum = max(t['absolute_error'] for t in checks)
        if maximum > 1e-7:
            raise ValueError(f'Real-data joint gradient mismatch: {maximum:g}')
        entries = model.prepare(parameters[:5]*model.scale)
        coupling = model.coupling(entries)
        refined = model.phase_solution(coupling)
        after = coupling.evaluate(refined)
        improvement = result['value']-after['value']
        record = {'interval_s': span, 'objective': result['value'], 'maximum_gradient_error': maximum,
            'directional_checks': checks, 'phase_refit_objective': after['value'],
            'phase_refit_maximum_gradient': float(abs(after['gradient']).max()),
            'phase_escape_improvement': float(improvement), 'restart_needed': bool(improvement > 1e-9),
            'parameters_after_phase_refit': np.r_[parameters[:5], refined].tolist()}
        output['trials'].append(record)
        args.output.write_text(json.dumps(output, indent=2)+'\n')
        print(json.dumps({k: record[k] for k in ('interval_s', 'maximum_gradient_error', 'phase_escape_improvement', 'restart_needed')}), flush=True)
    output['complete'] = True
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
