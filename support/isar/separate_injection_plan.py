"""Move the existing point along its exact bistatic equal-range contour.

Separation is measured from the coarse AIS centre at the training midpoint.
Velocity, acceleration, height, band gains and aperture are unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from recorded_injection_fields import load_fields
from report_measured_scatterer_phases import serial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('docs/data/injection_20261018/plan.json'))
    p.add_argument('--baseline', type=Path, default=Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    p.add_argument('--cross-range-m', type=float, default=500.)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    original = json.loads(args.source.read_text())
    _, rotation, rx, tx, coarse, mid = load_fields(args.baseline)
    first, last = original['interval_s']
    centre_time = (first+(first+.65*(last-first)))/2
    dt = centre_time-mid
    coarse = np.asarray(coarse)
    prior = coarse[0]+dt*coarse[1]+.5*dt**2*coarse[2]
    original_state = np.asarray(original['state_enu'])
    point = original_state[0]+dt*original_state[1]+.5*dt**2*original_state[2]

    def path(enu, channel):
        world = rx+enu@rotation
        return np.linalg.norm(world-rx)+np.linalg.norm(world-tx[channel])-np.linalg.norm(tx[channel]-rx)

    world = rx+prior@rotation
    gradient = sum((world-a)/np.linalg.norm(world-a) for a in [rx, tx[2]])@rotation.T
    normal = gradient[:2]/np.linalg.norm(gradient[:2])
    tangent = np.array([-normal[1], normal[0]])
    target_range = path(point, 2)

    def candidate(normal_offset):
        value = prior.copy()
        value[:2] += args.cross_range_m*tangent+normal_offset*normal
        return value

    normal_offset = brentq(lambda r: path(candidate(r), 2)-target_range, -1000., 1000., xtol=1e-9)
    moved = candidate(normal_offset)
    new_state = original_state.copy()
    new_state[0] += moved-point
    changed = dict(original)
    changed['state_enu'] = new_state
    changed.pop('first_profile_analysis', None)
    changed['separation_control'] = dict(training_midpoint_s=centre_time,
        cross_range_from_ais_m=args.cross_range_m,
        original_cross_range_from_ais_m=float((point-prior)[:2]@tangent),
        horizontal_separation_from_ais_m=float(np.linalg.norm((moved-prior)[:2])),
        original_to_new_enu_m=moved-point, contour_tangent_enu=np.r_[tangent, 0],
        reference_channel=2, reference_path_difference_m=path(moved, 2)-target_range,
        per_channel_path_changes_m=[path(moved,c)-path(point,c) for c in range(5)],
        velocity_and_acceleration_unchanged=True, band_gains_unchanged=True,
        description=__doc__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        out.write(json.dumps(serial(changed), indent=2, allow_nan=False)+'\n')
    print(json.dumps(serial(changed['separation_control'])), flush=True)


if __name__ == '__main__':
    main()
