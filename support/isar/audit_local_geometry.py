"""Compare relative-coordinate geometry with the old ECEF round trip.

The legacy calculation is a diagnostic wrapper only. Both paths use identical
physical motion, scenes, observations and analytic likelihood derivatives.
No fit is accepted or selected by this finite-difference probe.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from field import Field
from fit_marginal_motion import make_bands
from marginal_motion import MarginalMotion
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    fields = []
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as f:
            fields.append(Field(dict(f), f['pulse_samples']))
    bands = make_bands(fields, model)
    curve = SplineMotion(model)
    joint = MarginalMotion(bands, model['parameters'])
    yaw = curve.sections[1]
    free = (joint.active < yaw.start)|(joint.active >= yaw.stop)
    rng = np.random.default_rng(14731)
    directions = rng.normal(size=(3, len(joint.initial)))*free
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    original = Field.raw_trajectory

    def legacy(self, *a, **kw):
        kw['position_offset'] = None
        return original(self, *a, **kw)

    results, templates = {}, {}
    for name in ('local', 'ecef_round_trip'):
        with nullcontext() if name == 'local' else patch.object(Field, 'raw_trajectory', legacy):
            joint.last = None
            value = joint.evaluate(joint.initial)
            templates[name] = [b.design(joint.expand(joint.initial))[0] for b in bands]
            analytic = directions@value['gradient']
            probes = []
            for step in (1e-3, 1e-4, 1e-5, 1e-6):
                numeric = np.array([(joint.objective(joint.initial+step*d)[0]-
                    joint.objective(joint.initial-step*d)[0])/(2*step) for d in directions])
                probes.append({'step': step, 'finite_difference': numeric.tolist(),
                    'maximum_absolute_derivative_error': float(np.max(abs(numeric-analytic)))})
            results[name] = {'objective': value['value'], 'analytic_directional_derivative': analytic.tolist(),
                'probes': probes}
    changes = [float(np.linalg.norm(a-b)/np.linalg.norm(a)) for a, b in
        zip(templates['local'], templates['ecef_round_trip'])]
    result = {'scope': __doc__, 'motion': str(args.motion), 'motion_sha256': fingerprint(args.motion),
        'per_band_relative_template_change': changes, 'directions': directions.tolist(), 'results': results,
        'interpretation': 'This tests numerical consistency, not real ship motion or a physical SNR improvement.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'per_band_relative_template_change': changes, 'results': results}), flush=True)


if __name__ == '__main__':
    main()
