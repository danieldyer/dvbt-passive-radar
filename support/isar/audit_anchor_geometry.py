"""Check local observability separately from global planar mirror ambiguity.

This is a known-target/receiver geometry example, not a calibration result for
the recording. Four planar targets can have full local rank for a 3-D unknown
transmitter plus a common unknown range origin, despite an exact mirror twin.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def excess(targets, receiver, parameters):
    transmitter, origin = parameters[:3], parameters[3]
    vectors = transmitter-targets
    direct = transmitter-receiver
    distances = np.linalg.norm(vectors, axis=1)
    baseline = np.linalg.norm(direct)
    value = distances+np.linalg.norm(targets-receiver, axis=1)-baseline+origin
    derivative = np.c_[vectors/distances[:, None]-direct/baseline, np.ones(len(targets))]
    return value, derivative


def example():
    targets = np.array([[0., 0., 0.], [100., 0., 0.], [0., 120., 0.], [80., 60., 0.]])
    receiver = np.array([-40., -60., 0.])
    truth = np.array([35., 45., 80., 7.5])
    mirror = truth.copy(); mirror[2] *= -1
    observations, derivative = excess(targets, receiver, truth)
    singular = np.linalg.svd(derivative, compute_uv=False)
    fits = []
    for source in (truth, mirror):
        initial = source+np.array([1., -1., .8, .2])
        fit = least_squares(lambda p: excess(targets, receiver, p)[0]-observations,
            initial, jac=lambda p: excess(targets, receiver, p)[1],
            gtol=1e-12, ftol=1e-12, xtol=1e-12)
        fits.append({'parameters': fit.x.tolist(), 'success': bool(fit.success),
            'maximum_residual_m': float(np.max(abs(fit.fun))),
            'maximum_parameter_error_from_corresponding_truth_m': float(np.max(abs(fit.x-source)))})
    return {'scope': __doc__, 'targets_m': targets.tolist(), 'receiver_m': receiver.tolist(),
        'true_transmitter_and_origin_m': truth.tolist(), 'mirror_transmitter_and_origin_m': mirror.tolist(),
        'local_jacobian_singular_values': singular.tolist(),
        'local_condition_number': float(singular[0]/singular[-1]),
        'local_jacobian_determinant': float(np.linalg.det(derivative)),
        'maximum_mirror_observation_difference_m': float(np.max(abs(
            observations-excess(targets, receiver, mirror)[0]))),
        'fits': fits, 'recording_calibration_claimed': False,
        'interpretation': 'The four target anchors are exactly coplanar, yet the unknown transmitter xyz and common origin have full local rank. Reflection across the anchor/receiver plane leaves all observations identical. This does not prove uniqueness or adequate precision for unknown moving vehicles; it prevents treating planar targets alone as a proof of continuous unobservability.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing geometry audit')
    result = example()
    if (result['local_jacobian_singular_values'][-1] <= .1 or
            result['maximum_mirror_observation_difference_m'] > 1e-12 or
            any(not f['success'] or f['maximum_parameter_error_from_corresponding_truth_m'] > 1e-7
                for f in result['fits'])):
        raise ValueError('Independent geometry check failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'condition_number': result['local_condition_number'],
        'mirror_difference_m': result['maximum_mirror_observation_difference_m']}))


if __name__ == '__main__':
    main()
