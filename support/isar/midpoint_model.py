"""Bound a fast forward-model approximation; observations remain unchanged.

The exact original-row model is retained for validation and final backprojection.
The approximation evaluates a template at each information-weighted bin epoch.
It is useful only when a declared motion/support check bounds its mismatch.
"""
import json

import numpy as np

from field import Field, average


def representative(field):
    data = dict(field.data)
    weight = data['row_information']
    block = field.metadata['block_rows']
    for key in ('centre_ecef_m', 'body_to_ecef', 'centre_path_m',
                'centre_velocity_ecef_mps', 'centre_omega'):
        if np.all(data[key]==data[key][0]):
            # A fixed observer remains exactly fixed. Weighted arithmetic on
            # an identical value can otherwise add round-off that changes
            # its declared stationary-reference identity.
            data[key]=np.broadcast_to(data[key][0],(len(field.time),)+data[key].shape[1:]).copy()
            continue
        values, information = average(data[key], weight, block)
        if np.any(information == 0):
            fallback, _ = average(data[key], np.ones(len(weight)), block)
            values[information == 0] = fallback[information == 0]
        data[key] = values
    u, _, vh = np.linalg.svd(data['body_to_ecef'])
    data['body_to_ecef'] = u@vh
    data['row_time'] = field.time.copy()
    data['row_information'] = data['information'].copy()
    data['metadata_json'] = json.dumps({**field.metadata, 'block_rows': 1,
        'model_approximation': __doc__, 'source_model_block_rows': block})
    result = Field(data, field.pulse.samples)
    np.testing.assert_allclose(result.y, field.y, rtol=0, atol=0)
    return result


def discrepancy(exact, approximate):
    a, b = exact.ravel(), approximate.ravel()
    inner = np.vdot(a, b)
    loss = float(10*np.log10(np.vdot(a, a).real*np.vdot(b, b).real/abs(inner)**2))
    return {'matched_snr_loss_db': max(0., loss),
        'relative_complex_error': float(np.linalg.norm(a-b)/np.linalg.norm(a)),
        'constant_phase_difference_degrees': float(np.degrees(np.angle(inner))),
        'amplitude_ratio': float(np.linalg.norm(b)/np.linalg.norm(a))}
