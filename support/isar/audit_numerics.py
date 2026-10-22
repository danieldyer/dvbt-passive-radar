"""Record isolated numerical validation and actual-geometry attitude sensitivity.

This is validation tooling, not a fit to the recorded ship. In particular the
roll/pitch amplitudes below are declared probes, not measured vessel motion.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from ais import read_csv, utc_seconds
from imaging import physical_state, sites
from motion import C, excess_path, path_difference
from spline_motion import SplineMotion
from test_attitude_motion import independent_echo, setup
from test_channel_alignment import synthetic
from test_coherent_motion import CoherentMotionTests
from test_rigid_motion import fixture
from full_motion import Band


def antenna_recovery():
    """Known motion, two identifiable physical location directions, unknown gains/delays."""
    joint, truth, _ = CoherentMotionTests.geometry_fixture()
    active = np.r_[joint.nmotion+1, joint.nmotion+joint.nantenna-3,
                   np.arange(joint.nmotion+joint.nantenna, len(truth))]
    initial = truth.copy()
    initial[active[:2]] = 0
    initial[joint.range_slice] += [3., -2., 4.]
    h = joint.information(truth)[np.ix_(active, active)]/joint.energy
    eigen, vectors = np.linalg.eigh(h)
    transform = (vectors/np.sqrt(eigen))@vectors.T
    def objective(x):
        p = initial.copy()
        p[active] += transform@x
        value, gradient = joint.objective(p)
        return value, transform.T@gradient[active]
    result = minimize(objective, np.zeros(len(active)), jac=True, method='BFGS',
                      options={'gtol': 1e-8, 'maxiter': 80})
    recovered = initial.copy()
    recovered[active] += transform@result.x
    displacement = joint.antennas.factor@(recovered[joint.antenna_slice]-truth[joint.antenna_slice])
    before, after = joint.validation(initial), joint.validation(recovered)
    return {'scope': 'Independent direct-norm/Fourier/quadrature synthesis. Two identifiable antenna directions with known motion/point support; all complex band gains and all envelope origins unknown. Not a claim that all real antenna coordinates can be recovered from one ship.',
        'maximum_antenna_coordinate_error_m': float(abs(displacement).max()),
        'maximum_range_origin_error_m': float(abs(recovered[joint.range_slice]-truth[joint.range_slice]).max()),
        'maximum_scaled_gradient': float(abs(result.jac).max()), 'iterations': result.nit,
        'conditional_information_condition': float(eigen.max()/eigen.min()),
        'before_heldout_phase_rms_degrees': [c['prediction_phase_rms_degrees'] for c in before],
        'after_heldout_phase_rms_degrees': [c['prediction_phase_rms_degrees'] for c in after]}


def sensitivity(ais, geometry, epoch):
    track = read_csv(ais, reference_height_ellipsoid_m=5, mmsi=563052200)
    tx, rx, _ = sites(geometry)
    t = np.linspace(142, 172, 601)
    state = physical_state(track, epoch+t, 'course-prior')
    centre, frame = state['position_ecef_m'], state['body_to_ecef']
    _, gradient = excess_path(centre, tx[3], rx)
    direction = gradient/np.sum(gradient**2, axis=1)[:, None]
    probes = []
    for heights in ('20 m plane', 'mixed 5–45 m heights'):
        points = np.array([[-120., -15., 20.], [-70., 15., 20.], [-20., -15., 20.],
                           [30., 15., 20.], [80., -15., 20.], [130., 15., 20.]])
        if heights.startswith('mixed'):
            points[:, 2] = [5, 20, 40, 15, 45, 25]
        physical = centre[:, None, :]+np.einsum('nij,pj->npi', frame, points)
        tangent = points@np.array([[0, 1., 0], [-1., 0, 0], [0, 0, 0]])
        yaw_direction = np.einsum('nij,pj->npi', frame, tangent)
        for axis in (0, 1):
            angles = np.zeros((len(t), 3))
            angles[:, axis] = np.radians(.1)*np.sin(2*np.pi*(t-142)/10.)
            rotated = frame@Rotation.from_euler('xyz', angles).as_matrix()
            displacement = np.einsum('nij,pj->npi', rotated-frame, points)
            phases, derivatives = [], []
            for channel, rf in enumerate([177.5, 184.5, 191.625, 219.5, 226.5]):
                k = 2*np.pi*rf*1e6/C
                phase = -k*path_difference(physical, displacement, tx[channel], rx)
                phase -= phase.mean(axis=0)
                _, g = excess_path(physical, tx[channel], rx)
                derivative = -k*np.stack([np.einsum('npi,ni->np', g, direction),
                                           np.sum(g*yaw_direction, axis=2)], axis=-1)
                phases.append(phase); derivatives.append(derivative)
            phase, derivative = np.concatenate(phases, axis=1), np.concatenate(derivatives, axis=1)
            solution = np.array([np.linalg.lstsq(a, b, rcond=None)[0] for a, b in zip(derivative, phase)])
            residual = phase-np.einsum('npi,ni->np', derivative, solution)
            probes.append({'support': heights, 'rotation': ['roll', 'pitch'][axis],
                'probe_amplitude_degrees': .1, 'period_seconds': 10.,
                'phase_rms_degrees_before_common_path_yaw': float(np.degrees(np.sqrt(np.mean(phase**2)))),
                'phase_rms_degrees_after_common_path_yaw': float(np.degrees(np.sqrt(np.mean(residual**2)))),
                'scope': 'Exact geometric phase; remove constant per-point phases, then best linear common-path/yaw perturbations independently at each epoch. This is a sensitivity probe, not recovered attitude or a realistic motion prior.'})
    return probes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ais', type=Path, required=True)
    p.add_argument('--geometry-directory', type=Path, required=True)
    p.add_argument('--nominal-start-utc', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    field = fixture()
    model, prior = setup(field)
    curve = SplineMotion(model)
    points = np.array([[-100., -12., 8.], [20., 16., 35.], [115., -6., 45.]])
    amplitudes = np.array([.8+.3j, -.5+.4j, .4-.7j])
    state = curve.apply(field.data['row_time'], prior, field.data['transmitter_ecef_m'], field.data['receiver_ecef_m'])
    field.y = independent_echo(field, state, points, amplitudes)
    band = Band(field, prior, field.data['transmitter_ecef_m'], field.data['receiver_ecef_m'],
                points, model, np.arange(len(field.time))%2 == 0)
    design, _ = band.design(curve.parameters)
    forward_error = np.linalg.norm(design@amplitudes-field.y.ravel())/np.linalg.norm(field.y)
    check = band.validation(curve.parameters)
    fit, bias, gains = synthetic()
    p, result = fit.fit()
    actual_bias, actual_gain = fit.unpack(p)
    before, after = fit.validation(np.zeros(3*fit.count), [0, 16]), fit.validation(p, [0, 16])
    report = {'scope': __doc__, 'known_three_dimensional_scene': {
        'relative_complex_forward_error': float(forward_error),
        'heldout_phase_rms_degrees': check['prediction_phase_rms_degrees'],
        'heldout_conditional_loss_db': check['conditional_efficiency_loss_db'],
        'independent_synthesis': 'Direct Euclidean norms, Fourier spectrum pulse evaluation, Gauss–Legendre symbol quadrature.'},
        'known_range_phase_recovery': {'fit': result,
            'maximum_range_error_m': float(np.max(abs(actual_bias-bias))),
            'maximum_relative_phase_error_degrees': float(np.max(abs(np.degrees(np.angle(actual_gain/gains))))),
            'maximum_gain_amplitude_error': float(np.max(abs(abs(actual_gain)-abs(gains)))),
            'before_conditional_coherent_loss_db': before['conditional_coherent_loss_db'],
            'after_conditional_coherent_loss_db': after['conditional_coherent_loss_db'],
            'before_phase_rms_degrees_by_channel': [r['prediction_phase_rms_degrees'] for r in before['channels']],
            'after_phase_rms_degrees_by_channel': [r['prediction_phase_rms_degrees'] for r in after['channels']],
            'heldout_error_fraction': after['prediction_error']/after['stationary_only_error']},
        'known_antenna_location_recovery': antenna_recovery(),
        'actual_geometry_roll_pitch_sensitivity': sensitivity(args.ais, args.geometry_directory, utc_seconds(args.nominal_start_utc))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
