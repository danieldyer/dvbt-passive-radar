"""Conditional exact point-response cuts with training-fitted brightness.

No recorded image is replaced or normalised here. This separates the known
operator's point spread and the loss from an unweighted temporal aperture
from residual scene/model error. The motion, associations and brightness are
conditional fitted hypotheses, not an independently known physical truth.
All measured cuts use excluded epochs only.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from full_ship_field import load
from full_path_point import PointBand
from full_path_scene import SceneBand
from joint_ship_motion import JointMotion
from rigid_patch import profile_static
from report_measured_scatterer_phases import serial


def matched(template, data):
    """Complex Gaussian matched statistic and its original template norm."""
    energy = float(np.vdot(template, template).real)
    return np.vdot(template, data), energy


def coherent_fraction(template, signal):
    z, energy = matched(template, signal)
    signal_energy = float(np.vdot(signal, signal).real)
    return float(abs(z)**2/(energy*signal_energy)) if energy*signal_energy else 0.


def conditioned_information(values, derivatives):
    """Position Fisher matrix after one unknown constant complex amplitude."""
    h = np.concatenate([x.ravel() for x in values])
    d = np.concatenate([x.reshape(-1, 2) for x in derivatives])
    norm = np.vdot(h, h).real
    if norm <= 0:
        raise ValueError('The conditional point has no information')
    d = d-h[:, None]*(h.conj()@d/norm)[None, :]
    return 2*np.real(d.conj().T@d)


def width(distance, power):
    """Central, zero-coordinate FWHM, with unresolved cuts left explicit."""
    middle = int(np.argmin(abs(distance)))
    threshold = power[middle]/2
    left = np.flatnonzero(power[:middle] <= threshold)
    right = np.flatnonzero(power[middle+1:] <= threshold)+middle+1
    if not len(left) or not len(right):
        return None
    lo, hi = left[-1], right[0]
    a = np.interp(threshold, power[lo:lo+2], distance[lo:lo+2])
    b = np.interp(threshold, power[hi-1:hi+1][::-1], distance[hi-1:hi+1][::-1])
    return float(b-a)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--points', type=int, nargs='+', help='Default: three highest-information fitted points')
    p.add_argument('--extent-m', type=float, default=100.)
    p.add_argument('--step-m', type=float, default=1.)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    fit = json.loads(a.motion.read_text())
    if not fit['complete'] or 'scene_window_edges_s' in fit:
        raise ValueError('Use a complete isolated persistent-scene fit')
    motion = JointMotion(Path(fit['motion']))
    motion.interval = fit['interval_s']
    fields, coordinates, _ = load(fit['field'])
    saved = np.load(a.motion.parent/'trajectory.npz')
    points = np.asarray(fit['scene_points_body_m'])
    phase = np.asarray(fit['scene_point_phase_radians'])
    gains = np.array([complex(*z) for z in fit['physical_gains_real_imag']])
    bands = []
    predictions = []
    for c, (old, coordinate) in enumerate(zip(fields, coordinates)):
        field = old.windowed(*fit['interval_s'])
        index = np.searchsorted(old.time, field.time)
        coord = {k: coordinate[k].reshape(-1, 16)[index].ravel() for k in ['native_time_s', 'native_information']}
        coord.update(training=coordinate['training'][index])
        if 'stationary_groups' in coordinate:
            coord['stationary_groups'] = coordinate['stationary_groups'][index]
        b = PointBand(field, coord, motion, fit['point_body_m'], fit['path_knots_s'],
            fit['path_coefficients_m'], fit['anchor_path_time_s'], fit['amplitude_spacing_s'], fit['orientation'],
            angular_rate_offset_degps=fit.get('angular_rate_offset_degps', 0.),
            angular_acceleration_offset_degps2=fit.get('angular_acceleration_offset_degps2', 0.))
        np.testing.assert_allclose(b.t, saved[f'time_s_{c}'], atol=1e-10, rtol=0)
        scene = SceneBand(b, points)
        h = scene.design(np.zeros(b.b.shape[1]), c, False)
        predictions.append(np.sum(h*(saved[f'amplitude_{c}']*np.exp(1j*(phase+np.angle(gains[c]))))[:, None, :], axis=-1))
        bands.append(b)
    strength = np.zeros(len(points))
    for c, b in enumerate(bands):
        h = SceneBand(b, points).design(np.zeros(b.b.shape[1]), c, False)
        strength += np.sum(abs(h[~b.training])**2*saved[f'amplitude_{c}'][~b.training, None, :]**2, axis=(0, 1))
    selected = a.points or np.argsort(strength)[-3:][::-1].tolist()
    distance = np.arange(-round(a.extent_m/a.step_m), round(a.extent_m/a.step_m)+1)*a.step_m
    records = []
    arrays = dict(distance_m=distance)
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(len(selected), 3, figsize=(16, 4*len(selected)), layout='constrained', squeeze=False)
    for row, point_index in enumerate(selected):
        point = points[point_index]
        true = []
        derivative = []
        flat = []
        fractions = []
        supports = []
        for c, b in enumerate(bands):
            mask = ~b.training
            def project(x):
                return profile_static(x, b.field.root_information, mask, b.groups)*mask.reshape((-1,)+(1,)*(x.ndim-1))
            h, _, _, de, dn = b.waveform(np.zeros(b.b.shape[1]), c, False, point-b.reference_point, True)
            amplitude = saved[f'amplitude_{c}'][:, point_index]
            carrier = np.exp(1j*(phase[point_index]+np.angle(gains[c])))
            signal = project(h*amplitude[:, None]*carrier)
            dh = project(np.stack([de, dn], axis=-1)*amplitude[:, None, None]*carrier)
            hp = project(h)*gains[c]
            true.append(signal)
            derivative.append(dh)
            flat.append(hp)
            fractions.append(coherent_fraction(hp, signal))
            info = np.sum(abs(signal)**2, axis=1)
            cumul = np.cumsum(info)
            supports.append([float(np.interp(q*cumul[-1], cumul, b.t)) for q in [.05, .95]])
        fisher = conditioned_information(true, derivative)
        eigen, vectors = np.linalg.eigh(fisher)
        response = np.zeros((2, len(distance), 5, 4), complex)
        norms = np.zeros((2, len(distance), 5))
        point_response = np.zeros((2, len(distance), 5), complex)
        point_norm = np.zeros((2, len(distance), 5))
        for axis in range(2):
            for j, offset in enumerate(distance):
                candidate = point.copy()
                candidate[:2] += offset*vectors[:, axis]
                for c, b in enumerate(bands):
                    mask = ~b.training
                    h = b.waveform(np.zeros(b.b.shape[1]), c, False, candidate-b.reference_point)
                    hp = profile_static(h, b.field.root_information, mask, b.groups)*mask[:, None]*gains[c]
                    pred = profile_static(predictions[c], b.field.root_information, mask, b.groups)*mask[:, None]
                    data = b.data[False]
                    for k, value in enumerate([data, pred, data-pred, true[c]]):
                        response[axis, j, c, k], norms[axis, j, c] = matched(hp, value)
                    amplitude = saved[f'amplitude_{c}'][:, point_index]
                    optimal = profile_static(h*amplitude[:, None], b.field.root_information, mask, b.groups)*mask[:, None]*gains[c]
                    z, energy = matched(optimal, true[c])
                    point_response[axis, j, c] = z
                    point_norm[axis, j, c] = energy
            print(json.dumps(dict(point=point_index, axis=axis, seconds=time.monotonic()-started)), flush=True)
        denominator = norms.sum(axis=2)
        power = abs(response.sum(axis=2))**2/denominator[:, :, None]
        oracle_power = abs(point_response.sum(axis=2))**2/point_norm.sum(axis=2)
        # Unit-peak PSF normalization is confined to a known synthetic point.
        # Recorded/model/residual statistics retain their original noise units.
        middle = len(distance)//2
        for axis in range(2):
            ax = axes[row, axis]
            for k, label in enumerate(['Excluded recording', 'Full fitted scene', 'Unexplained residual']):
                ax.plot(distance, 10*np.log10(np.maximum(power[axis, :, k], 1e-30)), label=label, lw=.9)
            ax.set(xlabel='Displacement along conditional principal axis (m)', ylabel='Matched power / thermal variance (dB)',
                title=f'Point {point_index} · {"weak" if axis == 0 else "strong"} information axis')
            ax.legend(fontsize=7)
        ax = axes[row, 2]
        for axis, style in enumerate(['-', '--']):
            single = power[axis, :, 3]
            ax.plot(distance, 10*np.log10(np.maximum(single/single[middle], 1e-20)), style, label=f'Flat aperture, axis {axis+1}')
            ax.plot(distance, 10*np.log10(np.maximum(oracle_power[axis]/oracle_power[axis, middle], 1e-20)), style, color='black', alpha=.6,
                label=f'Known-envelope filter, axis {axis+1}')
        ax.set(xlabel='Displacement from the known conditional point (m)', ylabel='Known-point power relative to true location (dB)', ylim=[-35, 3], title='Synthetic point only: exact motion and fitted fading')
        ax.legend(fontsize=7)
        record = dict(point=point_index, position_body_m=point, conditional_position_information=fisher,
            position_information_eigenvalues=eigen, principal_axes_body_xy=vectors.T,
            flat_filter_fraction_of_known_envelope_information_by_band=fractions,
            ninety_percent_signal_information_time_s_by_band=supports,
            flat_known_point_central_fwhm_m=[width(distance, x) for x in power[:, :, 3]],
            known_envelope_point_central_fwhm_m=[width(distance, x) for x in oracle_power])
        records.append(record)
        arrays[f'power_{point_index}'] = power
        arrays[f'known_envelope_point_power_{point_index}'] = oracle_power
    fig.suptitle('Exact known-waveform response cuts; excluded epochs; original noise units\nConditional motion/scatterer model, not accepted truth. No phase correction or image normalization is applied.')
    fig.savefig(a.output/'fading-aperture-cuts.png', dpi=145)
    plt.close(fig)
    np.savez_compressed(a.output/'cuts.npz', **arrays)
    (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=True, scope=__doc__, motion=str(a.motion),
        point_selection='Requested indices or three highest conditional signal-information fitted points', records=records,
        seconds=time.monotonic()-started)), indent=2)+'\n')


if __name__ == '__main__':
    main()
