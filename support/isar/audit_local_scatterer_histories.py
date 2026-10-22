"""Measure deblended complex histories; apply no correction to the data.

Each one-second bin estimates the complex coefficients of the declared,
range-separated point responses by ordinary GLS in the existing whitened
metric. A full-record stationary profile is estimated from the frozen scene
residual. Training and excluded observations are treated separately. These
histories diagnose a motion model; they are not a fitted phase screen.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from full_path_point import PointBand
from full_path_scene import SceneBand
from full_ship_field import load
from joint_ship_motion import JointMotion
from rigid_patch import profile_static
from local_ship_motion import load_motion
from motion import excess_path
from full_path_point import carrier_bin


def deblend(h, y):
    a = np.asarray(h).reshape(-1, h.shape[-1])
    y = np.asarray(y).ravel()
    gram = a.conj().T @ a
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[0] <= eigenvalues[-1] * 1e-10:
        raise ValueError('Unresolved scattering responses; no diagnostic inverse')
    covariance = np.linalg.inv(gram)
    return covariance @ (a.conj().T @ y), covariance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strong-motion', type=Path, help='Diagnostic only: jointly deblend three nuisance responses on the separate stronger delay branch')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    d = json.loads(args.motion.read_text())
    fields, coordinates, _ = load(Path(d['field']))
    motion = JointMotion(Path(d['motion']))
    saved = np.load(args.motion.parent / 'trajectory.npz')
    points = np.asarray(d['scene_points_body_m'])
    edges = np.asarray(d['scene_window_edges_s'])
    phase = np.asarray(d['scene_point_phase_radians']).reshape(points.shape[:2])
    beta = np.angle([complex(*z) for z in d['physical_gains_real_imag']])
    windows, count, _ = points.shape
    strong = load_motion(args.strong_motion) if args.strong_motion else None
    times = []; histories = []; covariances = []; identities = []; folds = []; rows = {}
    for c, (field, coordinate) in enumerate(zip(fields, coordinates)):
        point = PointBand(field, coordinate, motion, d['point_body_m'],
                          np.asarray(d['path_knots_s']), np.asarray(d['path_coefficients_m']),
                          162., d['amplitude_spacing_s'], d.get('orientation', 'existing'))
        band = SceneBand(point, points, edges, d['amplitude_spacing_s'])
        h = band.design(np.zeros(point.b.shape[1]), c, False)
        h *= np.exp(1j * (band.phase_at_time(phase.ravel()) + beta[c]))[:, None, :]
        model = np.sum(h * saved[f'amplitude_{c}'][:, None, :], axis=-1)
        if strong is not None:
            physical = strong.state(point.t)
            source = np.asarray(strong.path_fit['point_body_m'])
            position = physical['position_ecef_m'] + physical['body_to_ecef']@source
            velocity = physical['velocity_ecef_mps'] + physical['frame_velocity']@source
            delta, omega, gradient = field.path_kinematics(position, velocity, transmitter=strong.tx[c], receiver=strong.rx,
                position_offset=position-field.data['centre_ecef_m'])
            speed = np.sum(gradient*velocity, axis=1)
            factor, _ = carrier_bin(point.t, speed, coordinate['native_time_s'], coordinate['native_information'], field.k)
            nuisance = [field.apply(field.path_response(delta+offset, omega)*factor[:, None]) for offset in [-60., 0., 60.]]
            h = np.concatenate([h, np.stack(nuisance, axis=-1)], axis=-1)
        for training in [True, False]:
            mask = point.training if training else ~point.training
            residual = field.y - model
            # frozen-scene residual mean, independently in each fold
            stationary = residual - profile_static(residual, field.root_information, mask, point.groups)
            y = field.y - stationary
            selected = mask & (field.root_information > 0)
            gram = np.einsum('trp,trq->tpq', h[selected].conj(), h[selected])
            eigenvalues = np.linalg.eigvalsh(gram)
            resolved = eigenvalues[:, 0] > eigenvalues[:, -1]*1e-10
            indices = np.flatnonzero(selected)[resolved]
            gram = gram[resolved]
            covariance_rows = np.linalg.inv(gram)
            rhs = np.einsum('trp,tr->tp', h[indices].conj(), y[indices])
            value_rows = np.einsum('tpq,tq->tp', covariance_rows, rhs)
            suffix = f'{c}_{int(training)}'
            rows[f'time_s_{suffix}'] = point.t[indices]
            rows[f'window_{suffix}'] = band.window_index[indices]
            rows[f'value_{suffix}'] = value_rows[:, :count]
            rows[f'covariance_{suffix}'] = covariance_rows[:, :count, :count]
            for w, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
                cuts = np.linspace(first, last, int(np.ceil(last-first))+1)
                for left, right in zip(cuts[:-1], cuts[1:]):
                    keep = mask & (point.t >= left) & (point.t < right)
                    if np.count_nonzero(keep) < 3:
                        continue
                    value, covariance = deblend(h[keep], y[keep])
                    times.append(np.average(point.t[keep], weights=field.root_information[keep]**2))
                    histories.append(value[:count]); covariances.append(covariance[:count, :count])
                    identities.append((c, w)); folds.append(training)
        print(json.dumps(dict(channel=c, bins=len(histories))), flush=True)
    time = np.asarray(times); identity = np.asarray(identities); training = np.asarray(folds)
    value = np.asarray(histories); covariance = np.asarray(covariances)
    information = 1 / np.diagonal(covariance, axis1=-2, axis2=-1).real
    np.savez_compressed(args.output/'histories.npz', time_s=time, identity=identity,
                        training=training, value=value, covariance=covariance,
                        points_body_m=points, edges_s=edges)
    np.savez_compressed(args.output/'row-histories.npz', **rows)
    from matplotlib import pyplot as plt
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    for w in range(windows):
        fig, axes = plt.subplots(count, 2, figsize=(14, 3.4*count), layout='constrained', sharex=True)
        for c, name in enumerate(names):
            for p in range(count):
                for fold, marker, alpha in [(True, '.', .3), (False, 'o', .8)]:
                    keep = (identity[:, 0] == c) & (identity[:, 1] == w) & (training == fold)
                    snr = abs(value[keep, p])**2 * information[keep, p]
                    high = snr > 4
                    x = time[keep]
                    axes[p, 0].scatter(x[high], np.degrees(np.angle(value[keep, p][high])),
                                       s=np.clip(snr[high], 5, 60), marker=marker, alpha=alpha,
                                       label=name if not fold else None, color=f'C{c}')
                    if not fold:
                        axes[p, 1].plot(x, 10*np.log10(np.maximum(snr, 1e-12)), '.-', lw=.7,
                                         label=name, color=f'C{c}')
            
        for p in range(count):
            axes[p, 0].set(ylabel=f'Point {p+1}: residual phase (deg)', ylim=[-180, 180])
            axes[p, 1].set(ylabel='Conditional one-second power (dB)')
        axes[0, 0].legend(ncol=5, fontsize=8)
        axes[-1, 0].set_xlabel('Capture time (s)'); axes[-1, 1].set_xlabel('Capture time (s)')
        fig.suptitle(f'Frozen physical motion: individual scattering histories · {edges[w]:.2f}–{edges[w+1]:.2f} s\n'
                     'No phase smoothing or correction; pale dots training, circles excluded; phase shown only above 6 dB conditional power')
        fig.savefig(args.output/f'window-{w:02d}.png', dpi=140); plt.close(fig)
    (args.output/'analysis.json').write_text(json.dumps(dict(complete=True, motion=str(args.motion),
        scope=__doc__, windows=windows, points_per_window=count, strong_branch_nuisance_motion=str(args.strong_motion) if strong else None,
        diagnostic_only=True, no_data_correction=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
