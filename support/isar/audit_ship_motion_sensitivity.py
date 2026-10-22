"""Exact conditional phase sensitivity at two hull-scale material points.

Saved best-aperture rays, frames, times and information are fixed. This is a
controlled perturbation study, not a fit or a measurement of real heading,
height or antenna errors. Relative point phases remove any phase common to
both cells. No phase correction or data normalization is adopted.
"""
import argparse
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


def path_change(ray, displacement):
    return np.sum(2*ray*displacement+displacement**2, axis=1)/(
        np.linalg.norm(ray+displacement, axis=1)+np.linalg.norm(ray, axis=1))


def metrics(path, weight, k):
    mean = np.average(path, weights=weight)
    rms = np.sqrt(np.average((path-mean)**2, weights=weight))*k
    fraction = abs(np.sum(weight*np.exp(-1j*k*(path-mean)))/weight.sum())**2
    return dict(phase_rms_degrees=float(np.degrees(rms)),
                coherence_loss_db=float(-10*np.log10(max(fraction, 1e-50))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('build/isar-20261015/heading-five-142-172'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    description = json.loads((a.source/'image-description.json').read_text())
    directory = Path(description['motion_model']['geometry_directory'])
    rotation = np.array(json.loads((directory/'geometry-prior.json').read_text())['ecef_to_enu'])
    points = np.array([[-150., 0., 20.], [150., 0., 20.]])
    rates = [.001, .003, .01, .03, .1]
    records = []
    for c in range(5):
        with np.load(a.source/f'channel-{c}-image.npz') as z: time = z['time_s']
        with (a.source/f'channel-{c}.bin').open('rb') as f:
            magic, rows, _, _, k = struct.unpack('<8sIIIf', f.read(24))
            if magic != b'ISARBP1\0' or rows != len(time): raise ValueError('Original time/geometry mismatch')
            g = np.fromfile(f, '<f4', count=rows*40).reshape(rows, 10, 4).astype(float)
        frame = g[:, 2:5, :3].transpose(0, 2, 1); weight = g[:, 9, 3]
        displacements = np.einsum('tij,pj->tpi', frame, points)
        rays = [g[:, arm, None, :3]+displacements for arm in [0, 1]]
        errors = {}
        for rate in rates:
            angle = np.radians(rate)*(time-157.)
            shifted = np.zeros_like(displacements)
            for j, point in enumerate(points):
                local = np.c_[point[0]*np.cos(angle), point[0]*np.sin(angle), np.full(rows, point[2])]
                shifted[:, j] = np.einsum('tij,tj->ti', frame, local)-displacements[:, j]
            errors[f'heading_rate_{rate:g}_degps'] = sum(np.column_stack([
                path_change(ray[:, j], shifted[:, j]) for j in range(2)]) for ray in rays)
        up = np.einsum('tij,j->ti', frame, [0., 0., 40.])
        errors['scatterer_height_plus_40m'] = sum(np.column_stack([
            path_change(ray[:, j], up) for j in range(2)]) for ray in rays)
        for arm, label, distance in [(0, 'TX', 100.), (1, 'RX', 10.)]:
            for axis, name in enumerate(['E', 'N', 'U']):
                delta = -distance*rotation[axis]
                errors[f'{label}_{distance:g}m_{name}'] = np.column_stack([
                    path_change(rays[arm][:, j], delta) for j in range(2)])
        for label, error in errors.items():
            for duration, first, last in [(30., 142., 172.), (3., 160., 163.)]:
                w = weight*((time >= first) & (time < last))
                record = dict(channel=c, perturbation=label, interval_s=[first, last], duration_s=duration,
                              point_metrics=[metrics(error[:, j], w, k) for j in range(2)],
                              differential_point_phase=metrics(error[:, 1]-error[:, 0], w, k))
                records.append(record)
        print(f'channel {c}: exact phase perturbations evaluated', flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for duration in [3., 30.]:
        values = [max(r['differential_point_phase']['phase_rms_degrees'] for r in records
                      if r['perturbation']==f'heading_rate_{rate:g}_degps' and r['duration_s']==duration) for rate in rates]
        axes[0].loglog(rates, values, 'o-', label=f'{duration:g} s aperture')
    labels = ['scatterer_height_plus_40m', *[f'{arm}_{distance}m_{axis}' for arm, distance in [('TX', '100'), ('RX', '10')] for axis in ['E', 'N', 'U']]]
    for duration in [3., 30.]:
        values = [max(r['differential_point_phase']['phase_rms_degrees'] for r in records
                      if r['perturbation']==label and r['duration_s']==duration) for label in labels]
        axes[1].semilogy(np.arange(len(labels)), values, 'o-', label=f'{duration:g} s aperture')
    axes[0].set(xlabel='Controlled heading-rate error (degrees/s)', ylabel='Worst-band differential phase RMS (degrees)', title='Two points separated by 300 m')
    axes[1].set(xticks=np.arange(len(labels)), xticklabels=['Height +40 m', 'TX E', 'TX N', 'TX U', 'RX E', 'RX N', 'RX U'],
                ylabel='Worst-band differential phase RMS (degrees)', title='TX offsets 100 m; RX offsets 10 m')
    axes[1].tick_params(axis='x', labelrotation=25)
    for ax in axes: ax.legend()
    fig.suptitle('Conditional sensitivity on the retained best-image geometry\nConstant phase removed; comparing the two points also cancels a common phase screen; this is not an estimated error budget')
    fig.savefig(a.output/'motion-versus-site-sensitivity.png', dpi=150); plt.close(fig)
    result = dict(complete=True, scope=__doc__, source=str(a.source), points_body_m=points.tolist(),
                  assumptions='Nominal legacy physical frame held fixed except for each explicit perturbation; no motion/site/scene refit',
                  records=records, production_changes=False)
    (a.output/'motion-sensitivity.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
