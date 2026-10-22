"""Test transfer of one fixed multichannel complex response across time/points.

This explicitly compares a shared spectral signature with independent channel
responses. It does not use complex image correlation: disjoint RF bands have
different complex point-spread functions, making that statistic misleading.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fit_autofocus import extract


def efficiency(template, observation, information):
    numerator = abs(np.sum(information*template.conj()*observation, axis=0))**2
    denominator = np.sum(information*abs(template)**2, axis=0)*np.sum(information*abs(observation)**2, axis=0)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0) or np.any(information < 0):
        raise ValueError('No finite multichannel phase information')
    return numerator/denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    args = parser.parse_args()
    description = json.loads((args.images/'image-description.json').read_text())
    # Fixed positions from the prior independent images, before any phase fit.
    positions = np.array([[-50., -2., 20., 0.], [25., 5., 20., 0.],
                          [90., 8., 20., 0.], [135., 13., 20., 0.]])
    positions[:, 2] = description['height_offset_m']
    args.output.mkdir(parents=True, exist_ok=True)
    average = np.zeros((2, 5, len(positions)), dtype=complex)
    information = np.zeros((2, 5, len(positions)))
    half = np.zeros_like(average);half_information = np.zeros_like(information)
    for channel in range(5):
        data = extract(args.images, args.output, channel, args.executable,
                       description['height_offset_m'], pixels=positions.copy())
        t, z, w = data['time'], data['value'], data['information']
        for destination, info, partition in ((average, information, ((t-description['interval_s'][0])//2)%2),
                                              (half, half_information, t >= np.mean(description['interval_s']))):
            for fold in (0, 1):
                use = partition == fold
                destination[fold, channel] = np.sum(w[use, None]*z[use], axis=0)/w[use].sum()
                info[fold, channel] = w[use].sum()
    response = average[0]
    test = average[1]
    loss = -10*np.log10(efficiency(response, test, information[1]))
    half_loss = -10*np.log10(efficiency(half[0], half[1], half_information[1]))
    transfer = np.array([-10*np.log10(efficiency(response[:, j, None], test, information[1])) for j in range(len(positions))])
    # Remove an unobservable common target phase when showing band drift.
    phase = np.angle(test*response.conj())
    common = np.angle(np.sum(information[1]*test*response.conj(), axis=0))
    phase = np.angle(np.exp(1j*(phase-common[None, :])))
    result = {'body_points_m': positions[:, :3].tolist(),
              'interleaved_2s_held_out_efficiency_loss_db': loss.tolist(),
              'first_half_to_second_half_efficiency_loss_db': half_loss.tolist(),
              'training_anchor_to_held_out_anchor_loss_db': transfer.tolist(),
              'held_out_relative_band_phase_drift_deg': np.degrees(phase).tolist(),
              'scope': 'Conditional coherence at fixed prior-selected points; not a phase-centre calibration. Independent-channel responses supply the upper bound, including their measurement noise.'}
    np.savez(args.output/'coherence-observations.npz', point_body_m=positions[:, :3],
             average=average, information=information, half_average=half, half_information=half_information)
    (args.output/'coherence.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
