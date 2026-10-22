"""Independent Vulkan round trip with nonzero channel envelope and phase offsets."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from numpy.polynomial.legendre import leggauss

from combine_images import combine
from imaging import corrected_sites, geometry_from_state, read_image, write_input
from motion import C, enu_basis
from test_rigid_motion import fixture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    point = np.array([47.25, -13.7, 31.])
    amplitude = .4+.7j
    biases = np.array([13., -9., 4.5, 0., -1.25])
    gains = np.array([.8*np.exp(.7j), .6*np.exp(-2j), 1.2*np.exp(1.7j), 1., .75*np.exp(-1.1j)])
    rotation = enu_basis(*np.radians([-37.87, 144.93])).T
    tx_offsets = np.array([[15., -21., 8.], [14., -19., 12.], [12., -18., 7.], [18., -15., 9.], [-17., 21., -6.]])
    rx_offset = np.array([7., -9., 3.])
    nominal_tx = np.array([fixture(c).data['transmitter_ecef_m'] for c in range(5)])
    nominal_rx = fixture().data['receiver_ecef_m']
    fit = {'ecef_to_enu': rotation, 'transmitter_offsets_enu_m': tx_offsets, 'receiver_offset_enu_m': rx_offset}
    fitted_tx, fitted_rx = corrected_sites(nominal_tx, nominal_rx, fit)
    estimates, information, analytic, results = [], [], [], []
    u, weight = leggauss(32); u, weight = u/2, weight/2
    with tempfile.TemporaryDirectory(prefix='isar-registered-') as temporary:
        root = Path(temporary)
        for channel in range(5):
            f = fixture(channel); d = f.data
            state = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
                     'body_to_ecef': d['body_to_ecef'],
                     'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
            position = state['position_ecef_m']+state['body_to_ecef']@point
            velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
            # Independent synthesis uses explicit Cartesian displacement; the
            # renderer consumes the metadata's ENU offsets through its helper.
            actual_tx = nominal_tx[channel]+rotation.T@tx_offsets[channel]
            actual_rx = nominal_rx+rotation.T@rx_offset
            vectors = [position-actual_tx, position-actual_rx]
            lengths = [np.sqrt(np.sum(v*v, axis=1)) for v in vectors]
            path = sum(lengths)-np.linalg.norm(actual_tx-actual_rx)
            rate = sum(np.sum(v*velocity, axis=1)/r for v, r in zip(vectors, lengths))
            first = int(np.floor(path.min()*8e6/C))-80
            coordinates = (path+biases[channel])*8e6/C-first
            pulse = np.sinc(.76*(np.arange(192)[None, :]-coordinates[:, None]))
            omega = -f.k*.001024*rate
            tone = np.exp(1j*omega[:, None]*u)
            zero, one = tone@weight, tone@(u*weight)
            raw = gains[channel]*amplitude*pulse*np.exp(-1j*f.k*path[:, None])
            moments = np.stack([zero[:, None]*raw, one[:, None]*raw])
            packed, _ = geometry_from_state(state, fitted_tx[channel], fitted_rx,
                f.metadata['rf_hz'], first, d['row_information'])
            packed[:, 9, 0] += biases[channel]*8e6/C
            source, target = root/f'{channel}.bin', root/f'{channel}-image.bin'
            write_input(source, packed, moments, np.array([[*point, 0.]]), f.metadata['rf_hz'])
            subprocess.run([str(args.executable), str(source), str(target)], check=True)
            value = read_image(target, 1)[0]
            estimate = (value[0]+1j*value[1])/value[2]
            expected = gains[channel]*amplitude*np.average(zero-1j*omega*one, weights=d['row_information'])
            error = abs(estimate-expected)
            phase_error = abs(np.degrees(np.angle(estimate/expected)))
            if error > 1e-4 or phase_error > .02:
                raise ValueError(f'Channel {channel}: registered backprojection fails independent forward model')
            estimates.append([estimate]); information.append([value[2]]); analytic.append([expected])
            results.append({'channel': channel, 'complex_error': float(error), 'phase_error_degrees': float(phase_error)})
    combined = combine(np.array(estimates), np.array(information), gains)[0]
    expected = combine(np.array(analytic), np.array(information), gains)[0]
    result = {'scope': __doc__+' Includes nonzero physical transmitter and receiver offsets.', 'channels': results,
        'combined_complex_error': float(abs(combined[0]-expected[0])),
        'combined_phase_error_degrees': float(abs(np.degrees(np.angle(combined[0]/expected[0])))),
        'passed': True}
    if result['combined_complex_error'] > 1e-4 or result['combined_phase_error_degrees'] > .02:
        raise ValueError('Combined registered backprojection fails independent reference')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
