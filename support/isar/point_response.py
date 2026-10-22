"""Evaluate the actual imaging aperture with a known coherent point target.

Uses each channel's archived spectral filter weights, original geometry/epochs
and information. This tests conditional image resolution, not AIS accuracy or
the assumption that real ship scattering is common across all five RF bands.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from imaging import read_image, write_input
from motion import C
from multipath import spectrum_at, index_archive
from symbol_moments import moments as symbol_moments


def synthesise(geometry, ranges, point, k, spectrum):
    g = geometry.astype(float)
    d = np.einsum('nai,a->ni', g[:, 2:5, :3], point)
    v = g[:, 5, :3]+np.einsum('nai,a->ni', g[:, 6:9, :3], point)
    delta, rate = np.zeros(len(g)), np.zeros(len(g))
    for antenna in (0, 1):
        a = g[:, antenna, :3]
        length = np.linalg.norm(a+d, axis=1)
        delta += (2*np.sum(a*d, axis=1)+np.sum(d*d, axis=1))/(length+np.linalg.norm(a, axis=1))
        rate += np.sum((a+d)*v, axis=1)/length
    coordinate = g[:, 9, 0]+delta*8e6/C
    phase = (g[:, 9, 1]-1j*g[:, 9, 2])*np.exp(-1j*k*delta)
    omega = -k*rate*.001024
    m0, m1, _, _ = symbol_moments(omega)
    oversample = 128
    n = 8192*oversample
    padded = np.zeros(n, dtype=complex)
    padded[np.arange(-3408, 3409)%n] = spectrum
    response = np.fft.ifft(padded)*n
    result = np.empty((2, len(g), ranges), dtype='<c8')
    for first in range(0, len(g), 1024):
        section = slice(first, first+1024)
        x = (np.arange(ranges)[None, :]-coordinate[section, None])*oversample
        base = np.floor(x).astype(int)
        pulse = response[base%n]*(1-(x-base))+response[(base+1)%n]*(x-base)
        signal = phase[section, None]*pulse
        result[0, section] = m0[section, None]*signal
        result[1, section] = m1[section, None]*signal
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', type=Path)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    args = parser.parse_args()
    description = json.loads((args.images/'image-description.json').read_text())
    archive = index_archive(args.capture, args.index)
    args.output.mkdir(parents=True, exist_ok=True)
    results, amplitudes, information = [], [], []
    for channel in range(5):
        with np.load(args.images/f'channel-{channel}-image.npz') as image:
            along, port, time = image['along_m'], image['port_m'], image['time_s']
        with (args.images/f'channel-{channel}.bin').open('rb') as source:
            magic, rows, ranges, count, k = struct.unpack('<8sIIIf', source.read(24))
            if magic != b'ISARBP1\0' or rows != len(time):
                raise ValueError('Mismatched real aperture input')
            geom = np.fromfile(source, '<f4', count=rows*40).reshape(rows, 10, 4)
            pixels = np.fromfile(source, '<f4', count=count*4).reshape(count, 4)
        point = np.array([0., 0., description['height_offset_m']])
        spectrum = spectrum_at(args.capture, archive, channel, time.mean())
        moments = synthesise(geom, ranges, point, k, spectrum)
        source, target = args.output/f'channel-{channel}.bin', args.output/f'channel-{channel}-image.bin'
        write_input(source, geom, moments, pixels, k*C/(2*np.pi))
        subprocess.run([str(args.executable), str(source), str(target)], check=True)
        value = read_image(target, count)
        shape = (len(port), len(along))
        amplitude = ((value[:, 0]+1j*value[:, 1])/value[:, 2]).reshape(shape)
        weight = value[:, 2].reshape(shape)
        np.savez(args.output/f'channel-{channel}-image.npz', along_m=along, port_m=port,
                 complex_image=amplitude, information=weight, time_s=time)
        amplitudes.append(amplitude);information.append(weight)
        peak = np.unravel_index(np.argmax(abs(amplitude)), shape)
        results.append({'channel': channel, 'peak_forward_m': float(along[peak[1]]),
                        'peak_port_m': float(port[peak[0]]), 'peak_amplitude': float(abs(amplitude[peak])),
                        'peak_phase_deg': float(np.angle(amplitude[peak])*180/np.pi)})
        print(json.dumps(results[-1]), flush=True)
    total = np.sum(information, axis=0)
    coherent = np.sum(np.array(amplitudes)*information, axis=0)/total
    np.savez(args.output/'coherent-point.npz', along_m=along, port_m=port,
             complex_image=coherent, information=total)
    (args.output/'point-test.json').write_text(json.dumps({'point_body_m': point.tolist(),
        'channels': results, 'maximum_coherent_amplitude': float(abs(coherent).max()),
        'spectrum': 'Actual mid-aperture archived |H|^2/noise weights; mean symbol energy',
        'scope': 'Conditional point response with known motion and phase centres; not a real-data calibration'}, indent=2))


if __name__ == '__main__':
    main()
