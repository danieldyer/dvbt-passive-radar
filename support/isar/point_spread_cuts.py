"""Known-point GPU ambiguity cuts for fixed and rotating physical apertures.

Uses original epochs/information, the archived mid-aperture |H|^2/noise
spectrum, and the same C0/C1 forward and GPU backprojection operators. This
is a conditional mean-symbol point response, not a measured ship image or a
proof that physical cross-band scattering phases are calibrated.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from extract_full_aperture import load
from geographic_movie import state_at
from imaging import geometry_from_state, read_image, sites, write_input
from motion import C, enu_basis
from multipath import spectrum_at
from point_response import synthesise
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def central_width(distance, power):
    centre = int(np.argmin(abs(distance)))
    peak = power[centre]
    below = power <= peak/2
    left, right = np.flatnonzero(below[:centre]), np.flatnonzero(below[centre+1:])+centre+1
    if not len(left) or not len(right):
        return None
    lo, hi = left[-1], right[0]
    a = np.interp(peak/2, power[lo:lo+2], distance[lo:lo+2])
    b = np.interp(peak/2, power[hi-1:hi+1][::-1], distance[hi-1:hi+1][::-1])
    return float(b-a)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('aperture', 'capture', 'index', 'information', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    p.add_argument('--extent-m', type=float, default=2000.)
    p.add_argument('--step-m', type=float, default=.25)
    args = p.parse_args()
    if not 0 < args.step_m < args.extent_m:
        raise ValueError('Need a positive, resolved point-spread cut')
    info = json.loads(args.information.read_text())
    archive = json.loads(args.index.read_text())
    identity = archive['identity']
    if (Path(identity[0]).resolve() != args.capture.resolve() or identity[1] != args.capture.stat().st_size
            or identity[2] != args.capture.stat().st_mtime_ns):
        raise ValueError('Capture index identity differs from the supplied recording')
    apertures = [load(args.aperture/f'channel-{c}') for c in range(5)]
    distance = np.arange(-round(args.extent_m/args.step_m), round(args.extent_m/args.step_m)+1)*args.step_m
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'scope': __doc__, 'information_source': str(args.information),
        'information_sha256': fingerprint(args.information), 'records': []}
    for variant, record in enumerate(info['results']):
        path = Path(record['motion'])
        model = json.loads(path.read_text())
        curve = SplineMotion(model)
        tx, rx, _ = sites(Path(model['geometry_directory']))
        centre_time = float(np.mean(info['requested_interval_s']))
        state = state_at(None, model, curve, tx, rx, np.array([centre_time]))
        body = state['body_to_ecef'][0]
        up = body[:, 2]
        east = np.cross([0., 0., 1.], up); east /= np.linalg.norm(east)
        local = np.column_stack([east, np.cross(up, east)])
        axes = np.array(record['without_stationary_nuisance']['known_relative_band_calibration']['principal_axes_east_north'])
        points = np.concatenate([distance[:, None]*(body.T@local@axis) for axis in axes])
        points += [0., 0., 20.]
        pixels = np.c_[points, np.zeros(len(points))].astype('<f4')
        gains = np.array([complex(*v) for v in model['channel_gains_real_imag']])
        values, weights = [], []
        for c, aperture in enumerate(apertures):
            interval = info['requested_interval_s']
            keep = np.flatnonzero((aperture['time_s'] >= interval[0])&(aperture['time_s'] < interval[1]))
            t = aperture['time_s'][keep]
            meta = json.loads(aperture['metadata_json'])
            rf = meta['channel_configuration']['rf_hz']
            g, _ = geometry_from_state(state_at(None, model, curve, tx, rx, t), tx[c], rx, rf,
                meta['first_range_sample'], aperture['information'][keep]*aperture['valid'][keep])
            g[:, 9, 0] += model['range_bias_m'][c]*8e6/C
            spectrum = spectrum_at(args.capture, archive, c, centre_time)
            samples = synthesise(g, aperture['moments'].shape[2], np.array([0., 0., 20.]), 2*np.pi*rf/C, spectrum)
            samples *= gains[c]
            phase = (g[:, 9, 1].astype(float)+1j*g[:, 9, 2])*gains[c].conjugate()
            g[:, 9, 1], g[:, 9, 2] = phase.real, phase.imag
            g[:, 8, 3] = abs(gains[c])**2
            source, target = args.output/f'point-{variant}-{c}.bin', args.output/f'image-{variant}-{c}.bin'
            write_input(source, g, samples, pixels, rf)
            subprocess.run([str(args.executable), str(source), str(target)], check=True)
            data = read_image(target, len(pixels))
            if np.any(data[:, 2] <= 0):
                raise ValueError('A requested point-spread cut left the recorded range support')
            values.append(data[:, 0]+1j*data[:, 1]); weights.append(data[:, 2])
            source.unlink(); target.unlink()
            print(json.dumps({'variant': variant, 'channel': c, 'rows': len(t), 'pixels': len(pixels)}), flush=True)
        values, weights = np.array(values), np.array(weights)
        coherent = abs(values.sum(axis=0))**2/weights.sum(axis=0)
        independent = np.sum(abs(values)**2/weights, axis=0)
        coherent, independent = [v.reshape(2, -1) for v in (coherent, independent)]
        centre = len(distance)//2
        coherent /= coherent[:, centre, None]
        independent /= independent[:, centre, None]
        output = args.output/f'variant-{variant}.npz'
        np.savez(output, distance_m=distance, coherent_power=coherent, independent_power=independent, axes_en=axes)
        report['records'].append({'motion': str(path), 'motion_sha256': fingerprint(path),
            'fixed_ne_frame': bool(model.get('fixed_ne_frame')), 'cuts': str(output),
            'axes_east_north': axes.tolist(),
            'coherent_central_fwhm_m': [central_width(distance, v) for v in coherent],
            'independent_central_fwhm_m': [central_width(distance, v) for v in independent],
            'normalisation': 'Unit power at the known point in this synthetic test only; no normalization of recorded images.'})
        (args.output/'point-spread.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
