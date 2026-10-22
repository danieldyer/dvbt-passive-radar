"""Form an explicit AIS-prior moving-body image with exact GPU backprojection.

This diagnostic does not infer antenna calibration or true ship heading. The
course-derived attitude is selected explicitly and recorded as a hypothesis.
Original complex profiles, acquisition epochs and both symbol moments are used.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from ais import read_csv, utc_seconds
from motion import C, body_basis, enu_basis, excess_path
from spline_motion import SplineMotion


def sites(directory):
    prior = json.loads((directory/'geometry-prior.json').read_text())
    main = json.loads((directory/'sites-main.json').read_text())
    receiver = np.array(prior['receiver_ecef_m'])
    rotation = np.array(prior['ecef_to_enu'])
    centre = np.array(main['site_positions_enu_m'])[main['channel_sites']]
    transmitters = receiver+(centre+np.array(main['channel_offsets_enu_m']))@rotation
    return transmitters, receiver, main


def course_basis(state):
    local = np.einsum('nji,nj->ni', enu_basis(state['latitude'], state['longitude']),
                      state['velocity_ecef_mps'])
    if np.any(np.linalg.norm(local[:, :2], axis=1) < .1):
        raise ValueError('Course attitude is undefined at very low speed')
    heading_hypothesis = np.arctan2(local[:, 0], local[:, 1])
    return body_basis(state['latitude'], state['longitude'], heading_hypothesis)


def attitude_basis(track, state, mode):
    if mode == 'course-prior':
        return course_basis(state)
    if mode != 'reported-course-prior':
        raise ValueError('Unknown explicit attitude hypothesis')
    known = np.isfinite(track.course)
    if known.sum() < 2:
        raise ValueError('Reported-course hypothesis requires reported COG')
    curve = PchipInterpolator(track.seconds[known]-track.epoch, np.unwrap(track.course[known]), extrapolate=False)
    heading = curve(state['seconds']-track.epoch)
    if not np.isfinite(heading).all():
        raise ValueError('Reported-course attitude extrapolation is not permitted')
    return body_basis(state['latitude'], state['longitude'], heading)


def physical_state(track, seconds, attitude, *, max_gap_s=180, reference_heading_degrees=None):
    seconds = np.atleast_1d(seconds).astype(float)
    def orientation(state):
        if reference_heading_degrees is None:
            return attitude_basis(track, state, attitude)
        return body_basis(state['latitude'], state['longitude'],
                          np.full(len(state['latitude']), np.radians(reference_heading_degrees)))
    state = track.sample(seconds, max_gap_s=max_gap_s)
    basis = orientation(state)
    step = .01
    before = track.sample(seconds-step, max_gap_s=max_gap_s)
    after = track.sample(seconds+step, max_gap_s=max_gap_s)
    derivative = (orientation(after)-orientation(before))/(2*step)
    return {**state, 'body_to_ecef': basis, 'frame_velocity': derivative}


def motion_state(track, seconds, model):
    """Explicit coarse reference for a physical radar motion correction.

    A linear reference needs no AIS samples after its single anchor. Position,
    velocity and heading evolution are then supplied by the fitted motion, not
    inherited from interpolation between sparse AIS reports.
    """
    reference = model.get('linear_reference')
    if reference is None:
        return physical_state(track, seconds, model['attitude_prior'],
            max_gap_s=model.get('max_ais_gap_s', 180.),
            reference_heading_degrees=model.get('fixed_reference_heading_deg'))
    time = np.atleast_1d(seconds)-reference['utc_seconds']
    position = np.asarray(reference['position_ecef_m'])
    velocity = np.asarray(reference['velocity_ecef_mps'])
    frame = np.asarray(reference['body_to_ecef'])
    return {'position_ecef_m': position+time[:, None]*velocity,
            'velocity_ecef_mps': np.broadcast_to(velocity, (len(time), 3)),
            'body_to_ecef': np.broadcast_to(frame, (len(time), 3, 3)),
            'frame_velocity': np.zeros((len(time), 3, 3))}


def geometry_from_state(state, transmitter, receiver, rf, first_sample, information):
    """Pack one physical state into the centre-relative GPU representation."""
    centre = state['position_ecef_m']
    basis, derivative = state['body_to_ecef'], state['frame_velocity']
    path, gradient = excess_path(centre, transmitter, receiver)
    k = 2*np.pi*rf/C
    values = np.zeros((len(centre), 10, 4), dtype='<f4')
    for i, antenna in enumerate((transmitter, receiver)):
        values[:, i, :3] = centre-antenna
        values[:, i, 3] = np.linalg.norm(centre-antenna, axis=1)
    values[:, 2:5, :3] = basis.transpose(0, 2, 1)
    values[:, 5, :3] = state['velocity_ecef_mps']
    values[:, 6:9, :3] = derivative.transpose(0, 2, 1)
    values[:, 9, 0] = path*8e6/C-first_sample
    values[:, 9, 1] = np.cos(k*path)
    values[:, 9, 2] = np.sin(k*path)
    values[:, 9, 3] = information
    return values, {'path_m': path,
                    'rate_mps': np.einsum('ni,ni->n', gradient, state['velocity_ecef_mps']),
                    'basis_ecef': basis, 'centre_ecef_m': centre}


def corrected_sites(transmitters, receiver, fit):
    """Apply fitted ENU offsets to rays without changing the motion reference."""
    if not fit:
        return transmitters, receiver
    rotation = np.asarray(fit['ecef_to_enu'])
    return (transmitters+np.asarray(fit['transmitter_offsets_enu_m'])@rotation,
            receiver+np.asarray(fit['receiver_offset_enu_m'])@rotation)


def geometry(track, seconds, transmitter, receiver, rf, first_sample, information, *, attitude='course-prior',
             motion=None, motion_epoch=None, reference_transmitter=None, reference_receiver=None):
    """Host-double geometry; the GPU receives centre-relative vectors only."""
    state = physical_state(track, seconds, attitude) if motion is None else motion_state(track, seconds, motion.description)
    if motion is not None:
        if motion_epoch is None or reference_transmitter is None:
            raise ValueError('Physical correction needs its epoch and reference transmitter')
        state = motion.apply(seconds-motion_epoch, state, reference_transmitter,
                             receiver if reference_receiver is None else reference_receiver)
    return geometry_from_state(state, transmitter, receiver, rf, first_sample, information)


def write_input(path, geom, moments, pixels, rf):
    if geom.shape != (moments.shape[1], 10, 4) or moments.shape[0] != 2 or pixels.shape[1] != 4:
        raise ValueError('Inconsistent operator arrays')
    with path.open('xb') as output:
        output.write(struct.pack('<8sIIIf', b'ISARBP1\0', moments.shape[1], moments.shape[2],
                                 len(pixels), 2*np.pi*rf/C))
        np.asarray(geom, dtype='<f4').tofile(output)
        np.asarray(pixels, dtype='<f4').tofile(output)
        np.asarray(moments, dtype='<c8').tofile(output)


def read_image(path, pixels):
    with path.open('rb') as source:
        magic, count = struct.unpack('<8sI', source.read(12))
        data = np.fromfile(source, '<f4')
    if magic != b'ISARIM1\0' or count != pixels or data.size != 4*pixels:
        raise ValueError('Invalid backprojection result')
    return data.reshape(-1, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aperture', type=Path, required=True)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--geometry-directory', type=Path, required=True)
    parser.add_argument('--nominal-start-utc', required=True)
    parser.add_argument('--reference-height-ellipsoid-m', type=float, required=True)
    parser.add_argument('--attitude', choices=['course-prior', 'reported-course-prior'], required=True)
    parser.add_argument('--motion', type=Path, help='Fitted physical spline correction JSON; changes range, RF phase and symbol Doppler together')
    parser.add_argument('--interval', type=float, nargs=2, required=True)
    parser.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--along', type=float, nargs=2, default=[-350, 350])
    parser.add_argument('--port', type=float, nargs=2, default=[-150, 150])
    parser.add_argument('--height-offset-m', type=float, default=20)
    parser.add_argument('--spacing-m', type=float, default=4)
    parser.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.spacing_m <= 100 or not args.interval[0] < args.interval[1]:
        raise ValueError('Invalid image spacing or aperture')
    track = read_csv(args.ais, reference_height_ellipsoid_m=args.reference_height_ellipsoid_m,
                     mmsi=563052200)
    epoch = utc_seconds(args.nominal_start_utc)
    tx, rx, configuration = sites(args.geometry_directory)
    reference_tx, reference_rx = tx[3].copy(), rx.copy()
    motion = SplineMotion(json.loads(args.motion.read_text())) if args.motion else None
    if motion is not None:
        model = motion.description
        if model.get('attitude_prior') != args.attitude or model.get('nominal_start_utc') != args.nominal_start_utc:
            raise ValueError('Fitted motion attitude or epoch differs from the image prior')
        if model.get('reference_height_ellipsoid_m') != args.reference_height_ellipsoid_m:
            raise ValueError('Fitted motion height differs from the image prior')
        tx, rx = corrected_sites(tx, rx, model.get('antenna_fit'))
    along = np.arange(*[args.along[0], args.along[1]+args.spacing_m*.1, args.spacing_m])
    port = np.arange(*[args.port[0], args.port[1]+args.spacing_m*.1, args.spacing_m])
    x, y = np.meshgrid(along, port)
    pixels = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, args.height_offset_m), np.zeros(x.size)])
    args.output.mkdir(parents=True, exist_ok=True)
    description = {'interval_s': args.interval, 'nominal_start_utc': args.nominal_start_utc,
                   'time_origin_calibrated': False, 'phase_centres': configuration,
                   'phase_centres_calibrated': False, 'attitude': args.attitude,
                   'transmitter_ecef_m': tx.tolist(), 'receiver_ecef_m': rx.tolist(),
                   'attitude_measured': False, 'reference_height_ellipsoid_m': args.reference_height_ellipsoid_m,
                   'height_offset_m': args.height_offset_m,
                   'operator': 'Exact bistatic moving-body BP; 64-tap range interpolation; original epochs and C0/C1; rectangular aperture; no DC projection',
                   'phase_correction': 'Physical spline translation and body yaw' if motion else 'AIS trajectory only; no radar-derived autofocus applied',
                   'motion_model': motion.description if motion else None,
                   'channels': []}
    for channel in args.channels:
        tic = time.perf_counter()
        with np.load(args.aperture/f'channel-{channel}.npz', allow_pickle=False) as aperture:
            meta = json.loads(str(aperture['metadata_json']))
            t = aperture['time_s']
            keep = (t >= args.interval[0])&(t < args.interval[1])
            if not keep.any():
                raise ValueError('No rows in requested image aperture')
            t = t[keep]
            moments = aperture['moments'][:, keep]
            information = aperture['information'][keep]*aperture['valid'][keep]
            rf = meta['channel_configuration']['rf_hz']
        geom, prior = geometry(track, epoch+t, tx[channel], rx, rf, meta['first_range_sample'], information,
                               attitude=args.attitude, motion=motion, motion_epoch=epoch,
                               reference_transmitter=reference_tx, reference_receiver=reference_rx)
        source, target = args.output/f'channel-{channel}.bin', args.output/f'channel-{channel}-image.bin'
        write_input(source, geom, moments, pixels, rf)
        subprocess.run([str(args.executable), str(source), str(target)], check=True)
        value = read_image(target, len(pixels))
        if np.any(value[:, 2] <= 0) or np.ptp(value[:, 2])/np.mean(value[:, 2]) > 1e-5:
            raise ValueError('Image exceeds range crop support; enlarge the original profile crop')
        amplitude = (value[:, 0]+1j*value[:, 1])/value[:, 2]
        np.savez(args.output/f'channel-{channel}-image.npz', along_m=along, port_m=port,
                 complex_image=amplitude.reshape(x.shape), information=value[:, 2].reshape(x.shape),
                 sum_squared_information=value[:, 3].reshape(x.shape),
                 time_s=t, centre_path_m=prior['path_m'], centre_rate_mps=prior['rate_mps'],
                 centre_ecef_m=prior['centre_ecef_m'], basis_ecef=prior['basis_ecef'])
        record = {'channel': channel, 'rf_hz': rf, 'rows': len(t), 'valid_rows': int((information > 0).sum()),
                  'pixels': len(pixels), 'maximum_amplitude': float(abs(amplitude).max()),
                  'seconds': time.perf_counter()-tic}
        description['channels'].append(record)
        (args.output/'image-description.json').write_text(json.dumps(description, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
