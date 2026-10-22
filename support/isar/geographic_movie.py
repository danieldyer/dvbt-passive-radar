"""North-up ship-following and fixed-geographic coherent BP movies.

All viewport pixels are backprojected under the same moving-rigid-body model.
Only the displayed power is reprojected for the camera. Complex values from
the five channels are combined before power detection. No image patches are
pasted onto a separately formed background; no DC notch or adaptive scale.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from ais import read_csv, utc_seconds
from extract_full_aperture import load
from imaging import corrected_sites, geometry_from_state, motion_state, sites
from motion import C, enu_basis
from scene_dictionary import fingerprint
from spline_motion import SplineMotion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'calibration'))
from terrain import geodetic

BLOCK = 128
FPS = 30
HEIGHT = 20.
SIZES = {'following': (800, 800), 'geographic': (800, 1200)}


class ImageRing:
    """Sequential, lossless complex-float ring without evicting capture pages."""
    def __init__(self, path, slots, pixels):
        self.descriptor = os.open(path, os.O_RDWR|os.O_CREAT|os.O_EXCL, 0o600)
        self.bytes = pixels*8
        os.ftruncate(self.descriptor, slots*self.bytes)
        if sys.platform == 'darwin':
            import fcntl
            # F_NOCACHE is 48 in the macOS SDK's sys/fcntl.h. This affects only
            # this disposable working file, not the capture or OS cache policy.
            fcntl.fcntl(self.descriptor, 48, 1)

    def read(self, slot):
        data = os.pread(self.descriptor, self.bytes, slot*self.bytes)
        if len(data) != self.bytes:
            raise IOError('Incomplete coherent window ring read')
        return np.frombuffer(data, '<c8')

    def write(self, slot, value):
        view = memoryview(np.asarray(value, dtype='<c8')).cast('B')
        position = slot*self.bytes
        while view:
            n = os.pwrite(self.descriptor, view, position)
            if not n:
                raise IOError('Incomplete coherent window ring write')
            position += n; view = view[n:]

    def close(self):
        os.close(self.descriptor)


def frame_transform(centre, body, origin, local, height=HEIGHT):
    """Exact affine orthographic EN projection of the fitted body plane."""
    matrix = local[:, :2].T@body[:, :2]
    shift = (centre-origin+height*body[:, 2])@local[:, :2]
    if abs(np.linalg.det(matrix)) < .5:
        raise ValueError('Body plane is unsuitable for this north-up projection')
    return matrix, shift


def body_corners(bounds, matrix, shift):
    east0, east1, north0, north1 = bounds
    points = np.array([[e, n] for e in (east0, east1) for n in (north0, north1)])
    return np.linalg.solve(matrix, (points-shift).T).T


def project_power(power, along, port, bounds, matrix, shift, size):
    """Display interpolation follows complex combination and power detection."""
    width, height = size
    east0, east1, north0, north1 = bounds
    world = np.array([[(east1-east0)/width, 0.], [0., -(north1-north0)/height]])
    offset = np.array([east0, north1])+world@np.array([.5, .5])
    inverse = np.linalg.inv(matrix)
    step = np.array([along[1]-along[0], port[1]-port[0]])
    a = (inverse@world)/step[:, None]
    b = (inverse@(offset-shift)-np.array([along[0], port[0]]))/step
    # Pillow's affine coordinates refer to pixel centres at half integers.
    affine = np.c_[a, b+.5-a@np.array([.5, .5])]
    source = Image.fromarray(np.asarray(power, dtype='float32'))
    return np.asarray(source.transform(size, Image.Transform.AFFINE, tuple(affine.ravel()),
        resample=Image.Resampling.BILINEAR, fillcolor=float('nan')))


def read_exact(stream, count):
    result = bytearray(count)
    view = memoryview(result)
    while view:
        n = stream.readinto(view)
        if not n:
            raise RuntimeError('Backprojection worker closed its output early')
        view = view[n:]
    return result


def state_at(track, model, curve, tx, rx, t):
    prior = motion_state(track, utc_seconds(model['nominal_start_utc'])+t, model)
    return curve.apply(t, prior, tx[model['reference_channel']], rx)


def channel_geometry(model, state, tx, rx, channel, rf, first_sample, information):
    """Correct propagation rays while preserving the nominal motion reference."""
    actual_tx, actual_rx = corrected_sites(tx, rx, model.get('antenna_fit'))
    geometry, _ = geometry_from_state(state, actual_tx[channel], actual_rx, rf, first_sample, information)
    geometry[:, 9, 0] += model['range_bias_m'][channel]*8e6/C
    gain = complex(*model['channel_gains_real_imag'][channel])
    rotation = (geometry[:, 9, 1].astype(float)+1j*geometry[:, 9, 2])*gain.conjugate()
    geometry[:, 9, 1], geometry[:, 9, 2] = rotation.real, rotation.imag
    geometry[:, 5, 3], geometry[:, 8, 3] = 2*np.pi*rf/C, abs(gain)**2
    return geometry


def encode_lossless(path, size):
    width, height = size
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'gray16le',
        '-video_size', f'{width}x{height}', '-framerate', str(FPS), '-i', '-', '-an',
        '-c:v', 'ffv1', '-level', '3', '-coder', '1', '-context', '1', '-g', '1',
        '-slicecrc', '0', '-threads', '2', '-n', str(path)], stdin=subprocess.PIPE)


def coherent_window_plan(epochs, centres, nominal_period, seconds):
    """Select fixed-size complete apertures, preserving actual channel epochs."""
    if not np.isfinite(seconds) or seconds <= 0 or not np.isfinite(nominal_period) or nominal_period <= 0:
        raise ValueError('Coherent duration and row period must be positive and finite')
    count = len(epochs)//BLOCK
    blocks = max(1, round(seconds/(nominal_period*BLOCK)))
    if count < blocks:
        raise ValueError(f'Insufficient full {seconds:g}-second aperture')
    start = np.arange(count)*BLOCK
    available = (epochs[start[blocks-1:]+BLOCK-1]+epochs[start[:count-blocks+1]])/2
    chosen = np.argmin(abs(available[:, None]-centres), axis=0)+blocks-1
    if np.max(abs(available[chosen-blocks+1]-centres)) > nominal_period*BLOCK:
        raise ValueError('Frame centre lacks complete coherent support')
    return blocks, chosen


def clip(apertures, model, track, tx, rx, times, camera, output, executable, spacing, packet_rows,
         *, coherent_seconds=30.):
    curve = SplineMotion(model)
    states = state_at(track, model, curve, tx, rx, times)
    origin, local = np.asarray(camera['origin_ecef_m']), np.asarray(camera['enu_to_ecef'])
    geographic = camera['geographic_bounds_m']
    centres = (states['position_ecef_m']-origin)@local
    transforms, followers, corners = [], [], []
    half_east, half_north = np.asarray(camera['following_extent_m'])/2
    for i in range(len(times)):
        matrix, shift = frame_transform(states['position_ecef_m'][i], states['body_to_ecef'][i], origin, local)
        following = [centres[i, 0]-half_east, centres[i, 0]+half_east,
                     centres[i, 1]-half_north, centres[i, 1]+half_north]
        transforms.append((matrix, shift)); followers.append(following)
        corners.extend([body_corners(geographic, matrix, shift), body_corners(following, matrix, shift)])
    corners = np.concatenate(corners)
    minimum = np.floor((corners.min(axis=0)-4*spacing)/spacing)*spacing
    maximum = np.ceil((corners.max(axis=0)+4*spacing)/spacing)*spacing
    along, port = [np.arange(a, b+spacing*.1, spacing) for a, b in zip(minimum, maximum)]
    xx, yy = np.meshgrid(along, port)
    pixels = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, HEIGHT), np.zeros(xx.size)].astype('<f4')
    if len(pixels) > 4_000_000:
        raise ValueError(f'Clip needs {len(pixels)} pixels; shorten the clip to bound memory')
    output.mkdir(parents=True, exist_ok=True)
    pixel_path = output/'pixels.bin'
    with pixel_path.open('wb') as stream:
        stream.write(struct.pack('<I', len(pixels))); pixels.tofile(stream)
    # Merge actual row epochs; inter-channel offsets remain intact. Fixed blocks
    # of the merged rows provide >30 distinct coherent images per second.
    channels, indices, epochs, geometry = [], [], [], []
    first = max(model['interval_s'][0], times[0]-coherent_seconds/2-.025)
    last = min(model['interval_s'][1], times[-1]+coherent_seconds/2+.025)
    for c, aperture in enumerate(apertures):
        keep = np.flatnonzero((aperture['time_s'] >= first)&(aperture['time_s'] <= last))
        t = aperture['time_s'][keep]
        if not len(t):
            raise ValueError('No original data in requested movie clip')
        meta = json.loads(aperture['metadata_json'])
        rf = meta['channel_configuration']['rf_hz']
        state = state_at(track, model, curve, tx, rx, t)
        geom = channel_geometry(model, state, tx, rx, c, rf, meta['first_range_sample'],
            aperture['information'][keep]*aperture['valid'][keep])
        channels.append(np.full(len(keep), c, dtype='u1')); indices.append(keep)
        epochs.append(t); geometry.append(geom)
    order = np.argsort(np.concatenate(epochs), kind='stable')
    channel = np.concatenate(channels)[order]
    row = np.concatenate(indices)[order]
    t = np.concatenate(epochs)[order]
    geom = np.concatenate(geometry)[order]
    del channels, indices, epochs, geometry, order
    count = len(t)//BLOCK
    start = np.arange(count)*BLOCK
    period = float(np.mean([np.median(np.diff(a['time_s'])) for a in apertures]))/5
    window_blocks, chosen = coherent_window_plan(t, times, period, coherent_seconds)
    # The ring is a bounded disk working file, not a scientific archive. Only
    # one complex image and the current window sum need resident memory.
    ring_path = output/'window-ring.dat'
    # A short preview only needs old contributions that will actually leave its
    # window. Long clips reuse the bounded circular file as usual.
    needed_old = max(0, int(chosen[-1])-window_blocks+1)
    ring = ImageRing(ring_path, min(window_blocks, max(1, needed_old)), len(pixels))
    weights = np.zeros(window_blocks)
    total, information = np.zeros(len(pixels), dtype=complex), 0.
    worker = subprocess.Popen([str(executable), '--stream-coherent', str(pixel_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(output/'gpu.log').open('w'))
    videos = {name: encode_lossless(output/f'{name}.mkv', size) for name, size in SIZES.items()}
    histogram = np.zeros(65536, dtype=np.uint64)
    frame_index, processed, peak = 0, 0, -np.inf
    started = time.perf_counter()
    actual = []
    ranges = apertures[0]['moments'].shape[2]
    try:
        for offset in range(0, count*BLOCK, packet_rows):
            end = min(count*BLOCK, offset+packet_rows)
            n = end-offset
            samples = np.empty((2, n, ranges), dtype='<c8')
            for c, aperture in enumerate(apertures):
                selected = np.flatnonzero(channel[offset:end] == c)
                samples[:, selected] = aperture['moments'][:, row[offset:end][selected]]
            worker.stdin.write(struct.pack('<II', n, ranges))
            worker.stdin.write(geom[offset:end].tobytes())
            worker.stdin.write(samples.tobytes()); worker.stdin.flush()
            blocks = struct.unpack('<I', read_exact(worker.stdout, 4))[0]
            for b in range(blocks):
                i = processed+b
                weight = struct.unpack('<d', read_exact(worker.stdout, 8))[0]
                value = np.frombuffer(read_exact(worker.stdout, len(pixels)*8), '<c8')
                if i > chosen[-1]:
                    continue
                at = i%window_blocks
                if i >= window_blocks:
                    total -= ring.read(at); information -= weights[at]
                total += value; information += weight
                if i < needed_old:
                    ring.write(at, value)
                weights[at] = weight
                while frame_index < len(times) and chosen[frame_index] == i:
                    power = (8192*abs(total)**2/information).reshape(xx.shape)
                    matrix, shift = transforms[frame_index]
                    for name, bounds in (('following', followers[frame_index]), ('geographic', geographic)):
                        displayed = project_power(power, along, port, bounds, matrix, shift, SIZES[name])
                        if not np.isfinite(displayed).all() or np.any(displayed < 0):
                            raise ValueError('Displayed viewport leaves its backprojected image')
                        db = 10*np.log10(np.maximum(displayed, 1e-30))
                        peak = max(peak, float(db.max()))
                        encoded = np.rint(np.clip((db+64)*256, 0, 65535)).astype('<u2')
                        histogram += np.bincount(encoded.ravel(), minlength=65536).astype('uint64')
                        videos[name].stdin.write(encoded.tobytes())
                    actual.append([float(t[start[i-window_blocks+1]]), float(t[start[i]+BLOCK-1])])
                    frame_index += 1
            processed += blocks
            if processed%128 < blocks:
                print(json.dumps({'clip': output.name, 'frames': frame_index, 'total_frames': len(times),
                    'pixels': len(pixels), 'elapsed_s': time.perf_counter()-started}), flush=True)
            if frame_index == len(times):
                break
        worker.stdin.close()
        if worker.wait() != 0:
            raise RuntimeError('Vulkan backprojection worker failed')
        for encoder in videos.values():
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Lossless display cache encoder failed')
        if frame_index != len(times):
            raise ValueError('Movie interval was not completely formed')
    except BaseException:
        worker.kill(); worker.wait()
        for encoder in videos.values():
            encoder.kill(); encoder.wait()
        raise
    ring.close()
    ring_path.unlink(); pixel_path.unlink()
    np.save(output/'histogram.npy', histogram)
    result = {'frame_centres_s': times.tolist(), 'actual_apertures_s': actual, 'fps': FPS,
        'coherent_seconds': coherent_seconds,
        'spacing_m': spacing, 'body_grid': [along[[0, -1]].tolist(), port[[0, -1]].tolist()],
        'pixels': len(pixels), 'maximum_display_power_db': peak, 'seconds': time.perf_counter()-started,
        'complete': True, 'scope': __doc__}
    (output/'clip.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('pixels', 'maximum_display_power_db', 'seconds', 'complete')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('aperture', 'ais', 'geometry-directory', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    p.add_argument('--spacing-m', type=float, default=2.)
    p.add_argument('--clip-seconds', type=float, default=30.)
    p.add_argument('--coherent-seconds', type=float, default=30.)
    p.add_argument('--packet-rows', type=int, default=512)
    p.add_argument('--clips', type=int, nargs='+')
    p.add_argument('--centres', type=float, nargs=2, help='Explicit short preview; otherwise all complete capture apertures')
    args = p.parse_args()
    if (args.packet_rows%BLOCK or not 0 < args.packet_rows <= 4096 or args.spacing_m <= 0
            or not np.isfinite(args.coherent_seconds) or args.coherent_seconds <= 0):
        raise ValueError('Invalid bounded GPU packet or image spacing')
    model = json.loads(args.motion.read_text())
    apertures = [load(args.aperture/f'channel-{c}') for c in range(5)]
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    first = math.ceil((model['interval_s'][0]+args.coherent_seconds/2+.05)*FPS)/FPS
    last = math.floor((model['interval_s'][1]-args.coherent_seconds/2-.05)*FPS)/FPS
    if args.centres:
        first, last = args.centres
    times = first+np.arange(round((last-first)*FPS)+1)/FPS
    sampled = state_at(track, model, SplineMotion(model), tx, rx, np.linspace(*model['interval_s'], 1201))
    origin = sampled['position_ecef_m'][0]
    latitude, longitude, _ = geodetic(origin)
    local = enu_basis(*np.radians([latitude, longitude]))
    enu = (sampled['position_ecef_m']-origin)@local
    east0, north0 = enu[:, :2].min(axis=0)-400
    east1, north1 = enu[:, :2].max(axis=0)+400
    aspect = SIZES['geographic'][0]/SIZES['geographic'][1]
    width = max(east1-east0, (north1-north0)*aspect)
    height = max(north1-north0, width/aspect)
    east_mid, north_mid = (east0+east1)/2, (north0+north1)/2
    east0, east1, north0, north1 = east_mid-width/2, east_mid+width/2, north_mid-height/2, north_mid+height/2
    camera = {'origin_ecef_m': origin.tolist(), 'enu_to_ecef': local.tolist(),
        'origin_latitude_longitude_deg': [float(latitude), float(longitude)],
        'geographic_bounds_m': [east0, east1, north0, north1],
        'following_extent_m': [800., 800.], 'fixed_north_up': True}
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'motion': str(args.motion), 'motion_sha256': fingerprint(args.motion),
        'operator_sha256': {str(path): fingerprint(path) for path in [Path(__file__),
            Path(__file__).with_name('backprojection.comp'), Path(__file__).with_name('interpolate_profiles.comp')]},
        'camera': camera, 'frame_centres_s': times.tolist(), 'sizes': SIZES, 'fps': FPS,
        'scientific_spacing_m': args.spacing_m, 'clip_seconds': args.clip_seconds,
        'coherent_seconds': args.coherent_seconds,
        'phase_calibration_validated': False,
        'antenna_fit_applied_to_propagation': model.get('antenna_fit'),
        'nominal_motion_reference_preserved': True,
        'motion_scope': model.get('scope'), 'no_dc_notch': True,
        'display_cache': 'Lossless FFV1 of displayed power in dB; uint16 step 1/256 dB, zero code -64 dB. Fixed final scale selected over the complete rendered sequence.'}
    manifest = args.output/'movie.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Movie inputs changed; choose a new output directory')
    manifest.write_text(json.dumps(identity, indent=2)+'\n')
    frames = round(args.clip_seconds*FPS)
    for i, start in enumerate(range(0, len(times), frames)):
        if args.clips is not None and i not in args.clips:
            continue
        directory = args.output/f'clip-{i:03d}'
        if (directory/'clip.json').exists():
            continue
        clip(apertures, model, track, tx, rx, times[start:start+frames], camera, directory,
            args.executable, args.spacing_m, args.packet_rows, coherent_seconds=args.coherent_seconds)


if __name__ == '__main__':
    main()
