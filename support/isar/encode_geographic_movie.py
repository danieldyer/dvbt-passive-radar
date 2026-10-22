"""Colour and encode complete geographic BP caches for smooth QuickTime scrubbing.

The colour scale is fixed over both complete videos. Its bottom sits 1.5 dB
below the dominant displayed background level; its top includes the measured
maximum. This is presentation only, after complex coherent image formation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ais import read_csv, utc_seconds
from ais_video_overlay import AisOverlay, CYAN
from geographic_movie import FPS, SIZES, read_exact
from imaging import sites
from scene_dictionary import fingerprint


def encoder(path, size):
    width, height = size
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '17', '-g', '1', '-bf', '0', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-n', str(path)], stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial-preview', action='store_true')
    p.add_argument('--scale-db', type=float, nargs=2,
        help='Use an explicit fixed display scale for comparison with an existing video')
    p.add_argument('--motion-label', default='conditional radar motion',
        help='Short description of the motion model used for these frames')
    p.add_argument('--ais', type=Path, help='Overlay the interpolated AIS antenna track in cyan')
    args = p.parse_args()
    description = json.loads((args.cache/'movie.json').read_text())
    coherent_seconds = description.get('coherent_seconds', 30.)
    if fingerprint(Path(description['motion'])) != description['motion_sha256']:
        raise ValueError('Cached image motion has changed; cannot place a reliable geographic overlay')
    model = json.loads(Path(description['motion']).read_text())
    clips = sorted(args.cache.glob('clip-*/clip.json'))
    if not clips:
        raise ValueError('No complete geographic image clips')
    records = [json.loads(path.read_text()) for path in clips]
    times = np.concatenate([r['frame_centres_s'] for r in records])
    if (not args.allow_partial_preview and not np.array_equal(times, description['frame_centres_s'])):
        raise ValueError('Complete the full capture render before selecting a global colour scale')
    if len(times)>1 and np.max(abs(np.diff(times)-1/FPS)) > 1e-8:
        raise ValueError('Display clips are not contiguous at the declared frame rate')
    apertures = np.concatenate([r['actual_apertures_s'] for r in records])
    duration = apertures[:, 1]-apertures[:, 0]
    if apertures.shape != (len(times), 2) or np.any(abs(duration-coherent_seconds) > .01):
        raise ValueError('A displayed frame does not have the full declared coherent aperture')
    histogram = sum(np.load(path.parent/'histogram.npy') for path in clips)
    grouped = histogram.reshape(-1, 64).sum(axis=1)
    levels = (np.arange(len(grouped))+.5)/4-64
    mode = levels[np.argmax(np.where((levels > -20)&(levels < 30), grouped, 0))]
    low = float(mode-1.5)
    high = float(np.ceil((max(r['maximum_display_power_db'] for r in records)+.05)*10)/10)
    if args.scale_db is not None:
        low, high = args.scale_db
    if not np.isfinite([low, high]).all() or not low < high:
        raise ValueError('Invalid fixed display scale')
    db = np.arange(65536)/256-64
    lut = (matplotlib.colormaps['magma'](np.clip((db-low)/(high-low), 0, 1))[:, :3]*255).astype('uint8')
    font_path = Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    font = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 17)
    overlay = None
    if args.ais is not None:
        track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
        tx, rx, _ = sites(Path(model['geometry_directory']))
        overlay = AisOverlay(description, model, track, tx, rx, times)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'frames': len(times), 'fps': FPS, 'frame_centres_s': times[[0, -1]].tolist(),
        'requested_cpi_s': coherent_seconds, 'scale_db': [low, high], 'background_mode_db': float(mode),
        'actual_aperture_extrema_s': [float(duration.min()), float(duration.max())],
        'maximum_window_centre_offset_s': float(np.max(abs(apertures.mean(axis=1)-times))),
        'source': str(args.cache), 'scientific_motion': description['motion'],
        'scientific_motion_sha256': description['motion_sha256'],
        'coherent_combination': 'Complex five-channel sum before power detection; constant declared calibration.',
        'motion_status': args.motion_label+'; absolute motion and cross-band calibration remain unvalidated.',
        'scale_selection': 'Explicit comparison scale' if args.scale_db is not None else 'Complete sequence background mode and maximum',
        'partial_preview': bool(args.allow_partial_preview),
        'cache_quantization_db': 1/256, 'encoding': 'H.264 yuv420p, all frames intra, no B-frames, faststart',
        'files': [], 'complete': False}
    if overlay is not None:
        summary['ais_overlay'] = {**overlay.metadata, 'source': str(args.ais), 'source_sha256': fingerprint(args.ais)}
    (args.output/'videos.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name, size in SIZES.items():
        width, height = size
        frame_size = (width+140, height+(230 if overlay is not None else 206))
        path = args.output/f'ship-{name}-north-up-{coherent_seconds:g}s-30fps.mp4'
        out = encoder(path, frame_size)
        base = Image.new('RGB', frame_size, '#111217')
        draw = ImageDraw.Draw(base)
        title = 'Ship-following, fixed north-up axes' if name == 'following' else 'Fixed geographic viewport, north up'
        draw.text((30, 17), title, font=font, fill='white')
        draw.text((30, 48), f'Five-channel coherent BP | {coherent_seconds:g} s sliding aperture | '+args.motion_label, font=small, fill='#dddddd')
        x, y = 55, 112
        draw.text((x+width*.7, y+height+8), 'East →', font=small, fill='white')
        draw.text((x-40, y-24), 'N ↑', font=small, fill='white')
        if name == 'following':
            labels = ['−400 m', '0', '+400 m']
            vertical = '+400 to −400 m from fitted centre'
        else:
            b = description['camera']['geographic_bounds_m']
            labels = [f'{v/1000:+.2f} km' for v in (b[0], (b[0]+b[1])/2, b[1])]
            vertical = f'North {b[2]/1000:+.2f} to {b[3]/1000:+.2f} km'
        for label, fraction in zip(labels, (0., .5, 1.)):
            draw.text((x+fraction*(width-70), y+height+8), label, font=small, fill='#dddddd')
        draw.text((30, y+height+39), vertical+'; no DC notch', font=small, fill='#dddddd')
        if overlay is not None:
            draw.text((30, y+height+63), 'Cyan: interpolated AIS antenna track; squares: reports; dashes: >60 s gaps', font=small, fill=CYAN)
        draw.text((30, y+height+(87 if overlay is not None else 63)),
            'Fixed calibration and motion are provisional; no frame-dependent scaling.', font=small, fill='#bfbfbf')
        bar = (matplotlib.colormaps['magma'](np.linspace(1, 0, height))[:, :3]*255).astype('uint8')
        base.paste(Image.fromarray(np.repeat(bar[:, None], 18, axis=1)), (x+width+12, y))
        for fraction in (0., .25, .5, .75, 1.):
            value = high-fraction*(high-low)
            draw.text((x+width+34, y+fraction*(height-22)), f'{value:.0f}', font=small, fill='white')
        epoch = utc_seconds(model['nominal_start_utc'])
        count = 0
        try:
            for clip_path, record in zip(clips, records):
                decoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(clip_path.parent/f'{name}.mkv'),
                    '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'], stdout=subprocess.PIPE)
                for t in record['frame_centres_s']:
                    raw = np.frombuffer(read_exact(decoder.stdout, width*height*2), '<u2').reshape(height, width)
                    frame = base.copy()
                    displayed = Image.fromarray(lut[raw])
                    if overlay is not None:
                        overlay.draw(displayed, name, count)
                    frame.paste(displayed, (x, y))
                    title_draw = ImageDraw.Draw(frame)
                    utc = datetime.fromtimestamp(epoch+t, timezone.utc).strftime('%H:%M:%S.%f')[:-3]
                    title_draw.text((30, 78), f'Capture {t:.3f} s | {utc} UTC (nominal)', font=small, fill='white')
                    out.stdin.write(frame.tobytes())
                    if count in (0, len(times)//2, len(times)-1):
                        frame.save(args.output/f'{name}-frame-{count:05d}.png')
                    count += 1
                if decoder.stdout.read(1) or decoder.wait() != 0:
                    raise ValueError('Unexpected display-cache frame count or decode failure')
                print(json.dumps({'video': name, 'encoded_frames': count, 'total': len(times)}), flush=True)
            out.stdin.close()
            if out.wait() != 0 or count != len(times):
                raise ValueError('Final movie encoding failed or missed frames')
        except BaseException:
            out.kill(); out.wait()
            raise
        summary['files'].append({'path': str(path), 'size': frame_size, 'bytes': path.stat().st_size})
        summary['complete'] = len(summary['files']) == 2
        (args.output/'videos.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
