"""Five-channel, boxed calibration-candidate clips from production preview caches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'video'))
import render_recording as recording
import render_video as video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--candidate', action='append', required=True, help='LABEL:CHANNEL:TRACK')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists')
    manifest = json.loads((args.cache / 'processing/manifest.json').read_text())
    tracks = json.loads(args.tracks.read_text())['tracks']
    seeds = []
    for spec in args.candidate:
        label, ch, index = spec.split(':')
        tr = np.array(tracks[int(ch)][int(index)])
        epoch = np.mean(tr[:, 0])
        seeds.append((label, epoch, np.polynomial.polynomial.polyfit(tr[:, 0] - epoch, tr[:, 1], 3),
                      np.polynomial.polynomial.polyfit(tr[:, 0] - epoch, tr[:, 2], 3),
                      (tr[0, 0] - 2, tr[-1, 0] + 2)))
    states = manifest['signals']
    # Display-only crop, preserving the already computed scientific map.
    limits = (.6, 2.6, -30, 30)
    cursors, sources, frames = [], [], []
    floors, maxima = [], []
    for state in states:
        source = Path(state['spool']).open('rb')
        floor, peak = [], -np.inf
        for meta, db in recording.read_previews(source):
            h, w = db.shape
            selected = db[round(h * (40 - limits[3]) / 80):round(h * (40 - limits[2]) / 80),
                          round(w * limits[0] / 4):round(w * limits[1] / 4)]
            floor.append(float(np.median(selected)))
            peak = max(peak, float(selected.max()))
        floors.append(float(np.median(floor)));maxima.append(peak)
        source.seek(0);sources.append(source)
        cursor = recording.read_previews(source);cursors.append(cursor);frames.append(next(cursor))
    lower = -1.5
    upper = max(p - f for p, f in zip(maxima, floors)) + .25
    starts = [m['phase_epoch_s'] for m, _ in frames]
    start = max(starts)
    stop = manifest['recording_start_s'] + manifest['recording_duration_s'] - .51
    fps = manifest['fps'];count = max(0, int((stop - start) * fps))
    if not count:
        raise ValueError('No common clip interval')
    following = [next(c, None) for c in cursors]
    colours = np.array([[.018,.032,.07],[.08,.22,.40],[.08,.65,.66],[.94,.65,.20],[1,.98,.82]])
    lut = (255 * np.stack([np.interp(np.linspace(0, 4, 4096), np.arange(5), colours[:, c])
                          for c in range(3)], axis=-1)).astype(np.uint8)
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    fonts = {s: ImageFont.truetype(font_path, s) for s in (16, 18, 20, 24, 28, 32)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1920x1080', '-r', str(fps), '-i', '-', '-an', *video.ENCODINGS['prores'][1],
               '-movflags', '+faststart', str(args.output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for n in range(count):
            t = start + n / fps
            image = Image.new('RGB', (1920, 1080), '#0c131e')
            draw = ImageDraw.Draw(image)
            draw.text((30, 20), 'PHASE-CENTRE CALIBRATION  /  CANDIDATE REVIEW', font=fonts[28], fill='#55d6c1')
            draw.text((30, 60), f'Capture {t:07.3f} s   |   1 s coherent window   |   {fps:.2f} fps', font=fonts[20], fill='#c2d0df')
            for ch, state in enumerate(states):
                while following[ch] is not None and following[ch][0]['phase_epoch_s'] <= t:
                    frames[ch] = following[ch];following[ch] = next(cursors[ch], None)
                meta, db = frames[ch]
                x0, y0 = 55 + (ch % 3) * 635, 140 + (ch // 3) * 455
                width, height = 560, 350
                h, w = db.shape
                selected = db[round(h * (40 - limits[3]) / 80):round(h * (40 - limits[2]) / 80),
                              round(w * limits[0] / 4):round(w * limits[1] / 4)]
                index = np.clip((selected - floors[ch] - lower) / (upper - lower) * 4095, 0, 4095).astype(int)
                panel = Image.fromarray(lut[index]).resize((width, height), Image.Resampling.NEAREST)
                image.paste(panel, (x0, y0));draw = ImageDraw.Draw(image)
                draw.text((x0, y0 - 34), f"{state['name']}   {state['rf_hz']/1e6:g} MHz", font=fonts[24], fill='#eef5f8')
                draw.rectangle((x0, y0, x0 + width, y0 + height), outline='#55616d')
                for hz in (-30, -15, 0, 15, 30):
                    yy = y0 + height * (30 - hz) / 60
                    draw.text((x0 - 7, yy), f'{hz:+d}', font=fonts[16], fill='#a8b7c6', anchor='rm')
                for km in (.6, 1, 1.5, 2, 2.6):
                    xx = x0 + width * (km - .6) / 2
                    draw.text((xx, y0 + height + 10), f'{km:g}', font=fonts[16], fill='#a8b7c6', anchor='mt')
                draw.text((x0 + width/2, y0 + height + 35), 'Excess path (km)', font=fonts[18], fill='#a8b7c6', anchor='mt')
                tm = meta['phase_epoch_s']
                for label, epoch, path, rate, support in seeds:
                    if not support[0] <= tm <= support[1]:
                        continue
                    length = np.polynomial.polynomial.polyval(tm - epoch, path)
                    speed = np.polynomial.polynomial.polyval(tm - epoch, rate)
                    fd = -state['rf_hz'] * speed / 299792458.
                    xx = x0 + width * (length / 1000 - .6) / 2
                    yy = y0 + height * (30 - fd) / 60
                    dx, dy = width * .15 / 2, height * 2 / 60
                    if x0 <= xx <= x0 + width and y0 <= yy <= y0 + height:
                        draw.rectangle((xx-dx, yy-dy, xx+dx, yy+dy), outline='#ff5cf3', width=2)
                        draw.text((xx-dx, yy-dy-23), label, font=fonts[18], fill='#ff8bf5')
            x, y = 1325, 625
            for line in ['Magenta boxes: proposed shared returns', 'Association is provisional.', '',
                         'Same path-rate seed; Doppler scaled', 'by each channel’s physical RF.', '',
                         'Fixed colour scale throughout clip.', 'Per-channel constant noise reference.',
                         'Stationary suppression remains enabled.']:
                draw.text((x, y), line, font=fonts[20], fill='#c2d0df');y += 29
            draw.text((30, 1035), f'Display: {lower:.1f} to {upper:.1f} dB relative to each channel’s fixed median background.  No frame autoscaling.',
                      font=fonts[18], fill='#a8b7c6')
            if n == count // 2:
                image.save(args.output.with_suffix('.png'))
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Video encoder failed')
    finally:
        if encoder.poll() is None:
            encoder.kill();encoder.wait()
        for source in sources:
            source.close()
    args.output.with_suffix('.json').write_text(json.dumps({'capture':manifest['source'], 'frames':count,
        'fps':fps, 'first_phase_epoch_s':start, 'last_phase_epoch_s':start+(count-1)/fps,
        'selection':args.candidate, 'box_half_width_m':150, 'box_half_height_hz':2,
        'display_noise_references_db':floors, 'display_limits_relative_db':[lower,upper],
        'status':'provisional association; not an accepted phase-centre calibration'}, indent=2))
    print(args.output, count, 'frames')


if __name__ == '__main__':
    main()
