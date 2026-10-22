"""Render independent and coherent five-channel ship BP on a fixed scale."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from combine_images import combine


def calibrated_images(samples, gains):
    """Combine original complex BP statistics with the declared channel gains."""
    values = np.array([s[0] for s in samples])
    information = np.broadcast_to(np.array([s[1] for s in samples])[:, None, None], values.shape)
    combined, total, _ = combine(values, information, gains)
    return np.concatenate([8192*information*abs(values)**2, (8192*total*abs(combined)**2)[None]])


class Channel:
    def __init__(self, directory, channel):
        self.prefix = np.load(directory/f'channel-{channel}-prefix.npy', mmap_mode='r')
        with np.load(directory/f'channel-{channel}-timing.npz') as d:
            self.timing = dict(d)

    def image(self, centre, requested_seconds):
        t = self.timing
        count = max(1, round(requested_seconds/(float(t['symbol_period_s'])*int(t['block_rows']))))
        # Use a fixed number of full blocks: no growing or variable-length CPI.
        full = np.flatnonzero(t['rows_per_block'] == t['block_rows'])
        if len(full) < count:
            raise ValueError('Coherent aperture exceeds retained complete blocks')
        starts = full[full+count-1 <= full[-1]]
        midpoint = (t['first_time'][starts]+t['last_time'][starts+count-1])/2
        index = int(np.argmin(abs(midpoint-centre)))
        first, last = starts[index], starts[index]+count
        if abs(midpoint[index]-centre) > float(t['symbol_period_s'])*int(t['block_rows']):
            raise ValueError('Requested movie centre has insufficient complete-aperture support')
        weight = float(t['weight_prefix'][last]-t['weight_prefix'][first])
        if weight <= 0:
            raise ValueError('No information in requested movie aperture')
        value = (self.prefix[last].astype(complex)-self.prefix[first])/weight
        return value, weight, [float(t['first_time'][first]), float(t['last_time'][last-1])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--coherent-seconds', type=float, nargs='+', default=[3., 10., 15.])
    p.add_argument('--centres', type=float, nargs=2, required=True)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--colour-scale-db', type=float, nargs=2, default=[5., 50.])
    p.add_argument('--label', default='Preliminary physical fit; nominal cross-channel calibration')
    p.add_argument('--overwrite', action='store_true', help='Replace videos in the specified output directory')
    args = p.parse_args()
    if (args.fps < 30 or not args.centres[0] < args.centres[1] or not 0 < min(args.coherent_seconds)
            or not args.colour_scale_db[0] < args.colour_scale_db[1]):
        raise ValueError('Need at least 30 fps, positive apertures and an increasing time interval')
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    font = ImageFont.truetype(str(font_path), 20)
    small = ImageFont.truetype(str(font_path), 16)
    channels = [Channel(args.cache, c) for c in range(5)]
    description = json.loads((args.cache/'description.json').read_text())
    gains = np.array([complex(*g) for g in description.get('channel_gains_real_imag', [[1., 0.]]*5)])
    for channel in channels[1:]:
        if any(not np.array_equal(channel.timing[key], channels[0].timing[key]) for key in ('along_m', 'port_m')):
            raise ValueError('All coherent channel images must have the same physical grid')
    cmap = (matplotlib.colormaps['magma'](np.linspace(0, 1, 4096))[:, :3]*255).astype('u1')
    width, height = 1440, 1000
    panel_width, panel_height = 650, 228
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC', 'Coherent five-channel sum']
    times = np.arange(round((args.centres[1]-args.centres[0])*args.fps)+1)/args.fps+args.centres[0]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for duration in args.coherent_seconds:
        path = args.output/f'ship-isar-{duration:g}s-{args.fps}fps.mp4'
        command = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                   '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '17',
                   '-g', '1', '-bf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y' if args.overwrite else '-n', str(path)]
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
        bounds, maximum = [], -np.inf
        try:
            for centre in times:
                samples = [c.image(centre, duration) for c in channels]
                power = calibrated_images(samples, gains)
                db = 10*np.log10(np.maximum(power, 1e-30))
                maximum = max(maximum, float(db.max()))
                low, high = args.colour_scale_db
                colour = cmap[np.rint(np.clip((db-low)/(high-low), 0, 1)*4095).astype(int)]
                canvas = Image.new('RGB', (width, height), '#111217')
                draw = ImageDraw.Draw(canvas)
                draw.text((35, 18), f'MAERSK RIO DELTA hypothesis | t = {centre:.3f} s | requested CPI {duration:g} s', font=font, fill='white')
                draw.text((35, 49), args.label, font=small, fill='#d6d6d6')
                for i, name in enumerate(names):
                    x, y = 55+(i%2)*710, 102+(i//2)*286
                    panel = Image.fromarray(colour[i, ::-1]).resize((panel_width, panel_height), Image.Resampling.NEAREST)
                    canvas.paste(panel, (x, y))
                    draw.text((x, y-26), name, font=font, fill='white')
                    along, port = channels[0].timing['along_m'], channels[0].timing['port_m']
                    draw.text((x, y+panel_height+4), f'Forward {along[0]:g} to {along[-1]:g} m; port {port[0]:g} to {port[-1]:g} m', font=small, fill='#cccccc')
                draw.text((35, 965), f'Origin: AIS antenna body frame | Fixed {low:g} to {high:g} dB matched power / conditional thermal variance | No DC notch', font=small, fill='#cccccc')
                pipe.stdin.write(canvas.tobytes())
                bounds.append([s[2] for s in samples])
            pipe.stdin.close()
            if pipe.wait() != 0:
                raise RuntimeError('Video encoding failed')
        except BaseException:
            pipe.kill(); pipe.wait()
            raise
        record = {'path': str(path), 'requested_aperture_s': duration, 'fps': args.fps,
                  'frame_centres_s': times.tolist(), 'actual_channel_windows_s': bounds,
                  'colour_scale_db': args.colour_scale_db, 'maximum_power_db': maximum,
                  'display_saturated': maximum > args.colour_scale_db[1], 'label': args.label,
                  'channel_gains_real_imag': np.c_[gains.real, gains.imag].tolist(),
                  'source_description': str(args.cache/'description.json'),
                  'scope': 'Display of retained complex BP contributions. Full fixed-length windows and declared complex channel calibration; no frame-dependent fitting or normalization. Body origin is the AIS antenna prior; this does not establish a hull centroid or absolute heading.'}
        records.append(record)
        (args.output/'videos.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps({k: record[k] for k in ('path', 'requested_aperture_s', 'maximum_power_db', 'display_saturated')}), flush=True)


if __name__ == '__main__':
    main()
