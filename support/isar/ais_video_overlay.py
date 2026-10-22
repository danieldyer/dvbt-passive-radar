"""Geographic AIS reference overlay, applied only after image formation.

Uses the same fixed EN camera as the scientific image. Fitted vessel attitude
never rotates the AIS path. A moving camera changes its translation only.
AIS denotes the reporting antenna, not a surveyed hull-centre location.
"""
import numpy as np
from PIL import Image, ImageDraw

from ais import utc_seconds
from geographic_movie import state_at
from spline_motion import SplineMotion


CYAN = '#00ffff'


def pixels(points, bounds, size):
    """Map physical EN coordinates onto image pixel centres, north upward."""
    x0, x1, y0, y1 = bounds
    width, height = size
    if not x0 < x1 or not y0 < y1:
        raise ValueError('Invalid geographic viewport')
    return (np.asarray(points)-[x0, y1])*[width/(x1-x0), -height/(y1-y0)]-.5


def following_bounds(centre, extent):
    east, north = centre
    width, height = extent
    return [east-width/2, east+width/2, north-height/2, north+height/2]


class AisOverlay:
    def __init__(self, description, model, track, transmitters, receiver, times):
        camera = description['camera']
        if not camera['fixed_north_up']:
            raise ValueError('AIS overlay requires a fixed north-up geographic frame')
        self.camera = camera
        origin, frame = np.array(camera['origin_ecef_m']), np.array(camera['enu_to_ecef'])
        epoch = utc_seconds(model['nominal_start_utc'])
        half = description.get('coherent_seconds', 30.)/2
        start, stop = times[0]-half, times[-1]+half
        # Drawing resolution only; no effect on the physical forward operator.
        time = np.linspace(start, stop, int(np.ceil((stop-start)*2))+1)
        sampled = track.sample(epoch+time, max_gap_s=400.)
        self.path = ((sampled['position_ecef_m']-origin)@frame)[:, :2]
        self.dashed = sampled['bracketing_interval_s'] > 60.
        self.dash_on = (np.floor((time-time[0])/4).astype(int)%2) == 0
        self.current = ((track.sample(epoch+times, max_gap_s=400.)['position_ecef_m']-origin)@frame)[:, :2]
        selected = (track.seconds >= epoch+start)&(track.seconds <= epoch+stop)
        observed = track.sample(track.seconds[selected], max_gap_s=400.)['position_ecef_m']
        self.reports = ((observed-origin)@frame)[:, :2]
        state = state_at(track, model, SplineMotion(model), transmitters, receiver, times)
        self.centres = ((state['position_ecef_m']-origin)@frame)[:, :2]
        self.fixed_layers = {}
        self.metadata = {'colour': CYAN, 'frame': 'Fixed geographic EN; north upward',
            'meaning': 'Interpolated AIS antenna position, not a radar fit or hull-centre constraint',
            'interpolation': 'Existing geodetic cubic Hermite track with reported speed/course tangents',
            'long_gap_dashes_above_s': 60., 'maximum_bracketing_interval_s': float(sampled['bracketing_interval_s'].max()),
            'capture_interval_s': [float(start), float(stop)],
            'maximum_interpolated_point_step_s': float(np.diff(time).max()),
            'nominal_start_utc': model['nominal_start_utc'],
            'display_only': True}

    def layer(self, bounds, size):
        layer = Image.new('RGBA', size)
        draw = ImageDraw.Draw(layer)
        xy = pixels(self.path, bounds, size)
        visible = ~self.dashed[:-1] | self.dash_on[:-1]
        edges = np.flatnonzero(np.diff(np.r_[False, visible, False]))
        for start, stop in zip(edges[::2], edges[1::2]):
            segment = [tuple(p) for p in xy[start:stop+1]]
            draw.line(segment, fill='black', width=4)
            draw.line(segment, fill=CYAN, width=2)
        for x, y in pixels(self.reports, bounds, size):
            if -5 < x < size[0]+5 and -5 < y < size[1]+5:
                draw.rectangle((x-3, y-3, x+3, y+3), fill='black')
                draw.rectangle((x-2, y-2, x+2, y+2), fill=CYAN)
        return layer

    def draw(self, image, name, frame_index):
        if name == 'following':
            bounds = following_bounds(self.centres[frame_index], self.camera['following_extent_m'])
            overlay = self.layer(bounds, image.size)
        elif name == 'geographic':
            bounds = self.camera['geographic_bounds_m']
            if image.size not in self.fixed_layers:
                self.fixed_layers[image.size] = self.layer(bounds, image.size)
            overlay = self.fixed_layers[image.size].copy()
        else:
            raise ValueError('Unknown geographic camera')
        x, y = pixels(self.current[frame_index], bounds, image.size)
        draw = ImageDraw.Draw(overlay)
        draw.ellipse((x-6, y-6, x+6, y+6), fill='black')
        draw.ellipse((x-4, y-4, x+4, y+4), fill=CYAN)
        draw.ellipse((x-1, y-1, x+1, y+1), fill='white')
        image.paste(overlay, (0, 0), overlay)

