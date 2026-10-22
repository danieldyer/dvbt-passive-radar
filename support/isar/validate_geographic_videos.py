"""Decode and check the finished fixed-rate geographic backprojection videos."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'video'))
from validate_capture_videos import mov_atoms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    description = json.loads((args.directory/'videos.json').read_text())
    if not description['complete']:
        raise ValueError('Complete both videos before validation')
    result = {'source': str(args.directory/'videos.json'), 'files': [], 'complete': False}
    for entry in description['files']:
        path = Path(entry['path'])
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames,duration,has_b_frames:frame=key_frame,pict_type,best_effort_timestamp_time',
            '-of', 'json', str(path)]))
        stream, frames = probe['streams'][0], probe['frames']
        time = np.array([float(frame['best_effort_timestamp_time']) for frame in frames])
        if (len(frames) != description['frames'] or int(stream['nb_frames']) != len(frames)
                or float(Fraction(stream['r_frame_rate'])) != description['fps']
                or stream['has_b_frames'] != 0 or stream['codec_name'] != 'h264'
                or stream['pix_fmt'] != 'yuv420p'
                or [stream['width'], stream['height']] != entry['size']
                or any(frame['key_frame'] != 1 or frame['pict_type'] != 'I' for frame in frames)
                or abs(float(stream['duration'])-len(frames)/description['fps']) > 2e-6
                or abs(time[0]) > 2e-6 or np.max(abs(np.diff(time)-1/description['fps']), initial=0.) > 2e-6):
            raise ValueError(f'Frame count, encoding or time-grid mismatch: {path}')
        atoms = mov_atoms(path)
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
        result['files'].append({'path': str(path), 'frames': len(frames), 'fps': description['fps'],
            'duration_s': float(stream['duration']), 'all_intra': True, 'b_frames': 0,
            'maximum_frame_period_error_s': float(np.max(abs(np.diff(time)-1/description['fps']), initial=0.)),
            'faststart': True, 'atoms': atoms, 'full_decode_passed': True})
    result['complete'] = True
    (args.directory/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
