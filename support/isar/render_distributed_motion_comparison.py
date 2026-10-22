"""Compare two fixed-N/E fits using identical original-row GPU backprojection.

The fitted scene is not rendered or substituted for the observations. Both
images use the complete original five-channel profiles, unchanged provisional
channel gains, the same physical viewport and one common fixed display scale.
"""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv, utc_seconds
from extract_full_aperture import load
from geographic_movie import SIZES, clip, state_at
from imaging import sites
from motion import enu_basis
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from terrain import geodetic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'aperture', 'ais', 'work', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--spacing-m', type=float, default=2.)
    p.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    args = p.parse_args()
    fitted = json.loads(args.motion.read_text())
    paths = [Path(fitted['source_motion']), args.motion]
    models = [json.loads(path.read_text()) for path in paths]
    for model in models:
        if not model['fixed_ne_frame'] or np.any(SplineMotion(model).yaw_coefficients) or not model['fit']['converged']:
            raise ValueError('Complete both fixed-N/E fits before rendering')
        for key in ('interval_s', 'range_bias_m', 'channel_gains_real_imag', 'geometry_directory', 'nominal_start_utc'):
            if model[key] != models[0][key]:
                raise ValueError('Paired images need identical support and calibration')
    apertures = [load(args.aperture/f'channel-{c}') for c in range(5)]
    tx, rx, _ = sites(Path(fitted['geometry_directory']))
    midpoint = np.mean(fitted['interval_s'])
    centre = state_at(None, models[0], SplineMotion(models[0]), tx, rx, np.array([midpoint]))['position_ecef_m'][0]
    latitude, longitude, _ = geodetic(centre)
    local = enu_basis(*np.radians([latitude, longitude]))
    camera = {'origin_ecef_m': centre.tolist(), 'enu_to_ecef': local.tolist(),
        'origin_latitude_longitude_deg': [float(latitude), float(longitude)],
        'geographic_bounds_m': [-250., 250., -350., 350.],
        'following_extent_m': [500., 700.], 'fixed_north_up': True}
    track = read_csv(args.ais, reference_height_ellipsoid_m=fitted['reference_height_ellipsoid_m'], mmsi=563052200)
    ais_time = utc_seconds(fitted['nominal_start_utc'])+np.linspace(*fitted['interval_s'], 101)
    ais = (track.sample(ais_time, max_gap_s=400.)['position_ecef_m']-centre)@local
    powers, records = [], []
    for name, path, model in zip(('sparse-motion', 'distributed-motion'), paths, models):
        output = args.work/name
        identity = {'model': str(path), 'model_sha256': fingerprint(path), 'camera': camera,
            'scientific_spacing_m': args.spacing_m,
            'renderer_sha256': fingerprint(Path(__file__).with_name('geographic_movie.py')),
            'shader_sha256': fingerprint(Path(__file__).with_name('backprojection.comp'))}
        if (output/'clip.json').exists():
            if json.loads((output/'identity.json').read_text()) != identity:
                raise ValueError('Existing image cache differs; choose a new work directory')
        else:
            clip(apertures, model, track, tx, rx, np.array([midpoint]), camera, output,
                args.executable, args.spacing_m, 512, coherent_seconds=float(np.diff(fitted['interval_s'])[0]))
            (output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
        data = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output/'geographic.mkv'),
            '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'])
        width, height = SIZES['geographic']
        power = np.frombuffer(data, '<u2').reshape(height, width)/256-64
        if not np.isfinite(power).all():
            raise ValueError('Incomplete geographic viewport')
        powers.append(power)
        rendered = json.loads((output/'clip.json').read_text())
        records.append({**identity, 'name': name, 'actual_aperture_s': rendered['actual_apertures_s'][0],
            'displayed_peak_db': float(power.max()), 'fixed_ne_frame': True})
    if records[0]['actual_aperture_s'] != records[1]['actual_aperture_s']:
        raise ValueError('The paired backprojections used different data support')
    low = float(np.quantile(powers, .01)-1.)
    high = float(np.ceil((max(v.max() for v in powers)+.05)*10)/10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), layout='constrained', sharex=True, sharey=True)
    for ax, power, label in zip(axes, powers, ('Original eight-point motion fit', 'Distributed-scene translation fit')):
        im = ax.imshow(power, extent=camera['geographic_bounds_m'], origin='upper', cmap='magma',
            vmin=low, vmax=high, interpolation='nearest')
        ax.plot(ais[:, 0], ais[:, 1], color='#00ffff', lw=1.4)
        ax.plot(ais[50, 0], ais[50, 1], 'o', color='#00ffff', ms=5)
        ax.set(xlabel='East (m)', ylabel='North (m)', title=label)
    fig.colorbar(im, ax=axes, label='Conditional coherent matched power (dB)', shrink=.8)
    fig.suptitle('Fixed N/E, yaw exactly zero, same 30 s original data\nFive-channel complex BP; cyan = interpolated AIS antenna track')
    fig.supxlabel('Provisional band calibration. The fitted scene is not substituted for the data; no DC notch.', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    result = {'scope': __doc__, 'motion': str(args.motion), 'records': records,
        'colour_scale_db': [low, high], 'ais_sha256': fingerprint(args.ais),
        'cross_band_phase_validated': False, 'production_changed': False,
        'peak_change_db': records[1]['displayed_peak_db']-records[0]['displayed_peak_db'],
        'validation': 'Image peaks are diagnostics, not independent sensitivity or hull-resolution evidence.'}
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'peak_change_db': result['peak_change_db'], 'colour_scale_db': [low, high]}))


if __name__ == '__main__':
    main()
