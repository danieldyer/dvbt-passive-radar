"""Render paired no-yaw controls on the same fixed geographic grid.

Starts are selected by training likelihood only. Original-row fixed-duration GPU
backprojection, calibration, physical viewport and display scale are identical
within each pair. These images do not certify cross-band calibration.
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
from fixed_frame_motion import geographic_frame
from geographic_movie import SIZES, clip, state_at
from imaging import sites
from motion import C, enu_basis, path_difference
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from terrain import geodetic


def phase_effect(model, transmitters, receiver, frequencies):
    """Path change from freezing attitude, before refitting translation.

    Constant point phase is unobservable with profiled complex reflectivities;
    the reported excursion removes it by taking the temporal peak-to-peak.
    """
    time = np.linspace(*model['interval_s'], 1001)
    frozen = geographic_frame(model)
    moving = state_at(None, model, SplineMotion(model), transmitters, receiver, time)
    fixed = state_at(None, frozen, SplineMotion(frozen), transmitters, receiver, time)
    records = []
    for c, rf in enumerate(frequencies):
        degrees = []
        for a, b in zip(model['selected_points_body_m'][c], frozen['selected_points_body_m'][c]):
            point = moving['position_ecef_m']+moving['body_to_ecef']@a
            offset = fixed['body_to_ecef']@b-moving['body_to_ecef']@a
            delta = path_difference(point, offset, transmitters[c], receiver)
            degrees.append(float(360*rf/C*np.ptp(delta)))
        records.append(degrees)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fits', 'aperture', 'ais', 'geometry-directory', 'work', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--windows', nargs='+', type=int, required=True)
    p.add_argument('--spacing-m', type=float, default=1.)
    p.add_argument('--extent-m', type=float, nargs=2, default=[500., 700.])
    p.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    args = p.parse_args()
    if args.spacing_m <= 0 or min(args.extent_m) <= 0:
        raise ValueError('Image spacing and extents must be positive')
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    apertures = [load(args.aperture/f'channel-{c}') for c in range(5)]
    frequencies = [json.loads(a['metadata_json'])['channel_configuration']['rf_hz'] for a in apertures]
    tx, rx, _ = sites(args.geometry_directory)
    fig, axes = plt.subplots(len(args.windows), 2, figsize=(12, 8*len(args.windows)),
        squeeze=False, layout='constrained')
    records = []
    for row, window in enumerate(args.windows):
        free_path = args.fits/f'window-{window:03d}-free.json'
        free = json.loads(free_path.read_text())
        candidates = [(path, json.loads(path.read_text())) for path in
                      sorted(args.fits.glob(f'window-{window:03d}-fixed-ne*.json'))]
        if not candidates:
            raise ValueError('The matched fixed-frame starts are missing')
        minimum = min(m['fit']['objective'] for _, m in candidates)
        stationary = [(path, m) for path, m in candidates
            if m['fit']['converged'] and m['fit']['objective'] <= minimum+1e-10]
        fixed_path, fixed = min(stationary or candidates, key=lambda item: item[1]['fit']['objective'])
        if not free['fit']['converged'] or not fixed['fit']['converged']:
            raise ValueError('Complete the paired fits and stationarity checks first')
        if fixed['interval_s'] != free['interval_s'] or not fixed['fixed_ne_frame']:
            raise ValueError('Paired fits have incompatible support or orientation')
        for key in ('range_bias_m', 'channel_gains_real_imag', 'geometry_directory', 'nominal_start_utc'):
            if fixed[key] != free[key]:
                raise ValueError('Paired image calibration must remain identical')
        midpoint = np.mean(free['interval_s'])
        centre = state_at(None, free, SplineMotion(free), tx, rx, np.array([midpoint]))['position_ecef_m'][0]
        lat, lon, _ = geodetic(centre)
        local = enu_basis(*np.radians([lat, lon]))
        camera = {'origin_ecef_m': centre.tolist(), 'enu_to_ecef': local.tolist(),
            'origin_latitude_longitude_deg': [float(lat), float(lon)],
            'geographic_bounds_m': [-args.extent_m[0]/2, args.extent_m[0]/2,
                                    -args.extent_m[1]/2, args.extent_m[1]/2],
            'following_extent_m': args.extent_m, 'fixed_north_up': True}
        track = read_csv(args.ais, reference_height_ellipsoid_m=free['reference_height_ellipsoid_m'], mmsi=563052200)
        epoch = utc_seconds(free['nominal_start_utc'])
        time = np.linspace(*free['interval_s'], 101)
        ais = (track.sample(epoch+time, max_gap_s=400.)['position_ecef_m']-centre)@local
        record = {'window': window, 'interval_s': free['interval_s'], 'selected_fixed_start': str(fixed_path),
            'start_selection': 'Minimum training objective; withheld values do not select starts',
            'starts': [{'path': str(path), 'training': m['fit']['objective'],
                'heldout_error': m['after_summary']['prediction_error']} for path, m in candidates],
            'heldout_error_change_percent': 100*(fixed['after_summary']['prediction_error']/free['after_summary']['prediction_error']-1),
            'heldout_matched_energy_change_db': 10*np.log10(fixed['after_summary']['matched_energy']/free['after_summary']['matched_energy']),
            'free_yaw_excursion_deg': float(np.degrees(np.ptp(SplineMotion(free).components(time)[:, 1]))),
            'frozen_attitude_point_phase_excursions_deg': phase_effect(free, tx, rx, frequencies),
            'camera': camera, 'variants': []}
        for column, (name, path, model) in enumerate((('free', free_path, free), ('fixed-ne', fixed_path, fixed))):
            output = args.work/f'window-{window:03d}-{name}'
            identity = {'model': str(path), 'model_sha256': fingerprint(path), 'camera': camera,
                'spacing_m': args.spacing_m, 'renderer_sha256': fingerprint(Path(__file__).with_name('geographic_movie.py'))}
            if (output/'clip.json').exists():
                if json.loads((output/'identity.json').read_text()) != identity:
                    raise ValueError('Existing image cache differs from this comparison')
            else:
                clip(apertures, model, track, tx, rx, np.array([midpoint]), camera, output,
                    args.executable, args.spacing_m, 512,
                    coherent_seconds=float(np.diff(model['interval_s'])[0]))
                (output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
            data = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output/'geographic.mkv'),
                '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'])
            width, height = SIZES['geographic']
            power = np.frombuffer(data, '<u2').reshape(height, width)/256-64
            ax = axes[row, column]
            im = ax.imshow(power, extent=camera['geographic_bounds_m'], origin='upper', cmap='magma',
                vmin=.875, vmax=48.8, interpolation='nearest')
            ax.plot(ais[:, 0], ais[:, 1], color='#00ffff', lw=1.5)
            ax.plot(ais[50, 0], ais[50, 1], 'o', color='#00ffff', ms=5)
            title = 'Free yaw' if name == 'free' else 'Fixed N/E, translation only'
            ax.set(xlabel='East (m)', ylabel='North (m)',
                title=f'{title} · {free["interval_s"][0]:.2f}–{free["interval_s"][1]:.2f} s')
            rendered = json.loads((output/'clip.json').read_text())
            record['variants'].append({'name': name, **identity,
                'fit': {key: model['fit'][key] for key in ('objective', 'converged', 'maximum_kkt_residual')},
                'parameters': model['fitted_real_parameters'], 'validation': model['after_summary'],
                'per_channel_validation': model['original_row_validation_after'],
                'actual_aperture_s': rendered['actual_apertures_s'][0],
                'displayed_peak_db': float(power.max())})
        if record['variants'][0]['actual_aperture_s'] != record['variants'][1]['actual_aperture_s']:
            raise ValueError('Compared images use different data support')
        records.append(record)
        print(json.dumps({key: record[key] for key in ('window', 'selected_fixed_start',
            'heldout_error_change_percent', 'heldout_matched_energy_change_db', 'free_yaw_excursion_deg')}), flush=True)
    high = float(np.ceil((max(v['displayed_peak_db'] for r in records for v in r['variants'])+.05)*10)/10)
    for ax in axes.ravel():
        ax.images[0].set_clim(.875, high)
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Conditional coherent matched power (dB)', shrink=.7)
    fig.suptitle('Original-row five-channel backprojection: removing fitted yaw\nSame data, calibration and fixed scale; cyan = interpolated AIS antenna track')
    fig.supxlabel('The image calibration is provisional. Geographic coordinates are relative to each aperture’s free-fit midpoint.', fontsize=10)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps({'scope': __doc__, 'records': records,
        'colour_scale_db': [.875, high], 'scientific_grid_spacing_m': args.spacing_m,
        'validation_scope': 'Independent per-band reflectivities, shared physical motion; this is not a calibration-efficiency certificate.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
