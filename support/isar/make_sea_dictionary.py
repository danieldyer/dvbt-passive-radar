"""Cache declared height/sea-path hypotheses without fitting measured phase.

One fixed osculating water sphere is tangent to the EGM96 geoid at the aperture
midpoint. Heights are body-up coordinates relative to the recorded AIS origin.
Terrain checks mark reflected variants whose sampled specular point is on land
or outside the DEM. Finite shoreline/Fresnel-zone effects are not modelled;
passing the centre check alone does not establish an infinite-water model.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from ais import read_csv, utc_seconds
from channel_alignment import AlignmentBand
from field import Field
from imaging import physical_state, sites
from motion import ecef, enu_basis, WGS84_A, WGS84_E2
from scene_dictionary import fingerprint
from spherical_sea import components
from spline_motion import SplineMotion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'calibration'))
from terrain import RasterTerrain, geodetic


def water_sphere(track, model, terrain):
    middle = np.mean(model['interval_s'])
    state = track.sample([utc_seconds(model['nominal_start_utc'])+middle], max_gap_s=180)
    lat, lon = state['latitude'][0], state['longitude'][0]
    height = float(terrain.undulation(np.degrees(lat), np.degrees(lon)))
    n = WGS84_A/np.sqrt(1-WGS84_E2*np.sin(lat)**2)
    meridian = WGS84_A*(1-WGS84_E2)/(1-WGS84_E2*np.sin(lat)**2)**1.5
    radius = float(np.sqrt(n*meridian)+height)
    surface = ecef(lat, lon, height)
    normal = enu_basis(lat, lon)[:, 2]
    return {'centre_ecef_m': (surface-radius*normal).tolist(), 'radius_m': radius,
        'tangent_latitude_longitude_deg': np.degrees([lat, lon]).tolist(), 'tangent_geoid_m': height,
        'orthometric_water_level_m': 0., 'water_level_measured': False,
        'scope': 'Fixed osculating sphere with geometric-mean WGS84 curvature, tangent to the EGM96 geoid. No atmospheric refraction or diffuse sea scattering.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'support', 'alignment', 'ais', 'geometry-directory', 'terrain-directory', 'dem', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--heights-m', type=float, nargs='+', default=[5., 15., 20., 25., 35., 45., 55.])
    p.add_argument('--relative-permittivity', type=float, default=80.)
    p.add_argument('--conductivity-s-per-m', type=float, default=4.)
    args = p.parse_args()
    model, support, alignment = [json.loads(path.read_text()) for path in (args.motion, args.support, args.alignment)]
    if model.get('antenna_fit') or model.get('rock_spacing_seconds') or np.any(model.get('channel_path_rate_bias_mps', 0.)):
        raise ValueError('Use fixed roll/pitch and sites, without channel path-rate corrections')
    if not support.get('complete') or not support.get('fit', {}).get('converged'):
        raise ValueError('Complete the sub-grid support first')
    points = np.array(support['points_body_m'])
    heights = np.asarray(args.heights_m)
    if len(np.unique(heights)) != len(heights) or np.any(heights <= 0):
        raise ValueError('Use distinct positive declared body-up heights')
    tx, rx, _ = sites(args.geometry_directory)
    prior = json.loads((args.geometry_directory/'geometry-prior.json').read_text())
    terrain = RasterTerrain(args.terrain_directory, prior, args.dem)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    water = water_sphere(track, model, terrain)
    source_paths = [args.motion, args.support, args.alignment, args.ais, args.dem,
        args.terrain_directory/'us_nga_egm96_15.tif', args.geometry_directory/'geometry-prior.json',
        args.geometry_directory/'sites-main.json', Path(__file__), Path(__file__).with_name('spherical_sea.py'),
        Path(__file__).with_name('field.py')]
    identity = {'sha256': {str(path): fingerprint(path) for path in source_paths},
        'heights_body_m': heights.tolist(), 'relative_permittivity': args.relative_permittivity,
        'conductivity_s_per_m': args.conductivity_s_per_m}
    families = {'direct': [0], 'transmit': [0, 1], 'receive': [0, 2], 'both': [0, 1, 2, 3]}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'dictionary.json'
    result = {'identity': identity, 'scope': __doc__, 'source_motion': str(args.motion),
        'source_support': str(args.support), 'water': water, 'points_body_m': points.tolist(),
        'families': families, 'column_order': 'point-major, then height', 'channels': [], 'complete': False}
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if result['identity'] != identity:
            raise ValueError('Changed sea-dictionary input; use a new output directory')
    curve = SplineMotion(model)
    started = time.perf_counter()
    for c in range(5):
        if any(ch['channel'] == c for ch in result['channels']):
            continue
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        state = physical_state(track, utc_seconds(model['nominal_start_utc'])+field.data['row_time'], model['attitude_prior'])
        state = curve.apply(field.data['row_time'], state, tx[3], rx)
        block = np.floor(field.time-curve.interval[0]).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        band = AlignmentBand(field, state, points, selected)
        shape = (field.y.size, len(points)*len(heights))
        matrices = {name: np.lib.format.open_memmap(args.output/f'channel-{c}-{name}.building.npy',
            mode='w+', dtype=np.complex128, shape=shape, fortran_order=True) for name in families}
        water_valid = np.ones((2, len(points), len(heights)), bool)
        diagnostics = []
        direct_error = 0.
        for j, point_body in enumerate(points):
            for h, height in enumerate(heights):
                point = point_body.copy(); point[2] = height
                values, paths = components(field, state, point, np.array(water['centre_ecef_m']), water['radius_m'],
                    relative_permittivity=args.relative_permittivity, conductivity_s_per_m=args.conductivity_s_per_m,
                    range_bias_m=model['range_bias_m'][c])
                for family, rays in families.items():
                    matrices[family][:, j*len(heights)+h] = values[rays].sum(axis=0).ravel()
                if j in (0, len(points)//2, len(points)-1) and h == 0:
                    direct = field.point(point, motion=state, range_bias_m=model['range_bias_m'][c])
                    direct_error = max(direct_error, float(np.linalg.norm(values[0]-direct)/np.linalg.norm(direct)))
                record = {'point': j, 'height_body_m': float(height), 'legs': []}
                for leg, reflection in enumerate(paths['reflected_legs']):
                    # All retained-bin centres plus both aperture edges.
                    keep = np.unique(np.r_[0, np.arange(8, len(field.data['row_time']), 16), len(field.data['row_time'])-1])
                    lat, lon, ellipsoid_height = geodetic(reflection['specular_ecef_m'][keep])
                    ground = terrain.elevation(lat, lon)
                    water_valid[leg, j, h] = np.isfinite(ground).all() and np.all(abs(ground) <= .5)
                    extra = paths['excess_path_m'][1+leg]-paths['excess_path_m'][0]
                    record['legs'].append({'specular_point_on_water': bool(water_valid[leg, j, h]),
                        'terrain_min_max_m': [float(np.nanmin(ground)), float(np.nanmax(ground))],
                        'extra_path_min_max_m': [float(extra.min()), float(extra.max())],
                        'grazing_min_max_degrees': np.degrees(np.arcsin([reflection['sin_grazing'].min(), reflection['sin_grazing'].max()])).tolist(),
                        'sphere_minus_geoid_min_max_m': [float(v) for v in [
                            np.min(ellipsoid_height-terrain.undulation(lat, lon)), np.max(ellipsoid_height-terrain.undulation(lat, lon))]],
                        'maximum_specular_sine_residual': float(reflection['specular_sine_residual'].max())})
                diagnostics.append(record)
            if j%16 == 15:
                print(json.dumps({'channel': c, 'points': j+1, 'seconds': time.perf_counter()-started}), flush=True)
        if direct_error > 1e-7:
            raise ValueError('Direct ray disagrees with the existing physical observation operator')
        for family, matrix in matrices.items():
            matrix.flush()
            (args.output/f'channel-{c}-{family}.building.npy').replace(args.output/f'channel-{c}-{family}.npy')
        matrices.clear()
        np.savez(args.output/f'channel-{c}-observation.npz', time=field.time, data=field.y,
            selected=selected, background=band.nuisance.design, water_valid=water_valid)
        entry = {'channel': c, 'shape': list(shape), 'direct_probe_relative_error': direct_error,
            'water_invalid_count_by_leg': (~water_valid).sum(axis=(1, 2)).tolist(), 'diagnostics': diagnostics}
        result['channels'].append(entry)
        result['complete'] = len(result['channels']) == 5
        result['seconds'] = time.perf_counter()-started
        manifest.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in entry.items() if k != 'diagnostics'}), flush=True)


if __name__ == '__main__':
    main()
