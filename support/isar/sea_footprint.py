"""Plot exact first-Fresnel surface zones for declared ship-height examples.

These are finite sphere-surface path contours, not fitted scattering weights.
DEM height near zero is only a provisional water classification. The figure
checks where the ideal unbounded-water approximation encounters the shoreline.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv, utc_seconds
from imaging import physical_state, sites
from motion import ecef, enu_basis, C
from spherical_sea import reflected_leg
from spline_motion import SplineMotion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'calibration'))
from terrain import RasterTerrain, geodetic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dictionary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ais', type=Path, default=Path('docs/data/isar_20261013/rio-delta-ais.csv'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--terrain', type=Path, default=Path('build/calibration-20261013'))
    args = p.parse_args()
    manifest = json.loads((args.dictionary/'dictionary.json').read_text())
    model = json.loads(Path(manifest['source_motion']).read_text())
    water = manifest['water']
    centre, radius = np.asarray(water['centre_ecef_m']), water['radius_m']
    lat, lon = np.radians(water['tangent_latitude_longitude_deg'])
    basis = enu_basis(lat, lon)
    origin = ecef(lat, lon, water['tangent_geoid_m'])
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry)
    middle = float(np.mean(model['interval_s']))
    state = physical_state(track, np.array([utc_seconds(model['nominal_start_utc'])+middle]), model['attitude_prior'])
    state = SplineMotion(model).apply(np.array([middle]), state, tx[3], rx)
    prior = json.loads((args.geometry/'geometry-prior.json').read_text())
    terrain = RasterTerrain(args.terrain, prior, args.terrain/'ga-footprint-dem.tif')
    # Twenty-metre surface sampling covers both visible first-Fresnel zones;
    # the footprint boundary gate prevents silently clipping the area metric.
    east = np.arange(-4000., 12000.1, 20.)
    north = np.arange(-4000., 12000.1, 20.)
    x, y = np.meshgrid(east, north)
    r2 = x*x+y*y
    z = -r2/(radius+np.sqrt(radius*radius-r2))
    surface = origin+np.stack([x, y, z], axis=-1)@basis.T
    normals = (surface-centre)/radius
    la, lo, _ = geodetic(surface.reshape(-1, 3))
    ground = terrain.elevation(la, lo).reshape(x.shape)
    ocean = np.isfinite(ground)&(abs(ground) <= .5)
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained', sharex=True, sharey=True)
    colours = ['#0072b2', '#d55e00', '#009e73']
    frequency = 219.5e6
    for leg, (antenna, label, ax) in enumerate(zip([tx[3], rx], ['Transmitter–ship reflection', 'Ship–receiver reflection'], axes)):
        ax.contourf(x/1000, y/1000, np.where(ocean, 0., 1.), levels=[-.5, .5, 1.5], colors=['#f2f8fc', '#dedede'])
        ax.contour(x/1000, y/1000, np.nan_to_num(ground, nan=50.), levels=[.5], colors=['#999999'], linewidths=.7)
        antenna_range = np.linalg.norm(surface-antenna, axis=-1)
        for height, colour in zip([5., 20., 55.], colours):
            point_body = np.array([0., 0., height])
            position = state['position_ecef_m']+state['body_to_ecef']@point_body
            velocity = state['velocity_ecef_mps']+state['frame_velocity']@point_body
            ray = reflected_leg(position, velocity, antenna, centre, radius)
            residual = antenna_range+np.linalg.norm(surface-position[0], axis=-1)-ray['length_m'][0]
            visible = (np.sum((antenna-surface)*normals, axis=-1) >= 0)&(np.sum((position[0]-surface)*normals, axis=-1) >= 0)
            zone = visible&(residual <= C/frequency/2)
            if not zone.any() or zone[0].any() or zone[-1].any() or zone[:, 0].any() or zone[:, -1].any():
                raise ValueError('First-Fresnel footprint is empty or clipped by the plotted surface')
            local = (ray['specular_ecef_m'][0]-origin)@basis
            ax.contour(x/1000, y/1000, zone.astype(float), levels=[.5], colors=[colour], linewidths=1.2)
            ax.plot(local[0]/1000, local[1]/1000, 'o', color=colour, ms=4, label=f'{height:g} m body-up')
            records.append({'leg': leg, 'height_body_m': height, 'zone_grid_points': int(zone.sum()),
                'zone_area_km2': float(zone.sum()*.02**2), 'water_area_fraction': float(ocean[zone].mean()),
                'unknown_terrain_fraction': float((~np.isfinite(ground[zone])).mean()),
                'specular_east_north_up_m': local.tolist(),
                'extra_path_m': float(ray['length_m'][0]-np.linalg.norm(position[0]-antenna))})
        ship = (state['position_ecef_m'][0]-origin)@basis
        ax.plot(ship[0]/1000, ship[1]/1000, 'k*', ms=10, label='AIS-origin hypothesis')
        local_rx = (rx-origin)@basis
        ax.plot(local_rx[0]/1000, local_rx[1]/1000, 'kx', ms=7, label='Receiver')
        ax.set(aspect='equal', xlabel='East of water tangent point (km)', title=label,
            xlim=(-1.5, 8), ylim=(-1.3, 9))
        ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('North of water tangent point (km)')
    fig.suptitle('Ten, aperture midpoint: exact λ/2 reflected-path surface contours\nGrey: DEM land or unmapped area; circles: specular centres. No atmospheric refraction.', fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    plt.close(fig)
    result = {'scope': __doc__, 'time_s': middle, 'channel': 3, 'rf_hz': frequency,
        'grid_spacing_m': 20., 'water': water, 'records': records}
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
