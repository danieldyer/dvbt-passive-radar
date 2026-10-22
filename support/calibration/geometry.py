"""Double-precision excess-path geometry and explicitly labelled site priors."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from terrain import RasterTerrain
from transmitter_config import ROOT, nominal_indices

C = 299792458.
RF = np.array([177.5, 184.5, 191.625, 219.5, 226.5]) * 1e6
SITES = np.array(nominal_indices())


def ecef(latitude, longitude, height):
    lat, lon = np.deg2rad(latitude), np.deg2rad(longitude)
    a, e2 = 6378137., 6.6943799901413165e-3
    n = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    return np.stack(((n+height)*np.cos(lat)*np.cos(lon),
                     (n+height)*np.cos(lat)*np.sin(lon),
                     (n*(1-e2)+height)*np.sin(lat)), axis=-1)


def enu_rotation(latitude, longitude):
    a, b = np.deg2rad([latitude, longitude])
    return np.array([[-np.sin(b), np.cos(b), 0],
                     [-np.sin(a)*np.cos(b), -np.sin(a)*np.sin(b), np.cos(a)],
                     [np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)]])


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def observables(x, velocity, transmitter, receiver):
    a, b = x-transmitter, x-receiver
    length = np.linalg.norm(a, axis=-1)+np.linalg.norm(b, axis=-1)-np.linalg.norm(transmitter-receiver, axis=-1)
    gradient = unit(a) + unit(b)
    rate = np.sum(gradient * velocity, axis=-1)
    return length, rate


def central_jacobian(fun, x, step=1e-5):
    """Deterministic central differences for support diagnostics and tests."""
    columns = []
    for i in range(len(x)):
        h = step*max(1., abs(x[i])); delta = np.zeros_like(x); delta[i] = h
        columns.append((fun(x+delta)-fun(x-delta))/(2*h))
    return np.stack(columns, axis=1)


def terrain_height(directory, lat, lon):
    latitude, longitude = int(np.floor(lat)), int(np.floor(lon))
    name = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}"
    values = np.load(directory / (name + '.npy'), mmap_mode='r')
    n = values.shape[0]-1
    point = np.array([[(latitude+1-lat)*n], [(lon-longitude)*n]])
    h = float(map_coordinates(values.astype(float), point, order=1)[0])
    if h < -500:
        raise ValueError('Void terrain post')
    return h


def receiver_prior(directory, metadata):
    """Read the physical receiver prior, independently of saved data references."""
    if 'receiver' not in metadata:
        raise ValueError('Receiver geometry is required in metadata: location and antenna_agl_m')
    source = metadata['receiver'];location = source['location']
    lat, lon = float(location['latitude_deg']), float(location['longitude_deg'])
    agl = float(source['antenna_agl_m'])
    if not np.all(np.isfinite([lat, lon, agl])) or abs(lat) > 90 or abs(lon) > 180:
        raise ValueError('Invalid receiver coordinates or height')
    geoid = np.asarray(Image.open(directory / 'us_nga_egm96_15.tif'))
    undulation = float(map_coordinates(geoid, [[(90-lat)/.25], [(lon+180)/.25]],
                                      order=1, output=np.float64)[0])
    height = terrain_height(directory, lat, lon)
    receiver = ecef(lat, lon, height+agl+undulation)
    rotation = enu_rotation(lat, lon)
    description = {'latitude_deg': lat, 'longitude_deg': lon, 'antenna_agl_m': agl,
        'terrain_egm96_m': height, 'geoid_undulation_m': undulation,
        'prior_sigma_enu_m': [15., 15., 7.5],
        'prior_sigma_reference': 'Analysis sensitivity assumption, not surveyed accuracy',
        'position_source': source.get('position_reference', 'Recording metadata'),
        'height_source': source.get('height_reference', 'Recording metadata; AGL')}
    return receiver, rotation, description


def prior(directory, metadata):
    receiver, rotation, description = receiver_prior(directory, metadata)
    geoid = np.asarray(Image.open(directory / 'us_nga_egm96_15.tif'))
    def undulation(a, b):
        return float(map_coordinates(geoid, [[(90-a)/.25], [(b+180)/.25]], order=1, output=np.float64)[0])
    terrain_description = metadata['transmitter_terrain']
    if terrain_description['vertical_datum'] != 'EGM96':
        raise ValueError('Transmitter DEM must have a declared EGM96 vertical datum')
    terrain = RasterTerrain(directory, {'receiver_ecef_m': receiver, 'ecef_to_enu': rotation},
                            ROOT/terrain_description['raster'])
    tx, descriptions = [], []
    for site in metadata['transmitters']:
        loc = site['location'];a, b = loc['latitude_deg'], loc['longitude_deg']
        agl = float(site['antenna_agl_m'])
        ground = float(terrain.elevation(a, b));geoid_height = undulation(a, b)
        if not np.all(np.isfinite([a, b, agl, ground, geoid_height])) or agl < 0:
            raise ValueError('Invalid transmitter coordinates, antenna height or DEM coverage')
        position = ecef(a, b, ground+agl+geoid_height)
        tx.append(rotation @ (position-receiver))
        descriptions.append({'id': site['id'], 'latitude_deg': a, 'longitude_deg': b,
                             'acma_site_id': site['acma_site_id'],
                             'terrain_egm96_m': ground, 'antenna_agl_m': agl,
                             'geoid_undulation_m': geoid_height,
                             'height_ellipsoid_m': ground+agl+geoid_height,
                             'prior_sigma_enu_m': [30., 30., 75.],
                             'position_source': site['position_reference'],
                             'height_status': site['height_reference']})
    return np.array(tx), {'receiver': description, 'transmitters': descriptions,
        'enu_transmitters_m':np.array(tx).tolist(), 'receiver_ecef_m':receiver.tolist(),
        'ecef_to_enu':rotation.tolist(), 'transmitter_terrain': terrain_description,
        'terrain_status': 'Transmitter ground: cached GA one-arcsecond DEM, EGM96, not native DTED2. Receiver ground: incumbent one-arcsecond mosaic.',
        'site_configuration': {key: metadata[key] for key in ('transmitters', 'signals')},
        'transmitter_height_model': 'one metadata-specified phase centre per site'}
