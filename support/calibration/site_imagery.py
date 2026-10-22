"""Plot the actual optimiser priors on georeferenced public aerial imagery.

Download once with --download, then reproduce the figures offline from cache.
Markers come from the model ECEF coordinates, not a geocoder or fitted result.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

from terrain import geodetic

SERVICE = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
NAMES = {'12013': 'Ornata Road', '12015': 'Eyre Road', '12014': 'Ridge Road'}
SLUGS = {'12013': 'ornata', '12015': 'eyre', '12014': 'ridge'}
RADIUS = 6378137.


def mercator(latitude, longitude):
    latitude, longitude = np.radians([latitude, longitude])
    return RADIUS*np.array([longitude, np.log(np.tan(np.pi/4+latitude/2))])


def priors(directory):
    prior = json.loads((directory/'geometry-prior.json').read_text())
    main = json.loads((directory/'sites-main.json').read_text())
    licensed = json.loads((directory/'acma-transmitters.json').read_text())
    absolute = np.array(prior['receiver_ecef_m'])+np.array(main['site_positions_enu_m'])@prior['ecef_to_enu']
    result = []
    for site_id, position in zip(('12013', '12015', '12014'), absolute):
        latitude, longitude, height = map(float, geodetic(position))
        site = licensed['sites'][site_id]
        if 'site_configuration' in prior:
            configured = next(s for s in prior['site_configuration']['transmitters'] if s['acma_site_id']==site_id)
            expected_lat, expected_lon = configured['location']['latitude_deg'], configured['location']['longitude_deg']
        else:
            expected_lat, expected_lon = float(site['LATITUDE']), float(site['LONGITUDE'])
        if max(abs(latitude-expected_lat), abs(longitude-expected_lon)) > 1e-9:
            raise ValueError('Model prior differs from its declared source location; inspect before rendering')
        channels = [{'callsign': d['CALL_SIGN'], 'role': d['STATION_TYPE'], 'height_agl_m': float(d['HEIGHT'])}
                    for d in licensed['devices'] if d['SITE_ID']==site_id and d['CALL_SIGN'] in
                    ('HSV6', 'SBS7', 'GTV8', 'ATV11', 'ABC12')]
        result.append({'site_id': site_id, 'slug': SLUGS[site_id], 'name': NAMES[site_id],
            'latitude_deg': latitude, 'longitude_deg': longitude,
            'configured_agl_m':configured['antenna_agl_m'] if 'site_configuration' in prior else None,
            'model_base_phase_centre_ellipsoid_m': height, 'licensed_site_description': site['NAME'],
            'licensed_position_precision': site['SITE_PRECISION'], 'channels': channels})
    return result


def views(sites):
    result = []
    for site in sites:
        centre = mercator(site['latitude_deg'], site['longitude_deg'])
        scale = np.cos(np.radians(site['latitude_deg']))
        half = 180./scale
        result.append({'slug': site['slug'], 'centre_mercator_m': centre.tolist(),
                       'latitude_deg': site['latitude_deg'], 'longitude_deg': site['longitude_deg'],
                       'bbox': [*(centre-half), *(centre+half)], 'pixels': 1536})
    latitude = np.mean([s['latitude_deg'] for s in sites]);longitude = np.mean([s['longitude_deg'] for s in sites])
    centre = mercator(latitude, longitude);half = 400./np.cos(np.radians(latitude))
    result.append({'slug': 'overview', 'centre_mercator_m': centre.tolist(), 'latitude_deg': latitude,
                   'longitude_deg': longitude, 'bbox': [*(centre-half), *(centre+half)], 'pixels': 2048})
    return result


def retrieve(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'dvbt-passive-radar-site-review/1.0'})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def download(view, cache):
    export = {'bbox': ','.join(map(str, view['bbox'])), 'bboxSR': 3857, 'imageSR': 3857,
              'size': f"{view['pixels']},{view['pixels']}", 'format': 'jpg', 'dpi': 96, 'f': 'json'}
    url = SERVICE+'/export?'+urllib.parse.urlencode(export)
    description = json.loads(retrieve(url))
    if 'error' in description:
        raise RuntimeError(description['error'])
    raw = retrieve(description['href'])
    (cache/(view['slug']+'.jpg')).write_bytes(raw)
    description['request_url'] = url
    description['downloaded_utc'] = datetime.now(timezone.utc).isoformat()
    description['sha256'] = hashlib.sha256(raw).hexdigest()
    (cache/(view['slug']+'-export.json')).write_text(json.dumps(description, indent=2)+'\n')
    metadata_query = {'geometry': f"{view['longitude_deg']},{view['latitude_deg']}",
        'geometryType': 'esriGeometryPoint', 'inSR': 4326, 'spatialRel': 'esriSpatialRelIntersects',
        'outFields': '*', 'returnGeometry': 'false', 'f': 'json'}
    metadata_url = SERVICE+'/9/query?'+urllib.parse.urlencode(metadata_query)
    metadata = json.loads(retrieve(metadata_url))
    if 'error' in metadata:
        raise RuntimeError(metadata['error'])
    metadata['request_url'] = metadata_url
    (cache/(view['slug']+'-metadata.json')).write_text(json.dumps(metadata, indent=2)+'\n')
    print('Downloaded imagery and metadata:', view['slug'], flush=True)


def source_description(cache, slug):
    rows = json.loads((cache/(slug+'-metadata.json')).read_text()).get('features', [])
    result = []
    for row in rows:
        a = row['attributes']
        timestamp = a.get('SRC_DATE2')
        date = datetime.fromtimestamp(timestamp/1000, timezone.utc).date().isoformat() if timestamp else str(a.get('SRC_DATE', 'unknown'))
        result.append({'provider': a.get('NICE_NAME'), 'description': a.get('NICE_DESC'),
                       'capture_date': date, 'resolution_m': a.get('SRC_RES'),
                       'accuracy_m': a.get('SRC_ACC')})
    return result


def panel(ax, view, sites, cache, *, close=False):
    description = json.loads((cache/(view['slug']+'-export.json')).read_text())
    scale = np.cos(np.radians(view['latitude_deg']));centre = np.array(view['centre_mercator_m'])
    extent = description['extent']
    low = (np.array([extent['xmin'], extent['ymin']])-centre)*scale
    high = (np.array([extent['xmax'], extent['ymax']])-centre)*scale
    image = np.asarray(Image.open(cache/(view['slug']+'.jpg')))
    if image.shape[:2] != (description['height'], description['width']):
        raise ValueError('Returned image dimensions disagree with its geographic extent')
    ax.imshow(image, extent=[low[0], high[0], low[1], high[1]], origin='upper', interpolation='none')
    for site in sites:
        if view['slug'] != 'overview' and site['slug'] != view['slug']:
            continue
        xy = (mercator(site['latitude_deg'], site['longitude_deg'])-centre)*scale
        if np.any(xy < low) or np.any(xy > high):
            continue
        mark, = ax.plot(*xy, '+', color='#fff05a', ms=16 if close else 13, mew=1.8)
        mark.set_path_effects([effects.Stroke(linewidth=3.5, foreground='black'), effects.Normal()])
        ax.add_patch(Circle(xy, 30., fill=False, color='#00e4ff', linewidth=1., linestyle=(0, (5, 4))))
        if view['slug']=='overview':
            label = ax.annotate(site['name'], xy, xytext=(10, 10), textcoords='offset points',
                                color='#fff05a', fontsize=10, fontweight='bold')
            label.set_path_effects([effects.Stroke(linewidth=2.8, foreground='black'), effects.Normal()])
    limit = 65. if close else (high[0]-low[0])/2
    ax.set_xlim(-limit, limit);ax.set_ylim(-limit, limit);ax.set_aspect('equal');ax.set_axis_off()
    bar = 20. if close else 100. if view['slug']=='overview' else 50.
    x, y = -limit*.87, -limit*.86
    line, = ax.plot([x, x+bar], [y, y], color='white', lw=3)
    line.set_path_effects([effects.Stroke(linewidth=5, foreground='black'), effects.Normal()])
    text = ax.text(x+bar/2, y+limit*.04, f'{bar:g} m', ha='center', color='white', fontsize=9)
    text.set_path_effects([effects.Stroke(linewidth=2, foreground='black'), effects.Normal()])
    ax.annotate('N', xy=(.93, .94), xytext=(.93, .84), xycoords='axes fraction',
                color='white', ha='center', fontweight='bold',
                arrowprops={'arrowstyle': '-|>', 'color': 'white', 'lw': 1.7},
                path_effects=[effects.Stroke(linewidth=2, foreground='black'), effects.Normal()])


def render(sites, view_list, cache, output, geometry):
    attribution = 'Imagery: Esri, Vantor, Earthstar Geographics and the GIS User Community'
    pages = []
    for site, view in zip(sites, view_list):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout='constrained')
        panel(axes[0], view, sites, cache);panel(axes[1], view, sites, cache, close=True)
        axes[0].set_title('360 m context');axes[1].set_title('130 m detail · same georeferenced imagery')
        fig.suptitle(f"{site['name']} · ACMA site {site['site_id']}\n"
                     f"Optimiser prior: {site['latitude_deg']:.6f}, {site['longitude_deg']:.6f}", fontsize=15)
        source = source_description(cache, site['slug'])
        source_text = '; '.join(f"{s['provider']} · captured {s['capture_date']} · resolution {s['resolution_m']} m · reported accuracy {s['accuracy_m']} m" for s in source)
        fig.supxlabel('Yellow cross: nominal model position · cyan ring: 30 m horizontal prior scale, not a confidence boundary\n'
                      'Tall mast tops can be displaced from their bases in oblique imagery; the marker is the vertical ground projection.\n'
                      +source_text+'\n'+attribution+' · Based on Australian Communications and Media Authority information', fontsize=8)
        name = site['slug']+'-prior.png';fig.savefig(output/name, dpi=160);plt.close(fig)
        pages.append(f'<h2>{html.escape(site["name"])}</h2><a href="{name}"><img src="{name}"></a>')
        site['imagery_source'] = source
    view = view_list[-1]
    fig, ax = plt.subplots(figsize=(10, 10), layout='constrained');panel(ax, view, sites, cache)
    ax.set_title('Mount Dandenong · all three transmitter position priors', fontsize=14)
    fig.supxlabel('Yellow crosses: actual optimiser priors · cyan rings: 30 m horizontal prior scale\n'
                  +attribution+'\nBased on Australian Communications and Media Authority information', fontsize=8)
    fig.savefig(output/'overview.png', dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), layout='constrained')
    for site, view, ax in zip(sites, view_list, axes):
        panel(ax, view, sites, cache, close=True)
        ax.set_title(f"{site['name']} · {site['site_id']}\n"
                     f"{site['latitude_deg']:.6f}, {site['longitude_deg']:.6f}", fontsize=11)
    fig.suptitle('Transmitter priors on satellite imagery · 130 m detail at each site', fontsize=15)
    fig.supxlabel('Yellow cross: exact optimiser prior · cyan ring: 30 m horizontal prior scale · north upwards\n'
                  +attribution+'\nBased on Australian Communications and Media Authority information', fontsize=8)
    fig.savefig(output/'three-sites.png', dpi=170);plt.close(fig)
    rows = ''.join(f'<tr><td>{s["name"]}</td><td>{s["site_id"]}</td><td>{s["latitude_deg"]:.9f}</td>'
                   f'<td>{s["longitude_deg"]:.9f}</td><td>{s["configured_agl_m"] if s["configured_agl_m"] is not None else "Historical licensed heights"}</td><td>'+', '.join(f'{c["callsign"]}: {c["height_agl_m"]:g} m ({c["role"]})' for c in s['channels'])+'</td></tr>' for s in sites)
    updated = any(s['configured_agl_m'] is not None for s in sites)
    position_source = ('Coordinates and shared per-site AGL heights are the user-supplied priors of 16 October 2026. Ground heights use the cached Geoscience Australia one-arcsecond DEM, EGM96; this is not native DTED2.' if updated else 'All three numerical coordinates agree with the cached ACMA site records.')
    body = f'''<!doctype html><meta charset="utf-8"><title>Transmitter prior imagery</title>
<style>body{{font:17px/1.5 system-ui;max-width:1450px;margin:2em auto;padding:0 1em;color:#222}}img{{width:100%}}td,th{{padding:.5em;text-align:left;border-bottom:1px solid #ddd}}.note{{padding:1em;background:#fff2cc}}</style>
<h1>Satellite/aerial views of the transmitter priors</h1>
<p>Each yellow marker is calculated from the exact ECEF coordinates in the geometry snapshot, converted back to latitude/longitude. It is not a geocoded street address or a fitted position. {position_source} The cyan ring shows a 30 m horizontal prior scale; it is not a surveyed error bound or a two-dimensional confidence contour.</p>
<p class="note">The previous fits did <b>not</b> fix the receiver latitude/longitude. They retained 15/15/7.5 m ENU prior standard deviations with ±3-standard-deviation parameter bounds. Both range-inclusive candidates put the receiver vertical correction at −22.5 m; the Ornata-SBS candidate also moved it approximately 24.4 m south. These bounded diagnostic fits lack a physically valid receiver-height constraint and are not accepted antenna positions.</p>
<a href="overview.png"><img src="overview.png"></a>
<p><a href="three-sites.png">Three-site detail comparison</a></p>
<table><tr><th>Site</th><th>ACMA ID</th><th>Latitude used</th><th>Longitude used</th><th>Configured AGL (m)</th><th>Historical licence records (not new height priors)</th></tr>{rows}</table>
{''.join(pages)}
<h2>Interpretation and reproducibility</h2>
<p>The marker represents the horizontal coordinate of the assumed vertical antenna axis. A tall mast may appear to lean in aerial imagery, and its shadow is not its base. Imagery alone cannot establish the electrical phase centre or whether an alternative antenna was transmitting. No transmitter coordinate was changed by this rendering.</p>
<p>The historical model directly used the numerical ACMA latitude/longitude values in its WGS84 ECEF calculation. A rigorous Australian-datum/epoch conversion has not been applied in this model; the views deliberately show the coordinates actually used. Imagery geolocation and tall-object displacement also limit metre-scale interpretation.</p>
<p>Imagery: <a href="{SERVICE}">Esri World Imagery</a>; acquisition metadata is queried at each site and listed on its image. The image acquisition date is separate from its download date. Based on <a href="https://www.acma.gov.au/radiocomms-licence-data">Australian Communications and Media Authority information</a>, using the cached RRL snapshot of 13 October 2026.</p>
<p><a href="site-priors.json">Exact marker coordinates, source imagery metadata and input hashes</a></p>'''
    (output/'index.html').write_text(body)
    manifest = {'sites': sites, 'geometry_directory': str(geometry), 'imagery_service': SERVICE,
        'source_images': {view['slug']: json.loads((cache/(view['slug']+'-export.json')).read_text())
                          for view in view_list},
        'views': view_list, 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [geometry/'geometry-prior.json', geometry/'sites-main.json', geometry/'acma-transmitters.json']}}
    (output/'site-priors.json').write_text(json.dumps(manifest, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261016_sites'))
    parser.add_argument('--cache', type=Path, default=Path('build/site-imagery-20261016'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261016/transmitter-updated-imagery'))
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args();args.cache.mkdir(parents=True, exist_ok=True);args.output.mkdir(parents=True, exist_ok=True)
    site_list = priors(args.geometry);view_list = views(site_list)
    if args.download:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda view: download(view, args.cache), view_list))
    render(site_list, view_list, args.cache, args.output, args.geometry)
    print(args.output/'index.html', flush=True)


if __name__ == '__main__':
    main()
