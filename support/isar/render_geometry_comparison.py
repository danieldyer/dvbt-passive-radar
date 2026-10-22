"""Compare physical motion/range/antenna models on one original-data EN grid.

Each channel is backprojected independently. A separate complex combination
uses the declared, unchanged provisional gains. No fitted scene is rendered.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv, utc_seconds
from combine_images import combine
from extract_full_aperture import load
from geographic_movie import channel_geometry, state_at
from imaging import read_image, sites, write_input
from motion import enu_basis
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from terrain import geodetic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', nargs=2, action='append', required=True, metavar=('LABEL', 'JSON'))
    p.add_argument('--aperture', type=Path, required=True)
    p.add_argument('--ais', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spacing-m', type=float, default=4.)
    p.add_argument('--extent-m', type=float, nargs=2, default=[500., 700.])
    p.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Use a new directory for the complete paired image experiment')
    paths = [Path(path) for _, path in args.model]
    models = [json.loads(path.read_text()) for path in paths]
    reference = models[0]
    for model in models:
        if not model['fit']['converged'] or not model['fixed_ne_frame'] or np.any(SplineMotion(model).yaw_coefficients):
            raise ValueError('Use completed fixed-N/E fits')
        for key in ('interval_s', 'nominal_start_utc', 'geometry_directory', 'channel_gains_real_imag'):
            if model[key] != reference[key]:
                raise ValueError('Compared models need identical observations, epoch and declared band gains')
    if args.spacing_m <= 0 or min(args.extent_m) <= 0:
        raise ValueError('Positive grid spacing and extent are required')
    tx, rx, _ = sites(Path(reference['geometry_directory']))
    midpoint = float(np.mean(reference['interval_s']))
    anchor = state_at(None, reference, SplineMotion(reference), tx, rx, np.array([midpoint]))['position_ecef_m'][0]
    latitude, longitude, _ = geodetic(anchor)
    rotation = enu_basis(*np.radians([latitude, longitude]))
    east, north = [np.arange(-length/2, length/2+args.spacing_m*.1, args.spacing_m) for length in args.extent_m]
    xx, yy = np.meshgrid(east, north)
    world = anchor+np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 20.)]@rotation.T
    apertures = [load(args.aperture/f'channel-{c}') for c in range(5)]
    gains = np.array([complex(*g) for g in reference['channel_gains_real_imag']])
    args.output.mkdir(parents=True)
    sources = [*paths, args.ais, args.executable, Path(__file__), Path('support/isar/geographic_movie.py'),
        Path('support/isar/imaging.py'), Path('support/isar/backprojection.comp')]
    result = {'scope': __doc__, 'complete': False, 'interval_s': reference['interval_s'],
        'grid_spacing_m': args.spacing_m, 'extent_m': args.extent_m, 'origin_ecef_m': anchor.tolist(),
        'enu_to_ecef': rotation.tolist(), 'origin_latitude_longitude_deg': [float(latitude), float(longitude)],
        'channel_gains_real_imag': reference['channel_gains_real_imag'], 'calibration_validated': False,
        'source_sha256': {str(path): fingerprint(path) for path in sources}, 'models': []}
    powers = []
    for index, ((label, _), model) in enumerate(zip(args.model, models)):
        curve = SplineMotion(model)
        at_midpoint = state_at(None, model, curve, tx, rx, np.array([midpoint]))
        frame = at_midpoint['body_to_ecef'][0]; centre = at_midpoint['position_ecef_m'][0]
        body = (world-centre)@frame
        np.testing.assert_allclose(centre+body@frame.T, world, atol=2e-9, rtol=0)
        pixels = np.c_[body, np.zeros(len(body))].astype('<f4')
        # Independent channel images carry no fitted band gain. Gains enter
        # once, only in the separately reported coherent combination.
        independent = {**model, 'channel_gains_real_imag': [[1., 0.]]*5}
        amplitudes, information, windows, channel_records = [], [], [], []
        for c, aperture in enumerate(apertures):
            keep = (aperture['time_s'] >= model['interval_s'][0])&(aperture['time_s'] < model['interval_s'][1])
            t = aperture['time_s'][keep]
            meta = json.loads(aperture['metadata_json']); rf = meta['channel_configuration']['rf_hz']
            state = state_at(None, model, curve, tx, rx, t)
            geom = channel_geometry(independent, state, tx, rx, c, rf, meta['first_range_sample'],
                aperture['information'][keep]*aperture['valid'][keep])
            source, target = [args.output/f'model-{index}-channel-{c}-{suffix}.bin' for suffix in ('input', 'image')]
            write_input(source, geom, aperture['moments'][:, keep], pixels, rf)
            with (args.output/f'model-{index}-channel-{c}-gpu.log').open('w') as log:
                subprocess.run([str(args.executable), str(source), str(target)], check=True, stdout=log, stderr=log)
            values = read_image(target, len(pixels))
            if not np.isfinite(values).all() or np.any(values[:, 2] <= 0) or np.ptp(values[:, 2])/values[:, 2].mean() > 1e-5:
                raise ValueError('Image has nonfinite values or leaves original range support')
            amplitude = (values[:, 0]+1j*values[:, 1])/values[:, 2]
            amplitudes.append(amplitude.reshape(xx.shape)); information.append(values[:, 2].reshape(xx.shape))
            windows.append(t[[0, -1]].tolist())
            channel_records.append({'channel': c, 'rows': len(t), 'valid_rows': int(aperture['valid'][keep].sum()),
                'information': float(values[0, 2]), 'maximum_power_db': float(10*np.log10((8192*values[:, 2]*abs(amplitude)**2).max()))})
            source.unlink(); target.unlink()
            print(json.dumps({'model': label, **channel_records[-1], 'seconds': time.monotonic()-started}), flush=True)
        amplitudes, information = np.array(amplitudes), np.array(information)
        coherent, total, efficiency = combine(amplitudes, information, gains)
        power = np.r_[8192*information*abs(amplitudes)**2, (8192*total*abs(coherent)**2)[None]]
        powers.append(10*np.log10(np.maximum(power, 1e-30)))
        np.savez(args.output/f'model-{index}-images.npz', east_m=east, north_m=north,
            complex_images=amplitudes, information=information, coherent_image=coherent,
            coherent_information=total, coherent_to_independent_energy=efficiency)
        result['models'].append({'label': label, 'source': str(paths[index]), 'channels': channel_records,
            'actual_channel_apertures_s': windows, 'antenna_fit': model.get('antenna_fit'),
            'range_bias_m': model['range_bias_m'], 'maximum_coherent_power_db': float(powers[-1][-1].max())})
        if index and windows != result['models'][0]['actual_channel_apertures_s']:
            raise ValueError('Compared images use different original rows')
        (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    samples = np.concatenate([v.ravel() for v in powers])
    edges = np.arange(-20., 30.25, .25); histogram, _ = np.histogram(samples, edges)
    low = float((edges[np.argmax(histogram)]+edges[np.argmax(histogram)+1])/2-1.5)
    high = float(np.ceil((samples.max()+.05)*10)/10)
    track = read_csv(args.ais, reference_height_ellipsoid_m=reference['reference_height_ellipsoid_m'], mmsi=563052200)
    ais_time = utc_seconds(reference['nominal_start_utc'])+np.linspace(*reference['interval_s'], 101)
    ais = (track.sample(ais_time, max_gap_s=400.)['position_ecef_m']-anchor)@rotation
    figure, axes = plt.subplots(len(models), 6, figsize=(20, 5*len(models)), squeeze=False,
        sharex=True, sharey=True, layout='constrained')
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC', 'Complex sum: provisional gains']
    for row, (power, (label, _)) in enumerate(zip(powers, args.model)):
        for c, ax in enumerate(axes[row]):
            im = ax.imshow(power[c], extent=[east[0], east[-1], north[0], north[-1]], origin='lower',
                vmin=low, vmax=high, cmap='magma', interpolation='nearest', aspect='equal')
            ax.plot(ais[:, 0], ais[:, 1], color='#00ffff', lw=1.)
            ax.plot(ais[50, 0], ais[50, 1], 'o', color='#00ffff', ms=3)
            ax.set(title=names[c], xlabel='East (m)')
            if c == 0:ax.set_ylabel(label+'\nNorth (m)')
    figure.colorbar(im, ax=axes.ravel().tolist(), label='Matched power / conditional thermal variance (dB)', shrink=.65)
    figure.suptitle('Same 30-second original data and physical grid; yaw exactly zero\nIndependent per-band BP and declared complex combination; cyan = interpolated AIS antenna track')
    figure.savefig(args.output/'comparison.png', dpi=150); plt.close(figure)
    result.update(complete=True, colour_scale_db=[low, high], seconds=time.monotonic()-started,
        limitation='Image peaks are descriptive, not independent SNR or geolocation validation. The combined panel uses inherited provisional complex gains, not newly calibrated common scattering.')
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
