"""Compare exact coherent backprojection with nominal and fitted channel registration.

Both images use the same physical motion, samples and fixed colour scale. This
is a diagnostic of a conditional fit, not a validated absolute hull image.
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
from combine_images import combine
from imaging import corrected_sites, geometry, read_image, sites, write_input
from motion import C
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'aperture', 'ais', 'geometry-directory', 'work', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    p.add_argument('--comparison-motion', type=Path, help='Compare a complete converged control model with this complete trial on identical original rows')
    p.add_argument('--title', default='Joint antenna locations, motion, gains and delays · trial only')
    p.add_argument('--comparison-title', default='Fitted motion, gains and delays · nominal antenna locations')
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    if not model['coherent_motion']['converged']:
        raise ValueError('Finish the numerical convergence check before rendering this candidate')
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    variants = [('nominal', 'Nominal channel alignment', model, False),
                ('registered', 'Fitted channel range origins and complex gains', model, True)]
    if args.comparison_motion:
        control = json.loads(args.comparison_motion.read_text())
        if not control['coherent_motion']['converged']:
            raise ValueError('Comparison model has not passed numerical convergence')
        for key in ('interval_s', 'nominal_start_utc', 'reference_height_ellipsoid_m'):
            if control[key] != model[key]:
                raise ValueError('The comparison must use the same aperture and coordinate reference')
        variants = [('control', args.comparison_title, control, True),
                    ('geometry', args.title, model, True)]
    x, y = np.arange(-220, 221, 2.), np.arange(-76, 77, 2.)
    xx, yy = np.meshgrid(x, y)
    pixels = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 20.), np.zeros(xx.size)]
    args.work.mkdir(parents=True, exist_ok=True)
    results, records = [], []
    channel_maps = []
    for label, title, current, registered in variants:
        if np.any(current.get('channel_path_rate_bias_mps', 0.)):
            raise ValueError('Validate GPU path-rate registration before rendering this experiment')
        directory = args.work/label
        directory.mkdir(parents=True, exist_ok=True)
        curve = SplineMotion(current)
        gains = np.array([complex(*z) for z in current['channel_gains_real_imag']])
        actual_tx, actual_rx = corrected_sites(tx, rx, current.get('antenna_fit') if registered else None)
        images, information = [], []
        for channel in range(5):
            with np.load(args.aperture/f'channel-{channel}.npz') as f:
                time = f['time_s']
                keep = (time >= curve.interval[0]+.02)&(time < curve.interval[1]-.02)
                time, moments = time[keep], f['moments'][:, keep]
                meta = json.loads(str(f['metadata_json']))
                weight = f['information'][keep]*f['valid'][keep]
            rf = meta['channel_configuration']['rf_hz']
            packed, _ = geometry(track, epoch+time, actual_tx[channel], actual_rx, rf, meta['first_range_sample'], weight,
                attitude=current['attitude_prior'], motion=curve, motion_epoch=epoch,
                reference_transmitter=tx[3], reference_receiver=rx)
            if registered:
                # Envelope origin is independent of the fitted carrier gain.
                # Do not also add k*bias to RF phase: that would count it twice.
                packed[:, 9, 0] += current['range_bias_m'][channel]*8e6/C
            source = directory/f'channel-{channel}.bin'
            target = directory/f'channel-{channel}-image.bin'
            write_input(source, packed, moments, pixels, rf)
            subprocess.run([str(args.executable), str(source), str(target)], check=True)
            value = read_image(target, len(pixels))
            if np.any(value[:, 2] <= 0) or np.ptp(value[:, 2])/value[:, 2].mean() > 1e-5:
                raise ValueError('Pixel leaves the retained range support')
            images.append((value[:, 0]+1j*value[:, 1])/value[:, 2])
            information.append(value[:, 2])
            np.savez(directory/f'channel-{channel}-image.npz', along_m=x, port_m=y, time_s=time,
                complex_image=images[-1].reshape(xx.shape), information=value[:, 2].reshape(xx.shape))
        (directory/'image-description.json').write_text(json.dumps({
            'motion_model': current, 'title': title, 'interval_s': current['interval_s'],
            'channel_gains_real_imag': np.c_[gains.real, gains.imag].tolist() if registered else [[1., 0.]]*5,
            'range_origins_applied': registered, 'operator': 'Original-row exact physical backprojection; rectangular aperture; no DC notch',
            'frame_origin': 'AIS antenna prior', 'absolute_heading_validated': False}, indent=2)+'\n')
        image, total, _ = combine(np.asarray(images), np.asarray(information), gains if registered else None)
        power = 8192*total*abs(image)**2
        results.append(10*np.log10(np.maximum(power.reshape(xx.shape), 1e-30)))
        channel_maps.append(10*np.log10(np.maximum(8192*np.array(information)*abs(np.array(images))**2, 1e-30)).reshape(5, *xx.shape))
        np.savez(args.work/f'{label}-coherent.npz', along_m=x, port_m=y,
            complex_image=image.reshape(xx.shape), information=total.reshape(xx.shape))
        records.append({'mode': label, 'title': title, 'maximum_conditional_matched_power_db': float(results[-1].max()),
            'transmitter_ecef_m': actual_tx.tolist(), 'receiver_ecef_m': actual_rx.tolist(),
            'individual_channel_peak_db': channel_maps[-1].max(axis=(1, 2)).tolist()})
    upper = max(z.max() for z in results)
    lower = upper-45.
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout='constrained')
    for ax, values, (_, title, _, _) in zip(axes, results, variants):
        im = ax.imshow(values, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='equal',
            cmap='viridis', vmin=lower, vmax=upper)
        ax.set(xlabel='Forward coordinate of fitted body frame (m)', ylabel='Port coordinate (m)', title=title)
    fig.colorbar(im, ax=axes, label='Conditional coherent matched power (dB)', shrink=.8)
    fig.suptitle('Five-channel complex backprojection · 142–172 s\nSame original samples and fixed colour scale; no DC notch')
    fig.supxlabel('Trial reconstruction. Frame origin remains the AIS antenna prior; absolute heading and hull centroid are unvalidated.', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True, layout='constrained')
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC', 'Coherent combination']
    last = np.concatenate([channel_maps[-1], results[-1][None]])
    for ax, values, title in zip(axes.ravel(), last, names):
        im = ax.imshow(values, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='equal',
            cmap='viridis', vmin=lower, vmax=upper)
        ax.set(title=title, xlabel='Forward (m)', ylabel='Port (m)')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Conditional matched power (dB)', shrink=.8)
    fig.suptitle('Independent channels and actual coherent combination · 30 s aperture\nTrial calibration; coordinates relative to the AIS antenna prior, not a measured hull centroid')
    fig.savefig(args.output.with_name(args.output.stem+'-channels.png'), dpi=150)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps({'source_motion': str(args.motion), 'scope': __doc__,
        'comparison_motion': str(args.comparison_motion) if args.comparison_motion else None,
        'colour_scale_db': [float(lower), float(upper)], 'results': records}, indent=2)+'\n')


if __name__ == '__main__':
    main()
