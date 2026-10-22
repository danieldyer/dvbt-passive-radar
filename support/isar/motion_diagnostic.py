"""Residual path-rate spectra after the explicit AIS prior, for diagnosis only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from field import Field
from motion import excess_path
from spline_motion import SplineMotion
from imaging import sites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, default=95.)
    parser.add_argument('--end', type=float, default=206.)
    parser.add_argument('--duration', type=float, default=8.)
    parser.add_argument('--step', type=float, default=1.)
    parser.add_argument('--motion', type=Path, help='Overlay the fitted physical centre path rate, without changing observations')
    args = parser.parse_args()
    rates = np.linspace(-6, 6, 301)
    centres = np.arange(args.start, args.end, args.step)
    duration = args.duration
    if duration <= 0 or args.step <= 0 or not len(centres):
        raise ValueError('Positive window duration/step and a nonempty interval are required')
    model = None if args.motion is None else json.loads(args.motion.read_text())
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), layout='constrained', sharex=True, sharey=True)
    result = {'scope': 'Diagnostic nonuniform-time Fourier spectra of the fixed whitened observations after AIS deramp; no estimated correction applied.',
              'coherence_s': duration, 'rates_mps': rates.tolist(), 'time_s': centres.tolist(), 'channels': []}
    for channel, name in enumerate(['Seven', 'SBS', 'Nine', 'Ten', 'ABC']):
        with np.load(args.fields/f'channel-{channel}.npz') as loaded:
            d = dict(loaded)
        field = Field(d, d['pulse_samples'])
        power = []
        for time in centres:
            selected = abs(field.time-time) < duration/2
            t = field.time[selected]-time
            window = .5+.5*np.cos(2*np.pi*t/duration)
            transform = (np.exp(1j*field.k*rates[:, None]*t[None, :])*window)@field.y[selected]
            power.append(np.mean(abs(transform)**2, axis=1)/np.sum(window**2))
        power = np.array(power)
        corner_rates = []
        for along in (-143.265, 143.265):
            for port in (-20., 20.):
                displacement = np.einsum('nij,j->ni', d['body_to_ecef'], [along, port, 20.])
                velocity = d['centre_velocity_ecef_mps']+np.gradient(displacement, d['row_time'], axis=0)
                _, gradient = excess_path(d['centre_ecef_m']+displacement, d['transmitter_ecef_m'], d['receiver_ecef_m'])
                rate = np.sum(gradient*velocity, axis=1)+d['centre_omega']/field.k/.001024
                corner_rates.append(np.interp(centres, d['row_time'], rate))
        corner_rates = np.array(corner_rates)
        db = 10*np.log10(np.maximum(power, 1e-12))
        artist = axes[channel].pcolormesh(centres, rates, db.T, shading='auto', vmin=0, vmax=25, cmap='magma')
        axes[channel].plot(centres, corner_rates.min(axis=0), color='cyan', lw=1.)
        axes[channel].plot(centres, corner_rates.max(axis=0), color='cyan', lw=1.)
        stationary_rate = np.interp(centres, d['row_time'], d['centre_omega']/field.k/.001024)
        axes[channel].plot(centres, stationary_rate, color='lime', ls='--', lw=1.)
        fitted_rate = None
        if model is not None:
            def sample(value):
                shape = value.shape[1:]
                flat = value.reshape(len(d['row_time']), -1)
                return np.column_stack([np.interp(centres, d['row_time'], v) for v in flat.T]).reshape((len(centres),)+shape)
            frame_rate = np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)
            prior = {'position_ecef_m': sample(d['centre_ecef_m']), 'velocity_ecef_mps': sample(d['centre_velocity_ecef_mps']),
                     'body_to_ecef': sample(d['body_to_ecef']), 'frame_velocity': sample(frame_rate)}
            tx, rx, _ = sites(Path(model['geometry_directory']))
            moved = SplineMotion(model).apply(centres, prior, tx[3], rx)
            _, gradient = excess_path(moved['position_ecef_m'], d['transmitter_ecef_m'], d['receiver_ecef_m'])
            fitted_rate = np.sum(gradient*moved['velocity_ecef_mps'], axis=1)+stationary_rate
            axes[channel].plot(centres, fitted_rate, color='white', lw=1.)
        axes[channel].set_ylabel(name+'\nResidual rate (m/s)')
        result['channels'].append({'channel': channel, 'peak_residual_rate_mps': rates[power.argmax(axis=1)].tolist(),
                                   'prior_hull_rate_min_mps': corner_rates.min(axis=0).tolist(),
                                   'prior_hull_rate_max_mps': corner_rates.max(axis=0).tolist(),
                                   'stationary_residual_rate_mps': stationary_rate.tolist(),
                                   'fitted_centre_residual_rate_mps': None if fitted_rate is None else fitted_rate.tolist()})
        print(name, 'peak-rate p10/50/90', np.percentile(rates[power.argmax(axis=1)], [10, 50, 90]), flush=True)
    fig.colorbar(artist, ax=axes, label='Range-mean spectral energy / conditional thermal level (dB)', shrink=.7)
    axes[-1].set_xlabel('Recording time (s)')
    fig.suptitle(f'Ship residual path-rate spectrum after AIS translation removal\nCyan: course-aligned hull; green: stationary; white: fitted centre; {duration:g} s Hann diagnostic')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
