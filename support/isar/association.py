"""Plot AIS-predicted path/rate against measured radar peaks before autofocus.

Uses licensed site hypotheses, not fitted phase centres. This first diagnostic
does not estimate an arbitrary delay, time shift or scattering offset to force
agreement. AIS height is an explicit reference-plane hypothesis; no attitude is
inferred from the supplied course.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from ais import read_csv, utc_seconds
from motion import C, excess_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'calibration'))
from candidates import load, NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--peaks', type=Path, required=True)
    parser.add_argument('--geometry-directory', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--nominal-start-utc', required=True)
    parser.add_argument('--reference-height-ellipsoid-m', type=float, required=True)
    parser.add_argument('--duration-s', type=float, default=601.8686)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    track = read_csv(args.ais, reference_height_ellipsoid_m=args.reference_height_ellipsoid_m,
                     mmsi=563052200)
    epoch = utc_seconds(args.nominal_start_utc)
    prior = json.loads((args.geometry_directory/'geometry-prior.json').read_text())
    main = json.loads((args.geometry_directory/'sites-main.json').read_text())
    receiver = np.array(prior['receiver_ecef_m'])
    rotation = np.array(prior['ecef_to_enu'])
    site_positions = np.array(main['site_positions_enu_m'])
    offsets = np.array(main['channel_offsets_enu_m'])
    transmitters = receiver+(site_positions[main['channel_sites']]+offsets)@rotation
    _, maps = load(args.peaks)
    candidate = json.loads(args.candidate.read_text())
    time = np.arange(0, args.duration_s, 1.)
    prior_motion = track.sample(epoch+time, max_gap_s=450)
    native_time = track.seconds-epoch
    native = track.sample(track.seconds, max_gap_s=450)
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 2, figsize=(15, 14), sharex=True, layout='constrained')
    report = {'ais': str(args.ais), 'positions': len(track.seconds),
              'median_interval_s': float(np.median(np.diff(track.seconds))),
              'maximum_interval_s': float(np.max(np.diff(track.seconds))),
              'true_heading_reports': int(np.isfinite(track.heading).sum()),
              'nominal_start_utc': args.nominal_start_utc,
              'start_time_is_calibrated': False,
              'reference_height_ellipsoid_m': args.reference_height_ellipsoid_m,
              'transmitter_configuration': main['channels'],
              'ais_interpolation': 'Hermite position with reported SOG/COG derivatives; uncertain prior only',
              'channels': []}
    save = {'time_s': time}
    for ch, name in enumerate(NAMES):
        path, gradient = excess_path(prior_motion['position_ecef_m'], transmitters[ch], receiver)
        rate = np.einsum('ni,ni->n', gradient, prior_motion['velocity_ecef_mps'])
        native_path, native_gradient = excess_path(native['position_ecef_m'], transmitters[ch], receiver)
        native_rate = np.einsum('ni,ni->n', native_gradient, native['velocity_ecef_mps'])
        points = np.array([[f['time'], p['range_m'], -C*p['doppler_hz']/f['rf_hz'], p['selection_snr_db']]
                           for f in maps[ch] for p in f['peaks']
                           if 13000 < p['range_m'] < 16000 and abs(p['doppler_hz']) < 6])
        a, b = axes[ch]
        if len(points):
            sizes = np.clip((points[:, 3]-12)*1.2, 3, 30)
            a.scatter(points[:, 0], points[:, 1]/1000, c=points[:, 3], vmin=14, vmax=35,
                      cmap='viridis', s=sizes, alpha=.65, linewidths=0)
            b.scatter(points[:, 0], points[:, 2], c=points[:, 3], vmin=14, vmax=35,
                      cmap='viridis', s=sizes, alpha=.5, linewidths=0)
        a.plot(time, path/1000, color='#b32119', lw=1.6, label='AIS direct-path prediction')
        b.plot(time, rate, color='#b32119', lw=1.6)
        a.scatter(native_time, native_path/1000, color='#b32119', marker='x', s=35)
        b.scatter(native_time, native_rate, color='#b32119', marker='x', s=35)
        for low, high in zip(native_time[:-1], native_time[1:]):
            if high-low > 90:
                for ax in (a, b):
                    ax.axvspan(low, high, color='grey', alpha=.09, zorder=-1)
        a.set(ylabel=name+'\nExcess path (km)', ylim=(13.4, 15.8), xlim=(0, args.duration_s))
        b.set(ylabel='Excess-path rate (m/s)', ylim=(-7, 4), xlim=(0, args.duration_s))
        observed = np.array(candidate['channels'][ch]['coarse_observations'])
        motion_d = track.sample(epoch+observed[:, 0], max_gap_s=450)
        ld, jd = excess_path(motion_d['position_ecef_m'], transmitters[ch], receiver)
        vd = np.einsum('ni,ni->n', jd, motion_d['velocity_ecef_mps'])
        report['channels'].append({'name': name,
            'candidate': candidate['label'], 'candidate_observations': len(observed),
            'median_candidate_minus_ais_path_m': float(np.median(observed[:, 1]-ld)),
            'median_candidate_minus_ais_rate_mps': float(np.median(observed[:, 2]-vd)),
            'mad_candidate_minus_ais_path_m': float(np.median(abs((observed[:, 1]-ld)-np.median(observed[:, 1]-ld))))})
        save[f'{ch}_path_m'], save[f'{ch}_rate_mps'] = path, rate
    axes[0, 0].legend(fontsize=9, loc='upper right')
    for ax in axes[-1]:
        ax.set_xlabel('Capture time (s), using file-creation UTC as a nominal origin')
    fig.suptitle('MAERSK RIO DELTA AIS versus measured radar peaks\n'
                 'Red: unshifted licensed-site geometry; crosses: AIS reports; grey: AIS gaps > 90 s\n'
                 'Peak colour/size indicates screening strength. No time, range or antenna-position fit applied.', fontsize=13)
    fig.savefig(args.output/'ais-radar-association.png', dpi=140)
    plt.close(fig)
    report['status'] = 'Association diagnostic; range offset unresolved, not an accepted ship or antenna calibration.'
    (args.output/'ais-radar-association.json').write_text(json.dumps(report, indent=2))
    np.savez(args.output/'ais-prediction.npz', **save)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
