"""Delay-scale tests for the ship's companion return; no ghost removal.

The flat-water image construction is a specular two-ray hypothesis, not a full
sea propagation model. The spectral PSF uses the archived filter power weights;
finite DVB-T waveform and distributed-target sidelobes require separate checks.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

from ais import read_csv, utc_seconds
from imaging import course_basis, sites
from motion import C, enu_basis, excess_path, path_difference

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'calibration'))
from phase_history import index_archive, exact
from candidates import load, NAMES
from geographic_io import read_profiles
from range_fit import Pulse


def spectrum_at(capture, archive, channel, time):
    record = min((r for r in archive['records'] if r[1] == channel),
                 key=lambda r: abs((r[2]+r[3])/2-time))
    with capture.open('rb') as source:
        source.seek(record[0])
        tag, size, payload = struct.unpack('<4sIQ', exact(source, 16))
        meta = json.loads(exact(source, size))
        if tag != b'PROF' or meta['response_carriers'] != 6817 or meta['noise_carriers'] != 6817:
            raise ValueError('Missing archived matched-filter spectrum')
        source.seek(8*(meta['count']*meta['moments']*meta['size']+meta['stationary_carriers']), 1)
        h = np.frombuffer(exact(source, 6817*8), '<c8').astype(complex)
        variance = np.frombuffer(exact(source, 6817*4), '<f4').astype(float)
    weight = abs(h)**2/variance
    return weight/weight.sum()


def flat_sea_extra_path(distance, antenna_height, scatterer_height):
    direct = np.hypot(distance, antenna_height-scatterer_height)
    reflected = np.hypot(distance, antenna_height+scatterer_height)
    return 4*antenna_height*scatterer_height/(direct+reflected)


def symbol_psf_summary(templates, interval=(58., 88.)):
    """Zero-relative-Doppler response including reconstructed symbol powers."""
    header, records = read_profiles(templates)
    if header.get('psf_samples') != 129:
        raise ValueError('Reconstructed-symbol autocorrelations are required')
    delay = np.arange(350., 751.)
    channels = []
    for channel, name in enumerate(NAMES):
        selected = records[(records['channel'] == channel)&(records['time'] >= interval[0])&
                           (records['time'] < interval[1])&(records['information'] > 0)]
        if not len(selected):
            raise ValueError('Missing channel template coverage')
        mean = np.average(selected['psf'], weights=selected['information'], axis=0)
        response = Pulse(mean)(delay*8e6/C)/mean[64].real
        channels.append({'channel': name, 'blocks': len(selected),
                         'maximum_response_350_to_750_m_db': float(20*np.log10(abs(response).max()))})
    return {'source': str(templates), 'interval_s': list(interval), 'channels': channels,
            'scope': 'Information-weighted circular point autocorrelation from replayed reconstructed DVB-T symbol powers, sampled over 350–750 m in one-metre steps at zero relative Doppler. This improves on the mean symbol-energy spectrum check, but is not a complete finite-waveform ambiguity surface, a distributed-target model or an unbiased companion RCS measurement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--peaks', type=Path, required=True)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--geometry-directory', type=Path, required=True)
    parser.add_argument('--nominal-start-utc', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive = index_archive(args.capture, args.index)
    _, maps = load(args.peaks)
    epoch = utc_seconds(args.nominal_start_utc)
    track = read_csv(args.ais, reference_height_ellipsoid_m=5, mmsi=563052200)
    time = np.arange(58, 88, .25)
    state = track.sample(epoch+time, max_gap_s=180)
    centre = state['position_ecef_m']
    basis = course_basis(state)
    tx, rx, _ = sites(args.geometry_directory)
    normal = enu_basis(state['latitude'], state['longitude'])[:, :, 2]
    # Conservative point heights for a simple delay-scale check. These are not
    # estimated reflector heights and are not fed into image formation.
    heights = np.array([10., 20., 40., 60.])
    hull = np.array([[x, y, 0] for x in (-286.53/2, 286.53/2) for y in (-20., 20.)])
    report = {'interval_s': [58, 88], 'flat_water_model': 'image construction at fixed nominal heights',
              'transmitter_height_hypothesis_m': 750, 'receiver_height_hypothesis_m': 50,
              'ship_height_hypotheses_m': heights.tolist(),
              'hull_hypothesis': '286.53 m by 40 m rectangle aligned to course; not measured true heading',
              'channels': []}
    offsets = np.linspace(-1600, 1600, 1601)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 2, figsize=(13, 14), sharex='col', layout='constrained')
    for ch, name in enumerate(NAMES):
        path, _ = excess_path(centre, tx[ch], rx)
        displacement = np.einsum('nij,kj->nki', basis, hull)
        hull_delta = path_difference(centre[:, None], displacement, tx[ch], rx)
        hull_spread = np.ptp(hull_delta, axis=1)
        extra = np.zeros((len(time), len(heights)))
        for antenna, h in ((tx[ch], 750), (rx, 50)):
            vector = centre-antenna
            horizontal = np.sqrt(np.maximum(0, np.sum(vector**2, axis=1)-np.sum(vector*normal, axis=1)**2))
            extra += flat_sea_extra_path(horizontal[:, None], h, heights[None, :])
        selected = []
        for frame in maps[ch]:
            if not 58 <= frame['time'] < 88:
                continue
            s = track.sample([epoch+frame['time']], max_gap_s=180)
            length, gradient = excess_path(s['position_ecef_m'], tx[ch], rx)
            rate = float(np.sum(gradient*s['velocity_ecef_mps']))
            candidates = [(p['range_m']-float(length[0]), -C*p['doppler_hz']/frame['rf_hz'], p)
                          for p in frame['peaks'] if abs(-C*p['doppler_hz']/frame['rf_hz']-rate) < 1.0]
            direct = [q for q in candidates if -180 < q[0] < 180]
            delayed = [q for q in candidates if 350 < q[0] < 750]
            for delta, speed, peak in candidates:
                if abs(delta) < 1600:
                    axes[ch, 0].scatter(frame['time'], delta, c=[peak['selection_snr_db']],
                                        vmin=14, vmax=35, cmap='viridis', s=12)
            if direct and delayed:
                a = max(direct, key=lambda q: q[2]['selection_snr_db'])
                b = max(delayed, key=lambda q: q[2]['selection_snr_db'])
                selected.append([frame['time'], b[0]-a[0], b[1]-a[1],
                                 10*np.log10(b[2]['power']/a[2]['power'])])
        weight = spectrum_at(args.capture, archive, ch, 73)
        frequency = np.arange(-3408, 3409)*8e6/8192
        psf = np.concatenate([np.exp(2j*np.pi*offsets[i:i+128, None]/C*frequency)@weight
                              for i in range(0, len(offsets), 128)])
        axes[ch, 1].plot(offsets, 20*np.log10(np.maximum(abs(psf), 1e-8)))
        axes[ch, 1].axvspan(350, 750, color='#b32119', alpha=.1)
        axes[ch, 1].axvspan(-750, -350, color='#b32119', alpha=.1)
        axes[ch, 0].axhline(0, color='#b32119', lw=1)
        axes[ch, 0].set(ylabel=name+'\nPath relative to AIS (m)', ylim=(-1600, 1600))
        axes[ch, 1].set(ylabel='Spectral point PSF (dB)', ylim=(-80, 3))
        entry = {'channel': name, 'paired_frames': len(selected),
                 'course_aligned_hull_path_spread_m': [float(hull_spread.min()), float(hull_spread.max())],
                 'double_sea_bounce_extra_path_max_m': float(extra.max()),
                 'spectral_psf_max_350_to_750_m_db': float(20*np.log10(abs(psf[(offsets > 350)&(offsets < 750)]).max()))}
        if selected:
            values = np.array(selected)
            entry.update({'median_measured_companion_separation_m': float(np.median(values[:, 1])),
                          'median_companion_minus_direct_path_rate_mps': float(np.median(values[:, 2])),
                          'median_companion_relative_power_db': float(np.median(values[:, 3])),
                          'paired_observations': selected})
        report['channels'].append(entry)
    for ax in axes[:, 0]:
        ax.set_xlim(58, 88)
    axes[-1, 0].set_xlabel('Capture time (s); unnotched 3 s maps')
    axes[-1, 1].set_xlabel('Offset from a single point return (m)')
    fig.suptitle('Companion-return delay checks\nLeft: measured peaks near the AIS rate. Right: archived filter spectral PSF.\n'
                 'Spectral PSF alone does not include finite waveform ambiguity or distributed scattering.', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'companion-delay-tests.png', dpi=140)
    (args.output/'companion-delay-tests.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({**report, 'channels': [{k: v for k, v in x.items() if k != 'paired_observations'} for x in report['channels']]}, indent=2))


if __name__ == '__main__':
    main()
