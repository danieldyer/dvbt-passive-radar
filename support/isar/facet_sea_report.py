"""Declared-geometry sensitivity of a finite ship side to HH sea interaction."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from compare_ais_phase import NAMES, COLOURS
from facet_sea import response
from flash_siting_report import EPOCH
from motion import C, ecef, enu_basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reflection', type=Path, default=Path('docs/data/isar_20261016/flash-reflection/analysis.json'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261017/facet-sea'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads(args.reflection.read_text())
    geometry_path = Path(previous['physical_geometry'])/'geometry-prior.json'
    prior = json.loads(geometry_path.read_text())
    track = read_csv('docs/data/isar_20261013/rio-delta-ais.csv', reference_height_ellipsoid_m=5.)
    state = track.sample([EPOCH+164.5], max_gap_s=180)
    lat, lon = state['latitude'][0], state['longitude'][0]
    water = previous['water']
    origin = ecef(lat, lon, water['tangent_geoid_m']); basis = enu_basis(lat, lon)
    rx_ecef = np.array(prior['receiver_ecef_m'])
    tx_ecef = rx_ecef+np.array(prior['enu_transmitters_m'])@np.array(prior['ecef_to_enu'])
    rx, tx = (rx_ecef-origin)@basis, (tx_ecef-origin)@basis
    sphere = (np.array(water['centre_ecef_m'])-origin)@basis
    radius = water['radius_m']
    frequency = np.array([177.5, 184.5, 191.625, 219.5, 226.5])*1e6
    heading = previous['specular_side'][0]['side_axis_heading_deg']
    velocity = state['velocity_ecef_mps'][0]@basis
    angles = np.linspace(-1.5, 1.5, 401)
    along = np.column_stack([np.sin(np.radians(heading+angles)), np.cos(np.radians(heading+angles)), np.zeros(len(angles))])
    centre = np.broadcast_to([0., 0., 20.], along.shape)
    values = response(centre, tx[0], rx, along, [0., 0., 1.], 286.53, 30., frequency, sphere, radius)
    models = [('Direct only', values[0]), ('Direct + transmit-side sea path', values[0]+values[1]),
              ('Ideal four paths', values.sum(axis=0))]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained', sharex=True)
    for col, (name, value) in enumerate(models):
        for ch in range(5):
            axes[0, col].plot(angles, 20*np.log10(abs(value[:, ch])), color=COLOURS[ch], label=NAMES[ch])
            axes[1, col].plot(angles, np.degrees(np.angle(value[:, ch])), color=COLOURS[ch])
        axes[0, col].set(title=name, ylim=(-45, 8))
        axes[1, col].set(xlabel='Side heading offset from specular direction (degrees)', ylim=(-185, 185))
    axes[0, 0].set_ylabel('Normalized field power (dB)'); axes[1, 0].set_ylabel('Phase relative to facet centre (degrees)')
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle('Finite ship side with HH sea interactions · Ornata geometry, all five RF frequencies\n286.5 × 30 m vertical surface, centre 20 m above water; illustrative shape, no radar fitting')
    fig.savefig(args.output/'surface-sea.png', dpi=160); plt.close(fig)
    # Follow the measured AIS translation, with no fitted antenna or trajectory
    # adjustments, at several declared slow side turn rates. Use a symmetric
    # five-point phase stencil and verify by halving its temporal step.
    records = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained', sharex=True)
    for col, turn in enumerate([0., .01, .05]):
        phase_derivatives = []
        for step in [.02, .01]:
            time = step*np.arange(-2, 3)
            offsets = np.linspace(-.75, .75, 61)
            positions = np.broadcast_to(np.array([0., 0., 20.])+time[:, None]*velocity, (len(offsets), 5, 3))
            angle = np.radians(heading+offsets[:, None]+turn*time)
            axes_vector = np.stack([np.sin(angle), np.cos(angle), np.zeros_like(angle)], axis=-1)
            rays = response(positions.reshape(-1, 3), tx[0], rx, axes_vector.reshape(-1, 3), [0., 0., 1.],
                286.53, 30., frequency, sphere, radius).reshape(4, len(offsets), 5, 5)
            total = np.stack([rays[0], rays[0]+rays[1], rays.sum(axis=0)])
            phase = np.angle(total*np.conj(total[:, :, 2:3, :]))
            k = 2*np.pi*frequency/C
            rate = -(phase[:, :, 0]-8*phase[:, :, 1]+8*phase[:, :, 3]-phase[:, :, 4])/(12*step*k)
            acceleration = -(-phase[:, :, 0]+16*phase[:, :, 1]+16*phase[:, :, 3]-phase[:, :, 4])/(12*step**2*k)
            phase_derivatives.append((rate, acceleration, total[:, :, 2, :]))
        rate, acceleration, strength = phase_derivatives[-1]
        # A plotted break near an aperture null prevents an undefined effective
        # phase centre from being mistaken for a continuous physical trajectory.
        keep = abs(strength) > .03
        for ch in range(5):
            for model, style in [(0, '--'), (1, '-')]:
                mask = keep[model, :, ch]
                axes[0, col].plot(offsets, np.where(mask, rate[model, :, ch], np.nan), style, color=COLOURS[ch], label=NAMES[ch] if model == 1 else None)
                axes[1, col].plot(offsets, np.where(mask, acceleration[model, :, ch], np.nan), style, color=COLOURS[ch])
        records.append({'turn_rate_deg_s': turn, 'heading_offset_deg': offsets.tolist(),
            'models': [m[0] for m in models], 'shape_amplitude': abs(strength).tolist(),
            'additional_phase_path_rate_mps': rate.tolist(), 'additional_phase_path_acceleration_mps2': acceleration.tolist(),
            'rate_step_halving_max_difference_above_shape_0_03_mps': float(np.max(abs(rate-phase_derivatives[0][0])[keep])),
            'acceleration_step_halving_max_difference_above_shape_0_03_mps2': float(np.max(abs(acceleration-phase_derivatives[0][1])[keep]))})
        main = (abs(offsets) <= .3)[:, None]&keep[0]&keep[1]
        records[-1]['main_lobe_tx_sea_added_rate_max_mps'] = float(np.max(abs(rate[1]-rate[0])[main]))
        records[-1]['main_lobe_tx_sea_added_acceleration_max_mps2'] = float(np.max(abs(acceleration[1]-acceleration[0])[main]))
        axes[0, col].set(title=f'Declared side turn {turn:g}°/s', ylim=(-.3, .3))
        axes[1, col].set(xlabel='Heading offset from specular direction (degrees)', ylim=(-.5, .5))
    axes[0, 0].set_ylabel('Extra phase path rate (m/s)'); axes[1, 0].set_ylabel('Extra phase path acceleration (m/s²)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fixed AIS translation: what the surface itself adds to centre-path kinematics\nSolid: direct + transmit-side sea; dashed: direct only. Breaks exclude shape amplitudes below 0.03.')
    fig.savefig(args.output/'surface-phase-dynamics.png', dpi=160); plt.close(fig)
    result = {'scope': __doc__, 'accepted_calibration': False, 'all_channels_use_ornata_for_sensitivity': True,
        'surface_dimensions_m': [286.53, 30.], 'centre_height_m': 20., 'reference_heading_deg': heading,
        'constant_translation_enu_mps': velocity.tolist(), 'records': records,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [args.reflection, geometry_path, Path(__file__), Path(__file__).with_name('facet_sea.py')]}}
    (args.output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.output/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Finite side and HH sea reflection</title>
<style>body{font:16px/1.5 system-ui;max-width:1250px;margin:35px auto;padding:0 20px}img{width:100%}.note{padding:15px;background:#fff1d3}</style>
<h1>Finite ship side and horizontal-polarisation sea reflection</h1>
<p class="note">This is a controlled physical sensitivity calculation, not a fit or accepted antenna calibration. A sea-reflected leg can substantially alter the complex ship echo. Strong reflection is compatible with both enhancement and destructive interference.</p>
<p>The four-path model evaluates the incident and outgoing directions separately at every quadrature point, with PEC physical-optics current, exact spherical path, geometric spreading, sphere divergence and HH Fresnel reflection. The direct part agrees with the independent existing finite-surface implementation; the point limit agrees with the four-ray reference. Quadrature refinement and rigid-coordinate changes are tested.</p>
<p>For this comparison all five frequencies use Ornata geometry so that the effect of frequency is visible without changing the transmitter. The surface dimensions and height, uniform current, seawater permittivity 80 and conductivity 4 S/m are declared hypotheses. No parameters have been learned from radar samples. Surface roughness, diffraction, ship detail, antenna patterns and shoreline truncation remain outside this test.</p>
<img src="surface-sea.png" alt="Finite-side amplitude and phase with direct, transmit-side reflected, and four-path illumination">
<p>Use the <a href="../../isar_20261016/flash-reflection/index.html">reflection-point geometry</a> to assess possible water paths. The four-path panel assumes unlimited water and is conditional: a calculated receive-side specular footprint can intersect land. The reflected field is not extra independent target energy; it coherently interferes with the direct component.</p>
<img src="surface-phase-dynamics.png" alt="Extra path rate and curvature from finite-side response and sea interaction">
<p>Phase rates use a five-point temporal derivative, checked by halving the step. They are corrections to the physical centre's path rate, not an arbitrary new motion variable. Constant heading and AIS translation alone give little phase curvature away from response nulls. Finite-side rotation can produce larger and frequency-dependent derivatives. The chosen turn rates are sensitivity controls, not estimates of the ship's true heading.</p>
<p>Within ±0.3° of the specular orientation, for the declared geometry, height and three turn rates, adding the transmit-side sea path changes the predicted phase rate by less than 0.0002 m/s and the phase acceleration by less than 0.00012 m/s². These small changes do not explain the measured flash's substantially larger interband disagreement. The conclusion is conditional on this surface hypothesis; it does not exclude rough-water dynamics or another ship structure.</p>
<p>The main-lobe plots do not establish that the flash is a specular side or identify the transmitters. The current direct-only finite-side fits still fail to predict all channels consistently; adding unconstrained phase curves would destroy the calibration information. Further candidates and a verified scattering model remain necessary before claiming a conclusive fit.</p>
<p>Primary construction: <a href="https://www.jpier.org/ac_api/download.php?id=11101108">Chen, Zhang and Yin (2012), four-path sea/ship interaction</a>; electrical parameters: <a href="https://www.itu.int/rec/R-REC-P.527/en">ITU-R P.527</a>. The former's microwave simulations do not validate this VHF target quantitatively. <a href="analysis.json">Numerical predictions and source hashes</a>.</p></html>''')
    print(json.dumps([{k: v for k, v in row.items() if 'step_halving' in k or k == 'turn_rate_deg_s'} for row in records], indent=2))


if __name__ == '__main__':
    main()
