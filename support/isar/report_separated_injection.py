"""Publish the 500 m relocation, acquisition bound and standard-window controls."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from recovery_experiment import SCALE


def read(path):
    return json.loads(path.read_text())


def position_error(analysis):
    return float(np.linalg.norm((np.asarray(analysis['selected']['parameters'])
                                -analysis['truth_parameters'])[:2]*SCALE[:2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('docs/data/injection_20261018'))
    args = p.parse_args(); root = args.root; output = root/'separated-500m'
    plan, original_plan = read(output/'plan.json'), read(root/'plan.json')
    old = read(root/'exact-profile-fit/analysis.json')
    profile, iq = [read(output/name/'analysis.json') for name in ['exact-profile-fit', 'iq-fit']]
    acquisition = read(output/'acquisition-validation.json')
    windows = read(output/'window-control.json')
    comparison = read(output/'operator-comparison.json')
    if not all(a['complete'] and a['passed'] for a in [profile, iq]) or not acquisition['complete']:
        raise ValueError('Paired injection and acquisition controls must finish')
    replays = [read(Path('build/injection-20261018')/name/'summary.json') for name in
               ['deterministic-baseline', 'separated-500m/exact-profile', 'separated-500m/iq']]
    if not all(r['channels'] == replays[0]['channels'] for r in replays):
        raise ValueError('Replay row/decoder identities changed')
    if not all(c['rs_bad'] == 0 for c in replays[0]['channels']):
        raise ValueError('Unexpected decoder errors')
    for a in [profile, iq]:
        limit = np.array([10, 10, 2, 2, 2, 2])
        if np.any(abs(np.asarray(a['selected']['parameters'])) >= limit-1e-4):
            raise ValueError('Position or motion fit reached a search bound')
    stages = acquisition['stages']; controls = acquisition['controls']
    clean_fit = np.asarray(controls['clean_refinement_from_acquisition']['parameters'])
    truth = np.asarray(profile['truth_parameters'])
    clean_error = float(np.linalg.norm((clean_fit-truth)[:2]*SCALE[:2]))
    nominal = read(root/'scene-strength.json')['channels']
    with np.load(output/'analytic-profile-fit/observations.npz') as data:
        energy_before = sum(np.vdot(data[f'injected_{c}'][:record['training_bins']*17],
                                      data[f'injected_{c}'][:record['training_bins']*17]).real
                            for c, record in enumerate(nominal))
    projection_loss = float(-10*np.log10(stages['oracle']['available_energy']/energy_before))
    rx = np.asarray(plan['receiver_ecef_m']); tx = np.asarray(plan['transmitters_ecef_m'])
    rotation = np.asarray(plan['ecef_to_enu']); original_state = np.asarray(original_plan['state_enu'])
    coarse = original_state.copy()
    coarse[:, :2] -= (np.asarray(old['truth_parameters'])*SCALE).reshape(3, 2)
    time = plan['separation_control']['training_midpoint_s']
    dt = time-plan['midpoint_s']
    centre = coarse[0]+dt*coarse[1]+.5*dt**2*coarse[2]

    def location(parameters):
        state = (np.asarray(parameters)*SCALE).reshape(3, 2)
        return state[0]+dt*state[1]+.5*dt**2*state[2]

    def path(offset):
        enu = centre+np.r_[offset, 0.]
        world = rx+enu@rotation
        return np.linalg.norm(world-rx)+np.linalg.norm(world-tx[2])-np.linalg.norm(tx[2]-rx)

    tangent = np.asarray(plan['separation_control']['contour_tangent_enu'])[:2]
    normal = np.array([tangent[1], -tangent[0]])
    target_range = path(location(old['truth_parameters']))
    curve = []
    for distance in np.linspace(-100, 800, 100):
        n = brentq(lambda value: path(distance*tangent+value*normal)-target_range, -1000, 1000)
        curve.append(distance*tangent+n*normal)
    curve = np.asarray(curve)
    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    ax.plot(curve[:, 0], curve[:, 1], '--', color='#888888', label='Same excess range (Nine)')
    for label, parameters, marker, colour in [
            ('AIS reference', np.zeros(6), '*', '#444444'),
            ('Original injection', old['truth_parameters'], 'o', '#0072b2'),
            ('Original joint fit', old['selected']['parameters'], 'x', '#0072b2'),
            ('500 m injection', truth, 'o', '#d55e00'),
            ('500 m profile joint fit', profile['selected']['parameters'], 'x', '#d55e00'),
            ('500 m IQ joint fit', iq['selected']['parameters'], '+', '#cc79a7'),
            ('500 m fit with known motion', controls['separated_known_motion']['fit']['parameters'], 's', '#009e73')]:
        point = location(parameters)
        if marker == 's':
            ax.scatter(*point, marker=marker, edgecolor=colour, facecolor='none', linewidth=2, s=180, label=label)
        else:
            ax.scatter(*point, marker=marker, color=colour, s=90, label=label)
    ax.set(xlabel='East of coarse AIS reference (m)', ylabel='North of coarse AIS reference (m)',
           aspect='equal', title=f'Point relocation and recovered positions at {time:g} s')
    ax.legend(fontsize=9)
    fig.savefig(output/'geographic-control.png', dpi=160);plt.close(fig)
    index = read(Path('build/calibration-20261013/histories/archive-index.json'))
    source = Path('captures/live-vhf-20261013-193334/capture.dvblive').stat()
    if [source.st_size, source.st_mtime_ns] != index['identity'][1:]:
        raise ValueError('Capture identity changed')
    summary = dict(relocation=plan['separation_control'], original_position_error_m=position_error(old),
        profile_position_error_m=position_error(profile), iq_position_error_m=position_error(iq),
        clean_refinement_position_error_m=clean_error, stationary_projection_point_loss_db=projection_loss,
        acquisition_loss_db=stages['coarse_bank_0']['mismatch_loss_db'],
        clean_refinement_loss_db=stages['clean_refined']['mismatch_loss_db'],
        mixed_refinement_point_loss_db=stages['mixed_acquisition_start']['mismatch_loss_db'],
        known_motion_position_error_m=controls['separated_known_motion']['horizontal_error_m'],
        nominal_thermal_snr_change_db=[10*np.log10(c['available_energy'])-o['nominal_thermal_coherent_snr_db']
            for c,o in zip(stages['oracle']['channels'], nominal)],
        worst_paired_operator_mismatch_db=max(c['coherence_loss_db'] for c in comparison),
        windows={k:{name:c[name] for name in ['horizontal_error_m','nominal_thermal_snr_penalty_db']}
                 for k,c in windows['controls'].items()}, original_capture_identity_unchanged=True,
        conclusion='The 500 m relocation does not reduce joint geographic error. Acquisition concentrates the known point nearly optimally under the declared metric; motion/position ambiguity and background mismatch remain. This is not a proof of optimal interference rejection.')
    paths = ['support/isar/separate_injection_plan.py', 'support/isar/validate_recorded_injection.py',
             'support/isar/validate_injection_acquisition.py', 'support/isar/window_injection_control.py',
             'support/isar/recovery_experiment.py', 'support/isar/compare_recorded_injections.py',
             'support/validation/recorded_injection_probe.cpp', str(Path(__file__))]
    summary['source_sha256'] = {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    rows = ''.join(f"<tr><td>{name}</td><td>{c['horizontal_error_m']:.2f}</td><td>{c['nominal_thermal_snr_penalty_db']:.2f}</td></tr>"
                   for name,c in windows['controls'].items())
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>500 m separated-point injection</title>
<style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:16px;background:#eef6fa}}pre{{overflow:auto;background:#eee;padding:12px}}</style>
<h1>500 m cross-range injection and acquisition verification</h1>
<p class="note"><strong>The relocation did not improve the joint geographic fit.</strong> Position error is {summary['profile_position_error_m']:.2f} m for finite-profile injection and {summary['iq_position_error_m']:.2f} m for IQ injection, compared with {summary['original_position_error_m']:.2f} m originally. It does not confirm that proximity to the ship alone explains the error. Holding motion known gives {summary['known_motion_position_error_m']:.3f} m position error.</p>
<h2>Controlled relocation</h2><p>The old point was 34.34 m cross-range from the coarse AIS reference. The new point is exactly 500 m cross-range from that reference at the training midpoint (151.75 s), at the same Nine excess range as before. Its horizontal distance from the reference is 513.19 m. The ship's particular material scatterer is not geographically identified: this is separation from the AIS prior, not a measured 500 m distance to a known bright reflector.</p>
<p>Velocity, acceleration, height and all complex band gains are unchanged; nominal coherent thermal SNR changes by less than 0.002 dB in each band. The exact equal-range constraint holds for the four co-sited test bands; ABC's excess path changes by 3.099 m, which is included in its model. The recorded background, native epochs, 16-symbol binning, covariance, stationary basis and 142–161.5 s fitting interval are retained. The final 10.5 s is excluded from fitting. Position bounds are expanded to ±1 km so the moved target is not clipped; all solutions remain interior.</p>
<img src="geographic-control.png" alt="Original and relocated points and joint geographic solutions">
<h2>Acquisition algorithm and energy bound</h2><p>This experiment uses the offline CPU support routine <code>Experiment.acquire</code>. It is a direct matched-chirp grid, not an explicit Radon/FRFT implementation. It shortlists three delays from incoherent range scores (18.74 m spacing), then searches a common residual path rate (0.02 m/s spacing) and acceleration (0.001 m/s² spacing), including RF-dependent quadratic phase. The coarse residual model does not include its additional range migration. The subsequent exact geometric fit does include physical range migration and original-symbol moment responses.</p>
<p>For a known injected point s in the declared range/noise metric, remove the fixed stationary nuisance subspace with P. The captured energy is Σ<sub>c</sub>|h<sub>c</sub>ᴴPs<sub>c</sub>|²/(h<sub>c</sub>ᴴPh<sub>c</sub>), bounded by Σ<sub>c</sub>||Ps<sub>c</sub>||² by Cauchy–Schwarz. Band coefficients remain independent constants. The bound is achieved by the true template. Stationary projection itself removes {projection_loss:.5f} dB of this point's profile energy, reported separately.</p>
<table><tr><th>Stage</th><th>Point energy loss against conditional bound</th></tr><tr><td>Best coarse chirp-bank candidate</td><td>{summary['acquisition_loss_db']:.5f} dB</td></tr><tr><td>Exact seed converted from that candidate</td><td>{stages['geometric_seed_0']['mismatch_loss_db']:.5f} dB</td></tr><tr><td>Exact refinement on point-only data</td><td>Numerical floor ({summary['clean_refinement_loss_db']:.2g} dB)</td></tr><tr><td>Refinement with recorded background</td><td>{summary['mixed_refinement_point_loss_db']:.6f} dB</td></tr></table>
<p>Point-only refinement starts from acquisition and recovers the injected geographic position to the numerical floor. With the recorded background present, starting at the exact truth returns the same displaced solution as all three acquisition starts. Thus the observed error is not explained by a missed acquisition basin in this experiment.</p>
<img src="acquisition-validation.png" alt="Acquisition energy and fixed-motion versus joint-motion position ambiguity">
<p>At the displaced 116 m position, changing position alone costs about 11.79 dB of point energy; allowing the fitted velocity/acceleration changes costs only 0.00201 dB after profiling constant band coefficients. This is why the 125–160 m fixed-motion imaging resolution cannot be used as the resolution of the full joint state fit.</p>
<h2>Windowing control</h2><table><tr><th>Standard aperture weighting</th><th>Joint position error (m)</th><th>Nominal thermal SNR penalty (dB)</th></tr>{rows}</table>
<p>These controls keep the same data, start at truth, and change only the declared temporal least-squares weights, rebuilding the stationary projection consistently. The SNR penalties use the actual weights/template and original thermal-noise variance, rather than assuming uniform illumination. Neither standard taper removes the error here. This does not establish that tapering is never useful, or that the present treatment of correlated background is optimal.</p>
<h2>IQ cross-check and scope</h2><p>New native profile and IQ replays have identical row identities and decoded-data checksums, with zero RS failures. The paired compressed-echo mismatch is below {summary['worst_paired_operator_mismatch_db']:.5f} dB, with no phase/gain alignment. The original capture's size and modification time remain unchanged. No production DSP formula was changed.</p>
<p>The energy bound is conditional on the supplied range fields, assumed thermal covariance, fixed stationary projection and one translating point. It verifies coherent concentration for this test; it does not prove global search optimality on arbitrary scenes, optimal rejection of correlated background, or unique real-ship localisation. The results leave a problem in the joint inference/background model to address. They do not establish an unavoidable localisation limit.</p>
<p><a href="summary.json">Summary and source hashes</a> · <a href="plan.json">Relocation plan</a> · <a href="exact-profile-fit/analysis.json">Finite-profile fit</a> · <a href="iq-fit/analysis.json">IQ fit</a> · <a href="acquisition-validation.json">Acquisition controls</a> · <a href="window-control.json">Window controls</a> · <a href="operator-comparison.json">Paired operator comparison</a> · <a href="../index.html">Original experiment</a></p></html>'''
    (output/'index.html').write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
