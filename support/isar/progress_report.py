"""Publish measured ship-autofocus progress, including rejected fits."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from imaging import physical_state
from motion import enu_basis
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ais', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = [('Fixed locations, free motion', 'full-likelihood-motion-142-172.json', 'full_likelihood'),
                  ('Joint locations, free motion', 'joint-likelihood-motion-142-172.json', 'joint_likelihood'),
                  ('Joint + kinematic bounds, offset start', 'joint-kinematic-motion-142-172.json', 'joint_likelihood'),
                  ('Joint + kinematic bounds, nominal start', 'joint-kinematic-nominal-142-172.json', 'joint_likelihood')]
    models = []
    pending = []
    for title, name, key in candidates:
        path = args.work/name
        if path.exists():
            data = json.loads(path.read_text())
            models.append((title, data, data[key], path))
        else:
            pending.append(title)
    seed = json.loads((args.work/'full-field-motion-validation-142-172.json').read_text())
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    stages = [('AIS/course prior', [c['before'] for c in seed['channels']]),
              ('Physical phase seed, full range evaluation', [c['after'] for c in seed['channels']])]
    stages.extend((title, result['after']) for title, _, result, _ in models)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
    x = np.arange(5)
    width = .8/len(stages)
    for i, (title, scores) in enumerate(stages):
        offset = x+(i-(len(stages)-1)/2)*width
        axes[0].bar(offset, [s['conditional_efficiency_loss_db'] for s in scores], width=width, label=title)
        axes[1].bar(offset, [s['prediction_phase_rms_degrees'] for s in scores], width=width)
    axes[0].set(ylabel='Conditional held-out coherent loss (dB)', xticks=x, xticklabels=names)
    axes[1].set(ylabel='Prediction phase RMS (degrees)', xticks=x, xticklabels=names)
    axes[0].legend(fontsize=8, ncols=2)
    figure.suptitle('Same observed data and point dictionary; alternating 2-second fit/validation blocks\nThese include scattering mismatch and noise, not pure motion error or a global SNR bound')
    figure.savefig(args.output/'heldout-comparison.png', dpi=145)
    plt.close(figure)
    if models:
        track = read_csv(args.ais, reference_height_ellipsoid_m=5., mmsi=563052200)
        figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout='constrained')
        for title, model, result, _ in models:
            m = SplineMotion(model)
            t = np.linspace(*m.interval, 601)
            state = physical_state(track, utc_seconds(model['nominal_start_utc'])+t, model['attitude_prior'])
            forward = np.einsum('nji,nj->ni', enu_basis(state['latitude'], state['longitude']), state['body_to_ecef'][:, :, 0])
            heading = np.degrees(np.unwrap(np.arctan2(forward[:, 0], forward[:, 1])))
            path, yaw = m.curves(t)
            rate, _ = m.curves(t, 1)
            axes[0].plot(t, path, label=title)
            axes[1].plot(t, rate)
            axes[2].plot(t, heading-np.degrees(yaw))
        axes[0].legend(fontsize=8, ncols=2)
        axes[0].set_ylabel('Reference path correction (m)')
        axes[1].set_ylabel('Correction rate (m/s)')
        axes[2].set(ylabel='Conditional heading (degrees)', xlabel='Capture time (s)')
        figure.suptitle('Motion curves: unrestricted fits can be numerically sharp but physically implausible\nHeading is conditional on the chosen body points; it is not an independent heading measurement')
        figure.savefig(args.output/'motion-comparison.png', dpi=145)
        plt.close(figure)
    rows = []
    for title, model, result, path in models:
        gradient = result['maximum_projected_gradient']
        change = [100*(b['prediction_error']/a['prediction_error']-1)
                  for a, b in zip(stages[1][1], result['after'])]
        bounds = 'yes' if result['bound_hit'] else 'no'
        link = os.path.relpath(path.resolve(), args.output.resolve())
        rows.append(f'<tr><td><a href="{escape(link)}">{escape(title)}</a></td><td>{result["objective"]:.7f}</td>'
                    f'<td>{result["converged"]}</td><td>{gradient:.3g}</td><td>{bounds}</td>'
                    f'<td>{", ".join(f"{v:+.1f}%" for v in change)}</td></tr>')
    calibration = []
    for title, model, result, path in models:
        if 'antenna_fit' not in model:
            continue
        a = model['antenna_fit']
        data = np.array(a['transmitter_offsets_enu_m']+[a['receiver_offset_enu_m']])
        body = ''.join(f'<tr><td>{name}</td><td>{r[0]:+.2f}</td><td>{r[1]:+.2f}</td><td>{r[2]:+.2f}</td></tr>'
                       for name, r in zip(names+['Receiver'], data))
        calibration.append(f'<details><summary>{escape(title)}: trial antenna offsets</summary>'
            f'<table><tr><th>Channel</th><th>East (m)</th><th>North (m)</th><th>Up (m)</th></tr>{body}</table>'
            '<p>Trial values only. The linked result contains conditional information eigenvalues and prior/posterior uncertainty. '
            'No production metadata or surveyed coordinates were changed.</p></details>')
    support_rows = []
    for channel, name in enumerate(names):
        dense_path = args.work/f'joint-kinematic-dense-142-172/channel-{channel}.json'
        scene_path = args.work/f'continuous-scene-142-172/channel-{channel}.json'
        if dense_path.exists() and scene_path.exists():
            dense, scene = json.loads(dense_path.read_text()), json.loads(scene_path.read_text())
            before, after = scene['before'], scene['after']
            ratios = [before['prediction_error']/before['stationary_only_error'],
                      dense['held_out_prediction_error']/dense['held_out_stationary_only_error'],
                      after['prediction_error']/after['stationary_only_error']]
            support_rows.append(f'<tr><td>{name}</td>'+''.join(f'<td>{r:.3f}</td>' for r in ratios)+'</tr>')
    exact_path = args.work/'joint-kinematic-exact-check.json'
    exact_text = ''
    if exact_path.exists():
        exact = json.loads(exact_path.read_text())
        exact_text = (f'<p>The dense exact-geometry check measures maximum path acceleration '
            f'{exact["maximum_absolute_path_acceleration_mps2"]:.6f} m/s², heading rate '
            f'{exact["maximum_absolute_heading_rate_deg_ps"]:.6f}°/s and heading acceleration '
            f'{exact["maximum_absolute_heading_acceleration_deg_ps2"]:.6f}°/s². '
            'The sampled linear bounds are therefore only approximate; this candidate is not an accepted calibrated trajectory.</p>')
    holdout_path = args.work/'holdout-duration-sensitivity.json'
    holdout_rows = []
    if holdout_path.exists():
        check = json.loads(holdout_path.read_text())
        for name, channel in zip(names, check['channels']):
            holdout_rows.append(f'<tr><td>{name}</td>'+''.join(
                f'<td>{m["conditional_efficiency_loss_db"]:.3f}</td>' for m in channel['masks'])+'</tr>')
    interleaved_trials = [p for p in (args.work/'joint-interleaved-motion-142-172.json',
                                    args.work/'joint-interleaved-from-control-142-172.json') if p.exists()]
    interleaved_path = (min(interleaved_trials,
        key=lambda p: json.loads(p.read_text())['joint_likelihood']['objective'])
        if interleaved_trials else args.work/'joint-interleaved-motion-142-172.json')
    interleaved_text = '<p>The interleaved-sample joint motion/location fit is running.</p>'
    if interleaved_path.exists():
        d = json.loads(interleaved_path.read_text())['joint_likelihood']
        interleaved_text = (f'<p>A separate joint fit trained on alternate retained 17.4 ms bins '
            f'has numerical convergence status {d["converged"]}, KKT residual '
            f'{d["maximum_projected_gradient"]:.3g}, and bound-hit status {d["bound_hit"]}. '
            'Held-out conditional dictionary losses by channel are '+
            ', '.join(f'{s["conditional_efficiency_loss_db"]:.3f}' for s in d['after'])+
            ' dB. These describe prediction within the chosen dictionary, not energy missing outside it or absolute phase-centre accuracy.</p>')
        interleaved_text += (f'<p>The displayed interleaved joint result has the lowest training objective '
            f'of {len(interleaved_trials)} completed starts: <code>{escape(interleaved_path.name)}</code>. '
            'Held-out samples were not used to select the start.</p>')
        information = np.array(d['information']['antenna_information_eigenvalues_before_prior'])
        interleaved_text += (f'<p>Even the optimistic conditional location-information calculation leaves '
            f'{np.sum(information < 1)} of {len(information)} joint antenna-coordinate directions with less '
            'data information than prior information. The trial locations are not independently established by this ship aperture.</p>')
        control_path = args.work/'fixed-interleaved-control-142-172.json'
        if control_path.exists():
            control = json.loads(control_path.read_text())['joint_likelihood']
            changes = [100*(b['prediction_error']/a['prediction_error']-1)
                       for a, b in zip(control['after'], d['after'])]
            total_change = 100*(1-sum(s['prediction_error'] for s in d['after'])/
                               sum(s['prediction_error'] for s in control['after']))
            interleaved_text += (f'<p>A matched fixed-location control uses the same motion basis, kinematic bounds, '
                f'point dictionary and sample split. Its training objective is {control["objective"]:.7f}, '
                f'versus {d["objective"]:.7f} with locations free. The held-out squared-error changes from freeing '
                'locations are '+', '.join(f'{v:+.3f}%' for v in changes)+
                f' by channel. The fixed-location control numerical convergence status is {control["converged"]} '
                f'(KKT residual {control["maximum_projected_gradient"]:.3g}). These are local-fit comparisons, '
                'not evidence of a unique global solution.</p>')
            interleaved_text += (f'<p>Across all channels, freeing the locations reduces held-out squared '
                f'prediction error by {total_change:.4f}%. This small result and the search-bound solutions '
                'do not justify adopting the fitted antenna positions. Prediction-phase RMS values are '+
                ', '.join(f'{s["prediction_phase_rms_degrees"]:.1f}°' for s in d['after'])+
                '; these include scattering mismatch, glint and noise, and are not pure motion-error measurements.</p>')
    movie_directory = Path('captures/live-vhf-20261013-193334/isar-trial-20261015')
    movie_text = ''
    if (movie_directory/'videos.json').exists():
        movies = json.loads((movie_directory/'videos.json').read_text())
        links = []
        for movie in movies:
            relative = os.path.relpath(Path(movie['path']).resolve(), args.output.resolve())
            bounds = np.array(movie['actual_channel_windows_s'])
            duration = bounds[:, :, 1]-bounds[:, :, 0]+.001088
            links.append(f'<li><a href="{escape(relative)}">{movie["requested_aperture_s"]:g} s trial video</a>: '
                f'{movie["fps"]} fps, actual CPI {duration.mean():.6f} s, fixed '
                f'{movie["colour_scale_db"][0]:g}–{movie["colour_scale_db"][1]:g} dB, '
                f'maximum {movie["maximum_power_db"]:.2f} dB; clipping: {movie["display_saturated"]}.</li>')
        movie_text = ('<h2>Diagnostic video previews</h2><p>These use the constrained offset-start trial, '
            'not an accepted physical autofocus result. The origin remains the AIS antenna prior. '
            'Each video has five independent channels and a nominal coherent complex sum, '
            '451 frames, H.264 all-intra encoding and no B frames. All files were decoded successfully.</p><ul>'+
            ''.join(links)+'</ul><figure><img src="joint-trial-video-frame.png"><figcaption>'
            'Example frame at capture time 157 s with a fixed 3 s aperture. Fine fringes in the nominal '
            'cross-band sum must not be interpreted as calibrated hull detail.</figcaption></figure>')
    state = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'Active ship-motion and autofocus investigation; no accepted absolute heading, hull-centroid calibration or coherent five-band reconstruction yet.',
             'models': [{'title': title, 'path': str(path), 'numerical_convergence': result['converged']}
                        for title, _, result, path in models], 'pending': pending,
             'report': str(args.output/'analysis.html'),
             'interleaved_joint_model': str(interleaved_path),
             'fixed_location_control': str(args.work/'fixed-interleaved-control-142-172.json'),
             'trial_videos': str(movie_directory),
             'joint_control_summary': str(args.work/'joint-versus-fixed-control-summary.json'),
             'numerical_audit_report': str(args.output/'numerical-audit.html'),
             'channel_alignment': str(args.work/'channel-alignment-142-172/alignment.json'),
             'subpixel_coherent_scene': str(args.work/'channel-alignment-142-172/subpixel.json'),
             'coherent_motion_control': str(args.work/'channel-alignment-142-172/coherent-motion.json'),
             'profiled_coherent_motion': str(args.work/'channel-alignment-142-172/profiled-coherent-motion-refined.json'),
             'registered_coherent_image': str(args.output/'registered-coherent-image.png'),
             'geometry_report': str(args.output/'geometry-refinement.html'),
             'amplitude_report': str(args.output/'amplitude-autofocus.html'),
             'amplitude_results': str(args.output/'amplitude-autofocus.json'),
             'coherent_motion_range_control': str(args.work/'channel-alignment-142-172/coherent-motion-range-control.json'),
             'coherent_geometry_joint_motion': str(args.work/'channel-alignment-142-172/coherent-geometry-joint-motion.json'),
             'corrected_transfer': str(args.work/'channel-alignment-142-172/coherent-geometry-transfer-projected.json'),
             'remaining': ['Keep roll and pitch fixed: flat-plane sensitivity is nearly degenerate with path/yaw. Refine shared physical motion with channel range/phase registration and common subpixel scattering support.',
                 'Joint motion/gain/range and joint antenna fits now pass numerical convergence. Antenna refinement adds only about 0.08 dB and reaches bounds; 14/18 location combinations remain weaker than their prior.',
                 'Earlier half-aperture transfer tests extrapolated poorly constrained stationary coefficients. Corrected projection retains the frozen target prediction, but the common constant scene still transfers poorly. Do not use the older scores alone to reject motion or phase corrections.',
                 'Resolve motion/scattering mismatch without interpreting conditional dictionary loss as overall SNR loss.',
                 'Establish a physical hull frame and transferable cross-band phase model before claiming calibrated coherent imaging.']}
    (args.output/'session.json').write_text(json.dumps(state, indent=2)+'\n')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship ISAR progress</title>
<style>body{{font:16px/1.55 system-ui;max-width:1250px;margin:35px auto;padding:0 24px;color:#20242a}}h1,h2{{line-height:1.2}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d7dce2;text-align:left}}.status{{padding:18px;background:#fff2d1;border-left:5px solid #ba7600}}figure{{margin:32px 0}}figcaption{{color:#555}}details{{margin:18px 0}}code{{background:#eee;padding:2px 5px}}</style>
<h1>Ship ISAR: motion and phase-centre fitting</h1><p>Updated {escape(state['updated_utc'])}. Capture: live-vhf-20261013-193334. Working interval: 142–172 s.</p>
<p class="status">{escape(state['status'])}</p>
<p><strong>Latest work:</strong> <a href="amplitude-autofocus.html">Separating changing reflector strengths from phase error: converged fits, range-transfer tests and registered coherent videos</a>. Roll and pitch remain fixed. The earlier <a href="geometry-refinement.html">antenna-location experiment</a> adds about 0.08 dB and remains weakly constrained; no new antenna coordinates were adopted.</p>
<p>The supplied AIS positions are usable as a trajectory prior. The file contains latitude, longitude, speed and course over ground; it contains no true heading or rate of turn. The ship's heading is fitted independently of its velocity direction.</p>
<h2>What has actually been formed</h2>
<p>Exact Vulkan backprojection forms each channel independently and sums their complex, information-weighted images on the same physical grid. Earlier figures below use nominal gains; the latest linked report includes fitted gains, envelope origins and physical antenna offsets. These are actual coherent sums, but their finer fringes do not prove calibrated hull detail. No aperture zero-Doppler notch is applied.</p>
<figure><img src="physical-motion-comparison.png"><figcaption>Initial physical phase seed, before the subsequent full-data and location fits. The fixed 30 s aperture and common colour scale show peak gains of 0.5–4.8 dB per channel and 4.6 dB in the nominal combined image. Peak gain alone is not autofocus validation.</figcaption></figure>
<h2>Full-data tests</h2>
<p>Both candidate trajectories predict the same retained complex range/slow-time observations. The forward model changes delay, carrier phase and symbol-moment Doppler together. The original row averaging and conditional range covariance are retained. Constant-delay nuisance responses are fitted jointly with the target. There is no moving-target Doppler notch or local image normalization.</p>
<p>Training uses alternating 2-second blocks. All complex point amplitudes and stationary nuisance coefficients are held fixed when predicting the other blocks. The initial point selection and phase seed used broader data, so this is conditional validation of subsequent fitting, not a fully independent test of target discovery. There are eight fixed points per channel with independent complex frequency responses.</p>
<table><tr><th>Model</th><th>Training objective</th><th>Converged</th><th>Projected/KKT gradient</th><th>Box bound hit</th><th>Held-out error change vs phase seed: Seven, SBS, Nine, Ten, ABC</th></tr>{''.join(rows)}</table>
<p>Negative error changes are improvements. An optimizer success flag is accepted only with the reported gradient/KKT residual and feasible constraints. Numerical convergence alone does not establish a physically correct motion fit.</p>
<figure><img src="heldout-comparison.png"><figcaption>Conditional coherent loss compares the fixed training prediction against independently fitted test amplitudes in the same dictionary. The phase metric projects one-second test blocks onto that prediction. Both include unresolved scattering and model mismatch; neither is a pure phase-motion error or a global theoretical loss bound.</figcaption></figure>
<figure><img src="motion-comparison.png"><figcaption>The unconstrained curves are retained as rejected controls. Rapid path-rate excursions are inconsistent with the intended smooth large-vessel model.</figcaption></figure>
<h2>Scattering support and validation sensitivity</h2>
<p>Increasing the dictionary from 8 to 128 points, or continuously refining the original point locations within 20 m, did not fix long-block prediction. The following values are held-out squared prediction error divided by the error of a stationary-only fit. Less than one means the moving model improves prediction. These are conditional diagnostics, not imaging SNR losses.</p>
<table><tr><th>Channel</th><th>8 fixed points</th><th>128 points, training GCV</th><th>8 continuously refined points</th></tr>{''.join(support_rows)}</table>
<p>The score also depends strongly on which times are withheld. Two-second gaps remove substantial viewing-angle information for an extended object. Holding out alternate short bins largely tests repeatability of the fitted dictionary, while long blocks additionally test extrapolation of its scattering model. Neither alone establishes that all available image information has been recovered. The table below changes only the amplitude-fit/validation mask at the existing motion: other masks are not independent validation of motion originally fitted with two-second blocks.</p>
<table><tr><th>Channel</th><th>Alternate 17.4 ms bins</th><th>0.25 s blocks</th><th>0.5 s blocks</th><th>1 s blocks</th><th>2 s blocks</th></tr>{''.join(holdout_rows)}</table>
<p>Entries are conditional dictionary losses in dB; they do not include energy the dictionary never explains.</p>
{interleaved_text}
<figure><img src="temporal-signal-audit.png"><figcaption>Measured short-time spectra and range profiles. White curves are the current point-model Dopplers. Strong amplitude changes and multiple unresolved components motivate checking the scattering model, rather than treating every residual phase fluctuation as antenna-position error.</figcaption></figure>
<h2>Joint transmitter/receiver refinement</h2>
<p>Five transmit phase centres and one receiver location are variable. Transmitters sharing a licensed tower have correlated site uncertainty, with additional per-frequency phase-centre uncertainty. The declared prior standard deviations are 30/30/75 m per site, 5/5/10 m per antenna, and 15/15/7.5 m for the receiver, in east/north/up order. The transmitter–receiver direct-path baseline changes together with the echo paths.</p>
<p>Several trial solutions reach the location search bounds, including implausible antenna-height offsets. They remain rejected diagnostics: the fit can use location error to absorb deficiencies of the motion/scattering model. No fitted coordinates have been written into capture or production configuration.</p>
<p>The constrained runs additionally bound actual reference bistatic path acceleration to 0.1 m/s², heading rate to 0.5°/s, and heading acceleration to 0.01°/s². These are stated analysis assumptions for the large-vessel hypothesis, not measured performance limits. The path-acceleration constraint is linearized about the prior and requires a final exact-geometry check.</p>
{exact_text}
{''.join(calibration)}
<p>Absolute heading has a coordinate ambiguity while the scattering-point coordinates are unknown: replacing <code>R(t)</code> by <code>R(t)Q</code> and every point <code>p</code> by <code>Qᵀp</code> leaves the physical trajectories identical. A fitted hull/body reference is needed before labelling the result an absolute ship heading. The present image origin is the AIS antenna prior, not a measured hull centroid.</p>
{movie_text}
<h2>Remaining work</h2><p>Resolve the held-out model mismatch; validate the constrained joint fit across starts and scattering support; compare coherent apertures; infer a defensible hull/body frame; then render the independent and coherent videos with the accepted motion. Video support now sums retained complex contributions over fixed-length windows and uses a fixed colour scale with all-intra encoding for scrubbing.</p>
<p>Pending runs at report generation: {escape(', '.join(pending) if pending else 'none in the current comparison; acceptance work remains')}.</p>
<h2>Method references</h2><p><a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5233">Garry and Smith, Passive ISAR part I</a> develops translational compensation, phase-gradient refinement and bistatic backprojection. <a href="https://scholar.afit.edu/etd/2240/">Evers, Generalized Phase Gradient Autofocus</a> motivates scatterer isolation and multi-point phase-gradient estimation. The joint physical likelihood, held-out diagnostics and constrained fits here are implementation-specific tests, not guarantees supplied by those papers.</p>
</html>'''
    (args.output/'analysis.html').write_text(html)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
