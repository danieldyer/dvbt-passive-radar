"""Report matched-likelihood and original-data image checks separately."""
import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np

from scene_dictionary import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'images', 'information', 'tests'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--joint-report', type=Path)
    p.add_argument('--video', type=Path, help='Completed preview directory, including playback validation')
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    motion = json.loads(args.motion.read_text())
    image_path = args.images/'comparison.json'; images = json.loads(image_path.read_text())
    information = json.loads(args.information.read_text())
    if not images['complete'] or not information['complete'] or images['models'][1]['source'] != str(args.motion):
        raise ValueError('Require the completed image and information comparisons')
    for result in (motion, images):
        for path, digest in result['source_sha256'].items():
            if fingerprint(Path(path)) != digest:
                raise ValueError('A source changed: '+path)
    if information['fit_sha256'] != motion['source_fit_sha256']:
        raise ValueError('The information audit uses a different physical fit')
    tests = args.tests.read_text(); count = re.search(r'Ran (\d+) tests', tests)
    if not count or tests.strip().splitlines()[-1] != 'OK':
        raise ValueError('Require a completed passing support suite')
    old, new = images['models']
    changes = [b['maximum_power_db']-a['maximum_power_db'] for a, b in zip(old['channels'], new['channels'])]
    information_cases = [{key: case[key] for key in ('ship_point_coordinates_eliminated', 'cutoff', 'combined')}
        for case in information['cases'] if case['cutoff'] == 1e-10]
    files = [args.motion, image_path, args.information, args.tests, args.images/'comparison.png', Path(__file__)]
    joint = json.loads(args.joint_report.read_text()) if args.joint_report else None
    if joint and not joint['joint_fit_completed']:
        raise ValueError('Use the completed joint comparison')
    video = json.loads((args.video/'videos.json').read_text()) if args.video else None
    if video:
        validation = json.loads((args.video/'validation.json').read_text())
        if not video['complete'] or not validation['complete']:
            raise ValueError('Finish rendering and playback validation')
        if fingerprint(Path(video['scientific_motion'])) != video['scientific_motion_sha256']:
            raise ValueError('Video motion source changed')
        files.extend([args.video/'videos.json', args.video/'validation.json', Path(video['scientific_motion'])])
    if joint:
        files.append(args.joint_report)
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'matched_likelihood_comparisons': motion['comparison'], 'image_comparison': images,
        'per_channel_image_peak_changes_db': changes,
        'coherent_image_peak_change_db': new['maximum_coherent_power_db']-old['maximum_coherent_power_db'],
        'fixed_location_information': information_cases, 'support_tests': int(count[1]),
        'accepted_phase_centres': False, 'cross_band_phase_validated': False, 'new_video_complete': bool(video),
        'joint_comparison': joint, 'video': video,
        'source_sha256': {str(path): fingerprint(path) for path in files}}
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.images/'comparison.png', args.output/'motion-origin-images.png')
    (args.output/'motion-origin-images.json').write_text(json.dumps(result, indent=2)+'\n')
    comparison_rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in
        [Path(c['source']).name, f"{c['ship_matched_energy_change_db']:+.3f}", f"{c['prediction_error_reduction_percent']:+.3f}"])
        +'</tr>' for c in motion['comparison'])
    peak_rows = ''.join(f'<tr><td>{name}</td><td>{change:+.3f}</td></tr>' for name, change in
        zip(['Seven', 'SBS', 'Nine', 'Ten', 'ABC'], changes))
    weak = '; '.join(f"{case['combined']['directions_weaker_than_declared_prior']}/18 weaker than the declared location prior "
        f"with ship reflector coordinates {'free' if case['ship_point_coordinates_eliminated'] else 'fixed'}"
        for case in information_cases)
    video_note = 'The new sliding video is not complete.'
    video_section = ''
    if video:
        video_note = 'A new short sliding preview is available below; it remains a diagnostic complex sum.'
        links = ''.join('<li><a href="'+html.escape(os.path.relpath(entry['path'], args.output))+'">'
            +html.escape(Path(entry['path']).name)+'</a></li>' for entry in video['files'])
        video_section = f'''<h2>Sliding 30-second preview</h2>
<p>The 45-second motion fit holds the five range origins at the values from the 30-second shared-target control. Antennas stay at the licensed main sites and yaw is exactly zero. Original-row stationarity passes the unchanged 1e−7 gate. Every viewport pixel is formed from the recorded data with the full coherent window; no scattering-point image is substituted.</p>
<ul>{links}</ul><p>There are {video['frames']} frames at {video['fps']} fps, covering frame centres {video['frame_centres_s'][0]:.3f}–{video['frame_centres_s'][1]:.3f} seconds. Actual coherent aperture is {video['actual_aperture_extrema_s'][0]:.6f}–{video['actual_aperture_extrema_s'][1]:.6f} seconds. The common fixed colour scale is {video['scale_db'][0]:.3f}–{video['scale_db'][1]:.3f} dB. Cyan is the interpolated AIS track. Both all-intra H.264 videos passed complete decoding and frame-timing checks. This is a short preview, not the full capture or accepted cross-band calibration.</p>'''
    joint_note = 'The free-antenna fit still needs its original-row stationarity and transfer checks.'
    if joint:
        fitted = next(row for row in joint['records'] if row['name'] == 'Fitted antennas')
        joint_note = (f"The completed <a href=\"shared-origin-validation.html\">free-antenna fit</a> "
            f"passes original-row stationarity but adds only {fitted['ship_matched_energy_change_db']:+.5f} dB "
            f"withheld ship matched energy and reaches {fitted['antenna_bound_hits']} antenna-coordinate bounds. "
            'Whole-target transfer remains a separate check; no correction is adopted.')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Motion and range-origin image check</title>
<style>body{{font:17px/1.55 system-ui;max-width:1500px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:16px}}</style>
<h1>Does the improved prediction become a better physical image?</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="shared-origin-validation.html">Origin sensitivity control</a> · <a href="motion-origin-images.json">Results and source identities</a> · {result['updated_utc']}</p>
<p class="note">The new fixed-site solution predicts the held-out data better. Its independent channel images change, but broad spatial ambiguity ridges remain. This is not a resolved hull or accepted coherent multi-band calibration. {video_note}</p>
<h2>Matched likelihood comparison</h2>
<p>Each listed baseline and the new solution is evaluated with the same original observation rows, train/test masks, covariance, stationary nuisance and eight-point per-band dictionary over 150.772–180.772 seconds. The comparison changes motion and envelope origins; it does not compare a sparse estimator against a distributed-scene estimator. Point supports and initialization were discovered from this recording, so these are conditional checks, not untouched external validation.</p>
<table><tr><th>Baseline</th><th>Withheld matched-energy change (dB)</th><th>Prediction-error reduction (%)</th></tr>{comparison_rows}</table>
<p>The previous origin experiment already found a better feasible state under the old likelihood. The gain is not evidence that the new shared-origin assumption, rather than initialization, is responsible. Antennas remain at nominal main-site coordinates in this paired image check.</p>
<h2>Original-data Vulkan backprojection</h2>
<p>Each band is independently backprojected on the same physical east/north grid at the aperture midpoint, using the full 30-second original data, exact stored epochs, both symbol moments and 64-tap range interpolation. All samples and actual channel windows are identical between the two rows. The fitted eight-point scene is not substituted for the observations. No DC notch is applied. The sixth panel combines the complex channel images with the same inherited provisional gains.</p>
<a href="motion-origin-images.png"><img src="motion-origin-images.png" alt="Five independent channel images and provisional coherent sum, before and after the motion/range-origin refinement"></a>
<p>The colour scale is fixed across all panels at {images['colour_scale_db'][0]:.3f} to {images['colour_scale_db'][1]:.3f} dB. Cyan shows the interpolated AIS antenna track. Image formation and the coherent sum precede colour mapping.</p>
<table><tr><th>Band</th><th>Maximum image-power change (dB)</th></tr>{peak_rows}<tr><td>Provisional complex sum</td><td>{result['coherent_image_peak_change_db']:+.3f}</td></tr></table>
<p>These are maxima over the entire shown viewport, which includes secondary returns. They are descriptive image statistics, not target-specific SNR gains or resolution measurements. In particular, the {result['coherent_image_peak_change_db']:+.3f} dB combined peak change must not be replaced by the larger sparse-prediction gain.</p>
<h2>Propagation geometry now reaches the renderer</h2>
<p>The geographic movie path previously ignored supplied antenna offsets. It now applies the licensed-site offset plus the fitted local offset to the propagation rays while preserving the nominal motion reference. The old nominal-site path remains bit-identical. Regression tests independently check envelope position, RF phase, path rate, complex gain and exported physical coordinates; the full support suite passes {count[1]} tests. No phase-only rotation of a formed image is introduced.</p>
<h2>What the location information says at this fixed-site solution</h2>
<p>{weak}. This local information calculation removes target motion and complex responses, retains the five common range origins while stacking targets, then eliminates those origins. No hard optimizer bound or motion prior is counted as an observation. It remains conditional on road associations, noise covariance and scene support, and is not an accepted location-error distribution.</p>
<p>{joint_note} The first 45-second initialization inadvertently included the 236-column diffuse Ten support in an unregularized point solver and failed the rank check; that result is retained. The replacement explicitly transfers the same eight physical core points per band from the 30-second experiment, without changing their midpoint world coordinates or silently truncating a matrix rank. This remains an intermediate sparse-motion experiment, not completion of the full imaging-efficiency goal.</p>{video_section}</html>'''
    (args.output/'motion-origin-images.html').write_text(document)
    print(json.dumps({'report': str(args.output/'motion-origin-images.html'), 'peak_changes_db': changes,
        'coherent_peak_change_db': result['coherent_image_peak_change_db'], 'tests': int(count[1])}))


if __name__ == '__main__':
    main()
