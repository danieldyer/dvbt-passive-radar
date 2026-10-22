"""Assemble the measured multiple-candidate phase plots and association controls."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('docs/data/isar_20261017')
    out = root/'phase-trajectories-review'
    out.mkdir(exist_ok=True)
    far = json.loads((root/'guided-abc-return/analysis.json').read_text())
    near = json.loads((root/'guided-abc-main/analysis.json').read_text())
    sections = ''.join(f'''<h2>Window {i}: {first}–{last} s</h2>
<a href="../measured-scatterer-phases/02-W{i}-trajectories.png"><img src="../measured-scatterer-phases/02-W{i}-trajectories.png" alt="Direct measured phases across five channels in window {i}"></a>'''
        for i, first, last in ((1,151,155),(2,158,162),(4,320,328)))
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Selected returns: measured five-channel phase trajectories</title>
<style>body{{font:16px/1.55 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}}img{{width:100%}}.note{{background:#eef4f8;padding:16px}}li{{margin:8px 0}}h2{{margin-top:2em}}</style>
<h1>Selected returns and their measured phase trajectories</h1>
<p class="note"><b>Latest temporal-model correction:</b> <a href="../flexible-flash-path/index.html">Shared nonquadratic flash paths, prediction comparisons and independent path plots</a>. Different brightness weights can give different quadratic coefficients for the same underlying motion; an original-row synthetic counterexample verifies this. A shared 1 s-knot spline improves unused prediction, while finer knots overfit and produce implausible endpoint acceleration. The earlier coefficient comparisons are conditional diagnostics, not direct physical antenna measurements. 163 numerical tests pass.</p>
<p class="note"><b>Newest phase plots:</b> <a href="../multiple-scatterer-phase-plots/index.html">Four marked returns with independently measured trajectories in all five channels</a>: original fixed-grid range/rate maps, conditional separation of neighbouring returns, physical path changes, and A−B/C−B/D−B comparisons. B has the clearest shared bulk motion; A fades and Ten's C is almost stationary. No across-band trajectory equality is imposed. <a href="../fixed-grid-phase-review/index.html">Stationary separation and noise controls at three intervals</a> reproduce 68 objectives exactly. 158 numerical tests pass across the operator and measurement-isolation suites.</p>
<p class="note">Multipath modelling is stopped. Ten local feature selections are shown in four matching time windows across all five channels. The central flash gives the clearest common path-change trend. The figures retain the disagreement and fades in other cells; no time-dependent phase correction is used to align them.</p>
<p class="note"><b>Latest follow-up:</b> <a href="../adjacent-scatterer-phases/index.html">Three adjacent windows with selected-point phase-sharing controls</a>. The large earlier all-return prediction penalty comes primarily from imposing a common trajectory on trailing D. Sharing A/B/C while leaving D independent changes unused error by less than 0.32% in these windows. B gives the strongest phase measurement; A remains unstable. Conditional phase errors and measured innovations are shown separately from total-field prediction error.</p>
<p><a href="#flash">Start with the flash</a> · <a href="#differences">Between-point differences</a> · <a href="#association">Conditional association tests</a> · <a href="../measured-scatterer-phases/measurements.json">Numerical measurements</a> · <a href="../phase-trajectory-tests.log">Numerical test log</a></p>
<h2>Selected cells in every channel</h2>
<p>Cyan boxes mark the selected features in direct range/rate periodograms. Columns use common windows; rows are Seven, SBS, Nine, Ten and ABC. Letters label local features and do not establish a persistent material point between windows. The saved AIS range reference is retained, including its original geometry.</p>
<a href="../measured-scatterer-phases/01-selected-points.png"><img src="../measured-scatterer-phases/01-selected-points.png" alt="Selected range and rate cells, marked in all five channels and four windows"></a>
<h2 id="flash">Flash window: 163–167 s</h2>
<p>Rows correspond to the three selected cells. From left to right: measured phase change; excess-path change using each channel’s wavelength; difference from Nine; and measured signal strength against conditional thermal noise. Solid curves are measurements; dashed curves are the nominal AIS/geometry prediction. Only constant phase origins are removed. No fitted rate or curvature enters these plots.</p>
<a href="../measured-scatterer-phases/02-W3-trajectories.png"><img src="../measured-scatterer-phases/02-W3-trajectories.png" alt="Three selected cells: measured phase and path change across all five channels during the flash"></a>
<p>Candidate B has a shared five-band trend through the full window. The other cells show portions of similar motion, with larger differential deviations. A cell may contain a mixture of returns: a common bulk phase trend does not alone identify one physical point. Lines stop at unsupported phase connections; gaps are not bridged through fades.</p>
<h2 id="differences">Compare phase changes between selected points</h2>
<p>Within each band, A−B and C−B subtract two measured path changes, removing one constant at a common available epoch. This suppresses their shared translation. Similar remaining changes across bands would support consistent associations under slowly varying heading. The final column repeats B with shorter averaging as a control. These overlapping measurements are correlated.</p>
<a href="../measured-scatterer-phases/03-between-points.png"><img src="../measured-scatterer-phases/03-between-points.png" alt="Between-point path-change differences compared across the channels"></a>
<p><a href="../measured-scatterer-phases/04-local-increments.png">Local 0.25 s phase increments without a long phase connection</a> preserve useful segments on either side of a fade. <a href="../measured-scatterer-phases/index.html">The raw-measurement report</a> records the exact averaging, deramp, phase-origin and quality policies.</p>
{sections}
<h2 id="association">Resolve overlapping returns before interpreting the phase differences</h2>
<p>The following are conditional model checks, kept separate from the direct measurements above. Four channels estimate a guide trajectory from training data. ABC is searched using only a constant phase and positive brightness; its rate and curvature are fixed to that guide. A subsequent joint fit retains four independently moving ABC returns as nuisance components. All prediction parameters are frozen on unused blocks.</p>
<ul><li><a href="../guided-abc-return/index.html">Weaker ABC candidate near {far['abc_range_origin_m']:+.0f} m</a>: unused squared prediction error improves {far['heldout_error_reduction_percent']:.2f}%; component/sum energy ratio {far['component_to_sum_energy_ratio']:.2f}, brightness condition {far['active_brightness_condition']:.1f}. This is a useful phase association, still unconfirmed as a hull point.</li>
<li><a href="../guided-abc-main/index.html">Trying the guide at the bright main ABC return</a>: only {near['heldout_error_reduction_percent']:.2f}% prediction improvement, with nearly coincident components separated by {near['nearest_component_separation_m']:.3f} m and component/sum energy ratio {near['component_to_sum_energy_ratio']:.1f}. Rejected as an identified second point.</li>
<li><a href="../phase-fingerprint-search/index.html">Full measured range search</a>: every candidate cell, fixed-guide residual phases and prediction scores. <a href="../phase-fingerprint-reference/index.html">Subtracting a fixed residual mean from 30–90 s</a> has little effect on these phase differences; no correction is adopted.</li>
<li><a href="../relative-delay-geometry/index.html">Relative-delay geometric consistency</a>: a receiver-independent check demonstrates that some independently fitted envelope origins cannot all represent the same compact group of points under the declared site bounds. It rejects those point associations, not the recording.</li></ul>
<p><a href="../multiple-scatterer-review/index.html">Polynomial fits and joint multi-return controls</a> include 38 short arcs and explicit rejected decompositions. Carrier phase can be measured more precisely than an envelope centre, but that does not make an incorrect envelope-to-point association valid.</p>
<p class="note">No new transmitter/receiver coordinates or physical hull-point assignments have been accepted. Production processing, captures and approved recording metadata are unchanged. The larger autofocus/image-formation goal remains active.</p>
<p><a href="../index.html">Analysis archive</a> · <a href="manifest.json">Report provenance</a></p></html>''')
    sources = [Path(__file__), out/'index.html']
    for folder, files in (
            ('measured-scatterer-phases', ['measurements.json', 'index.html', '01-selected-points.png',
                '02-W3-trajectories.png', '03-between-points.png']),
            ('guided-abc-return', ['fits.json', 'analysis.json', 'index.html']),
            ('guided-abc-main', ['fits.json', 'analysis.json', 'index.html']),
            ('phase-fingerprint-search', ['search.json']),
            ('phase-fingerprint-reference', ['search.json']),
            ('multiple-scatterer-phase-plots', ['measurements.json', 'index.html',
                '01-selected-points.png', '02-measured-trajectories.png', '03-between-points.png', 'component-tests.log']),
            ('fixed-grid-phase-review', ['analysis.json', 'index.html', 'tests.log']),
            ('flexible-flash-path', ['analysis.json', 'index.html', 'fits.json']),
            ('piecewise-geometry-review', ['analysis.json', 'index.html', 'tests.log']),
            ('relative-delay-geometry', ['analysis.json'])):
        sources.extend(root/folder/name for name in files)
    (out/'manifest.json').write_text(json.dumps(dict(
        accepted_calibration=False, multipath_modelling=False,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2))
    print(out/'index.html')


if __name__ == '__main__':
    main()
