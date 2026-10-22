"""Plot measured five-band phase histories and the ABC association control.

The four-band trajectory is fixed before examining the ABC predictions. These
plots retain the rejected near-coincident fit alongside the resolved candidate.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES, COLOURS
from fit_shared_echo_cluster import prepare
from motion import C
from report_background_extrapolation import original_values
from report_measured_scatterer_phases import load_fields, serial
from report_shared_echo_cluster import parts, component_phase
from shared_echo_cluster import SharedEchoCluster


def main():
    root = Path('docs/data/isar_20261017')
    geometry = Path('docs/data/calibration_20261016_sites')
    fields = load_fields(Path('build/isar-20261015/full-fields'), geometry)
    guide_source = root/'phase-fingerprint-search/search.json'
    guide = json.loads(guide_source.read_text())
    scene = json.loads((root/'shared-echo-cluster-context/fits.json').read_text())
    first, last = scene['interval_s']
    bands, partitions = prepare(fields, geometry, first, last, range_extent=16)
    four = SharedEchoCluster(bands[:4], scene['selected_origins_samples'], origin_radius=2.)
    guide_fit = guide['guide_fits'][guide['selected_guide']]
    gp = np.asarray(guide_fit['parameters'])
    gr, _ = four.evaluate(gp)
    if abs(gr@gr*four.energy-guide_fit['cost']) > 1e-7:
        raise ValueError('Four-band guide reproduction failed')
    four_phase = [component_phase(c, gp[i], p == 3)
                  for c, i, p in zip(four.clusters, four.indices, partitions[:4])]

    for folder, label, centre, radius in (
            ('guided-abc-return', 'D', 15., 1.),
            ('guided-abc-main', 'B', 0., 2.)):
        out = root/folder
        source = out/'fits.json'
        result = json.loads(source.read_text())
        if not result['complete']:
            raise ValueError('Fit is incomplete')
        model = SharedEchoCluster([bands[4]], result['selected_origins_samples'], origin_radius=2.)
        n = model.components
        model.lower[2*(n-1)] = centre-radius
        model.upper[2*(n-1)] = centre+radius
        errors = []
        for trial in result['trials']:
            q = np.asarray(trial['parameters'])
            residual, _ = model.evaluate(q)
            errors.append(abs(residual@residual*model.energy-trial['cost'])/trial['cost'])
            if not np.array_equal(q[-2:], result['fixed_external_path_coefficients_m']):
                raise ValueError('External guide was changed')
        if max(errors) > 1e-9:
            raise ValueError('Saved fit reproduction failed')
        selected = result['trials'][result['best_index']]
        p = np.asarray(selected['parameters'])
        baseline = SharedEchoCluster([bands[4]], scene['selected_origins_samples'], origin_radius=2.)
        bp = np.asarray(result['baseline']['parameters'])
        br, _ = baseline.evaluate(bp)
        if abs(br@br*baseline.energy-result['baseline']['cost']) > 1e-7:
            raise ValueError('Saved baseline reproduction failed')
        signals = parts(model.clusters[0], p[model.indices[0]])
        predicted = signals.sum(axis=0).reshape(bands[4].field.y.shape)
        before = parts(baseline.clusters[0], bp[baseline.indices[0]]).sum(axis=0).reshape(predicted.shape)
        f = bands[4].field
        raw_prediction, raw_before = [original_values(f, v) for v in (predicted, before)]
        observed = f.data['value']
        offsets = f.offsets*C/8e6
        local = p[model.indices[0]].reshape(n, 4)
        target = n-1
        peak = np.max(abs(observed))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained', sharex=True, sharey=True)
        for ax, value, title in zip(axes, (observed, observed-raw_before, observed-raw_prediction),
                ('Original ABC observations', 'Four-return prediction residual', 'With fixed-guide candidate: residual')):
            im = ax.imshow(20*np.log10(np.maximum(abs(value.T)/peak, 1e-6)),
                origin='lower', aspect='auto', cmap='magma', vmin=-45, vmax=0,
                extent=[f.time[0], f.time[-1], offsets[0]-C/16e6, offsets[-1]+C/16e6])
            ax.axhline(local[target, 0]*C/8e6, color='#00e5ff', lw=1.2)
            ax.set(title=title, xlabel='Capture time (s)')
        axes[0].set_ylabel('Range offset from saved AIS reference (m)')
        fig.colorbar(im, ax=axes, label='Magnitude relative to the same ABC measured maximum (dB)', shrink=.8)
        fig.suptitle(f'Guide {label}: selected ABC candidate at {local[target,0]*C/8e6:+.1f} m (cyan)\n'
                     'Frozen training fits; identical scale across original data and both residuals')
        fig.savefig(out/'01-selected-candidate.png', dpi=170)
        plt.close(fig)

        abc_phase = component_phase(model.clusters[0], p[model.indices[0]], partitions[4] == 3)[target]
        s = scene['labels'].index(label)
        histories = [v[s] for v in four_phase]+[abc_phase]
        coefficients = np.asarray(result['fixed_external_path_coefficients_m'])
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.3), layout='constrained')
        rms = []
        for ch, (history, b, colour) in enumerate(zip(histories, bands, COLOURS)):
            time = np.asarray(history['time_s'])
            u = (time-(first+last)/2)/b.scale
            curve = coefficients[0]*u+coefficients[1]*u*u
            phase = np.asarray(history['phase_residual_deg'])
            test = np.asarray(history['testing'])
            measured = curve-np.radians(phase)/b.field.k
            # These are circular innovations around the external guide. No
            # phase cycle is connected across a fade or inferred as observed.
            axes[0].plot(time[test], measured[test]*100, '.', ms=3, color=colour, alpha=.65, label=NAMES[ch])
            axes[1].plot(time[test], phase[test], '.', ms=3, color=colour, alpha=.75)
            rms.append(history['test_phase_rms_deg'])
        t = bands[0].field.time
        u = (t-(first+last)/2)/bands[0].scale
        axes[0].plot(t, 100*(coefficients[0]*u+coefficients[1]*u*u), 'k--', lw=1.5, label='Four-band guide')
        axes[0].set(title='Unused measured innovations on the guide branch',
                    xlabel='Capture time (s)', ylabel='Residual excess-path change (cm)')
        axes[1].set(title='Unused phase differences from fixed guide',
                    xlabel='Capture time (s)', ylabel='Conditional phase innovation (degrees)', ylim=(-180, 180))
        axes[1].axhline(0, color='#555', lw=.6)
        axes[2].bar(NAMES, rms, color=COLOURS)
        axes[2].set(title='Unused circular phase RMS', ylabel='Degrees')
        axes[2].tick_params(axis='x', rotation=25)
        axes[0].legend(ncol=2, fontsize=8)
        fig.suptitle(f'Candidate {label}: compare measured phase changes across all five channels\n'
                     'Neighbouring returns subtracted using frozen training fits; ABC rate and curvature fixed from the other four channels')
        fig.savefig(out/'02-five-channel-phases.png', dpi=170)
        plt.close(fig)

        old_error = result['baseline']['prediction']['testing'][0]['heldout_error']
        new_error = selected['prediction']['testing'][0]['heldout_error']
        reduction = 100*(1-new_error/old_error)
        separation = np.min(abs(local[:-1,0]-local[-1,0]))*C/8e6
        analysis = dict(accepted_calibration=False, guide_label=label,
            abc_range_origin_m=local[-1,0]*C/8e6, nearest_component_separation_m=separation,
            heldout_error_before=old_error, heldout_error_after=new_error,
            heldout_error_reduction_percent=reduction,
            active_brightness_condition=selected['active_brightness_condition'][0],
            component_to_sum_energy_ratio=selected['component_to_sum_energy_ratio'][0],
            all_trial_maximum_relative_cost_reproduction_error=max(errors),
            conditional_unused_phase_rms_deg=dict(zip(NAMES, rms)),
            scope='Prediction and phase-association control. The guide is independent of ABC, but these data have previously been inspected. Conditional phase innovations subtract other training-fitted returns and retain the guide phase branch. They are not independent raw phase unwraps or physical point identifications.',
            source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (source, guide_source, Path(__file__))})
        (out/'analysis.json').write_text(json.dumps(serial(analysis), indent=2, allow_nan=False))
        conclusion = ('This weaker candidate supports the four-band phase fingerprint without forcing the bright ABC return to share it. '
                      'It remains an unconfirmed range association: common phase change alone does not establish a material point on the hull.'
                      if folder == 'guided-abc-return' else
                      'Rejected as an identified association. The two nearly coincident components cancel strongly. '
                      'The small prediction improvement does not identify a second point or validate the common phase for the bright ABC return.')
        followup = ('<p><b>Later control:</b> this test rejects an extra nearly coincident component. It does not reject the existing bright main return. '
                    '<a href="../adjacent-scatterer-phases/index.html">The adjacent-window fits</a> share the existing A/B/C components while leaving D independent, '
                    'and incur only a small total-field prediction change. Those tests separate the main return from the trailing association.</p>'
                    if folder == 'guided-abc-main' else '')
        rows = ''.join(f'<tr><td>{i}</td><td>{v["cost"]:.3f}</td><td>{v["prediction"]["testing"][0]["heldout_error"]:.3f}</td><td>{v["maximum_projected_gradient"]:.3g}</td></tr>'
                       for i, v in enumerate(result['trials']))
        (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Guided phase association {label}</title>
<style>body{{font:16px/1.5 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:7px;border:1px solid #ddd}}table{{border-collapse:collapse}}.note{{padding:15px;background:#eef4f8}}</style>
<h1>Five-channel phase trajectories: candidate {label}</h1>
<p class="note">{conclusion}</p>
{followup}
<p>The selected ABC envelope origin is {analysis['abc_range_origin_m']:+.2f} m from the saved AIS range. Adding the fixed-guide candidate reduces final unused-quarter squared prediction error by {reduction:.2f}%. Brightness condition: {analysis['active_brightness_condition']:.2f}; component/sum energy ratio: {analysis['component_to_sum_energy_ratio']:.3f}; nearest other component: {separation:.3f} m.</p>
<a href="01-selected-candidate.png"><img src="01-selected-candidate.png" alt="Selected ABC cell and frozen prediction residuals"></a>
<a href="02-five-channel-phases.png"><img src="02-five-channel-phases.png" alt="Measured five-channel phase histories with fixed external guide"></a>
<p>The guide uses only Seven, SBS, Nine and Ten training data. ABC gets a constant phase, a bounded range origin and positive real brightness, with its residual path rate and curvature fixed. Four pre-existing ABC components remain independently fitted nuisance returns. This uses the exact original C0/C1 response, averaging and range covariance; no new background, sea or multipath model enters. Test points are repeating unused 125 ms blocks. Brightness is interpolated from training, and all other parameters remain frozen.</p>
<p>The phase dots are conditional circular innovations after subtracting fitted neighbouring returns. They are displayed on the guide branch; the raw single-cell histories are available in the <a href="../measured-scatterer-phases/index.html">unfitted measurement report</a>. These residual errors are not a radar SNR gain or a physical-coordinate confidence bound.</p>
<table><tr><th>Start</th><th>Training cost</th><th>Unused cost</th><th>Projected gradient</th></tr>{rows}</table>
<p>All saved objective values reproduce to relative error {max(errors):.2g}. <a href="fits.json">All starts</a> · <a href="analysis.json">Metrics and provenance</a> · <a href="tests.log">Numerical tests</a> · <a href="../phase-trajectories-review/index.html">Review index</a></p></html>''')
        print(out/'index.html', flush=True)


if __name__ == '__main__':
    main()
