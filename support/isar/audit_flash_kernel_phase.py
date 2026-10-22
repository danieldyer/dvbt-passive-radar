"""Measure known range-template phase bias at the selected flash returns.

Use the previously exported transmitted-waveform autocorrelations with the
original PROF weights. Bind each known 16-symbol kernel to its original rows;
retain the current observations, range covariance, moment operator and paths.
This is an operator check, not a new phase correction or location estimate.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from block_pulse import BlockPulse
from field import average
from fit_fixed_grid_phases import prepare_fixed
from report_measured_scatterer_phases import load_fields, serial
from shared_echo_cluster import ReferencedAmplitudeArc, SharedEchoCluster
from waveform_kernels import KernelField, read_kernels


def main():
    root = Path('docs/data/isar_20261017')
    out = root/'flash-kernel-phase'
    out.mkdir(exist_ok=True)
    source = root/'fixed-grid-multiple-phases/fits.json'
    fits = json.loads(source.read_text())
    geometry = Path('docs/data/calibration_20261016_sites')
    fields = load_fields(Path('build/isar-20261015/full-fields'), geometry)
    bands, _, _ = prepare_fixed(fields, geometry, *fits['interval_s'], stationary=True)
    kernel_root = Path('build/isar-20261016/waveform-kernels-30s')
    kernel_path = kernel_root/'response.kernels'
    header, records, pulses = read_kernels(kernel_path)
    request = np.load(kernel_root/'rows.npz')
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), layout='constrained')
    results = []
    for c, (band, entry) in enumerate(zip(bands, fits['independent'])):
        grid = request[f'channel_{c}']
        indices = np.searchsorted(grid['time'], band.field.data['row_time'])
        if np.any(indices >= len(grid)):
            raise ValueError('Known waveform kernels do not cover the flash')
        np.testing.assert_allclose(grid['time'][indices], band.field.data['row_time'], rtol=0, atol=2e-8)
        np.testing.assert_array_equal(grid['information'][indices], band.field.data['row_information'])
        original_blocks = indices//header['block_rows']
        used, mapping = np.unique(original_blocks, return_inverse=True)
        if not all(records[c][i]['valid'] for i in used):
            raise ValueError('An unavailable reconstructed waveform cannot be fitted')
        corrected_field = KernelField(band.field, BlockPulse(pulses[c][used]), mapping)
        np.testing.assert_array_equal(corrected_field.y, band.field.y)
        corrected = ReferencedAmplitudeArc(corrected_field, band.training, band.knots,
            *fits['interval_s'], band.known_path, band.known_rate,
            stationary_design=band.design, curve=band.base.curve)
        model = SharedEchoCluster([band], fits['selected_origins_samples'],
            origin_radius=fits['origin_radius_samples'])
        best = entry['trials'][entry['best_index']]
        local = np.asarray(best['parameters'])[model.indices[0]].reshape(4, 4)
        for s, (label, p) in enumerate(zip(fits['labels'], local)):
            raw_mean = band.response(p)[0]
            raw_known = corrected.response(p)[0]
            mean = average(raw_mean, band.field.data['row_information'], band.field.metadata['block_rows'])[0]
            known = average(raw_known, band.field.data['row_information'], band.field.metadata['block_rows'])[0]
            a, b = band.field.apply(mean), band.field.apply(known)
            product = np.sum(a.conj()*b, axis=1)
            phase = np.unwrap(np.angle(product))
            phase -= np.interp(band.epoch, band.field.time, phase)
            # A single range cell can have a different phase sensitivity from
            # a full covariance-weighted pulse coefficient. Keep both visible.
            column = np.argmax(np.mean(abs(mean)**2, axis=0))
            cell_phase = np.unwrap(np.angle(mean[:, column].conj()*known[:, column]))
            cell_phase -= np.interp(band.epoch, band.field.time, cell_phase)
            axes[s, 0].plot(band.field.time, np.degrees(phase), label=entry['name'])
            axes[s, 1].plot(band.field.time, np.degrees(cell_phase), label=entry['name'])
            power = np.sum(abs(a)**2, axis=1)
            loss = 10*np.log10(np.vdot(a, a).real*np.vdot(b, b).real/abs(np.vdot(a, b))**2)
            results.append(dict(channel=c, name=entry['name'], return_label=label,
                origin_samples=p[0], known_kernel_blocks=len(used),
                time_s=band.field.time, matched_phase_change_deg=np.degrees(phase),
                single_cell_phase_change_deg=np.degrees(cell_phase),
                matched_phase_rms_deg=np.degrees(np.sqrt(np.average(phase**2, weights=power))),
                maximum_matched_phase_change_deg=np.degrees(np.max(abs(phase))),
                maximum_single_cell_phase_change_deg=np.degrees(np.max(abs(cell_phase))),
                conditional_template_mismatch_db=float(loss)))
        print(entry['name'], 'known original row identities checked', flush=True)
    for s, label in enumerate(fits['labels']):
        for ax in axes[s]:
            ax.set(xlabel='Capture time (s)', ylabel='Template-induced phase change (°)', xlim=fits['interval_s'])
            ax.axhline(0, color='#aaa', lw=.5)
        axes[s, 0].set_title(label+' · covariance-weighted full-pulse coefficient')
        axes[s, 1].set_title(label+' · strongest individual range cell')
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle('Can the mean range-pulse approximation explain the measured flash phase differences?\n'
        'Known waveform kernels and original capture weights, fixed paths and observations. '
        'One constant at 165 s removed.', fontsize=12)
    fig.savefig(out/'phase-bias.png', dpi=150)
    plt.close(fig)
    sources = [Path(__file__), source, kernel_path, kernel_root/'rows.npz',
        *[Path('support/isar')/(n+'.py') for n in ('block_pulse','waveform_kernels',
        'fit_fixed_grid_phases','field','shared_echo_cluster')]]
    report = dict(scope=__doc__, complete=True, interval_s=fits['interval_s'],
        production_changed=False, phase_correction_applied=False, results=results,
        limits='Kernels are exact weighted circular autocorrelations per exported 16-symbol block, not per-row finite-waveform ambiguity functions. Rows retain their original identities even though the current field averaging groups differ. The unchanged mean covariance isolates forward-template effects. This excludes neither other operator errors nor uncertainty in fitted neighbour/static subtraction.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'analysis.json').write_text(json.dumps(serial(report), indent=2, allow_nan=False))
    print(out/'analysis.json', flush=True)


if __name__ == '__main__':
    main()
