"""Review the strongest legacy ship aperture without changing its data or fit.

The six-panel image uses saved complex backprojections, one fixed colour scale
and the original nominal band gains. New original-row histories test whether
constant band calibration and a common phase error suffice. Point selection
uses the full image, so temporal folds are conditional diagnostics, not blind
target discovery or independent-scatterer certification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from fit_autofocus import anchors, extract

NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']


def statistics(histories, first, last):
    """Integrate each channel at its own epochs; never interpolate RF phase."""
    scores, weights = [], []
    for h in histories:
        use = (h['time'] >= first) & (h['time'] < last)
        w = h['information'][use]
        scores.append(w @ h['value'][use])
        weights.append(w.sum())
    return np.array(scores), np.array(weights)


def coherence(scores, weights, gains):
    independent = np.sum(abs(scores)**2/weights[:, None], axis=0)
    coherent = abs(gains.conj() @ scores)**2/np.sum(abs(gains)**2*weights)
    return coherent, independent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('build/isar-20261015/heading-five-142-172'))
    p.add_argument('--replay', type=Path, default=Path('docs/data/isar_20261017/ship-baseline-replay'))
    p.add_argument('--executable', type=Path, default=Path('build/release/support/isar/isar_backprojection'))
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True); a.scratch.mkdir(parents=True, exist_ok=True)
    description = json.loads((a.replay/'analysis.json').read_text())
    amplitude, information = [], []
    for c in range(5):
        with np.load(a.replay/f'legacy-channel-{c}.npz') as z:
            amplitude.append(z['complex_image']); information.append(z['information'])
            along, port = z['along_m'], z['port_m']
    amplitude, information = np.array(amplitude), np.array(information)
    with np.load(a.replay/'legacy-combined.npz') as z:
        combined_power = 8192*z['information']*abs(z['complex_image'])**2
    power = 8192*information*abs(amplitude)**2
    independent = power.sum(axis=0)
    # The point selector takes power, not a formed image amplitude.
    pixels = anchors(np.sqrt(independent), along, port, count=8)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout='constrained', sharex=True, sharey=True)
    for ax, name, value in zip(axes.ravel(), NAMES+['Complex sum · nominal band calibration'], [*power, combined_power]):
        im = ax.imshow(10*np.log10(np.maximum(value, 1e-30)), origin='lower',
            extent=[along[0], along[-1], port[0], port[-1]], aspect='equal', cmap='magma',
            vmin=5, vmax=50, interpolation='nearest')
        ax.set(title=name, xlabel='Forward from AIS reference (m)', ylabel='Port (m)')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Matched power / conditional thermal variance (dB)', shrink=.8)
    fig.suptitle('Strongest retained ship-like aperture · 142–172 s\nOriginal complex images and nominal sum; fitted orientation is unvalidated; one fixed colour scale')
    fig.savefig(a.output/'best-ship-six-panels.png', dpi=150); plt.close(fig)
    histories, closure = [], []
    for c in range(5):
        saved = a.scratch/f'channel-{c}-histories.npz'
        if saved.exists():
            with np.load(saved) as z: h = dict(z)
            np.testing.assert_array_equal(h['pixels'], pixels)
        else:
            h = extract(a.source, a.scratch, c, a.executable, 20., pixels=pixels.copy())
        histories.append(h)
        score, weight = statistics([h], 142., 172.)
        expected = []
        for point in pixels:
            row = np.flatnonzero(port == point[1])[0]; col = np.flatnonzero(along == point[0])[0]
            expected.append(amplitude[c, row, col])
        measured = score[0]/weight[0]
        error = float(np.linalg.norm(measured-expected)/np.linalg.norm(expected))
        if error > 2e-5: raise AssertionError(f'Original image/history mismatch: {c}, {error}')
        closure.append(error)
        np.savez_compressed(a.output/f'channel-{c}-histories.npz', **h)
        print(f'channel {c}: history/image closure {error:.3g}', flush=True)
    # A contiguous temporal split exposes interpolation leakage across the
    # rapid flash. It is deliberately not used to select an accepted correction.
    train, wt = statistics(histories, 142., 157.)
    test, we = statistics(histories, 157., 172.)
    brightest = int(np.argmax(np.sum(abs(train)**2/wt[:, None], axis=0)))
    gains = np.exp(1j*np.angle(train[:, brightest]))
    complex_response = train[:, brightest]/wt
    complex_response /= max(abs(complex_response))
    rows = []
    for label, score, weight in [('first_half', train, wt), ('second_half', test, we)]:
        nominal, separate = coherence(score, weight, np.ones(5))
        calibrated, _ = coherence(score, weight, gains)
        response_calibrated, _ = coherence(score, weight, complex_response)
        rows.append(dict(fold=label, nominal_fraction_by_point=(nominal/separate).tolist(),
                         frozen_calibration_fraction_by_point=(calibrated/separate).tolist(),
                         frozen_complex_response_fraction_by_point=(response_calibrated/separate).tolist(),
                         nominal_sum_fraction=float(nominal.sum()/separate.sum()),
                         frozen_complex_response_sum_fraction=float(response_calibrated.sum()/separate.sum()),
                         frozen_calibration_sum_fraction=float(calibrated.sum()/separate.sum())))
    edges = np.arange(142., 172.01, 2.)
    block_score = []; block_weight = []
    for left, right in zip(edges[:-1], edges[1:]):
        score, weight = statistics(histories, left, right)
        block_score.append(score); block_weight.append(weight)
    block_score = np.array(block_score); block_weight = np.array(block_weight)
    # Choose a second spatial cell using only the first half. Same-channel
    # phase differences cancel any multiplicative phase common to both cells.
    order = np.argsort(np.sum(abs(train)**2/wt[:, None], axis=0))[::-1]
    neighbour = next(int(j) for j in order if j != brightest)
    relative = np.angle(block_score[:, :, neighbour]*block_score[:, :, brightest].conj())
    origin = np.angle(train[:, neighbour]*train[:, brightest].conj())
    relative = np.angle(np.exp(1j*(relative-origin)))
    power_blocks = 8192*abs(block_score)**2/block_weight[:, :, None]
    duration = 2*power_blocks[:, :, brightest].sum(axis=0)**2/np.sum(power_blocks[:, :, brightest]**2, axis=0)
    differential_checks = []
    for c in range(5):
        use = ((edges[:-1] >= 158.) & (edges[1:] <= 166.) &
               (power_blocks[:, c, brightest] >= 100) & (power_blocks[:, c, neighbour] >= 100))
        phase = relative[use, c]
        resultant = abs(np.mean(np.exp(1j*phase))) if len(phase) else np.nan
        sigma = np.degrees(np.sqrt(.5*(1/power_blocks[use, c, brightest]+1/power_blocks[use, c, neighbour])))
        differential_checks.append(dict(channel=c, blocks_used=int(use.sum()),
            circular_phase_spread_degrees=float(np.degrees(np.sqrt(-2*np.log(min(1.,resultant))))),
            maximum_conditional_thermal_phase_sigma_degrees=float(max(sigma)) if len(sigma) else None,
            scope='Both two-second cells at least 20 dB conditional power; 158–166 s. Noise formula assumes independent thermal errors; cell correlation and model error are not estimated.'))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    mid = .5*(edges[1:]+edges[:-1])
    for c, name in enumerate(NAMES):
        phase = np.angle(block_score[:, c, brightest]*gains[c].conj())
        axes[0, 0].plot(mid, np.degrees(phase), '.-', label=name)
        axes[0, 1].plot(mid, np.degrees(relative[:, c]), '.-', label=name)
        conditional_power = 8192*abs(block_score[:, c, brightest])**2/block_weight[:, c]
        axes[1, 0].plot(mid, 10*np.log10(np.maximum(conditional_power, 1e-30)), '.-', label=name)
    for record, style in zip(rows, ['o-', 's--']):
        axes[1, 1].plot(np.arange(len(pixels))+1, record['nominal_fraction_by_point'], style,
                       label=record['fold']+' nominal')
        axes[1, 1].plot(np.arange(len(pixels))+1, record['frozen_calibration_fraction_by_point'], style,
                       label=record['fold']+' fixed phases')
        axes[1, 1].plot(np.arange(len(pixels))+1, record['frozen_complex_response_fraction_by_point'], style,
                       label=record['fold']+' fixed complex response')
    axes[0, 0].set(title='Bright cell: constant band phases learned on first half', ylabel='Residual phase (degrees)', ylim=[-180, 180])
    axes[0, 1].set(title='Same-channel phase difference between two bright cells', ylabel='Change from first-half origin (degrees)', ylim=[-180, 180])
    axes[1, 0].set(title='Bright-cell two-second matched power', ylabel='Conditional power (dB)')
    axes[1, 1].set(title='Five-band coherent / independent matched energy', xlabel='Selected cell', ylabel='Fraction', ylim=[0, 1.01])
    for ax in axes.ravel(): ax.legend(fontsize=8, ncol=2)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]: ax.set_xlabel('Capture time (s)'); ax.axvline(157., color='grey', ls=':')
    fig.suptitle('New diagnostics on the exact best-image observations and motion\nCells were selected on the full image; histories are correlated mixtures; no correction is adopted')
    fig.savefig(a.output/'phase-and-calibration-transfer.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained', sharex=True, sharey=True)
    for ax, title, value in zip(axes, ['Sum of independent-band energies · diagnostic', 'Nominal complex sum'], [independent, combined_power]):
        im = ax.imshow(10*np.log10(np.maximum(value, 1e-30)), origin='lower', extent=[along[0], along[-1], port[0], port[-1]],
                       aspect='equal', cmap='magma', vmin=5, vmax=50, interpolation='nearest')
        ax.scatter(pixels[:, 0], pixels[:, 1], marker='+', color='cyan')
        for j, point in enumerate(pixels): ax.annotate(str(j+1), point[:2], xytext=(3, 5), textcoords='offset points', color='cyan')
        ax.set(title=title, xlabel='Forward from AIS reference (m)', ylabel='Port (m)')
    fig.colorbar(im, ax=axes.tolist(), label='Matched power / conditional thermal variance (dB)')
    fig.suptitle('Eight common cells selected on the full aperture\nTheir histories are correlated; the independent sum profiles more nuisance amplitudes and is not an SNR-efficiency bound')
    fig.savefig(a.output/'selected-cells.png', dpi=150); plt.close(fig)
    peak = np.unravel_index(np.argmax(independent), independent.shape)
    selected = (abs(np.meshgrid(along, port)[0]) < 200) & (abs(np.meshgrid(along, port)[1]) < 60)
    selected &= independent >= np.quantile(independent[selected], .99)
    sources = [Path(__file__), Path(__file__).with_name('fit_autofocus.py'), a.executable,
               a.replay/'analysis.json', *[a.replay/f'legacy-channel-{c}.npz' for c in range(5)],
               *[a.source/f'channel-{c}.bin' for c in range(5)]]
    source_hashes = {}
    for source in sources:
        with source.open('rb') as f: source_hashes[str(source)] = hashlib.file_digest(f, 'sha256').hexdigest()
    result = dict(complete=True, scope=__doc__, interval_s=[142., 172.],
        pixels_body_m=pixels.tolist(), calibration_anchor_point=brightest, comparison_point=neighbour,
        training_complex_response_real_imag=np.c_[complex_response.real,complex_response.imag].tolist(),
        bright_cell_two_second_power_participation_duration_s=duration.tolist(),
        participation_scope='2 * (sum P)^2 / sum P^2 across fifteen two-second matched-power bins. This describes measured power concentration, not an effective coherent aperture or a target information bound.',
        differential_phase_checks=differential_checks,
        history_image_relative_errors=closure, temporal_calibration_diagnostics=rows,
        nominal_fraction_at_independent_peak=float(combined_power[peak]/independent[peak]),
        nominal_fraction_bright_corridor=float(combined_power[selected].sum()/independent[selected].sum()),
        original_heading_rate_extrema_degps=description['heading_rate_extrema_degrees_per_s'],
        source_sha256=source_hashes, calibration_adopted=False, production_changes=False)
    (a.output/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
