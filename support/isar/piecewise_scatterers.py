"""Discover and validate short scatterer arcs without joining their phases.

Discovery is a training-only multiband matched-likelihood search over delay
and residual path rate. Each band has its own constant complex gain. Twelve
range/time regions provide distinct candidate observations, not a claim of
twelve identified permanent physical reflectors. Half of every arc remains
withheld through both discovery and subsequent joint delay/phase refinement.
"""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

from compare_ais_phase import NAMES, COLOURS
from motion import C
from phase_cluster_report import run_fit


def discover(directory):
    seconds = np.arange(90, 330)
    origins = np.arange(-6, 7)
    rates = np.linspace(-4, 4, 97)
    score = np.zeros((len(origins), len(seconds), len(rates)))
    for ch in range(5):
        with np.load(directory/f'channel-{ch}.npz') as data:
            time, info, value = data['time'], data['information'], data['value']
            pulse, offsets = data['pulse_samples'], data['offset_samples']
            k = 2*np.pi*json.loads(str(data['metadata_json']))['rf_hz']/C
            for j, origin in enumerate(origins):
                use = np.flatnonzero(abs(offsets-origin)<=2)
                r = offsets[use]
                covariance = pulse[r[:, None]-r[None, :]+64]
                whiten = solve_triangular(np.linalg.cholesky((covariance+covariance.conj().T)/2), np.eye(len(r)), lower=True)
                h = whiten@pulse[r-origin+64]
                product = (value[:, use]@whiten.T)@h.conj()
                norm = float(np.vdot(h, h).real)
                for n, second in enumerate(seconds):
                    selected = (time>=second)&(time<second+1)&((((time-second)//.125).astype(int)%2)==0)&(info>0)
                    t = time[selected]-(second+.5)
                    weight = 8192*info[selected]
                    if len(t)<4:
                        continue
                    cross = (weight*product[selected])@np.exp(1j*k*t[:, None]*rates[None, :])
                    score[j, n] += abs(cross)**2/(np.sum(weight)*norm)
        print('discovery', NAMES[ch], flush=True)
    maximum = score.max(axis=2)
    chosen = []
    for first, last in ((90, 150), (150, 210), (210, 270), (270, 330)):
        for low, high in ((-6, -3), (-2, 2), (3, 6)):
            gate = ((origins[:, None]>=low)&(origins[:, None]<=high)&
                    (seconds[None, :]>=first)&(seconds[None, :]<last)&(abs(seconds[None, :]+.5-164.5)>=4))
            index = np.argmax(np.where(gate, maximum, -np.inf))
            r, t = np.unravel_index(index, maximum.shape)
            chosen.append({'label': f'P{len(chosen)+1:02d}', 'first_s': int(seconds[t]), 'last_s': int(seconds[t]+1),
                           'centre_sample': int(origins[r]), 'training_matched_energy': float(maximum[r, t]),
                           'seed_residual_path_rate_mps': float(rates[np.argmax(score[r, t])]),
                           'discovery_region_s': [first, last], 'discovery_region_samples': [low, high]})
    return {'seconds': seconds.tolist(), 'origins': origins.tolist(), 'maximum_training_matched_energy': maximum.tolist(),
            'candidates': chosen, 'scope': __doc__}


def refine(directory, discovered):
    candidates = [{**candidate, 'channels': []} for candidate in discovered['candidates']]
    for ch in range(5):
        with np.load(directory/f'channel-{ch}.npz') as data:
            for candidate in candidates:
                p, fit, row = run_fit(data, candidate['first_s'], candidate['last_s'],
                                     centre=candidate['centre_sample'], rate_bound=4.)
                row['rf_hz'] = p.field.metadata['rf_hz']
                row['channel'] = NAMES[ch]
                row['range_search_bound_hit'] = bool(abs(fit.parameters[0]-candidate['centre_sample'])>.7999)
                candidate['channels'].append(row)
        print('piecewise fits', NAMES[ch], flush=True)
    return candidates


def plots(discovery, candidates, output):
    energy = np.array(discovery['maximum_training_matched_energy'])
    times, origins = np.array(discovery['seconds'])+.5, np.array(discovery['origins'])*C/8e6
    fig, ax = plt.subplots(figsize=(15, 5), layout='constrained')
    view = ax.imshow(10*np.log10(np.maximum(energy, 1e-30)), origin='lower', aspect='auto', cmap='magma',
                     extent=[times[0]-.5, times[-1]+.5, origins[0]-C/16e6, origins[-1]+C/16e6])
    for c in candidates:
        x, y = (c['first_s']+c['last_s'])/2, c['centre_sample']*C/8e6
        ax.plot(x, y, 's', mfc='none', mec='#00ddff', ms=12)
        ax.annotate(c['label'], (x, y), xytext=(6, 8), textcoords='offset points', color='#00ddff', fontsize=9)
    ax.set_xlabel('Recording time (s)');ax.set_ylabel('Excess-path offset from AIS (m)')
    ax.set_title('Twelve short candidate arcs · selected using training samples only\n'
                 'Fixed range/time regions; unknown band phases profiled, residual rate shared during discovery')
    fig.colorbar(view, ax=ax, label='Summed conditional matched energy (dB)')
    fig.savefig(output/'04-piecewise-candidates.png', dpi=170);plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), layout='constrained')
    for c, ax in zip(candidates, axes.flat):
        for row, colour in zip(c['channels'], COLOURS):
            d = row['diagnostics'];t = np.array(d['time_s']);training = np.array(d['training'])
            phase = np.array(d['model_residual_phase_rad']);origin = np.interp((c['first_s']+c['last_s'])/2, t, phase)
            k = 2*np.pi*row['rf_hz']/C
            observed = phase+np.radians(d['phase_residual_deg'])
            ax.plot(t, -100*(phase-origin)/k, color=colour, lw=1, label=row['channel'])
            ax.plot(t[~training], -100*(observed[~training]-origin)/k, '.', color=colour, ms=2, alpha=.65)
        ax.set_title(f"{c['label']} · {c['first_s']}–{c['last_s']} s · {c['centre_sample']*C/8e6:+.0f} m")
        ax.set_xlabel('Recording time (s)');ax.set_ylabel('Residual path change (cm)')
    axes[0, 0].legend(ncol=3, fontsize=7)
    fig.suptitle('Independent local path fits and withheld phase observations\n'
                 'Constant phase origins removed separately in each arc/band; no phase connection between arcs is imposed')
    fig.savefig(output/'05-piecewise-phase-fits.png', dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout='constrained')
    rms = np.array([[r['diagnostics']['test_phase_rms_deg'] for r in c['channels']] for c in candidates])
    prediction = np.array([[100*r['diagnostics']['test_complex_prediction_fraction'] for r in c['channels']] for c in candidates])
    for ax, matrix, label, low, high in zip(axes, (rms, prediction),
            ('Withheld phase RMS (degrees)', 'Withheld complex prediction (%)'), (0, 0), (90, 100)):
        view = ax.imshow(matrix, cmap='viridis_r' if ax is axes[0] else 'viridis', aspect='auto', vmin=low, vmax=high)
        ax.set_xticks(np.arange(5), NAMES)
        ax.set_yticks(np.arange(12), [f"{c['label']} · {c['first_s']} s" for c in candidates])
        for i in range(12):
            for j in range(5):
                ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', color='white' if matrix[i,j] < (low+high)/2 else 'black', fontsize=9)
        fig.colorbar(view, ax=ax, label=label)
    fig.suptitle('Prediction quality for every selected arc · weak/inconsistent candidates retained for inspection')
    fig.savefig(output/'06-piecewise-validation.png', dpi=170);plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261016/flash-siting'))
    p.add_argument('--reuse', action='store_true')
    args = p.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    discover_path = args.output/'piecewise-discovery.json'
    discovery = json.loads(discover_path.read_text()) if args.reuse and discover_path.exists() else discover(args.fields)
    discover_path.write_text(json.dumps(discovery, indent=2))
    result_path = args.output/'piecewise-fits.json'
    candidates = json.loads(result_path.read_text()) if args.reuse and result_path.exists() else refine(args.fields, discovery)
    result_path.write_text(json.dumps(candidates, indent=2))
    plots(discovery, candidates, args.output)
    print('Completed 12 arcs × 5 independent band fits', flush=True)


if __name__ == '__main__':
    main()
