"""Report paired no-yaw statistics across coherent integration durations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fits', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = []
    for root in args.fits:
        for free_path in sorted(root.glob('window-*-free.json')):
            free = json.loads(free_path.read_text())
            prefix = free_path.name.removesuffix('-free.json')
            candidates = [(path, json.loads(path.read_text())) for path in sorted(root.glob(prefix+'-fixed-ne*.json'))]
            if not candidates:
                raise ValueError('Each free-yaw fit needs its matching fixed-frame starts')
            minimum = min(m['fit']['objective'] for _, m in candidates)
            # A checked solution is preferred only within a declared training
            # numerical tie. Withheld scores never enter start selection.
            stationary = [(path, m) for path, m in candidates
                if m['fit']['converged'] and m['fit']['objective'] <= minimum+1e-10]
            fixed_path, fixed = min(stationary or candidates, key=lambda v: v[1]['fit']['objective'])
            if free['interval_s'] != fixed['interval_s']:
                raise ValueError('Compared models use different intervals')
            records.append({'window': int(prefix.split('-')[1]), 'duration_s': float(np.diff(free['interval_s'])[0]),
                'interval_s': free['interval_s'], 'free_path': str(free_path), 'fixed_path': str(fixed_path),
                'input_sha256': {str(path): fingerprint(path) for path in [free_path, *[p for p, _ in candidates]]},
                'paired_convergence': bool(free['fit']['converged'] and fixed['fit']['converged']),
                'heldout_error_change_percent': float(100*(fixed['after_summary']['prediction_error']/free['after_summary']['prediction_error']-1)),
                'heldout_matched_energy_change_db': float(10*np.log10(fixed['after_summary']['matched_energy']/free['after_summary']['matched_energy'])),
                'free': {'fit': free['fit'], 'validation': free['after_summary']},
                'fixed': {'fit': fixed['fit'], 'validation': fixed['after_summary']},
                'fixed_starts': [{'path': str(path), 'objective': m['fit']['objective'], 'converged': m['fit']['converged'],
                    'kkt': m['fit']['maximum_kkt_residual'], 'validation': m['after_summary']} for path, m in candidates]})
    records.sort(key=lambda r: (r['duration_s'], r['window']))
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    x = np.arange(len(records))
    labels = [f'{r["duration_s"]:g} s\nwindow {r["window"]}' for r in records]
    for ax, key, ylabel in zip(axes, ('heldout_error_change_percent', 'heldout_matched_energy_change_db'),
            ('Change in withheld squared error (%)', 'Change in withheld matched energy (dB)')):
        for i, r in enumerate(records):
            ax.bar(i, r[key], color='#25659b' if r['paired_convergence'] else '#dddddd',
                hatch=None if r['paired_convergence'] else '///', edgecolor='#555')
        ax.set(xticks=x, xticklabels=labels, ylabel=ylabel)
        ax.axhline(0, color='black', lw=.7)
    axes[0].set_title('Higher is worse')
    axes[1].set_title('Lower is worse')
    fig.suptitle('Fixed N/E translation versus fitted yaw\nHatched bars: provisional fits that fail the numerical convergence gate')
    fig.savefig(args.output/'fixed-ne-duration-statistics.png', dpi=170)
    plt.close(fig)
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'records': records,
        'selection': 'Minimum training objective; prefer a stationary solution only within an absolute 1e-10 objective tie.',
        'scope': 'Conditional paired model comparison with previously selected reflectors, independent band amplitudes and nominal antenna geometry. Shorter and longer windows are nested, not independent trials.'}
    (args.output/'fixed-ne-duration-statistics.json').write_text(json.dumps(result, indent=2)+'\n')
    rows = ''.join(f'<tr><td>{r["duration_s"]:g}</td><td>{r["interval_s"][0]:.2f}–{r["interval_s"][1]:.2f}</td>'
        f'<td>{r["heldout_error_change_percent"]:+.3f}%</td><td>{r["heldout_matched_energy_change_db"]:+.3f} dB</td>'
        f'<td>{"Pass" if r["paired_convergence"] else "Provisional: convergence failed"}</td>'
        f'<td>{r["free"]["fit"]["maximum_kkt_residual"]:.2g} / {r["fixed"]["fit"]["maximum_kkt_residual"]:.2g}</td></tr>' for r in records)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Yaw-free imaging versus coherence length</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#243040}}img{{max-width:100%;height:auto}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse;font-size:15px}}.note{{background:#fff0ce;padding:18px}}</style>
<h1>Fixed north/east imaging: 3-, 10- and 30-second controls</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="fixed-ne-validation.html">Original paired geographic images and method</a> · <a href="joint-sites-validation.html">Joint site and motion sweep</a> · <a href="fixed-ne-duration-statistics.json">All fits and input hashes</a></p>
<p class="note">Removing yaw has not improved the converged 10- or 30-second comparisons. The 3-second comparisons remain provisional: the point model is poorly conditioned and some fits fail the stationarity test. Their apparent gain or loss cannot be treated as a resolved model comparison.</p>
<p>Every fixed-frame fit uses geographic east/north axes throughout its integration interval. Physical translation is refitted. Five translation starts are tested; selection uses training likelihood only. A converged solution is preferred among numerical ties within 10⁻¹⁰ in the normalized training objective. The table retains failures rather than quietly selecting a worse training solution that happens to converge.</p>
<table><tr><th>Duration (s)</th><th>Capture interval (s)</th><th>Withheld error change</th><th>Matched-energy change</th><th>Paired numerical check</th><th>KKT residual, free / fixed</th></tr>{rows}</table>
<img src="fixed-ne-duration-statistics.png" alt="Conditional no-yaw statistics for three integration durations">
<h2>Interpretation</h2>
<p>These are matched comparisons within each duration, not an absolute SNR comparison across durations. Shorter windows contain fewer samples, use a correspondingly rebased motion spline and retain the same previously selected reflector support. They are nested in the 30-second windows. A negative matched-energy change means the yaw-free training template captures less of the projected withheld observation.</p>
<p>The fixed support was selected in earlier processing. A free-yaw parameter may compensate for a poor scattering model, incorrect support or geometry rather than measure the ship's heading. These results therefore do not vindicate the implausible heading oscillations in the earlier video. They do show that simply freezing that operator and refitting translation does not recover the missing signal in the converged comparisons.</p>
<p>The 3-second runs expose a separate numerical/model-order limitation: eight complex reflectors per band can become difficult to distinguish over a short aperture. Several fits reach large dictionary condition numbers or different local stationary points. The free-yaw family contains the no-yaw family, so a worse training optimum is itself evidence that a local search cannot certify the comparison. No numerical threshold was relaxed and no 3-second gain is adopted.</p>
<p>The 48-site joint refits test whether the yaw-free result can be rescued by transmitter/receiver and envelope-origin corrections. Their separate report retains the unknown-reflector-coordinate sensitivity, prior dependence and all trial locations. Cross-band phase calibration and a single physical trajectory throughout the recording remain unresolved work.</p></html>'''
    (args.output/'fixed-ne-duration-validation.html').write_text(page)
    print(json.dumps([{k:r[k] for k in ('window', 'duration_s', 'paired_convergence', 'heldout_error_change_percent', 'heldout_matched_energy_change_db')} for r in records], indent=2))


if __name__ == '__main__':
    main()
