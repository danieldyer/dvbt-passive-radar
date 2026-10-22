"""Compare one and two local echoes on identical untouched complex samples."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cluster_pair import Pair
from compare_ais_phase import NAMES
from motion import C
from phase_cluster_report import run_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261016/flash-pair'))
    parser.add_argument('--interval', type=float, nargs=2, default=[164., 165.])
    parser.add_argument('--widths', type=int, nargs='+', default=[5,9])
    parser.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for width in args.widths:
        if width%2 != 1 or width < 3 or width > 13:
            raise ValueError('Use odd range widths from 3 through 13')
        for channel in args.channels:
            path = args.output/f'channel-{channel}-width-{width}.json'
            if path.exists() and args.reuse:
                record = json.loads(path.read_text())
                if record['interval_s'] != args.interval:
                    raise ValueError('Do not reuse a different aperture')
            else:
                if path.exists():
                    raise ValueError(f'Existing result: {path}; use a new output directory or --reuse')
                with np.load(args.fields/f'channel-{channel}.npz') as data:
                    single, first, row = run_fit(data, *args.interval, radius=width//2, origin_width=1.)
                problem = Pair(single.field, single.training)
                secondary_bounds = (-3.,3.)
                starts = problem.initial(first.parameters, secondary_bounds, rate_bound=2., count=6)
                fit = problem.fit(starts, [secondary_bounds,secondary_bounds])
                diag = problem.diagnostics(fit)
                error_single = (1-row['diagnostics']['test_complex_prediction_fraction'])*np.sum(abs(problem.field.y[~problem.training])**2)
                record = {'channel':channel, 'name':NAMES[channel], 'width_cells':width,
                    'interval_s':args.interval, 'epoch_s':single.epoch, 'scale_s':single.scale,
                    'single':row, 'pair':{'parameters':fit.parameters.tolist(), 'objective':fit.objective,
                        'maximum_gradient':fit.gradient, 'trials':fit.trials,
                        'conditional_parameter_covariance':fit.covariance.tolist(),
                        'information_rank':fit.information_rank, 'diagnostics':diag,
                        'origin_bounds_samples':[secondary_bounds,secondary_bounds]},
                    'withheld_error_reduction_fraction':float(1-diag['test_squared_error']/error_single),
                    'source_fields':str(args.fields), 'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__),Path(__file__).with_name('cluster_pair.py'),Path(__file__).with_name('cluster_phase.py')]},
                    'accepted_geometric_calibration':False}
                path.write_text(json.dumps(record, indent=2, allow_nan=False))
            records.append(record)
            print(NAMES[channel], width, 'heldout error reduction',round(100*record['withheld_error_reduction_fraction'],2),
                  'pair prediction',round(record['pair']['diagnostics']['test_complex_prediction_fraction'],3),
                  'gradient',record['pair']['maximum_gradient'],flush=True)
    fig, axes = plt.subplots(1,2,figsize=(13,5),layout='constrained')
    def usable(r):
        best=min(r['pair']['trials'],key=lambda t:t['objective'])
        return (r['pair']['information_rank']==8 and best.get('maximum_projected_gradient',np.inf)<1e-6
                and not best.get('bound_parameter_indices',[0]))
    for width in args.widths:
        rows = [r for r in records if r['width_cells']==width]
        x = np.array([r['channel'] for r in rows])
        axes[0].plot(x,[100*r['withheld_error_reduction_fraction'] for r in rows],'.-',label=f'{width} range cells')
        rejected=[r for r in rows if not usable(r)]
        axes[0].plot([r['channel'] for r in rejected],[100*r['withheld_error_reduction_fraction'] for r in rejected],
            'x',color='black',ms=10,mew=1.5,label='Not a stable localized pair' if width==args.widths[0] else None)
        for component,style in [(0,'o-'),(1,'s--')]:
            delay = [r['pair']['parameters'][4*component]*C/8e6 for r in rows]
            axes[1].plot(x,delay,style,label=f'{width} cells, component {component+1}')
    for ax in axes:
        ax.set_xticks(args.channels,[NAMES[c] for c in args.channels]);ax.axhline(0,color='#777777',lw=.7);ax.legend(fontsize=8)
    axes[0].set_ylabel('Reduction in withheld squared error (%)')
    axes[1].set_ylabel('Fitted envelope offset from saved AIS reference (m)')
    fig.suptitle(f'{args.interval[0]:g}–{args.interval[1]:g} s · two echoes versus one\nConstant phase per echo; training-only amplitudes predict withheld 125 ms blocks')
    fig.savefig(args.output/'comparison.png',dpi=170);plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    table=''
    for r in records:
        trial=min(r['pair']['trials'],key=lambda t:t['objective'])
        table+=f"<tr><td>{r['name']}</td><td>{r['width_cells']}</td><td>{100*r['withheld_error_reduction_fraction']:.1f}%</td><td>{r['pair']['information_rank']}/8</td><td>{trial.get('maximum_projected_gradient',np.nan):.2g}</td><td>{trial.get('bound_parameter_indices',[])}</td><td>{'Pass numerical checks' if usable(r) else 'Reject for calibration'}</td></tr>"
    (args.output/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>One and two local echoes</title>
<style>body{{font:16px/1.55 system-ui;max-width:1150px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:7px;border:1px solid #ccc}}</style>
<h1>One versus two local echoes</h1><p>Both models use the same unmodified complex samples, original-symbol forward operator and conditional thermal covariance. Two nonnegative real strengths per training bin are profiled exactly. Every component retains a constant phase and quadratic physical path. Unused 125 ms blocks use frozen parameters and interpolated training strengths. No test strength is refitted.</p>
<img src="comparison.png" alt="Withheld error improvement and fitted component delays">
<p>Improved prediction is evidence against an isolated-point model, not proof of two uniquely identified physical points. In particular, two almost coincident components with opposing phases can develop very large cancelling strengths. These fits lose information rank or fail stationarity and are rejected. Range-search boundary hits also prevent a component being treated as localized. Even the remaining fits must be stable to support width and agree with a shared physical multichannel model before calibrating antenna positions.</p>
<table><tr><th>Band</th><th>Cells</th><th>Withheld error reduction</th><th>Information rank</th><th>Projected gradient</th><th>Active parameter bounds</th><th>Numerical status</th></tr>{table}</table>
<p>Numerical status requires full conditional rank, projected gradient below 10⁻⁶ and no active search bounds. This is not a physical model validation or a declaration of Gaussian residuals. <a href="summary.json">All fits, covariance matrices, source hashes and diagnostics</a>.</p></html>''')


if __name__ == '__main__':
    main()
