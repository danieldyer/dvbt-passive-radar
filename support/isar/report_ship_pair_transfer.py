"""Report genuinely frozen-antenna interval transfer, including all failures."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/ship-pair-frozen-transfer/fits.json'))
    args=parser.parse_args();out=args.source.parent;data=json.loads(args.source.read_text())
    if not data['complete']:raise ValueError('Finish all declared transfer fits')
    if not all(r['antennas_unchanged'] for r in data['trials']):raise ValueError('Calibration was not frozen')
    best=[min([r for r in data['trials'] if r['assignment']==c],key=lambda r:r['cost']) for c in data['assignments']]
    error=lambda row:sum(r['heldout_error'] for r in row['prediction'])
    calibration_path=out.parent/'ship-physical-pair-sites-32/fits.json';cal=json.loads(calibration_path.read_text())
    source_best={c:min([r for r in cal['trials'] if r['assignment']==c],key=lambda r:r['cost']) for c in data['assignments']}
    refit_path=out.parent/'ship-physical-pair-Flash/stationarity.json';refit=json.loads(refit_path.read_text())
    reference=min(refit['trials'],key=lambda r:r['cost'])
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    labels=data['assignments']
    for i,selected in enumerate(best):
        for r in [r for r in data['trials'] if r['assignment']==selected['assignment']]:
            style=dict(marker='o' if r['audit']['strict_local_minimum'] else 'x',color='#24699b' if r is selected else '#9fa8ae')
            axes[0].plot(i,r['cost'],**style);axes[1].plot(i,error(r),**style)
        axes[2].plot(error(source_best[selected['assignment']]),error(selected),'o',label=selected['assignment'])
        axes[2].annotate(selected['assignment'],(error(source_best[selected['assignment']]),error(selected)),xytext=(4,4),textcoords='offset points',fontsize=8)
    for ax in axes[:2]:ax.set_xticks(range(len(labels)),labels,rotation=45)
    axes[0].set(ylabel='Flash training objective with frozen antennas',title='Two target starts per calibration')
    axes[1].set(ylabel='Flash unused-data squared error',title='Frozen moving and stationary test coefficients')
    axes[1].axhline(error(reference),color='#bd6b22',ls='--',label='00002 separately refitted antennas');axes[1].legend(fontsize=8)
    axes[2].set(xlabel='F01 unused-data error',ylabel='Flash unused-data error',title='Does one antenna solution transfer?')
    axes[2].margins(x=.18,y=.10)
    fig.suptitle('Coordinates estimated in F01 are held exactly fixed in Flash\nBlue: start selected by training · grey: alternate · ×: incomplete stationarity',fontsize=12)
    fig.savefig(out/'frozen-transfer.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{r["assignment"]}</td><td>{r["seed_rank"]}</td><td>{r["cost"]:.2f}</td><td>{error(r):.2f}</td><td>{r["audit"]["strict_local_minimum"]}</td><td>{r["bound_indices"]}</td></tr>' for r in best)
    summary=dict(scope=__doc__,accepted_calibration=False,assignments=labels,trials=len(data['trials']),
        antennas_unchanged=True,strict=sum(r['audit']['strict_local_minimum'] for r in data['trials']),
        selected=[dict(assignment=r['assignment'],seed_rank=r['seed_rank'],cost=r['cost'],heldout_error=error(r),
            strict=r['audit']['strict_local_minimum'],bounds=r['bound_indices']) for r in best],
        independently_refitted_00002_heldout_error=error(reference))
    summary['paired_interval_prediction_errors']={r['assignment']:error(source_best[r['assignment']])+error(r) for r in best}
    paired=summary['paired_interval_prediction_errors']
    difference=100*abs(paired['10002']-paired['00002'])/min(paired['10002'],paired['00002'])
    summary['00002_vs_10002_combined_error_difference_percent']=difference
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen-antenna ship interval transfer</title>
<style>body{{font:16px/1.55 system-ui;max-width:1250px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:8px}}</style>
<h1>Can antennas fitted in F01 predict the separate flash?</h1>
<p>These fits hold all transmitter and receiver offsets exactly fixed at their F01 values. The flash primary point state, constant phases, real brightnesses, stationary coefficients and secondary trajectories are estimated only from the flash training samples. Every unused-sample prediction freezes those coefficients. This separates transfer of antenna geometry from an unrestricted new coordinate fit; it does not require the same material scatterer to dominate both intervals.</p>
<p>Selection policy: {data['selection_policy']} Two distinct flash associations are initialized for every selected calibration. Starts are ranked by training cost; incomplete convergence remains visible.</p>
<img src="frozen-transfer.png" alt="Frozen-antenna training and unused-data prediction errors in the separate flash interval">
<table><tr><th>Assignment</th><th>Selected target start</th><th>Training objective</th><th>Unused-data error</th><th>Strict minimum</th><th>Bound indices</th></tr>{rows}</table>
<p>All {len(data['trials'])} declared transfer trials are complete; {summary['strict']} pass the actual-objective curvature and KKT audit. The antenna vector remains bit-for-bit unchanged in every fit. The separately refitted 00002 control has unused-data error {error(reference):.2f}; the horizontal reference is diagnostic, not a universal optimum, and does not refit test coefficients.</p>
<p>The combined unused-data squared errors in F01 and Flash are {paired['00002']:.2f} for 00002 and {paired['10002']:.2f} for 10002, a difference of only {difference:.3f}%. Those two hypotheses differ in Seven's site (Ornata versus Eyre). This is a descriptive prediction comparison, not a likelihood-ratio significance result or proof that every other association has been excluded.</p>
<p>No calibration is accepted merely because it produces a converged short-interval solution. Scene association, remaining bounds and information must be considered together with these errors. These intervals have already been inspected, so this is a conditional transfer test, not a blind external validation dataset.</p>
<p><a href="fits.json">All frozen-coordinate fits</a> · <a href="analysis.json">Comparisons and provenance</a> · <a href="../ship-physical-pair-sites-32/index.html">All 32 source assignments</a> · <a href="../ship-physical-pair-Flash-review/index.html">Separate flash model diagnosis</a>.</p></html>'''
    (out/'index.html').write_text(page)
    paths=[args.source,calibration_path,refit_path,Path(__file__)]
    summary['sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    (out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('report',out/'index.html','trials',len(data['trials']),'strict',summary['strict'],flush=True)


if __name__=='__main__':main()
