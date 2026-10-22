"""Review physical unknown-image/common-path acquisition without focus claims."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--synthetic',type=Path,required=True)
    p.add_argument('--recorded',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    synthetic=json.loads((a.synthetic/'analysis.json').read_text());recorded=json.loads((a.recorded/'analysis.json').read_text())
    u=np.linspace(-1,1,301);basis=np.c_[u,u*u,u*u*u]
    truth=np.asarray(synthetic['truth']['truth_residual_path_coefficients_m']);fitted=np.asarray(synthetic['parameters_m'])
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    axes[0].plot(u,basis@truth,label='Hidden physical truth');axes[0].plot(u,basis@fitted,'--',label='Unknown-image joint estimate')
    axes[0].set(xlabel='Normalized time over 30 s aperture',ylabel='Residual bistatic path (m)',title='Physical off-grid eleven-point control');axes[0].legend()
    names=['Seven','SBS','Nine','Ten','ABC']
    values=[max(r['coherence_loss_db'] for r in synthetic['known_point_coherence'] if r['channel']==c) for c in range(5)]
    axes[1].bar(names,values);axes[1].set(ylabel='Worst point matched-power loss (dB)',title='Given coarse trajectory, attitude and calibration',ylim=(0,max(.03,max(values)*1.15)))
    axes[1].tick_params(axis='x',rotation=30)
    for d,label in [(synthetic,'Synthetic'),(recorded,'Recorded')]:
        print(json.dumps(dict(case=label,initial_outer=d['initial_outer'],outer=d['outer'],exact_physical_outer=d['exact_physical_outer'])),flush=True)
    axes[2].plot(u,basis@np.asarray(recorded['parameters_m']),color='#9f1239')
    axes[2].set(xlabel='Normalized time over 30 s aperture',ylabel='Additional fitted path (m)',title='Recorded flash: small residual correction')
    fig.suptitle('Unknown image and nonlinear motion are fitted together; original epochs, one physical path across all RF bands')
    fig.savefig(a.output/'physical-acquisition-control.png',dpi=150);plt.close(fig)
    relative=lambda target:Path(__import__('os').path.relpath(target,a.output)).as_posix()
    html=f'''<!doctype html><meta charset="utf-8"><title>Joint image and phase acquisition</title>
<style>body{{font:17px/1.5 system-ui;max-width:1300px;margin:35px auto;padding:0 20px}}img{{width:100%}}.status{{background:#fff1d6;padding:15px}}</style>
<h1>Joint unknown-image and common-path acquisition</h1>
<p class="status">The whole-capture focused-ship goal is still active. This is a validated local acquisition component, not an accepted recorded hull reconstruction or transmitter-position calibration.</p>
<p>Each trajectory trial re-estimates one complex image across all five bands with a uniform complex-Laplace prior. Point positions are not given to the estimator. A data-operator modal reduction and Chebyshev phase moments are numerical accelerations; the objective, gradient and nuisance projection are checked against the original rows. The linear phase/image-shift gauge is fixed at its coarse acquisition value; nonlinear phase is estimated.</p>
<img src="physical-acquisition-control.png">
<p>The physical synthetic control uses eleven off-grid points, the recorded waveform kernels, all original sample times, independent proper complex noise and a stationary nuisance. A hidden path error with coefficients [0, −0.4, 0.2] metres is recovered sufficiently well that the worst point matched-power loss is {max(values):.4f} dB. This loss describes motion coherence at the known truth locations after fitting; it is not total image efficiency. The inverse image also has finite-noise, discretization and regularization errors.</p>
<p>The recorded local test changes original excluded-crop energy prediction from {100*recorded['initial_outer']['explained_fraction']:.4f}% to {100*recorded['exact_physical_outer']['explained_fraction']:.4f}%. That is a small improvement and does not explain the broad image. No fitted alignment is used for the exact physical waveform check. The near-identical matched images below are retained to make the limited gain explicit.</p>
<h2>Matched images, separate from sparse coefficient pictures</h2>
<p>These images use AᴴPy and diag(AᴴPA), without the sparse coefficients or prior. All five bands and the calibrated complex sum are shown. Their physical body-coordinate rectangle is declared, and the same colour scale is used before and after each fit. They are local diagnostics, not the requested final north-up full-viewport videos.</p>
<h3>Synthetic</h3><img src="{relative(a.synthetic/'matched-before-after.png')}">
<h3>Recorded</h3><img src="{relative(a.recorded/'matched-before-after.png')}">
<p><a href="{relative(a.synthetic/'analysis.json')}">Synthetic parameters and exact physical checks</a> · <a href="{relative(a.recorded/'analysis.json')}">Recorded parameters and exact physical checks</a> · <a href="../moment-phase-scene-validation.log">Original-row algebra, calibration-gradient and sparse-profile tests</a>.</p>'''
    (a.output/'index.html').write_text(html)


if __name__=='__main__':main()
