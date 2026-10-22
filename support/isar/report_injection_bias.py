"""Render the measured displacement, its perturbation prediction and controls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/data/injection_20261018/separated-500m/bias-explanation'))
    args = parser.parse_args(); root = args.root
    x = json.loads((root/'analysis.json').read_text())
    names = ['Seven','SBS','Nine','Ten','ABC']
    fig, axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax = axes[0,0]
    e = [h['background_voltage_scale'] for h in x['homotopy']]+[1.]
    y = [h['cross_range_error_m'] for h in x['homotopy']]+[x['observed_cross_range_m']]
    order = np.argsort(e); e = np.asarray(e)[order]; y = np.asarray(y)[order]
    ax.plot(e,e*x['linear_predicted_cross_range_m'],'--',label='Normal-equation prediction')
    ax.plot(e,y,'o-',label='Full nonlinear physical fit')
    ax.axhline(0,color='#aaa',lw=.6); ax.set(xlabel='Recorded background voltage multiplier',
        ylabel='Position displacement along range contour (m)',title='Displacement follows background voltage')
    ax.legend(fontsize=9)
    ax = axes[0,1]
    v = [r['predicted_cross_range_m'] for r in x['attribution']]
    ax.bar(names,v,color=['#628bb0']*4+['#d57540']); ax.axhline(0,color='#555',lw=.7)
    ax.set(ylabel='Signed contribution to predicted displacement (m)',title='ABC supplies the second transmitter phase centre')
    for i,z in enumerate(v): ax.text(i,z+(3 if z>0 else -8),f'{z:.1f}',ha='center',fontsize=9)
    ax = axes[1,0]
    for record,name in zip(x['attribution'],names):
        t = np.asarray(record['time_s']); b = np.asarray(record['contribution_per_time_bin_m'])
        ax.plot(t,np.cumsum(b),label=name,lw=1)
    ax.set(xlabel='Capture time (s)',ylabel='Cumulative signed influence (m)',title='Contributions are distributed across the aperture')
    ax.legend(fontsize=9)
    ax = axes[1,1]
    for record,name in zip(x['attribution'],names):
        ax.plot(record['delay_relative_ais_m'],record['contribution_per_range_cell_m'],'o-',label=name,ms=3,lw=1)
    ax.axvspan(150,300,color='#eee',label='Injected range region')
    ax.set(xlabel='Excess path relative to AIS centre (m)',ylabel='Signed influence on displacement (m)',
        title='Exact adjoint attribution; these are not source powers')
    ax.legend(fontsize=8,ncol=2)
    fig.savefig(root/'bias-mechanism.png',dpi=160); plt.close(fig)
    fraction=x['exact_score_component_fraction']*100
    controls=x['controls']
    rows=''.join(f'<tr><td>{label}</td><td>{z["horizontal_error_m"]:.6g} m</td></tr>' for label,z in controls.items())
    info=x['position_information']
    (root/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Quantitative cause of injected-point displacement</title>
<style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:16px;background:#eef6fa}}code{{background:#eee}}</style>
<h1>Why a bright injected point moves by 116 metres</h1>
<p class="note">The error is reproducible from the recorded background and the exact motion Jacobian. The first-order calculation predicts {x['linear_predicted_cross_range_m']:.2f} m along the equal-range contour; the full fit moves {x['observed_cross_range_m']:.2f} m. This is not a weak injected point, a failed chirp search or a profile/IQ discrepancy. It exposes a mismatch between the thermal-only likelihood and the structured recorded scene.</p>
<p>These controls use the 500 m relocated point, the same five channels, 142–161.5 s fitting interval, original native epochs and C0/C1 waveform response. The final 10.5 s remains excluded. The synthetic point has six free planar kinematic coordinates and five independent, constant complex band coefficients; transmitter/receiver positions and height motion are fixed. All fits are interior to their bounds.</p>
<h2>A numerical prediction, before fitting the mixture</h2>
<p>After eliminating stationary echoes and complex point amplitudes, let <code>b</code> be the recorded background residual at the true trajectory, and <code>J</code> the residual Jacobian of the clean point. Linearised least squares gives <code>delta = -pinv(J) b</code>. The prediction is [+57.68, −96.98] m east/north; the actual nonlinear fit gives [+59.51, −99.21] m. Its velocity error is [−0.0919, +0.0862] m/s and acceleration error [−0.00896, +0.01444] m/s². Position, velocity and acceleration compensate one another, leaving only 0.00201 dB loss of point-template energy after constant band coefficients are eliminated.</p>
<p>The geometry here has two physical transmitter centres: four channels at Ornata Road and ABC at Ridge Road, about 326 m apart. Allowing unknown velocity increases the weak-direction nominal thermal position standard deviation from {info['motion_known']['principal_sigma_m'][-1]:.4f} m to {info['velocity_free']['principal_sigma_m'][-1]:.2f} m; freeing all four motion coordinates gives {info['all_motion_free']['principal_sigma_m'][-1]:.2f} m. The actual background projection in that weak position direction is {abs(x['weak_position_score_thermal_sigmas']):.2f} nominal thermal standard deviations. The fixed-motion 125–160 m image resolution therefore does not describe this joint inference problem.</p>
<img src="bias-mechanism.png" alt="Predicted and fitted displacement, channel contributions, and exact time and range influence">
<h2>Causal controls</h2>
<p>The background residual energy is {x['background_residual_energy']:,.1f}. Its component in the six-dimensional mean-Jacobian span has energy {x['exact_score_component_energy']:.2f}: only {fraction:.5f}% of the background residual energy. Removing exactly this component makes the wrong solution return to truth; retaining only it, with the same already-profiled nuisance component, nearly reproduces the full displacement.</p>
<table><tr><th>Control</th><th>Horizontal error</th></tr>{rows}</table>
<p>Reversing background voltage reverses displacement; reducing its voltage scales the displacement almost linearly. The plots also show that ABC supplies +123.79 m of the linear prediction, while the other four channels sum to −11.00 m. Range and time contributions are signed adjoint sensitivities with large cancellations. They are not physical source powers and do not uniquely identify ship, sea, RF interference or receiver error.</p>
<h2>What this does and does not justify</h2>
<p>The score-component removal is a truth-dependent causal diagnostic, not a deployable filter. Enforcing the injected motion or its relative band phases would also be an oracle constraint. The next estimator must treat recorded background correlations or other echoes explicitly, apply the same metric to observations/templates/derivatives, and recover point position and motion without either shortcut.</p>
<p>A standard basis for that work is covariance-weighted adaptive matched filtering (<a href="https://ece.iisc.ac.in/~cmurthy/E1_244/Slides/RFKN.pdf">Robey, Fuhrmann, Kelly and Nitzberg, 1992</a>) with regularised covariance estimation (<a href="https://www.ledoit.net/Well-conditioned2004.pdf">Ledoit and Wolf, 2004</a>). These references justify a model to test, not an assumption that this particular background is stationary or adequately trained.</p>
<p><a href="analysis.json">Numerical results</a> · <a href="../index.html">Paired profile/IQ and acquisition validation</a> · Reproduce with <code>support/isar/explain_injection_bias.py</code> and <code>support/isar/report_injection_bias.py</code>.</p></html>''')


if __name__=='__main__': main()
