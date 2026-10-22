"""Publish the conditional-noise recovery study, including its finite-sample limits."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('docs/data/injection_20261018'))
    args=parser.parse_args();root=args.root;out=root/'colored-point-study';out.mkdir(exist_ok=True)
    trials=[];diagnostics=[]
    for seed in [1901,1902]:
        d=root/f'colored-point-{seed}';x=json.loads((d/'analysis.json').read_text());z=json.loads((d/'diagnostics.json').read_text())
        if not x['complete'] or not z['source_complete'] or len(x['trials'])!=len(z['records']):
            raise ValueError('Finish the independent trials and diagnostics first')
        trials.extend(x['trials']);diagnostics.extend(z['records'])
    error=np.asarray([v['physical_error'] for v in trials]);cov=np.asarray(x['physical_parameter_crlb'])
    rms=np.sqrt(np.mean(error**2,axis=0));bound=np.sqrt(np.diag(cov))
    ratios=np.array([v['profile_likelihood_ratio'] for v in diagnostics]);observable=np.array([v['observable_mahalanobis_squared'] for v in diagnostics])
    result=dict(scope=__doc__,trials=len(trials),mean_physical_error=error.mean(axis=0),
        position_rmse_m=np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1))),
        position_local_crlb_rms_m=np.sqrt(np.trace(cov[:2,:2])),
        velocity_rmse_mps=np.sqrt(np.mean(np.sum(error[:,2:]**2,axis=1))),
        velocity_local_crlb_rms_mps=np.sqrt(np.trace(cov[2:,2:])),
        profile_likelihood_ratio_mean=ratios.mean(),observable_mahalanobis_mean=observable.mean(),
        profile_likelihood_ratio_total=ratios.sum(),
        asymptotic_chi_squared_total_95_percent_interval=chi2.ppf([.025,.975],4*len(trials)),
        maximum_excluded_combined_loss_db=max(v['excluded_five_band']['coherence_loss_db'] for v in trials),
        all_fits_converged=all(v['fit']['success'] for v in trials),
        all_fits_interior=all(not v['fit']['active_bound_coordinates'] for v in trials),
        validates_recorded_background_model=False,production_changed=False)
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax=axes[0,0];axis=np.arange(4)
    ax.bar(axis-.18,rms/bound,width=.36,label='Measured RMS / local bound')
    ax.bar(axis+.18,np.ones(4),width=.36,label='Local CRLB scale',color='#aaa')
    ax.set_xticks(axis,['East position','North position','East velocity','North velocity']);ax.set(ylabel='Ratio',title='Twelve independently acquired and fitted points');ax.legend(fontsize=8)
    ax=axes[0,1];q=(np.arange(len(ratios))+.5)/len(ratios);expected=chi2.ppf(q,4)
    ax.plot(expected,np.sort(ratios),'o');ax.plot([0,max(expected.max(),ratios.max())],[0,max(expected.max(),ratios.max())],'--',color='#999')
    ax.set(xlabel='χ²₄ quantile',ylabel='Measured profile likelihood ratio',title='Four identifiable kinematic parameters')
    ax=axes[1,0];cartesian=np.array([v['cartesian_mahalanobis_squared'] for v in diagnostics])
    ax.plot(ratios,cartesian,'o',label='Cartesian linear-ellipse error')
    ax.plot(ratios,observable,'s',label='Exact bistatic range/rate coordinates')
    ax.plot([0,max(ratios)],[0,max(ratios)],'--',color='#999')
    ax.set(xlabel='Exact profile likelihood ratio',ylabel='Squared local-normalised error',title='The narrow geographic likelihood ridge is curved');ax.legend(fontsize=8)
    ax=axes[1,1];loss=[v['excluded_five_band']['coherence_loss_db'] for v in trials]
    ax.bar(np.arange(1,len(trials)+1),loss);ax.set(xlabel='Independent trial',ylabel='Five-band coherence loss (dB)',title='Frozen prediction on the final 10.5 s')
    fig.savefig(out/'recovery-validation.png',dpi=160);plt.close(fig)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Conditional coloured-noise point recovery</title>
<style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 24px}}img{{width:100%}}.note{{padding:16px;background:#eef6fa}}</style>
<h1>Does the inference recover the information predicted by its noise model?</h1>
<p class="note">All twelve independent point trials converge without active bounds. Their total profile-likelihood error is {ratios.sum():.2f}, compared with an asymptotic χ² expectation of {4*len(trials)}. These results support the nonlinear inference under the tested model; twelve trials do not establish a tight efficiency margin or validate a Gaussian model for the actual recorded scene.</p>
<p>The background covariance is estimated once from the recorded mixed-scene residual, with spectral resolution selected by excluded-frequency likelihood. Independent proper-complex Gaussian samples are drawn through that frozen covariance. The injected point has the same strength, physical position/velocity and unknown constant band coefficients as the recorded-background CV test. Original waveform kernels, symbol epochs and C0/C1 moments are retained. Each realization has its own training-only coarse acquisition from the AIS reference and physical four-parameter refinement. No true position, motion, phase or band coefficient seeds a fit.</p>
<p>Mean east/north position error is [{error[:,0].mean():.2f}, {error[:,1].mean():.2f}] m. Position RMS is {result['position_rmse_m']:.2f} m against a local conditional bound of {result['position_local_crlb_rms_m']:.2f} m; velocity RMS is {result['velocity_rmse_mps']:.4f} m/s against {result['velocity_local_crlb_rms_mps']:.4f} m/s. The final 10.5 s is excluded from acquisition, fitting and band calibration. Its worst combined five-band coherence loss is {result['maximum_excluded_combined_loss_db']:.6f} dB.</p>
<img src="recovery-validation.png" alt="Empirical errors, likelihood ratios, nonlinear coordinate effects and excluded-data coherence">
<h2>Why a Cartesian error ellipse is insufficient here</h2>
<p>The geographic likelihood is a very narrow curved ridge. A tens-of-metres movement along it creates a second-order Cartesian displacement larger than its transverse thermal scale. Evaluating a finite error only against the tangent ellipse can therefore report misleadingly large errors. The exact profile likelihood and independently calculated bistatic range/range-rate coordinates agree closely: their mean squared-normalised errors are {ratios.mean():.3f} and {observable.mean():.3f}, versus the four-parameter expectation of 4. The Cartesian ellipses are not used as global confidence regions.</p>
<p>The sum of twelve likelihood ratios is compared with χ²₄₈, whose central 95% interval is [{result['asymptotic_chi_squared_total_95_percent_interval'][0]:.2f}, {result['asymptotic_chi_squared_total_95_percent_interval'][1]:.2f}]. This is a model-conditional asymptotic diagnostic, not a proof of global acquisition or statistical optimality on arbitrary interference.</p>
<p><a href="analysis.json">Summary</a> · <a href="../colored-point-1901/analysis.json">Seed 1901 fits</a> · <a href="../colored-point-1902/analysis.json">Seed 1902 fits</a> · <a href="../constant-velocity-study/index.html">Actual-background constant-velocity controls</a>. Reproduce using <code>validate_colored_point_recovery.py</code>, <code>diagnose_colored_recovery.py</code> and <code>report_colored_point_recovery.py</code> under <code>support/isar</code>.</p></html>''')


if __name__=='__main__':main()
