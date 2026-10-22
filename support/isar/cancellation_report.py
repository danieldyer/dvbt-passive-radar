"""Publish measured forward-model controls and conditional cancellation bounds."""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import re

import numpy as np

from finite_scene_report import digest,load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('information','observed','tests'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--implementation-archive',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,default=Path('docs/data/isar_20261015'))
    args=p.parse_args()
    info,first=load(args.information,args.implementation_archive)
    observed,second=load(args.observed,args.implementation_archive)
    tests=args.tests.read_text();match=re.search(r'Ran (\d+) tests',tests)
    if not match or tests.strip().splitlines()[-1]!='OK':raise ValueError('Require completed passing tests')
    for record in observed['records']:
        if digest(Path(record['arrays']))!=record['arrays_sha256']:raise ValueError('Observed matrices changed')
    if [c['channel'] for c in info['channels']]!=[c['channel'] for c in observed['records']]:
        raise ValueError('Compare the same channels')
    totals={};channels=[]
    for record in observed['records']:
        sums={}
        for c in record['checks']:
            key=f"{c['template']}/stationary={c['unknown_stationary']}/finite_covariance={c['finite_covariance']}"
            sums[key]=sums.get(key,0.)+c['prediction_error']
            totals[key]=totals.get(key,0.)+c['prediction_error']
        channels.append({'channel':record['channel'],'prediction_error_by_model':sums})
    def value(template,stationary=False,finite=False):
        return totals[f'{template}/stationary={stationary}/finite_covariance={finite}']
    comparison={'known_pulse_error_change_percent_without_stationary':100*(value('known')/value('mean')-1),
        'finite_symbol_error_change_percent_without_stationary':100*(value('finite')/value('mean')-1),
        'finite_symbol_error_change_percent_with_stationary':100*(value('finite',True)/value('mean',True)-1),
        'saved_covariance_error_change_percent_without_stationary':100*(value('mean',False,True)/value('mean')-1),
        'unknown_stationary_error_change_percent_mean_model':100*(value('mean',True)/value('mean')-1)}
    result={'complete':True,'updated_utc':datetime.now(timezone.utc).isoformat(),'scope':__doc__,
        'information':info,'observed_channels':channels,'prediction_error_totals':totals,'comparison':comparison,
        'support_tests':int(match[1]),'source_sha256':{str(f):digest(f) for f in [args.information,args.observed,args.tests,Path(__file__)]},
        'verified_original_implementation_substitutions':[first,second],
        'production_changed':False,'accepted_calibration':False,
        'scope_limit':'Thirty sparse 16-symbol blocks per channel span thirty seconds but retain only 0.52224 s of symbol time. Fixed motion and reflector support were fitted earlier using this recording. This is neither full-CPI efficiency certification nor independent validation of motion, support or sites.'}
    names=['Seven','SBS','Nine','Ten','ABC'];rows=''
    for c in info['channels']:
        record=[names[c['channel']],
            f"{c['compression_with_cancellation'][1]['worst_information_loss_db']:.6f}",
            f"{10*np.log10(c['optimal_profile_mse_over_raw_crlb']):.6f}",
            f"{10*np.log10(c['thermal_weighted_profile_mse_over_raw_crlb']):.6f}"]
        rows+='<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in record)+'</tr>'
    actual=''
    for name in ('mean','known','finite'):
        for stationary in (False,True):
            actual+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                [name,'fitted' if stationary else 'none',f'{value(name,stationary):.3f}',f'{value(name,stationary,True):.3f}'])+'</tr>'
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen cancellation and finite waveform audit</title>
<style>body{{font:17px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 22px;color:#243040}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Frozen cancellation uncertainty and exact finite-symbol predictions</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="cancellation-validation.json">Results and source identities</a> · {result['updated_utc']}</p>
<p>The saved cancellation uncertainty is correlated across time. Propagating it preserves almost all conditional eight-point scene information in the sampled test. On the recorded samples, neither this covariance nor the exact finite-symbol template repairs the prediction mismatch. No production correction or antenna calibration is adopted.</p>
<h2>What is measured and what is assumed</h2>
<p>The audit uses 480 original rows per channel: thirty complete 16-symbol blocks distributed through the 150.772–180.772 s aperture. Across five channels that is 2,400 rows. The retained symbol duration is 0.52224 s per band, not a continuous 30-second CPI. Exact original PROF response/noise weights, source-clock offsets and frozen cancellation variances are read by saved row epoch. Oracle QAM symbols are independently reconstructed. The actual-data test reads original saved C0/C1 profiles, rather than substituting a reprocessed receive stream.</p>
<p>The stochastic model conditions on the fitted response and known symbols. Primary carrier noise is independent proper-complex Gaussian, and each frozen training-coefficient error has the saved diagonal variance. Its value is shared across time. Unknown response-estimation error and correlated external interference are not certified by this model.</p>
<h2>Propagation and information calculation</h2>
<p>For each carrier k, write the primary observations as yₖ = Qₖa + zₖbₖ + nₖ, with bₖ of unit complex variance and zₖ = HₖXₖ√Vβ,ₖ. Thus Rₖ = Dₖ + zₖzₖᴴ. A rank-one square-root whitening applies this covariance to all sampled epochs together. Treating it as independent added noise in each block would be incorrect.</p>
<p>The saved range/moment operator L gives C = LDLᴴ + (LZ)(LZ)ᴴ. Raw and compressed scene information are QᴴR⁻¹Q and (LQ)ᴴC⁻¹(LQ). The compression loss uses their generalized eigenvalues for raw modes above relative power 1e−10. The JSON retains additional thresholds and the information cost of the frozen uncertainty itself. Estimator risk includes the actual covariance of the implemented linear inverse and is normalized by the raw marginal CRLB.</p>
<table><tr><th>Band</th><th>Worst core compression loss (dB)</th><th>Optimal profile estimator risk (dB)</th><th>Ignoring cancellation covariance: risk (dB)</th></tr>{rows}</table>
<p>These are conditional fixed-scene results, not a bound on every possible target or on nonlinear motion/geometry estimation. Both algebraic operators are checked against independent dense covariance solves.</p>
<h2>Prediction on original captured profiles</h2>
<p>Each of the thirty sparse blocks alternates between training and prediction; both parity choices are reported. The trajectory, eight-point per-band support, observations and range units are held fixed. Mean-pulse, actual known-block pulse and exact finite current/previous-symbol models are compared. The last includes the saved fractional clock, finite guard interval and original C0/C1 operator.</p>
<p>For each template, a second control fits unknown constant-delay amplitudes on the same physical delay grid. The finite-covariance variants additionally condition the frozen Gaussian coefficient errors on training data and predict their correlated contribution in withheld blocks. No covariance scale is trained on the ship return. All table entries are summed squared errors on identical withheld observations; lower is better.</p>
<table><tr><th>Template</th><th>Unknown stationary response</th><th>Thermal metric</th><th>Saved finite covariance</th></tr>{actual}</table>
<p>Without an unrestricted stationary fit, the exact finite-symbol model changes error by {comparison['finite_symbol_error_change_percent_without_stationary']:+.4f}% relative to the mean pulse; with one, the change is {comparison['finite_symbol_error_change_percent_with_stationary']:+.4f}%. Adding saved cancellation covariance to the mean-pulse control changes error by {comparison['saved_covariance_error_change_percent_without_stationary']:+.4f}%. Fitting unrestricted stationary coefficients changes the mean-model error by {comparison['unknown_stationary_error_change_percent_mean_model']:+.3f}% in these sparse blocks. These sampled results do not justify changing the existing full-aperture nuisance model without its own matched test.</p>
<h2>References and validation</h2>
<p><a href="https://gaussianprocess.org/gpml/chapters/RW2.pdf">Rasmussen and Williams, chapter 2</a> supplies the Gaussian coefficient marginalization and conditioning framework. The complex covariance information formula is treated by <a href="https://arxiv.org/abs/1306.6415">Besson and Abramovich</a>. <a href="https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-rsn.2016.0410">Tang et al.</a> analyses how imperfect reference signals affect passive-radar cancellation; its reference-noise model is distinct from the saved coefficient-uncertainty model tested here.</p>
<p>{int(match[1])} support tests pass. New checks cover dense versus rank-one whitening, covariance-aware amplitude risk, capture identity/truncation, original moment order and complex Doppler combination, and Gaussian prediction against an independent augmented penalized solve. The larger end-to-end imaging and phase-centre goal remains open.</p></html>'''
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'cancellation-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'cancellation-validation.html').write_text(document+'\n')


if __name__=='__main__':main()
