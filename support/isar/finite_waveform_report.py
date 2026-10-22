"""Publish the realised finite-waveform operator and information audit."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--information',type=Path,required=True)
    p.add_argument('--tests',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('docs/data/isar_20261015'))
    args=p.parse_args();a,b=[json.loads(x.read_text()) for x in (args.audit,args.information)]
    if not a['complete'] or not b['complete'] or a['input_sha256']!=b['input_sha256']:
        raise ValueError('Use completed controls on the identical reconstructed waveform')
    if fingerprint(Path(a['input']))!=a['input_sha256']:raise ValueError('Probe archive changed')
    for data in (a,b):
        for group in ('source_sha256','implementation_sha256'):
            for path,digest in data.get(group,{}).items():
                if fingerprint(Path(path))!=digest:raise ValueError('Implementation changed: '+path)
    log=args.tests.read_text();count=re.search(r'Ran (\d+) tests',log)
    if not count or log.strip().splitlines()[-1]!='OK':raise ValueError('Need a completed passing test log')
    names=['Seven','SBS','Nine','Ten','ABC'];rows=[]
    for c,name in enumerate(names):
        summaries=[r for r in a['summaries'] if r['channel']==c];information=[r for r in b['cases'] if r['channel']==c]
        rows.append({'channel':name,'probes':a['probe_count_by_channel'][str(c)],
            'maximum_ensemble_snr_loss_db':max(r['models']['mean_circular_mean_covariance']['snapshot_ensemble_loss_db'] for r in summaries),
            'maximum_single_row_snr_loss_db':max(r['models']['mean_circular_mean_covariance']['maximum_single_row_loss_db'] for r in summaries),
            'maximum_delay_rate_information_loss_db':max(r['worst_information_loss_db'] for r in information),
            'maximum_absolute_first_order_range_bias_m':max(abs(r['mean_template_first_order_bias_metres_and_mps'][0]) for r in information),
            'maximum_absolute_first_order_rate_bias_mps':max(abs(r['mean_template_first_order_bias_metres_and_mps'][1]) for r in information),
            'maximum_squared_bias_per_parameter_at_40db':max(r['squared_bias_in_crlb_units_per_parameter_at_snr_db']['40'] for r in information)})
    args.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');x=np.arange(5)
    axes[0].bar(x-.18,[r['maximum_ensemble_snr_loss_db'] for r in rows],.36,label='Isolated-echo SNR')
    axes[0].bar(x+.18,[r['maximum_delay_rate_information_loss_db'] for r in rows],.36,label='Worst delay/rate information direction')
    axes[0].set(xticks=x,xticklabels=names,ylabel='Maximum conditional loss (dB)',title='Finite profiles versus occupied-FFT bound')
    axes[0].legend(fontsize=8)
    axes[1].bar(names,[1000*r['maximum_absolute_first_order_range_bias_m'] for r in rows],color='#578776')
    axes[1].set(ylabel='Maximum absolute first-order delay bias (mm)',title='Mean waveform model; exact conditional covariance')
    fig.suptitle('Recorded reference symbols; controlled synthetic point echoes\nKnown response/noise, sparse original rows; not an actual-ship CRLB certificate')
    fig.savefig(args.output/'finite-waveform-validation.png',dpi=160);plt.close(fig)
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'channels':rows,'tests':int(count[1]),
        'maximum_gpu_absolute_error':max(r['maximum_absolute_error'] for r in a['gpu_comparison']),
        'maximum_gpu_relative_vector_error':max(r['relative_error'] for r in a['gpu_comparison']),
        'scope':a['efficiency_scope'],'motion_scope':b['bias_scope'],
        'source_sha256':{str(f):fingerprint(f) for f in (args.audit,args.information,args.tests)},
        'implementation_sha256':{str(Path(__file__)):fingerprint(Path(__file__))},
        'accepted_calibration':False,'production_changed':False}
    (args.output/'finite-waveform-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['channel'],r['probes'],
        f"{r['maximum_ensemble_snr_loss_db']:.6f}",f"{r['maximum_delay_rate_information_loss_db']:.6f}",
        f"{1000*r['maximum_absolute_first_order_range_bias_m']:.3f}"] )+'</tr>' for r in rows)
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Finite waveform validation</title>
<style>body{{font:17px/1.55 system-ui;color:#243040;max-width:1150px;margin:35px auto;padding:0 22px}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}pre{{white-space:pre-wrap;background:#f4f6f7;padding:15px}}</style>
<h1>Does the averaged waveform model discard motion information?</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="finite-waveform-validation.json">Results and source identities</a> · {result['updated_utc']}</p>
<p class="note">This control rules out a large isolated-point loss from the tested finite-reference/range-crop approximation. It does not certify the complete imaging pipeline, the interference model, a distributed ship scene, or transmitter coordinates.</p>
<h2>Production versus an independent calculation</h2>
<p>The saved channel IQ was replayed through the production demodulation, reconstruction, response fit, noise estimator and finite range compressor. Time-only selection retained 335 complete current/previous symbol pairs over five channels: vehicle intervals at 29–66 s and the ship interval at 150.772–180.772 s. Each record contains both reconstructed symbols, the response, carrier variance, measured spectrum, source-clock offsets and production C0/C1 profiles.</p>
<p>An independent double-precision time/Fourier operator reproduces the 64-tap range interpolator and Doppler-moment combination. Against 1,340 production profile comparisons, the largest absolute complex error is {result['maximum_gpu_absolute_error']:.3g}; the largest relative vector error is {result['maximum_gpu_relative_vector_error']:.3g}. The comparison reproduces the GPU's adjacent-row fractional-clock interpolation. The theoretical echo tests use an exact Fourier shift of the previous reference symbol.</p>
<h2>The bound being tested</h2>
<p>Let y have known covariance D, q be the exact finite echo in occupied measured FFT carriers, and A be the fixed profile/crop operator. Then the retained mean is p = A q and its covariance is C = A D Aᴴ. The optimal raw-observation SNR per unit echo power is qᴴD⁻¹q. The best retained-profile SNR is pᴴC⁻¹p. The following ratio is bounded above by one because it is an orthogonal projection after whitening:</p>
<pre>retained information = (qᴴ Aᴴ (A D Aᴴ)⁻¹ A q) / (qᴴ D⁻¹ q)
candidate filter SNR fraction = |wᴴ A q|² / [(wᴴ A D Aᴴ w)(qᴴ D⁻¹ q)]</pre>
<p>The candidate w uses the mean circular pulse and corresponding mean range covariance, with the analytical useful-symbol Doppler moments. This comparison includes actual QAM/pilot symbols, finite previous-symbol context, source offsets, 17 retained range cells, interpolation, response shape and noise colour. Four excess delays from about 0.60 to 29.98 km are tested at −100, 0 and +100 Hz: 4,020 single-row cases in total. Snapshot ensembles combine declared synthetic echoes; they are not uninterrupted physical coherent apertures.</p>
<h2>Delay and velocity information</h2>
<p>A separate 50-case test uses physically consistent constant path rate: fD = −RF × delay-rate / sample-rate, including range migration, carrier-phase derivatives and within-symbol Doppler. Extraction coordinates stay frozen when differentiating the physical echo. Unknown complex amplitude is eliminated before comparing the two-parameter Fisher matrices. Generalized eigenvalues compare every linear combination of delay and rate, rather than only their separate diagonal entries.</p>
<table><tr><th>Band</th><th>Probes</th><th>Max ensemble SNR loss (dB)</th><th>Max Fisher-information loss (dB)</th><th>Max local delay bias (mm)</th></tr>{table}</table>
<img src="finite-waveform-validation.png" alt="Per-channel conditional matched-filter and delay/rate-information loss and first-order delay bias">
<p>The reported bias is a first-order local mean-mismatch calculation after fitting unknown complex amplitude, using exact conditional covariance. It is not a nonlinear-estimator efficiency measurement. The maximum absolute path-rate bias is {max(r['maximum_absolute_first_order_rate_bias_mps'] for r in rows):.3g} m/s; the maximum squared bias per parameter in CRLB units at 40 dB total synthetic SNR is {max(r['maximum_squared_bias_per_parameter_at_40db'] for r in rows):.3g}.</p>
<h2>What remains outside this result</h2>
<p>Raw-channel/OFDM losses, discarded cyclic-prefix observations, uncertainty in fitted response/noise, correlated residual interference, 16-symbol or 20-ms binning, target association, distributed scattering, aperture acquisition and antenna-location nuisance are excluded here. Several are covered by separate conditional tests; they must not be silently included in this certificate. A mean pulse may predict an isolated point adequately while failing to predict sidelobes from strong surrounding clutter. That distributed-scene question remains open.</p>
<p>The next geometry sensitivity test should also address the current asymmetric range-origin assumption: vehicle origins are fixed while ship origins can move. Small point-template loss supplies no justification for treating that assumption or an inferred road association as measured calibration.</p>
<h2>Literature and verification</h2>
<p><a href="https://arxiv.org/abs/1908.03418">Baquero Barneto et al., Full-Duplex OFDM Radar With LTE and 5G NR Waveforms</a>, Section II-A, explicitly distinguishes matched filtering's single-target SNR advantage from its data-dependent range response and sidelobe behaviour. This motivates testing both information retention and the realised mean response; it does not validate our capture's covariance.</p>
<p><a href="https://arxiv.org/abs/1306.3604">Zhang and Xia, OFDM Synthetic Aperture Radar Imaging with Sufficient Cyclic Prefix</a>, analyse interference-free range reconstruction under a sufficient-CP construction and discuss the noise cost of dividing by small subcarrier values. Those sufficient-CP assumptions do not hold automatically for every excess delay in this recording. Our finite previous-symbol reference is tested directly, without importing their zero-interference conclusion.</p>
<p>All {result['tests']} support tests pass. New independent tests cover scalar Fourier sums, direct time correlation, proper-complex noise covariance, clock-fraction wrapping, analytic echo derivatives, complex-amplitude nuisance elimination and archive completion/truncation. No physical coordinate or production algorithm is changed by this diagnostic.</p></html>'''
    (args.output/'finite-waveform-validation.html').write_text(text)
    print(json.dumps(result))


if __name__=='__main__':main()
