"""Evidence report for common-clock reference, uncertainty and coherence tests."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.signal import welch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);a=p.parse_args();r=a.results
    metrology=json.loads((r/'capture-metrology-v2/analysis.json').read_text())
    old=np.genfromtxt(r/'controller-incumbent.csv',delimiter=',',names=True)
    fixed=np.genfromtxt(r/'controller-weighted-common-horizon.csv',delimiter=',',names=True)
    information=json.loads((r/'weighted-information.json').read_text())
    bandwidth=json.loads((r/'controller-bandwidth.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for amplitude in sorted({v['amplitude_ns'] for v in bandwidth['cases']}):
        case=[v for v in bandwidth['cases'] if v['amplitude_ns']==amplitude]
        f=np.array([v['frequency_hz'] for v in case])
        axes[0].semilogx(f,20*np.log10([v['gain'] for v in case]),'o-',label=f'{amplitude:g} ns input')
        axes[1].semilogx(f,np.degrees(np.unwrap(np.radians([v['phase_degrees'] for v in case]))),'o-',label=f'{amplitude:g} ns input')
    axes[0].axhline(0,color='grey',ls=':');axes[0].set(ylabel='Timing response gain (dB)',title='Remaining predictor peaking')
    axes[1].set(ylabel='Timing response phase (degrees)',title='Production controller; sinusoidal input')
    for ax in axes:ax.set_xlabel('Temporal frequency (Hz)');ax.legend()
    fig.savefig(r/'controller-bandwidth.png',dpi=150);plt.close(fig)
    phase=[]
    for name in ['retiming-wander-phase-vulkan','retiming-wander-10ns']:
        d=json.loads((r/(name+'.json')).read_text());loss=[]
        for c in d['channels']:
            for e in c['echoes']:
                ratio=e['amplitude_ratio'];real=ratio*math.cos(math.radians(e['phase_error_degrees']))
                actual_power=e['relative_complex_rms']**2-1+2*real
                loss.append(-10*math.log10(ratio*ratio/actual_power))
        phase.append(dict(file=name+'.json',amplitude_ns=d['common_time_warp_amplitude_ns'],passed=d['passed'],
            worst_phase_rms_degrees=max(e['phase_rms_degrees'] for c in d['channels'] for e in c['echoes']),
            failed_fec_words=sum(c['failed_packets'] for c in d['channels']),maximum_conditional_coherence_loss_db=max(loss)))
    rows=[]
    for lo,hi in [(10,100),(140,190),(300,400),(500,600)]:
        before=old['rate_ppm'][(old['time_s']>=lo)&(old['time_s']<hi)]
        after=fixed['rate_ppm'][(fixed['time_s']>=lo)&(fixed['time_s']<hi)]
        rows.append(dict(interval_s=[lo,hi],incumbent_std_ppm=float(np.std(before)),fixed_std_ppm=float(np.std(after)),
            rate_variation_reduction=float(np.std(before)/np.std(after))))
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for d,label in [(old,'Incumbent'),(fixed,'One ensemble horizon')]:
        axes[0,0].plot(d['time_s'],d['rate_ppm'],lw=.6,label=label)
        keep=(d['time_s']>=560)&(d['time_s']<565)
        axes[0,1].plot(d['time_s'][keep],d['rate_ppm'][keep],lw=1,label=label)
        f,s=welch(d['rate_ppm'],fs=1/.032,nperseg=4096)
        axes[1,0].loglog(f[1:],s[1:],label=label)
    for ax in axes[0]:ax.set(xlabel='Capture time (s)',ylabel='Applied rate (ppm)');ax.legend()
    axes[0,0].set_title('Same original observations; controller only, no IQ interpolation')
    axes[0,1].set_title('Identical five-second interval and common vertical units')
    axes[1,0].set(xlabel='Temporal frequency (Hz)',ylabel='Rate PSD (ppm²/Hz)',title='Controller oscillation suppression');axes[1,0].legend()
    for filename,label in [('horizon-stress-incumbent','Incumbent'),('horizon-stress-fixed','Fixed')]:
        d=np.genfromtxt(r/(filename+'.csv'),delimiter=',',names=True);t=d['time_s'];truth=lambda t:4e-6*56e6*t+.5*.15e-6*56e6*t*t
        error=d['offset_samples'][32:]-d['offset_samples'][:-32]-(truth(t[32:])-truth(t[:-32]));keep=t[32:]>5
        axes[1,1].plot(t[32:][keep],error[keep],lw=.5,label=label)
    axes[1,1].set(xlabel='Time (s)',ylabel='1.024 s timing-increment error (ADC samples)',title='Independent zero-sum nuisance regression');axes[1,1].legend()
    fig.suptitle('Fixed clock-reference definition: average clock states before choosing the controller horizon')
    fig.savefig(r/'controller-reference-fix.png',dpi=150);plt.close(fig)
    data=dict(scope=__doc__,complete=True,capture=metrology['capture'],rate_comparisons=rows,waveform_coherence=phase,
        hardware_allan_model_validated=False,allan_kalman_enabled_in_production=False,
        production_bandwidth_sweep=bandwidth,
        production_changes=['One shared controller horizon after arithmetic clock-state averaging','Conditional pilot Fisher covariance','Within-channel precision-weighted offset regression','Explicit raw-rate and timing-uncertainty TUI labels'],
        limitations=['Archive-derived time observations include transmitter and propagation effects','A residual common-clock time error is not directly equal to radar carrier phase error','A real-data CRLB-tightness claim is not established by Gaussian synthetic tests'])
    paths=['src/sample_clock.cpp','src/sample_clock.hpp','src/pilot_phase.hpp','src/clock_estimator.hpp','src/gpu_receiver.cpp','tests/common_clock_tests.cpp','tests/clock_estimator_tests.cpp','support/validation/information_probe.cpp','support/validation/pipeline_coherence_probe.cpp']
    data['source_sha256']={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    (r/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    table=''.join(f'<tr><td>{v["interval_s"][0]}–{v["interval_s"][1]} s</td><td>{v["incumbent_std_ppm"]:.5f}</td><td>{v["fixed_std_ppm"]:.5f}</td><td>{v["rate_variation_reduction"]:.1f}×</td></tr>' for v in rows)
    clock_before=json.loads((r/'baseline-information.json').read_text())['common_clock'][0]
    clock_after=information['common_clock'][0];rate=information['pilot_rate_information']
    timing=''.join(f'<tr><td>{v["nominal_snr_db"]:g}</td><td>{v["coloured_noise"]}</td><td>{v["anchor_average_count"]}</td><td>{v["production"]["mse_over_crlb"]:.4f}</td></tr>' for v in information['pilot_likelihood'])
    waveform=''.join(f'<tr><td>{v["amplitude_ns"]:g} ns at 3 Hz</td><td>{v["worst_phase_rms_degrees"]:.4f}°</td><td>{v["maximum_conditional_coherence_loss_db"]:.6f} dB</td><td>{"PASS" if v["passed"] else "FAIL — retained"}</td></tr>' for v in phase)
    html=f'''<!doctype html><meta charset="utf-8"><title>Shared-clock recovery audit</title>
<style>body{{font:17px/1.55 system-ui;max-width:1250px;margin:35px auto;padding:0 22px}}img{{width:100%}}table{{border-collapse:collapse}}th,td{{padding:8px 15px;border-bottom:1px solid #ddd}}.note{{background:#fff1d6;padding:15px}}code{{font-size:90%}}</style>
<h1>Shared-clock recovery: a confirmed controller defect and statistical validation</h1>
<p class="note"><strong>The incumbent was not an Allan/covariance-optimal clock estimator.</strong> A reference-weighting defect is reproduced and fixed. Conditional timing/rate efficiency is validated, but the real SDR's oscillator model is not independently calibrated. The Allan/Kalman core is tested and remains separate from the live controller until its model is justified. The production predictor still has measured response peaking; optimal-bandwidth deployment is unfinished. No capture was modified.</p>
<h2>1. The actual production path</h2>
<p>Every channel uses one continuous ADC time map. Separate receivers steer their integer FFT windows and describe the transmitter waveform's timing relative to that map; they do not independently resample the measured channels. Channel NCOs use the common corrected coordinate. The TUI's old “DRIFT ppm” field was an adjacent-symbol pilot-rate observation, not the actual applied resampler rate. It is now labelled raw rate, while the detailed panel identifies the common applied rate and conditional pilot timing sigma separately.</p>
<p>The incumbent clock loop used a one-second quadratic offset fit per channel, fixed acquisition smoothing, and a curvature-based extrapolation horizon between 25 and 100 ms. It had no oscillator Allan model and no observation-covariance propagation. Published segments meet to {metrology['maximum_continuity_error_samples']:.2g} ADC samples in the saved history: continuity alone did not certify estimator quality.</p>
<h2>2. Confirmed defect: hidden phase weights</h2>
<p>The old controller averaged individual channel rate commands. A phase term in each command was <code>(x_c − x_applied) / H_c</code>. Although the final arithmetic weights were equal, different per-channel horizons <code>H_c</code> therefore gave the phases hidden weights proportional to <code>1/H_c</code>. Independent transmitter frequency offsets accumulate phase separation; horizon fluctuations then convert that separation into clock excursions. This was a changing time reference, not evidence that the TCXO really wandered at that rate.</p>
<p>The production fix averages the fitted channel clock states first, then uses one common controller horizon. Within-channel timing fits now use the supplied conditional measurement precision. Confidence never changes the arithmetic definition of the transmitter ensemble.</p>
<img src="controller-reference-fix.png">
<table><tr><th>Interval</th><th>Incumbent rate std (ppm)</th><th>Fixed rate std (ppm)</th><th>Reduction</th></tr>{table}</table>
<p>Both replays use the same original-epoch, feedback-removed timing observations. Their input noise variances are estimated from original-symbol second differences in the first sixty seconds; these diagnostic variances are not advertised as real-data Fisher bounds. The controller-only incumbent reproduces the capture's growing excursions. The independent zero-sum regression has three different transmitter rates and channel disturbances summing to zero, so their arithmetic ensemble must remain unchanged. Its direct production test now has about 3 × 10<sup>−10</sup> ADC-sample RMS one-second increment error in that noiseless, exactly representable model. This is an algebraic regression, not physical timing accuracy.</p>
<h2>3. Recorded timing spectra and Allan deviation</h2>
<img src="capture-metrology-v2/clock-metrology.png">
<p>These statistics describe the SDR relative to the received transmitter ensemble, including propagation, receiver and pilot-estimation errors. They do not uniquely identify the SDR TCXO. Measurement interpolation is used only for plotted spectra; original epochs remain saved, and replay observations use those original epochs. Periodic controller artifacts must not be fitted as intrinsic oscillator noise. Static ppm accuracy is not an Allan-deviation specification.</p>
<p>The noise definitions and interpretation follow <a href="https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication1065.pdf">NIST SP 1065, Handbook of Frequency Stability Analysis</a>. For clock time error <code>x</code> and fractional frequency <code>y</code>, a white-FM plus random-walk-FM model has <code>AVAR(τ) = q_x/τ + q_y τ/3</code>, excluding deterministic drift. The conditional optimal causal estimator is a Kalman filter with those process covariances and the actual timing observation covariance. For a stationary noncausal comparator the Wiener response is <code>S_clock / (S_clock + S_measurement)</code>. An update cadence alone is not a correction bandwidth.</p>
<img src="controller-bandwidth.png">
<p>A deterministic sweep through the fixed production controller uses five channels, 1 ms observation spacing, 32 ms updates and a 2 ms command lead, matching the audit replay. The small-signal timing response reaches <strong>2.48× at 1 Hz</strong>. This is predictor peaking, not a measured oscillator fluctuation. The adaptive horizon also makes the response amplitude dependent. Removing hidden ensemble weights therefore fixes a major defect but does not establish optimal noise rejection. This test has no injected noise and cannot choose an optimal bandwidth by itself; <a href="controller-bandwidth.json">all cases and scope</a> are saved.</p>
<h2>4. Pilot timing and rate Fisher checks</h2>
<p>The complex-Gaussian pilot likelihood profiles unknown complex amplitude and phase. Its delay information is the weighted frequency-spread Schur complement: <code>J_delay = 2 |a|² [Σ w |h|² ω² − (Σ w |h|² ω)² / Σ w |h|²]</code>, where <code>ω = 2π(k−3408)/8192</code>. Delay is measured in 8 MS/s samples. Production now reports this conditional covariance and transforms it into ADC coordinates with the square of the decimation and applied time-map scale.</p>
<table><tr><th>SNR dB</th><th>Coloured noise</th><th>Anchor average</th><th>Timing MSE / nuisance-aware CRLB</th></tr>{timing}</table>
<p>There are 6,000 trials per timing case. They include frequency-selective multipath, unknown complex gain, 20 dB extra noise on one fifth of pilots, independent 408-observation variance estimates, and finite noisy anchors. Ratios near one are consistent with Monte Carlo sampling uncertainty (about ±3.6% at 95%). This validates the stated statistical cases, not arbitrary interference or a time-varying unknown channel.</p>
<p>A separate {rate['trials']:,}-trial, {rate['epochs']}-epoch rate fit gives MSE/CRLB <strong>{rate['rate_fit']['mse_over_crlb']:.4f}</strong>, with a common noisy 32-symbol anchor profiled as an unknown intercept. Adjacent differential-rate errors have correlation <strong>{rate['adjacent_differential_rate_error_correlation']:.4f}</strong>. They must not be counted as independent extra clock information. The raw adjacent-symbol rate has {rate['instantaneous_rate_variance_to_64_epoch_crlb']:,.0f} times the variance of this 68.5 ms constant-rate bound; that comparison concerns different observation intervals, not receiver inefficiency or radar SNR loss.</p>
<h2>5. Clock-model covariance and real-data limitation</h2>
<p>The new <code>ClockEstimator</code> jointly estimates common phase/frequency/drift and zero-sum differential transmitter phases/frequencies. Its stochastic covariances are exact discretizations. Transmitter fluctuations include their covariance with the ensemble mean, and future predictions include process uncertainty. Measurement-confidence changes cannot redefine the standard. The Joseph covariance update protects precise pilot measurements against broad acquisition priors.</p>
<p><a href="clock-estimator-tests.log">Independent tests</a> match a batch polynomial Fisher calculation for both timing and rate, and check Monte Carlo posterior and 200 ms forecast covariance with heterogeneous measurement precision and independent transmitter noise. That establishes conditional minimum-variance estimation. It does not choose the physical noise parameters of this SDR.</p>
<p>A first model applied to the old recording has strongly correlated innovations and is rejected as calibrated: <a href="replay-reference-full.json">innovation diagnostics</a>. Its default Allan values are design examples, not measured hardware parameters. The live controller has the reference fix and covariance-weighted observations; the experimental Allan/Kalman core is not silently substituted. A clean controller-fixed recording is the next useful measurement for oscillator/noise identification and held-out innovation validation. Independent absolute TCXO characterization would additionally require a known reference.</p>
<h2>6. Phase preservation and the limit of this explanation</h2>
<p>The five-channel end-to-end waveform test uses independent DVB-T signals, independent source-clock offsets, two moving physical echoes including one beyond the guard interval, and a common time warp including its channel-NCO phase. It compares against an independently generated CPU finite-waveform oracle without phase/gain alignment.</p>
<table><tr><th>Common warp</th><th>Worst echo phase RMS</th><th>Worst conditional coherence loss</th><th>Original 0.37° gate</th></tr>{waveform}</table>
<p>Every FEC word passes in both cases. The larger case fails the phase regression and is retained as a stress limit; the smaller case passes without relaxing any threshold. The quoted coherence loss is conditional on the compared echoes and oracle, not a whole-pipeline sensitivity bound. Shared phase can cancel through the direct-path reference, so multiplying clock time error by the RF span is not a measured radar coherence loss. Conversely, intra-symbol dilation, CFO/ICI and cancellation leakage can remain. These checks do not yet establish that this defect caused the ship-image smear.</p>
<p>The default three-degree observation-loop stress test improves from {clock_before['phase_rms_degrees']:.3f}° to <strong>{clock_after['phase_rms_degrees']:.3f}° RMS</strong> over a three-second interval after the common-reference fix and covariance weighting. This is a simulated observation model; the separate physical-waveform test and GPU interpolation regressions cover different parts of the pipeline.</p>
<h2>Files and reproduction</h2>
<p><a href="summary.json">Summary and source hashes</a> · <a href="weighted-information.json">Pilot/rate information and clock stress</a> · <a href="cpu-regressions.log">CPU regressions</a> · <a href="gpu-regressions.log">GPU regressions</a> · <a href="capture-metrology-v2/analysis.json">Recorded metrology</a>. The data reader skips IQ and profile payloads; no original data or archive metadata are rewritten. Analysis tools live under <code>support/validation</code>.</p>'''
    (r/'index.html').write_text(html);print(json.dumps(data),flush=True)


if __name__=='__main__':main()
