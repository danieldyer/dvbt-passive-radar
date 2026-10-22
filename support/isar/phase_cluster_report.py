"""Reproducible, staged prominent-point phase refinement and review plots.

Run from the repository root with the validation venv. This diagnostic writes
only support artifacts; it never updates capture metadata or site positions.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cluster_phase import Cluster, SplinePath, crop
from compare_ais_phase import NAMES, COLOURS
from motion import C
from test_cluster_phase import synthetic


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def run_fit(data, first, last, radius=2, degree=2, centre=0, rate_bound=2., origin_width=.8):
    f = crop(data, first, last, np.arange(centre-radius, centre+radius+1))
    # Contiguous eighth-second holdout blocks. Neither the rate seed nor any
    # optimization iteration sees these samples. These test phase/interpolation,
    # not extrapolation across many seconds or generalization to another ship.
    training = ((f.time-first)//.125).astype(int)%2 == 0
    problem = Cluster(f, training, degree=degree)
    fit = problem.fit(origin_bounds=(centre-origin_width, centre+origin_width), rate_bound=rate_bound)
    result = {'interval_s': [first, last], 'offset_samples': f.offsets.tolist(),
              'epoch_s': problem.epoch, 'scale_s': problem.scale, 'degree': degree,
              'parameters': fit.parameters.tolist(), 'objective': fit.objective,
              'maximum_gradient': fit.gradient, 'iterations': fit.iterations, 'trials': fit.trials,
              'conditional_parameter_covariance': fit.covariance.tolist(),
              'information_rank': fit.information_rank, 'diagnostics': problem.diagnostics(fit)}
    return problem, fit, result


def synthetic_stage(output):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), layout='constrained')
    results = []
    for column, (name, kwargs) in enumerate([
            ('Isolated point', {'noise': .15}),
            ('Deep fade + missing samples', {'noise': .15, 'fade': True, 'gaps': True}),
            ('Brighter competing echo', {'noise': .15, 'neighbour': 2.})]):
        p, truth = synthetic(**kwargs)
        f = p.fit()
        d = p.diagnostics(f)
        t = p.field.time;test = ~p.training
        truth_phase = truth[1]-p.field.k*p.path(truth)
        model_phase = f.parameters[1]-p.field.k*p.path(f.parameters)
        # Plot only a constant-cycle alignment; no fitted trend removal.
        model_phase += 2*np.pi*np.round(np.mean(truth_phase-model_phase)/(2*np.pi))
        axes[0, column].plot(t, np.degrees(truth_phase), color='black', label='Known truth')
        axes[0, column].plot(t, np.degrees(model_phase), '--', color='#d55e00', label='Training fit')
        axes[0, column].set_title(name)
        axes[0, column].legend(fontsize=8)
        q = np.array(d['phase_residual_deg'])
        axes[1, column].plot(t[p.training], q[p.training], '.', color='#999999', label='Training')
        axes[1, column].plot(t[test], q[test], '.', color='#0072b2', label='Withheld')
        axes[1, column].axhline(0, color='black', lw=.5)
        axes[1, column].legend(fontsize=8)
        axes[2, column].plot(t, d['amplitude'], color='#0072b2')
        axes[2, column].set_title(f"Withheld complex prediction: {100*d['test_complex_prediction_fraction']:.1f}%\n"
                                f"Withheld phase RMS: {d['test_phase_rms_deg']:.1f}°")
        axes[2, column].set_xlabel('Synthetic recording time (s)')
        results.append({'case': name, 'true_parameters': truth.tolist(),
                        'fit_parameters': f.parameters.tolist(), 'diagnostics': d})
    for row, label in enumerate(['Phase (degrees)', 'Phase error (degrees)', 'Matched amplitude']):
        axes[row, 0].set_ylabel(label)
    fig.suptitle('Step 1 · independent forward simulation and withheld-time prediction\n'
                 'One physical PSF and one linked delay/phase history; a competing echo must remain visible as model error')
    fig.savefig(output/'01-synthetic-validation.png', dpi=170);plt.close(fig)
    # Empirical uncertainty check of the conditional likelihood, no waveform
    # or phase perturbation chosen after observing any of these noise draws.
    estimates, variances = [], []
    for seed in range(40):
        p, truth = synthetic(seed=seed+100, noise=1.)
        f = p.fit(seeds=[truth+np.array([.03, .05, .01, -.01])])
        estimates.append(f.parameters-truth);variances.append(np.diag(f.covariance))
    standard = np.std(estimates, axis=0, ddof=1)
    predicted = np.sqrt(np.mean(variances, axis=0))
    results.append({'case': '40 independent Gaussian noise trials, local convergence test',
                    'empirical_parameter_std': standard.tolist(), 'conditional_predicted_std': predicted.tolist(),
                    'empirical_to_predicted_std_ratio': (standard/predicted).tolist(),
                    'mean_error': np.mean(estimates, axis=0).tolist(),
                    'scope': 'Correct isolated-point model and supplied covariance; not a claim about the real ship or global convergence.'})
    save_json(output/'01-synthetic-validation.json', results)
    return results


def burst_stage(directory, output):
    results = []
    fig, axes = plt.subplots(5, 3, figsize=(17, 15), layout='constrained')
    for channel in range(5):
        source = directory/f'channel-{channel}.npz'
        with np.load(source) as data:
            problem, fit, result = run_fit(data, 164., 165.)
            f = problem.field;d = result['diagnostics']
            time = f.time;train = problem.training
            model = np.array(d['model_residual_phase_rad'])
            observed = model+np.radians(d['phase_residual_deg'])
            origin = np.average(model[train])
            factor = -100/f.k
            axes[channel, 0].plot(time[train], factor*(observed[train]-origin), '.', color='#999999', ms=3)
            axes[channel, 0].plot(time[~train], factor*(observed[~train]-origin), '.', color=COLOURS[channel], ms=4, label='Withheld observations')
            axes[channel, 0].plot(time, factor*(model-origin), color='black', label='Training-only quadratic')
            axes[channel, 0].set_ylabel(NAMES[channel]+'\nPath residual (cm)')
            axes[channel, 0].set_title(f"Withheld phase RMS {d['test_phase_rms_deg']:.1f}°")
            axes[channel, 1].plot(time, d['phase_residual_deg'], '.', color=COLOURS[channel])
            axes[channel, 1].axhline(0, color='black', lw=.6)
            axes[channel, 1].set_ylabel('Phase error (degrees)')
            axes[channel, 1].set_title(f"Withheld complex prediction {100*d['test_complex_prediction_fraction']:.1f}%")
            raw = data['value'][(data['time']>=164)&(data['time']<165)]
            x = data['offset_samples']*C/8e6
            power = np.mean(abs(raw)**2, axis=0)
            axes[channel, 2].plot(x, 10*np.log10(power/max(power)), '.-', color=COLOURS[channel], label='Observed power')
            axes[channel, 2].axvspan(-2.5*C/8e6, 2.5*C/8e6, color='#777777', alpha=.1, label='Fitted cluster')
            axes[channel, 2].axvline(fit.parameters[0]*C/8e6, color='black', ls='--', label='Fitted PSF delay')
            axes[channel, 2].set_ylim(-35, 2)
            axes[channel, 2].set_ylabel('Relative range power (dB)')
            axes[channel, 2].set_title(f"Delay origin {fit.parameters[0]*C/8e6:+.1f} m from AIS")
            sensitivity = []
            for radius in (1, 3):
                _, _, variant = run_fit(data, 164., 165., radius=radius)
                sensitivity.append(variant)
            result['range_width_controls'] = sensitivity
            result['channel'] = NAMES[channel]
            result['rf_hz'] = f.metadata['rf_hz']
            result['source'] = str(source)
            with source.open('rb') as handle:
                result['source_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
            results.append(result)
            print('burst', NAMES[channel], 'phase RMS', round(d['test_phase_rms_deg'], 2),
                  'prediction', round(d['test_complex_prediction_fraction'], 3), flush=True)
    axes[0, 0].legend(fontsize=8);axes[0, 2].legend(fontsize=8)
    for col in (0, 1):
        axes[-1, col].set_xlabel('Recording time (s)')
    axes[-1, 2].set_xlabel('Excess path relative to each band’s AIS prediction (m)')
    fig.suptitle('Step 2 · the same AIS-centred cluster in all five bands, 164–165 s\n'
                 'Physical range response; alternate 125 ms blocks withheld; only one constant phase origin removed per band')
    fig.savefig(output/'02-bright-cluster.png', dpi=170);plt.close(fig)
    save_json(output/'02-bright-cluster.json', results)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for row, colour in zip(results, COLOURS):
        variants = [row['range_width_controls'][0], row, row['range_width_controls'][1]]
        widths = [3, 5, 7]
        rate = [v['parameters'][2]/v['scale_s'] for v in variants]
        errors = [np.sqrt(v['conditional_parameter_covariance'][2][2])/v['scale_s'] for v in variants]
        axes[0, 0].errorbar(widths, rate, yerr=errors, fmt='.-', color=colour, label=row['channel'])
        axes[0, 1].plot(widths, [v['parameters'][0]*C/8e6 for v in variants], '.-', color=colour)
        axes[1, 0].plot(widths, [v['diagnostics']['test_phase_rms_deg'] for v in variants], '.-', color=colour)
        axes[1, 1].plot(widths, [100*v['diagnostics']['test_complex_prediction_fraction'] for v in variants], '.-', color=colour)
    axes[0, 0].set_ylabel('Residual path rate at midpoint (m/s)')
    axes[0, 0].legend(ncol=3, fontsize=9)
    axes[0, 1].set_ylabel('Delay origin relative to AIS (m)')
    axes[1, 0].set_ylabel('Withheld phase RMS (degrees)')
    axes[1, 1].set_ylabel('Withheld complex prediction (%)')
    for ax in axes.flat:
        ax.set_xticks([3, 5, 7]);ax.set_xlabel('Number of adjacent range cells')
    fig.suptitle('Step 3 · sensitivity to cluster width\n'
                 'Error bars: conditional thermal 1σ only. Width-dependent shifts expose unresolved signal/model error.')
    fig.savefig(output/'03-range-width-control.png', dpi=170);plt.close(fig)
    return results


def extended_stage(directory, output):
    # This interval brackets two actual AIS reports; no extrapolation of the
    # bright 1-second polynomial to those reports is performed.
    first, last = 132., 204.
    centres = np.arange(first+1.5, last-1.49, 1.5)
    results = []
    fig, axes = plt.subplots(5, 3, figsize=(17, 15), layout='constrained')
    for channel in range(5):
        with np.load(directory/f'channel-{channel}.npz') as data:
            fits = []
            for centre in centres:
                _, _, r = run_fit(data, centre-1.5, centre+1.5)
                fits.append(r)
            time = data['time'];keep = (time>=first)&(time<last)
            value = data['value'][keep]
            power = 20*np.log10(np.maximum(abs(value), 1e-30))
            top = np.percentile(power, 99.8)
            axes[channel, 0].imshow(power.T, origin='lower', aspect='auto',
                extent=[time[keep][0], time[keep][-1], -8.5*C/8e6, 8.5*C/8e6],
                vmin=top-35, vmax=top, cmap='magma')
            axes[channel, 0].plot(centres, [r['parameters'][0]*C/8e6 for r in fits], color='#00ddff', lw=1, label='Fitted envelope origin')
            axes[channel, 0].set_ylim(-240, 240)
            axes[channel, 0].set_ylabel(NAMES[channel]+'\nRange offset from AIS (m)')
            rate = [r['parameters'][2]/r['scale_s'] for r in fits]
            error = [np.sqrt(r['conditional_parameter_covariance'][2][2])/r['scale_s'] for r in fits]
            axes[channel, 1].errorbar(centres, rate, yerr=error, fmt='.', color=COLOURS[channel], ms=4)
            axes[channel, 1].set_ylabel('Local residual path rate (m/s)')
            axes[channel, 2].plot(centres, [r['diagnostics']['test_phase_rms_deg'] for r in fits], '.-', color=COLOURS[channel], label='Phase RMS')
            axes[channel, 2].set_ylabel('Withheld phase RMS (degrees)')
            for ax in axes[channel]:
                for report in (139.892381, 195.892381):
                    ax.axvline(report, color='#44aa99', ls=':', lw=1.5)
                ax.axvspan(164, 165, color='#999999', alpha=.15)
            results.append({'channel': NAMES[channel], 'fits': fits})
            print('extension', NAMES[channel], 'windows', len(fits), flush=True)
    for ax in axes[-1]:
        ax.set_xlabel('Recording time (s)')
    axes[0, 0].set_title('Unmodified range magnitude and local delay fits')
    axes[0, 1].set_title('Independent 3 s quadratic fits; thermal 1σ bars')
    axes[0, 2].set_title('Phase prediction on withheld 125 ms blocks')
    fig.suptitle('Step 4 · extend the cluster to actual AIS epochs\n'
                 'Dotted vertical lines: AIS reports. Local fits are not yet joined across fades or competing returns.')
    fig.savefig(output/'04-extension-to-ais.png', dpi=170);plt.close(fig)
    save_json(output/'04-extension-to-ais.json', results)
    return results


def spline_stage(directory, output):
    first, last = 154., 184.
    results = []
    fig, axes = plt.subplots(5, 3, figsize=(17, 15), layout='constrained')
    for channel in range(5):
        with np.load(directory/f'channel-{channel}.npz') as data:
            f = crop(data, first, last, np.arange(-2, 3))
            partition = ((f.time-first)//.125).astype(int)%4
            training, validation, testing = partition%2 == 0, partition == 1, partition == 3
            alternatives = []
            for spacing in (1., 2., 4.):
                problem = Cluster(f, training, curve=SplinePath(f.time[0], f.time[-1], spacing))
                seeds = [problem.phase_seed(origin) for origin in (-.3, .3, .6)]
                fit = problem.fit(seeds=seeds, rate_bound=4.)
                v = problem.diagnostics(fit, validation)
                t = problem.diagnostics(fit, testing)
                alternatives.append((problem, fit, v, t, spacing))
            # Select flexibility using the validation quarter only. Test data
            # enters neither the seed, fit, nor this comparison.
            problem, fit, v, d, spacing = max(alternatives, key=lambda x:x[2]['test_complex_prediction_fraction'])
            model = np.array(d['model_residual_phase_rad'])
            observed = model+np.radians(d['phase_residual_deg'])
            anchor = np.interp(164.5, f.time, model)
            factor = -100/f.k
            axes[channel, 0].plot(f.time[testing], factor*(observed[testing]-anchor), '.', color=COLOURS[channel], ms=2, alpha=.5)
            axes[channel, 0].plot(f.time, factor*(model-anchor), color='black', lw=1)
            axes[channel, 0].set_ylabel(NAMES[channel]+'\nResidual path (cm)')
            axes[channel, 0].set_title(f"Chosen knot spacing {spacing:g} s; test phase RMS {d['test_phase_rms_deg']:.1f}°")
            axes[channel, 1].plot(f.time[testing], np.array(d['phase_residual_deg'])[testing], '.', color=COLOURS[channel], ms=2)
            axes[channel, 1].axhline(0, color='black', lw=.5)
            axes[channel, 1].set_ylabel('Held-out phase error (degrees)')
            axes[channel, 1].set_title(f"Test complex prediction {100*d['test_complex_prediction_fraction']:.1f}%")
            steps = np.arange(len(fit.iterations))
            axes[channel, 2].plot(steps, [r['objective'] for r in fit.iterations], color=COLOURS[channel])
            axes[channel, 2].set_ylabel('Training residual / data energy')
            axes[channel, 2].set_title(f"Gradient max {fit.gradient:.1e}; conditional rank {fit.information_rank}/{len(fit.parameters)}")
            alternatives_json = []
            for p, r, val, test, knot in alternatives:
                trials = []
                for trial in r.trials:
                    parameters = np.array(trial['parameters'])
                    phase = parameters[1]-f.k*p.path(parameters)
                    phase -= np.interp(164.5, f.time, phase)
                    trials.append({**trial, 'phase_from_164_5_rad': phase.tolist()})
                alternatives_json.append({'knot_spacing_s': knot, 'parameters': r.parameters.tolist(),
                    'maximum_gradient': r.gradient, 'objective': r.objective, 'trials': trials,
                    'validation_prediction_fraction': val['test_complex_prediction_fraction'],
                    'test_prediction_fraction': test['test_complex_prediction_fraction'],
                    'test_phase_rms_deg': test['test_phase_rms_deg']})
            results.append({'channel': NAMES[channel], 'interval_s': [first, last],
                            'epoch_s': problem.epoch, 'parameters': fit.parameters.tolist(),
                            'chosen_knot_spacing_s': spacing, 'diagnostics': d,
                            'validation': validation.tolist(), 'testing': testing.tolist(),
                            'alternatives': alternatives_json, 'iterations': fit.iterations,
                            'conditional_parameter_covariance': fit.covariance.tolist(),
                            'warning': 'A spline is a local phase estimate, not an identified physical reflector or a resolved integer-cycle connection through a fade.'})
            print('spline', NAMES[channel], 'spacing', spacing, 'test RMS', round(d['test_phase_rms_deg'], 1),
                  'prediction', round(d['test_complex_prediction_fraction'], 3), 'gradient', fit.gradient, flush=True)
    for ax in axes[-1, :2]:
        ax.set_xlabel('Recording time (s)')
    axes[-1, 2].set_xlabel('Joint delay/phase optimization iteration')
    fig.suptitle('Step 5 · recursive cluster/trajectory refinement over 30 s\n'
                 'Cubic path drives envelope and phase together; 50% fit, 25% choose knot spacing, 25% final prediction')
    fig.savefig(output/'05-joint-trajectory.png', dpi=170);plt.close(fig)
    save_json(output/'05-joint-trajectory.json', results)
    return results


def report(output, synthetic_result, bursts, extended, splines):
    rows = ''.join(f"<tr><td>{html.escape(r['channel'])}</td>"
                   f"<td>{r['diagnostics']['test_phase_rms_deg']:.2f}°</td>"
                   f"<td>{100*r['diagnostics']['test_complex_prediction_fraction']:.1f}%</td>"
                   f"<td>{r['parameters'][0]*C/8e6:+.2f} m</td></tr>" for r in bursts)
    ratio = synthetic_result[-1]['empirical_to_predicted_std_ratio']
    content = f'''<!doctype html><meta charset="utf-8"><title>Ship phase refinement: staged validation</title>
<style>body{{max-width:1250px;margin:2em auto;padding:0 1em;font:17px/1.5 system-ui;color:#222}}img{{width:100%}}
table{{border-collapse:collapse}}td,th{{padding:.4em 1em;border-bottom:1px solid #ccc}}code{{font-size:14px}}</style>
<h1>Ship phase refinement: staged validation</h1>
<p>This report starts again from the retained complex range data. It tests a small physical echo cluster, then asks whether its phase can be predicted and extended to actual AIS reports. No antenna positions have been adopted from these tests.</p>
<p><b>Model:</b> y(t) = A(t) exp(iφ₀ − ikΔL(t)) p[r − L₀ − ΔL(t)] + noise, with A(t) ≥ 0. One quadratic path increment drives both range migration and carrier phase. The exact original row times, OFDM symbol moments, range covariance and 16-row binning are included. The range origin and constant reflectivity phase have separate identifiable roles. All bands use their own RF wavenumber.</p>
<p>The positive real amplitude can vary during a glint or fade; it cannot absorb phase errors. The optimizer profiles this amplitude, while delay and phase are fitted together. Initial seeds, association and all optimization iterations use training samples only. Withheld complex prediction additionally interpolates amplitudes from training samples.</p>
<h2>1. Numerical controls</h2>
<p>Independent Fourier-domain pulse synthesis and numerical symbol integration test fractional range, phase sign, physical rate and exact binning. Other tests change withheld data, include erasures/fades, and add a brighter neighbouring echo. Forty local noise trials give empirical/predicted parameter standard-deviation ratios {', '.join(f'{x:.2f}' for x in ratio)}. These are conditional on the correct point model and covariance, not a real-data CRLB claim.</p>
<a href="01-synthetic-validation.json">Numerical results</a><img src="01-synthetic-validation.png">
<h2>2. Bright 164–165 s cluster</h2>
<p>Five adjacent cells around the same AIS-aligned range coordinate. Alternate 125 ms blocks are withheld. The displayed residual paths remove only one constant phase origin, and include no antenna-location correction.</p>
<table><tr><th>Band</th><th>Withheld phase RMS</th><th>Complex prediction</th><th>Range origin</th></tr>{rows}</table>
<p>Complex prediction means 1 − ||y − prediction||²/||y||² on withheld data; it is not the percentage of ship energy recovered or a coherence guarantee.</p>
<a href="02-bright-cluster.json">Fits, convergence histories and covariance</a><img src="02-bright-cluster.png">
<h2>3. Range-selection sensitivity</h2>
<p>Refitting with 3, 5 and 7 adjacent cells tests whether the inferred trajectory depends on unresolved nearby energy. The error bars include conditional thermal uncertainty only. Variation exceeding those bars is evidence that this uncertainty alone is insufficient.</p>
<img src="03-range-width-control.png">
<h2>4. Extend to actual AIS observations</h2>
<p>Actual AIS reports at 139.892381 s and 195.892381 s bracket the bright feature. Independent overlapping 3 s fits expose fades and association problems; their polynomials are not extrapolated from the bright second to the AIS reports. A visually smooth joined curve would not by itself establish a consistent reflector or resolve integer phase cycles.</p>
<a href="04-extension-to-ais.json">All local fits</a><img src="04-extension-to-ais.png">
<h2>5. Joint trajectory and range-cluster refinement</h2>
<p>For a 30 s arc, a cubic spline replaces a high-degree global polynomial. At each iteration its path changes both the physical delay template and carrier phase; the real amplitudes are reprofiled. Training-only phase unwrapping provides several seeds and is not treated as a resolved cycle connection. A separate validation quarter chooses 1, 2 or 4 s knot spacing; the last quarter supplies the plotted prediction errors. These are short withheld blocks within this arc, not a long-gap extrapolation test.</p>
<p>The path traces remove only a constant phase at 164.5 s. Grey phase origins and unknown integer cycles must not become antenna calibration measurements merely because the resulting curve is smooth. Competing initializations and their trajectories are retained in the JSON.</p>
<a href="05-joint-trajectory.json">Trajectories, alternate initializations, validation and test statistics</a><img src="05-joint-trajectory.png">
<h2>Scope and reproducibility</h2>
<p>These are offline support diagnostics. Source SHA-256 hashes and all parameters are in the JSON files. The captured data and production processing are unchanged. The archived profiles already contain the capture-time frozen stationary subtraction; this report adds no stationary notch or data normalization. The likelihood covariance is conditional thermal range covariance, not an independently validated model of residual clutter.</p>
<p>The joint range/phase reference follows the physical requirement discussed by <a href="https://isprs-annals.copernicus.org/articles/IV-3/61/2018/index.html">Chen, Jin and Dong (2018)</a>; this implementation uses a concentrated complex likelihood and explicit withheld-time tests. It does not infer that different bands necessarily see different reflectors.</p>
<p><code>build/validation-venv/bin/python support/isar/phase_cluster_report.py --fields build/isar-20261015/full-fields --output {html.escape(str(output))}</code></p>
'''
    (output/'index.html').write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    def stage(name, run):
        path = args.output/name
        return json.loads(path.read_text()) if args.reuse and path.exists() else run()
    synth = stage('01-synthetic-validation.json', lambda:synthetic_stage(args.output))
    burst = stage('02-bright-cluster.json', lambda:burst_stage(args.fields, args.output))
    extend = stage('04-extension-to-ais.json', lambda:extended_stage(args.fields, args.output))
    splines = stage('05-joint-trajectory.json', lambda:spline_stage(args.fields, args.output))
    report(args.output, synth, burst, extend, splines)
    print(args.output/'index.html', flush=True)


if __name__ == '__main__':
    main()
