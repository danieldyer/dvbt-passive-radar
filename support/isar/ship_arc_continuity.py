"""Test ship phase continuity while profiling positive, variable brightness.

Candidate regions predate this experiment. Amplitude knot spacing is selected
on a validation quarter; the remaining quarter tests frozen complex prediction.
All models include the same explicit stationary nuisance and original-row
forward operator. No geographic calibration is inferred from these arc fits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from amplitude_arc import AmplitudeArc
from cluster_phase import crop
from coherent_arc import CoherentArc
from compare_ais_phase import NAMES, COLOURS
from motion import C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'fits.json').exists():
        raise ValueError('Use a fresh output to preserve the declared comparison')
    intervals = [('P02 context',146.,150.), ('P05 context',158.,162.),
                 ('P05 to flash',160.,166.), ('Flash context',163.,166.), ('Flash',164.,165.)]
    result = {'scope': __doc__, 'complete': False, 'accepted_calibration': False,
              'intervals': [{'label': label, 'first_s': first, 'last_s': last, 'channels': []}
                            for label, first, last in intervals]}
    sources = [Path(__file__), Path(__file__).with_name('amplitude_arc.py'),
               Path(__file__).with_name('coherent_arc.py'), Path(__file__).with_name('field.py')]
    for ch in range(5):
        source = args.fields/f'channel-{ch}.npz'; sources.append(source)
        with np.load(source) as data:
            for interval in result['intervals']:
                first, last = interval['first_s'], interval['last_s']
                field = crop(data, first, last, np.arange(-2,3))
                partition = ((field.time-first)//.125).astype(int)%4
                training, validation, testing = partition%2 == 0, partition == 1, partition == 3
                base = CoherentArc(field, training); baseline = base.fit((-.8,.8),4.)
                constant = {'parameters': baseline.parameters.tolist(), 'objective': baseline.objective,
                    'test': base.diagnostics(baseline, testing), 'validation': base.diagnostics(baseline, validation)}
                alternatives = []
                for spacing in (.25, .5, 1.):
                    knots = np.r_[field.data['row_time'][0], np.arange(first+spacing,last,spacing),field.data['row_time'][-1]]
                    model = AmplitudeArc(field, training, knots)
                    seeds = []
                    for reduced in model.initial((-.8,.8),4.):
                        _, _, _, amplitude = CoherentArc.evaluate(model, reduced, True)
                        seeds.append(np.r_[reduced[0], np.angle(amplitude), reduced[1:]])
                    fit = model.fit((-.8,.8),4.,seeds=[baseline.parameters,*seeds])
                    alternatives.append({'spacing_s': spacing, 'parameters': fit.parameters.tolist(),
                        'objective': fit.objective, 'maximum_gradient': fit.gradient,
                        'information_rank': fit.information_rank, 'conditional_parameter_covariance': fit.covariance.tolist(),
                        'trials': fit.trials, 'validation': model.diagnostics(fit, validation),
                        'test': model.diagnostics(fit, testing)})
                chosen = int(np.argmax([v['validation']['test_complex_prediction_fraction'] for v in alternatives]))
                best = alternatives[chosen]
                interval['channels'].append({'channel': NAMES[ch], 'rf_hz': field.metadata['rf_hz'],
                    'epoch_s': base.epoch, 'scale_s': base.scale, 'constant': constant,
                    'alternatives': alternatives, 'chosen_index': chosen})
                print(interval['label'],NAMES[ch],'constant',round(constant['test']['test_complex_prediction_fraction'],3),
                    'variable',round(best['test']['test_complex_prediction_fraction'],3),'retained',
                    round(best['test']['training_target_information_fraction_after_stationary'],3),flush=True)
        (args.output/'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    result.update(complete=True, source_sha256={str(s):hashlib.file_digest(s.open('rb'),'sha256').hexdigest() for s in sources})
    (args.output/'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    fig, axes = plt.subplots(len(intervals),2,figsize=(13,15),layout='constrained')
    for i, interval in enumerate(result['intervals']):
        selected = [r['alternatives'][r['chosen_index']] for r in interval['channels']]
        axes[i,0].plot(NAMES,[100*r['constant']['test']['test_complex_prediction_fraction'] for r in interval['channels']],
                       'o--',label='Constant amplitude')
        axes[i,0].plot(NAMES,[100*r['test']['test_complex_prediction_fraction'] for r in selected],
                       'o-',label='Positive linear amplitude')
        axes[i,0].set_ylim(-10,100); axes[i,0].set_ylabel('Held-out energy explained (%)');axes[i,0].legend(fontsize=8)
        axes[i,1].plot(NAMES,[100*r['constant']['test']['training_target_information_fraction_after_stationary'] for r in interval['channels']],'o--')
        axes[i,1].plot(NAMES,[100*r['test']['training_target_information_fraction_after_stationary'] for r in selected],'o-')
        axes[i,1].set_ylim(-2,102);axes[i,1].set_ylabel('Template energy retained (%)')
        for ax in axes[i]:
            ax.set_title(f"{interval['label']} · {interval['first_s']:g}–{interval['last_s']:g} s")
    fig.suptitle('Amplitude changes versus phase continuity · fixed unseen quarter for final scores\n'
                 'Stationary projection can make short near-zero-Doppler fits ambiguous; longer spans test that ambiguity')
    fig.savefig(args.output/'model-comparison.png',dpi=150);plt.close(fig)
    interval = result['intervals'][2]
    fig, axes = plt.subplots(5,3,figsize=(15,14),layout='constrained')
    for i,row in enumerate(interval['channels']):
        fit = row['alternatives'][row['chosen_index']]; d = fit['test'];t=np.array(d['time_s']);test=np.array(d['testing'])
        model = np.array(d['model_residual_phase_rad']);observed=model+np.radians(d['phase_residual_deg'])
        origin=np.interp(164.5,t,model); factor=-1/(2*np.pi*row['rf_hz']/C)
        axes[i,0].plot(t[test],factor*(observed[test]-origin),'.',ms=3,color=COLOURS[i],label='Unused data')
        axes[i,0].plot(t,factor*(model-origin),color='black',label='Fitted quadratic')
        axes[i,0].set_ylabel(row['channel']+'\nPath residual (m)');axes[i,0].legend(fontsize=8)
        axes[i,1].plot(t[test],np.array(d['phase_residual_deg'])[test],'.',ms=3,color=COLOURS[i]);axes[i,1].axhline(0,color='black',lw=.6)
        axes[i,1].set_ylabel('Phase residual (°)');axes[i,1].set_ylim(-185,185)
        axes[i,2].plot(t,d['amplitude'],color=COLOURS[i]);axes[i,2].set_ylabel('Fitted nonnegative amplitude')
        axes[i,2].set_title(f"{fit['spacing_s']:g} s knots; test {100*d['test_complex_prediction_fraction']:.1f}%")
        for ax in axes[i]:ax.axvspan(164,165,color='#777777',alpha=.12)
    for ax in axes[-1]:ax.set_xlabel('Capture time (s)')
    fig.suptitle('160–166 s: one linked range/phase trajectory per band, with changing brightness\n'
                 'Only a constant phase origin removed; no inter-band registration or unwrapping of observations')
    fig.savefig(args.output/'phase-continuity.png',dpi=150);plt.close(fig)
    (args.output/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship phase continuity with varying brightness</title>
<style>body{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}img{max-width:100%}</style>
<h1>Ship phase continuity with varying brightness</h1>
<p>Each band has one quadratic range/phase trajectory and one constant phase origin. Nonnegative amplitude knots are applied at the original symbol epochs before averaging, while stationary complex range coefficients are fitted jointly. No time-dependent phase is hidden in those amplitudes. Training uses alternate 125 ms blocks. The other blocks are divided into a validation quarter (which chooses 0.25, 0.5 or 1 s amplitude knots) and a final prediction quarter. All prediction coefficients are frozen.</p>
<img src="model-comparison.png" alt="Constant and variable amplitude prediction and information retention">
<p>Both methods project the same stationary nuisance. Energy explained is relative to that nuisance-only predictor; it is not a percentage of physical ship energy, an image-SNR gain, or an antenna calibration. High explained energy with almost no surviving template energy exposes a nearly stationary ambiguity. Conditional thermal covariance excludes target-model error.</p>
<img src="phase-continuity.png" alt="Unused-sample phase residuals across the six-second interval">
<p><a href="fits.json">All models, initialization trials, final test scores and source hashes</a>. The intervals are chosen around previously discovered features, so this is a controlled local comparison, not an independent detection trial. Source observations and production processing are unchanged.</p>
<p>The separation of glint and physical motion matters for narrowband passive ISAR; see <a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5236">Garry et al., Passive ISAR part II (2019)</a>. This experiment tests the stated amplitude/phase model and does not assume that its phase is necessarily the hull trajectory.</p></html>''')


if __name__ == '__main__':
    main()
