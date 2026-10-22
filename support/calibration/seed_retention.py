"""Audit information retained by calibration deramping, binning and gating.

The counterfactual echo follows the chosen source track in every band. This is
an extraction-operator diagnostic, not a claim that every real echo is a point
target or that the production pipeline has this loss. Range interpolation and
physical transmitter differences are outside this conditional calculation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from complex_likelihood import gate
from phase_history import C, fit_track_seed


def retention(time, information, residual_phase, start, end, bandwidth_hz=1.):
    bins = np.floor((time-start)/.02).astype(int)
    count = int(np.ceil((end-start)/.02))
    keep = (bins>=0)&(bins<count)
    t, w, b = time[keep], information[keep], bins[keep]
    signal = np.exp(1j*residual_phase[keep])
    weight = np.bincount(b, weights=w, minlength=count)
    good = weight>0
    average = (np.bincount(b, weights=w*signal.real, minlength=count)
               +1j*np.bincount(b, weights=w*signal.imag, minlength=count))[good]/weight[good]
    times = np.bincount(b, weights=w*t, minlength=count)[good]/weight[good]
    q, _ = gate(times, 1/weight[good], bandwidth_hz, (start+end)/2, end-start)
    total = w.sum()
    binned = np.sum(weight[good]*abs(average)**2)/total
    projected = np.sum(abs(q@average)**2)/total
    return {'binning_retained_fraction': float(binned),
            'binning_and_gate_retained_fraction': float(projected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('histories', 'tracks', 'jobs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    tracks = json.loads(args.tracks.read_text())['tracks']
    results = []
    for label, channel, index, start, end in json.loads(args.jobs.read_text()):
        description = json.loads((args.histories/f'{label}.json').read_text())
        reference = fit_track_seed(np.asarray(tracks[channel][index]), start, end)
        bands = []
        for c, old in enumerate(description['channels']):
            with np.load(args.histories/f'{label}-{c}.npz') as source:
                time, weight = source['row_time'], source['row_information']
            def variation(seed):
                return np.polynomial.polynomial.polyval(
                    (time-seed['epoch'])/seed['scale'],seed['path'])-seed['path'][0]
            phase = -2*np.pi*old['rf_hz']/C*(variation(reference)-variation(old))
            previous = retention(time, weight, phase, start, end)
            shared = retention(time, weight, np.zeros(len(time)), start, end)
            bands.append({'channel': c, 'old_independent_seed': previous,
                          'common_source_seed': shared})
        results.append({'label': label, 'source_channel': channel, 'bands': bands})
        print(label, 'old retained percent', [round(100*r['old_independent_seed']['binning_and_gate_retained_fraction'], 2) for r in bands], flush=True)
    report = {'targets': results,
              'scope': 'Conditional extraction-operator audit for an ideal common-path echo following the actually selected source track. Uses original row epochs and information weights, 20 ms bins and the existing 1 Hz half-band gate. Independent row noise with variance proportional to inverse archived information is assumed. Range interpolation, physical transmitter differences and real scattering mismatch are excluded; these are not measured production SNR losses.',
              'consequence': 'Old calibration rejection and information estimates that use independently reselected per-band seeds require re-evaluation. No coordinates from those fits were adopted.'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
