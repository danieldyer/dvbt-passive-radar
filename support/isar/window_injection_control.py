"""Compare standard aperture tapers with identical data and truth-started fits.

This deliberately changes the temporal weighting; it is not claimed to be a
noise-whitening operation or a statistically optimal interference model.
"""
import argparse
from copy import copy
import json
from pathlib import Path

import numpy as np
from scipy.signal.windows import hann, tukey

from recorded_injection_fields import load_fields
from recovery_experiment import Experiment, SCALE
from report_measured_scatterer_phases import serial
from validate_injection_acquisition import geometric_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/data/injection_20261018/separated-500m'))
    args = parser.parse_args(); root = args.root
    prior = json.loads((root/'analytic-profile-fit/analysis.json').read_text())
    fields, rotation, rx, tx, state, mid = load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    training = [f.time < prior['training_end_s'] for f in fields]
    original = Experiment(fields, rotation, rx, tx, state, mid, training=training)
    truth = np.asarray(prior['truth_parameters'])
    with np.load(root/'analytic-profile-fit/observations.npz') as archive:
        clean = [archive[f'injected_{c}'] for c in range(5)]
        data = [archive[f'data_{c}'] for c in range(5)]
    result = dict(scope=__doc__, controls={})
    for name, generator in [('untapered', lambda n: np.ones(n)), ('tukey_025', lambda n: tukey(n, .25, sym=False)),
                            ('hann', lambda n: hann(n, sym=False))]:
        selected, weights, weighted = [], [], []
        for f, train, y in zip(fields, training, data):
            weight = np.ones(len(f.time)); weight[train] = generator(int(train.sum()))
            clone = copy(f)
            clone.root_information = f.root_information*np.sqrt(weight)
            clone.y = f.y*np.sqrt(weight[:, None])
            selected.append(clone); weights.append(weight)
            weighted.append(y*np.repeat(np.sqrt(weight), len(f.offsets)))
        ex = Experiment(selected, rotation, rx, tx, state, mid, training=training)
        if name == 'untapered':
            fit = json.loads((root/'acquisition-validation.json').read_text())['controls']['mixed_truth_start']
        else:
            fit, _ = ex.fit(weighted, truth, 120, position_halfwidth_m=1000.)
        snr = 0.; reference_snr = 0.
        for c, (f, train, q, w) in enumerate(zip(fields, training, ex.q, weights)):
            use = np.repeat(train, len(f.offsets)); wf = np.repeat(w[train], len(f.offsets))
            signal = clean[c][use]*np.sqrt(wf)
            signal -= q@(q.conj().T@signal)
            energy = np.vdot(signal, signal).real
            snr += energy**2/np.sum(wf*abs(signal)**2)
            ref = clean[c][use]-original.q[c]@(original.q[c].conj().T@clean[c][use])
            reference_snr += np.vdot(ref, ref).real
        point = np.asarray(fit['parameters'])
        result['controls'][name] = dict(fit=fit,
            horizontal_error_m=np.linalg.norm((point-truth)[:2]*SCALE[:2]),
            nominal_thermal_snr_penalty_db=-10*np.log10(snr/reference_snr),
            untapered_point_template_loss_db=geometric_score(original, point, clean)['mismatch_loss_db'])
        (root/'window-control.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n')
        print(name, serial(result['controls'][name]), flush=True)


if __name__ == '__main__':
    main()
