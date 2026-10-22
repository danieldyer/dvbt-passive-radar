"""Compare finite-profile and IQ injections using one frozen baseline metric.

The paired background is removed only for this operator diagnostic. No phase,
gain or time correction is fitted. Motion estimation uses the full backgrounds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recorded_injection_fields import load_fields,read_profiles,common_projection
from validate_recorded_injection import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, default=Path('build/injection-20261018'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--iq', type=Path)
    parser.add_argument('--allow-window-step-samples',type=int,default=0,
        help='Permit a measured integer FFT-window change within the same DVB symbol; report every changed epoch')
    parser.add_argument('--output', type=Path,
                        default=Path('docs/data/injection_20261018/exact-operator-comparison.json'))
    args = parser.parse_args()
    paths = [args.baseline or args.replay/'deterministic-baseline/profiles.bin',
             args.profile or args.replay/'exact-profile/profiles.bin',
             args.iq or args.replay/'deterministic-iq/profiles.bin']
    if args.allow_window_step_samples<0:raise ValueError('Window-step allowance cannot be negative')
    fields=load_fields(paths[0])[0]
    raw=[read_profiles(path) for path in paths]
    result = []
    for c,baseline in enumerate(fields):
        channels=[value[1][c] for value in raw]
        times=[ch['rows'][:,1] for ch in channels]
        if any(t.shape!=times[0].shape for t in times):raise ValueError('Native symbol counts changed')
        difference=max(np.max(abs(t-times[0])) for t in times)
        if difference>args.allow_window_step_samples/8e6+3e-14:
            raise ValueError('Native observation epochs changed beyond the declared FFT-window allowance')
        steps=[np.max(abs(ch['rows'][:,0]-channels[0]['rows'][:,0])) for ch in channels]
        if max(steps)>args.allow_window_step_samples:raise ValueError('Different native symbol identities')
        if any(not np.array_equal(ch['rows'][:,4],channels[0]['rows'][:,4]) for ch in channels):
            raise ValueError('Native FEC-valid identities changed')
        # Apply the same fixed range covariance and information at both levels.
        projected=[common_projection(baseline,ch,head['first_range_sample']) for ch,(head,_) in zip(channels,raw)]
        np.testing.assert_allclose(projected[0],baseline.y.ravel(),rtol=1e-12,atol=1e-10)
        expected=projected[1]-projected[0]
        actual=projected[2]-projected[0]
        gain = abs(np.vdot(expected, actual))/np.vdot(expected, expected).real
        result.append(dict(channel=c, **metrics(actual, expected),
                           amplitude_ratio=float(gain),native_epoch_difference_s=float(difference),
                           maximum_fft_window_step_samples=float(max(steps)),
                           changed_iq_native_epochs=int(np.count_nonzero(times[2]!=times[0])),
                           projection='One fixed baseline range/phase/binning map; no fitted alignment; actual native epochs retained separately'))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))
    if any(r['coherence_loss_db'] >= .05 or abs(20*np.log10(r['amplitude_ratio'])) >= .1 for r in result):
        raise SystemExit('Paired injection comparison failed')


if __name__ == '__main__':
    main()
