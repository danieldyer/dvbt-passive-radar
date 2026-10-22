"""Motion recovery after adding known echoes to unchanged recorded range fields.

The same acquisition and physical variable-projection fitter used for the ship
sees only the summed observations and a declared coarse search region. Truth is
used by the independent Cartesian/quadrature generator and scoring, never seeds.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from recovery_experiment import Experiment, SCALE, load_fields
from report_measured_scatterer_phases import serial


def metrics(actual, expected):
    ea, ee = np.vdot(actual, actual).real, np.vdot(expected, expected).real
    cross = np.vdot(expected, actual)
    return dict(coherence_loss_db=float(-10*np.log10(abs(cross)**2/(ea*ee))),
                phase_bias_deg=float(np.degrees(np.angle(cross))),
                relative_complex_error=float(np.linalg.norm(actual-expected)/np.sqrt(ee)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first', type=float, default=142.)
    p.add_argument('--last', type=float, default=172.)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--contrast-db', type=float, default=0.)
    p.add_argument('--max-evaluations', type=int, default=100)
    p.add_argument('--position-halfwidth-m', type=float, default=500.)
    p.add_argument('--profiles', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--level', choices=['profile', 'linear', 'iq'], default='profile')
    p.add_argument('--gain-db', type=float, default=0., help='Additional injected power for profile-level controls only')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.profiles:
        from recorded_injection_fields import load_fields as replay_fields
        fields, rotation, rx, tx, state, mid = replay_fields(args.profiles, args.stride)
    else:
        fields, rotation, rx, tx, state, mid = load_fields(args.first, args.last, args.stride)
    if args.level != 'profile' and args.gain_db:
        raise ValueError('Replay gain is defined by its injection plan')
    old_mid = mid
    mid = (args.first+args.last)/2
    fields = [f.windowed(args.first, args.last) for f in fields]
    dt = mid-old_mid
    state = [state[0]+dt*state[1]+.5*dt*dt*state[2], state[1]+dt*state[2], state[2]]
    split = args.first+.65*(args.last-args.first)
    training = [f.time < split for f in fields]
    ex = Experiment(fields, rotation, rx, tx, state, mid, training=training)
    # A target near, but separate from, the original ship crop. These values
    # are intentionally not used by acquisition or the physical fitter.
    background = [f.y.ravel().copy() for f in fields]
    if args.plan:
        plan = json.loads(args.plan.read_text())
        elapsed = mid-plan['midpoint_s']
        physical = np.asarray(plan['state_enu'])
        true_state = np.array([physical[0]+elapsed*physical[1]+.5*elapsed**2*physical[2],
                               physical[1]+elapsed*physical[2], physical[2]])
        if not np.allclose(true_state[:, 2], np.asarray(state)[:, 2], atol=1e-8, rtol=0):
            raise ValueError('Injection plan does not have the declared fixed-height motion')
        truth = ((true_state-np.asarray(state))[:, :2]).ravel()/SCALE
        gains = np.array([complex(*z) for z in plan['gains_real_imag']])
    else:
        physical = np.array([-90., -90., .073, -.061, .0017, -.0013]).reshape(3, 2)
        truth = np.r_[physical[0]+dt*physical[1]+.5*dt*dt*physical[2],
                      physical[1]+dt*physical[2], physical[2]]/SCALE
        gains = np.exp(1j*np.array([.1, -.7, 1.2, .4, -.3]))
        unit = [ex.independent_signal(truth, c, [g]) for c, g in enumerate(gains)]
        gains *= [np.sqrt(np.vdot(y, y).real*10**(args.contrast_db/10)/np.vdot(z, z).real)
                  for y, z in zip(background, unit)]
    gains *= 10**(args.gain_db/20)
    clean = [ex.independent_signal(truth, c, [g]) for c, g in enumerate(gains)]
    data = [y+z for y, z in zip(background, clean)] if args.level == 'profile' else background
    forward = [np.linalg.norm(z-ex.templates(truth, c)[:, 0]*g)/np.linalg.norm(z)
               for c, (z, g) in enumerate(zip(clean, gains))]
    if max(forward) > 3e-6:
        raise ValueError(f'Independent physical forward check failed: {forward}')
    result = dict(scope=__doc__, complete=False, level=args.level,
                  interval_s=[args.first, args.last], training_end_s=split,
                  contrast_db=args.contrast_db, additional_gain_db=args.gain_db,
                  midpoint_s=mid, truth_parameters=truth,
                  position_halfwidth_m=args.position_halfwidth_m,
                  independent_forward_relative_error=forward,
                  acquisition_seed=np.zeros(6), gains_real_imag=np.c_[gains.real, gains.imag],
                  source_profiles=str(args.profiles), fits=[])
    def save():
        result['seconds'] = time.monotonic()-started
        (args.output/'analysis.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n')
    save()
    proposals = ex.acquire(data, np.zeros(6), count=3)
    result['acquisition'] = proposals
    save()
    print('Acquired blind proposals', serial(proposals), flush=True)
    for proposal in proposals:
        fit, _ = ex.fit(data, proposal['parameters'], args.max_evaluations,
                       position_halfwidth_m=args.position_halfwidth_m)
        result['fits'].append(fit)
        save()
        print('Fit', fit['training_rss'], fit['gradient'], serial(fit['physical_correction']), flush=True)
    selected = min(result['fits'], key=lambda fit: fit['training_rss'])
    result['selected'] = selected
    output = {}
    channels = []
    all_prediction, all_truth, excluded_prediction, excluded_truth = [], [], [], []
    for c, f in enumerate(fields):
        alpha = np.asarray(selected['coefficients_real_imag'][c])
        alpha = alpha[:, 0]+1j*alpha[:, 1]
        prediction = ex.templates(np.asarray(selected['parameters']), c)@alpha
        holdout = np.repeat(~training[c], len(f.offsets))
        channels.append(dict(channel=c, samples=len(f.time),
                             full=metrics(prediction, clean[c]),
                             excluded=metrics(prediction[holdout], clean[c][holdout])))
        all_prediction.append(prediction);all_truth.append(clean[c])
        excluded_prediction.append(prediction[holdout]);excluded_truth.append(clean[c][holdout])
        output.update({f'{name}_{c}': values for name, values in
                       [('input', background[c]), ('injected', clean[c]),
                        ('data', data[c]), ('prediction', prediction)]})
    result['channels'] = channels
    result['combined'] = dict(full=metrics(np.concatenate(all_prediction), np.concatenate(all_truth)),
                              excluded=metrics(np.concatenate(excluded_prediction), np.concatenate(excluded_truth)))
    result['passed'] = bool(selected['success'] and max(ch['excluded']['coherence_loss_db'] for ch in channels) < .2
                            and result['combined']['excluded']['coherence_loss_db'] < .2)
    result['complete'] = True
    save()
    np.savez(args.output/'observations.npz', **output)
    print(json.dumps(serial({k: result[k] for k in ['passed', 'seconds', 'channels']})), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
