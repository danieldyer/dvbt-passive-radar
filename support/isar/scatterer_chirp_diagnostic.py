"""Original-epoch residual chirp likelihood for deblended point histories.

The FFT is an acquisition initializer only. Final scores and derivatives use
the actual epochs. Coefficients are frozen before evaluating excluded data.
No estimated chirp is applied to the capture or adopted as a motion solution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import fftfreq, ifft, next_fast_len
from scipy.optimize import minimize


def chirp_score(time, value, information, k, parameters):
    t = np.asarray(time)
    basis = np.c_[t, .5*t*t]
    phase = np.exp(1j*k*(basis@parameters))
    cross = np.sum(information*value*phase)
    derivative = np.sum((information*value*phase)[:, None]*(1j*k*basis), axis=0)
    norm = information.sum()
    return abs(cross)**2/norm, 2*np.real(cross.conjugate()*derivative)/norm, cross/norm


def acquire(time, value, information, k, *, maximum_rate=5., maximum_acceleration=.2):
    t = np.asarray(time); duration = np.ptp(t); wavelength = 2*np.pi/k
    step = float(np.median(np.diff(t)))
    index = np.rint((t-t[0])/step).astype(int)
    size = next_fast_len(8*(index.max()+1))
    frequency = fftfreq(size, step); rates = wavelength*frequency
    keep = abs(rates) <= maximum_rate
    acceleration_step = wavelength/(2*duration**2)
    accelerations = np.linspace(-maximum_acceleration, maximum_acceleration,
                               int(np.ceil(2*maximum_acceleration/acceleration_step))+1)
    candidates = []
    for acceleration in accelerations:
        grid = np.zeros(size, complex)
        np.add.at(grid, index, information*value*np.exp(.5j*k*acceleration*t*t))
        power = abs(ifft(grid)*size)**2/information.sum()
        maxima = np.flatnonzero(keep & (power >= np.roll(power, 1)) & (power >= np.roll(power, -1)))
        for at in maxima[np.argsort(power[maxima])[-3:]]:
            candidates.append((float(power[at]), np.array([rates[at], acceleration])))
    scale = np.array([wavelength/duration, wavelength/duration**2])
    energy = np.sum(information*abs(value)**2)
    def objective(q):
        score, gradient, _ = chirp_score(t, value, information, k, q*scale)
        return -score/energy, -gradient*scale/energy
    selected = []; results = []
    for _, seed in sorted(candidates, key=lambda x: -x[0]):
        if any(np.linalg.norm((seed-other)/scale) < 1 for other in selected):
            continue
        selected.append(seed)
        fit = minimize(objective, seed/scale, jac=True, method='L-BFGS-B',
                       bounds=[(-maximum_rate/scale[0], maximum_rate/scale[0]),
                               (-maximum_acceleration/scale[1], maximum_acceleration/scale[1])],
                       options=dict(gtol=1e-10, ftol=1e-13, maxiter=100))
        parameters = fit.x*scale
        score, gradient, alpha = chirp_score(t, value, information, k, parameters)
        results.append(dict(parameters=parameters, score=score, alpha=alpha,
                            coherence=score/energy, maximum_gradient=float(max(abs(gradient*scale))/energy),
                            numerical_convergence=bool(fit.success)))
        if len(selected) == 6:
            break
    return max(results, key=lambda r: r['score'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--histories', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--windows', type=int, nargs='*')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    d = np.load(args.histories); names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    rf = np.array([177.5, 184.5, 191.625, 219.5, 226.5])*1e6
    all_windows = np.unique(d['window_0_1']) if args.windows is None else args.windows
    reports = []
    from matplotlib import pyplot as plt
    for window in all_windows:
        fig, axes = plt.subplots(3, 2, figsize=(15, 11), layout='constrained')
        for c in range(5):
            train = d[f'window_{c}_1'] == window; test = d[f'window_{c}_0'] == window
            epoch = (d[f'time_s_{c}_1'][train][0]+d[f'time_s_{c}_1'][train][-1])/2
            for point in range(3):
                t = d[f'time_s_{c}_1'][train]-epoch
                y = d[f'value_{c}_1'][train, point]
                weight = 1/d[f'covariance_{c}_1'][train, point, point].real
                k = 2*np.pi*rf[c]/299792458.
                result = acquire(t, y, weight, k)
                rate, acceleration = result['parameters']
                tt = d[f'time_s_{c}_0'][test]-epoch; yy = d[f'value_{c}_0'][test, point]
                ww = 1/d[f'covariance_{c}_0'][test, point, point].real
                score, _, alpha = chirp_score(tt, yy, ww, k, result['parameters'])
                phase_bias = float(np.degrees(np.angle(alpha/result['alpha'])))
                coherence = float(score/np.sum(ww*abs(yy)**2))
                report = dict(window=int(window), channel=c, point=point, epoch_s=epoch,
                    residual_rate_mps=float(rate), residual_acceleration_mps2=float(acceleration),
                    training_coherence=float(result['coherence']), excluded_coherence=coherence,
                    excluded_phase_bias_deg=phase_bias, score_db=float(10*np.log10(score)),
                    maximum_gradient=result['maximum_gradient'], numerical_convergence=result['numerical_convergence'])
                reports.append(report); print(json.dumps(report), flush=True)
                corrected = yy*np.exp(1j*k*(rate*tt+.5*acceleration*tt**2))/result['alpha']
                # Display weighted 0.5-second means only after the measured
                # chirp; the final likelihood above used every original epoch.
                cuts = np.arange(tt.min(), tt.max()+.5, .5); x=[]; z=[]; signal=[]
                for left, right in zip(cuts[:-1], cuts[1:]):
                    keep = (tt>=left)&(tt<right)
                    if not keep.any(): continue
                    x.append(np.average(tt[keep], weights=ww[keep])+epoch)
                    v = np.average(corrected[keep], weights=ww[keep]); z.append(np.degrees(np.angle(v)))
                    signal.append(abs(v)**2*np.sum(ww[keep])*abs(result['alpha'])**2)
                axes[point, 0].scatter(x, z, s=np.clip(signal, 3, 30), color=f'C{c}', alpha=.7, label=names[c])
                axes[point, 1].scatter(rate, acceleration, color=f'C{c}', s=30+10*point)
                axes[point, 1].annotate(names[c], (rate, acceleration), fontsize=8)
        for point in range(3):
            axes[point, 0].set(ylabel=f'Point {point+1}: chirp residual phase (deg)', ylim=[-180, 180])
            axes[point, 1].set(xlabel='Measured residual rate (m/s)', ylabel='Measured residual acceleration (m/s²)')
        axes[0, 0].legend(ncol=5, fontsize=8); axes[-1, 0].set_xlabel('Capture time (s)')
        fig.suptitle(f'Independent-band quadratic phase diagnostic · window {window}\n'
                     'Training-only chirp; excluded-epoch phase shown. No correction adopted.')
        fig.savefig(args.output/f'window-{window:02d}.png', dpi=140); plt.close(fig)
        (args.output/'analysis.json').write_text(json.dumps(dict(scope=__doc__, complete=False, results=reports), indent=2)+'\n')
    (args.output/'analysis.json').write_text(json.dumps(dict(scope=__doc__, complete=True, results=reports), indent=2)+'\n')


if __name__ == '__main__':
    main()
