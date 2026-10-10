#!/usr/bin/env python3
"""Fixed-CPI template loss against the exact known-waveform white-noise bound.

Uses reconstructed recording symbols, a frozen fitted response and a specified
finite delay/frequency-shift operator. No motion trajectory or noise-covariance
estimate is inferred. Results are controlled template tests, not real-target SNR.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sensitivity import BINS, DT, FS, K, NFFT, blocks, window


def efficiency(coefficients, template, noise_variance=1):
    return float(abs(np.vdot(coefficients, template))**2 /
                 (np.sum(abs(coefficients)**2*noise_variance) * np.sum(abs(template)**2/noise_variance)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--noise',type=Path)
    args=parser.parse_args()
    variance=np.load(args.noise)['heldout_noise'] if args.noise else 1
    b=next(blocks(args.input))
    first=10;rows=int(1/DT);guard=512;length=NFFT+guard;origin=128
    if first+rows>b['last']:
        raise ValueError('Need a full one-second complete reference block')
    if not np.asarray(b['valid'][first-1:first+rows],bool).all():
        raise ValueError('Missing finite-waveform reference context')
    spectra=np.asarray(b['s'],dtype=np.complex128)
    packed=np.zeros((len(spectra),NFFT),complex);packed[:,K%NFFT]=spectra
    useful=np.fft.ifft(packed,axis=1)
    waveform=np.concatenate((useful[:,-guard:],useful),axis=1).reshape(-1)
    h=np.asarray(b['h'],dtype=np.complex128)
    reference=spectra[first:first+rows]*np.exp(-2j*np.pi*K*origin/NFFT)
    times=(np.arange(first,first+rows)*length+guard-origin)/FS
    tapers={name:window(name,rows) for name in ('rect','hann','hamming')}
    results=[]
    for delay in (0,128,384,512,800):
        starts=np.arange(first,first+rows)*length+guard-origin-delay
        samples=waveform[starts[:,None]+np.arange(NFFT)]
        for frequency in (0.,50.,100.,200.):
            phase=np.exp(2j*np.pi*frequency*(times[:,None]+np.arange(NFFT)/FS))
            exact=np.fft.fft(samples*phase,axis=1)[:,K%NFFT]*h
            diagonal=reference*np.exp(-2j*np.pi*K*delay/NFFT)*np.exp(2j*np.pi*frequency*times[:,None])
            # The current cross-spectrum uses X, whereas H X is the shaped
            # template. The exact, untapered template attains the upper bound.
            loss={name:-10*np.log10(efficiency(w[:,None]*diagonal,exact,variance)) for name,w in tapers.items()}
            shaped_loss=-10*np.log10(efficiency(diagonal*h/variance,exact,variance))
            optimum=efficiency(exact/variance,exact,variance)
            exact_hann_loss=-10*np.log10(efficiency(tapers['hann'][:,None]*exact/variance,exact,variance))
            if abs(optimum-1)>1e-12:raise ValueError('Optimum normalization failed')
            results.append(dict(delay_samples=delay,doppler_hz=frequency,
                current_template_loss_db=loss,shaped_diagonal_rect_loss_db=float(shaped_loss),
                exact_hann_loss_db=float(exact_hann_loss),
                gain_from_exact_template_with_hann_db=float(loss['hann']-exact_hann_loss),
                exact_rect_efficiency=optimum))
    output=dict(cpi_s=rows*DT,rows=rows,reference_origin_samples=origin,
        operator='finite integer delay plus frequency shift; no baseband range migration',
        dc_removal=False,
        noise='held-out diagonal physical carrier noise; white slow time' if args.noise else 'white independent equal-variance measured carriers',
        noise_source=str(args.noise) if args.noise else None,response='frozen first-batch fitted H',
        cases=results)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
