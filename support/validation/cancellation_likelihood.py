#!/usr/bin/env python3
"""Conditional Gaussian cancellation bounds; no fitted map normalisation.

This tests the likelihood algebra, not an end-to-end radar sensitivity claim.
The covariance and nuisance model are declared independently of the primary data.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def precision(diagonal, modes, covariance):
    """Woodbury precision for R = diag(D) + U C U^H, without inverting R."""
    weighted = modes / diagonal[:, None]
    middle = np.linalg.inv(covariance) + modes.conj().T @ weighted
    def apply(value):
        base = value / diagonal[:, None] if value.ndim == 2 else value / diagonal
        return base - weighted @ np.linalg.solve(middle, modes.conj().T @ base)
    return apply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('figure', type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(74601)
    n, trials = 256, 16000
    time = (np.arange(n)-(n-1)/2)*3/n
    signal = np.exp(2j*np.pi*12.4*time)/np.sqrt(n)
    nuisance = np.exp(2j*np.pi*12*time)[:, None]/np.sqrt(n)
    cases = []
    for power in [1, 10, 100, 1000]:
        diagonal = np.ones(n)
        covariance = np.array([[float(power)]])
        R = np.diag(diagonal) + nuisance @ covariance @ nuisance.conj().T
        apply = precision(diagonal, nuisance, covariance)
        optimal = apply(signal)
        dense = np.linalg.solve(R, signal)
        assert np.max(abs(optimal-dense)) < 1e-11
        information = np.vdot(signal, optimal).real
        efficiency = abs(np.vdot(signal, signal))**2 / (np.vdot(signal, R@signal).real * information)
        noise = (rng.normal(size=(n, trials))+1j*rng.normal(size=(n, trials)))/np.sqrt(2)
        coefficient = np.sqrt(power/2)*(rng.normal(size=trials)+1j*rng.normal(size=trials))
        noise += nuisance * coefficient
        measured = np.mean(abs(optimal.conj() @ noise)**2)
        assert abs(measured/information-1) < .04
        cases.append(dict(coherent_mode_power=power, diagonal_snr_loss_db=float(-10*np.log10(efficiency)),
                          optimal_score_variance_over_information=float(measured/information)))
    # Frozen constant coefficient: its error is shared across all primary rows.
    # Equal training/primary energy gives unit excess eigenvalue, NOT 1/N added
    # independently to every sample. The latter gives the wrong DC uncertainty.
    fd = np.linspace(.001, 5, 5000)
    overlap = abs(np.exp(2j*np.pi*time[:, None]*fd).mean(axis=0))**2
    finite_loss = -10*np.log10(1-.5*overlap)
    unknown_loss = -10*np.log10(1-overlap)
    outside = fd >= 4/3
    clocks = []
    for rate in [40e-9, .25e-6]:
        phase = np.pi*3408*rate
        clocks.append(dict(residual_ppm=rate*1e6, maximum_within_symbol_phase_degrees=float(phase*180/np.pi),
                           worst_case_waveform_error_power_bound_db=float(20*np.log10(phase)),
                           uniform_band_and_time_phase_rms_degrees=float(phase/3*180/np.pi)))
    output = dict(model='Known diagonal plus declared low-rank proper complex Gaussian covariance; 3 seconds',
                  qualification='Conditional likelihood and Monte Carlo tests, not a certificate for arbitrary RF interference or fitted H',
                  trials=trials, correlated_interference=cases,
                  frozen_coefficient=dict(training_to_primary_energy_ratio=1,
                                          max_loss_outside_4_over_T_db=float(finite_loss[outside].max()),
                                          unknown_nuisance_max_loss_outside_4_over_T_db=float(unknown_loss[outside].max())),
                  source_clock_approximation=clocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fd, finite_loss, label='Frozen fit, equal training and primary energy')
    axes[0].plot(fd, unknown_loss, label='Completely unknown constant nuisance')
    axes[0].axvline(4/3, color='gray', linestyle=':', label='4 / CPI')
    axes[0].set(xlabel='Doppler (Hz)', ylabel='Information loss vs known coefficient (dB)', ylim=(0, 3))
    axes[0].legend(fontsize=8)
    axes[1].semilogx([c['coherent_mode_power'] for c in cases], [c['diagonal_snr_loss_db'] for c in cases], marker='o')
    axes[1].set(xlabel='Correlated-mode excess eigenvalue', ylabel='Diagonal-weight SNR gap (dB)',
                title='Target 12.4 Hz; independent random interferer 12 Hz')
    fig.suptitle('Declared covariance models: uncertainty is not independent white noise')
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)


if __name__ == '__main__':
    main()
