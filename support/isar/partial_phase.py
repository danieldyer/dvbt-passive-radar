"""Calibrate declared common reflectors while profiling independent band scattering.

Unshared reflector amplitudes are linear nuisance parameters, eliminated
exactly before the common-phase solve and restored afterwards. This tests a
mixed scattering hypothesis without forcing every grid point to share phase.
"""
import numpy as np

from phase_coupling import PhaseCoupling


def common_visibility(designs, observations):
    """Rank points by the weakest-band conditional coefficient information.

    |a_j|^2 / [(A^H A)^-1]_jj is the fitted energy attributable to j after
    eliminating all other columns. No channel rescaling or test data is used.
    This is a conditional selection statistic, not a calibrated target SNR.
    """
    information = []
    for design, observation in zip(designs, observations):
        u, s, vh = np.linalg.svd(design, full_matrices=False)
        if s[-1] <= s[0]*1e-10:
            raise ValueError('Anchor selection needs a full-rank declared scene')
        amplitude = vh.conj().T@((u.conj().T@observation)/s)
        variance = np.sum(abs(vh.conj().T/s)**2, axis=1)
        information.append(abs(amplitude)**2/variance)
    information = np.array(information)
    return np.argsort(-information.min(axis=0)), information


def fit(designs, observations, anchors, *, reference=3, coordinate_sweeps=0):
    anchors = np.asarray(anchors, int)
    count = designs[0].shape[1]
    if not len(anchors) or len(np.unique(anchors)) != len(anchors) or anchors.min() < 0 or anchors.max() >= count:
        raise ValueError('Need distinct valid common reflector indices')
    other = np.setdiff1d(np.arange(count), anchors)
    projected, clean = [], []
    for design, observation in zip(designs, observations):
        target, data = design[:, anchors], observation
        if len(other):
            q, r = np.linalg.qr(design[:, other], mode='reduced')
            if np.linalg.cond(r) >= 1e10:
                raise ValueError('Independent scattering nuisance is rank deficient')
            target = target-q@(q.conj().T@target)
            data = data-q@(q.conj().T@data)
        projected.append(target); clean.append(data)
    coupling = PhaseCoupling(projected, clean, reference=reference)
    common, details = coupling.fit(coordinate_sweeps=coordinate_sweeps)
    coefficients = np.zeros((len(designs), count), complex)
    for c, (design, observation, amplitude) in enumerate(zip(designs, observations, common)):
        coefficients[c, anchors] = amplitude
        if len(other):
            coefficients[c, other] = np.linalg.lstsq(design[:, other], observation-design[:, anchors]@amplitude, rcond=None)[0]
    details.update(common_reflector_indices=anchors.tolist(), independent_reflector_indices=other.tolist(),
        partial_phase_scope=__doc__)
    return coefficients, details
