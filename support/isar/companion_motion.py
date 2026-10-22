"""Explicit delayed-return nuisance for estimating the primary target motion.

Companion path hypotheses are fixed from training data in a separate range
crop. One unknown complex amplitude per declared block is eliminated jointly
with stationary clutter. This changes only the autofocus likelihood; the raw
profiles and backprojection input retain every return. No Doppler notch or
data-dependent image normalization is applied.
"""
import numpy as np

from delayed_path import DelayedPath
from stationary import Stationary


def add_companion(band, parameters, records):
    component = DelayedPath(band, parameters, include_main=False)
    coverage = np.zeros(len(band.field.time), int)
    columns = []
    for record in records:
        first, last = record['interval_s']
        selected = (band.field.time >= first)&(band.field.time < last)
        coverage += selected
        component.epoch = record['phase_epoch_s']
        value = component.response(record['parameters'])
        columns.append(value*np.repeat(selected, band.field.y.shape[1]))
    if not np.all(coverage == 1):
        raise ValueError('Companion hypotheses must partition every retained time sample exactly once')
    extra = np.column_stack(columns)
    target = band.design(parameters)[0]
    before = band.nuisance.remove(target[band.selected])
    design = np.column_stack([band.nuisance.design, extra])
    nuisance = Stationary(design, band.selected)
    after = nuisance.remove(target[band.selected])
    retained = np.sum(abs(after)**2, axis=0)/np.sum(abs(before)**2, axis=0)
    basis, singular, _ = np.linalg.svd(before, full_matrices=False)
    if singular[-1] <= singular[0]*1e-10:
        raise ValueError('The declared primary scene is numerically rank deficient')
    projected = nuisance.remove(basis)
    mode_retained = np.linalg.eigvalsh(projected.conj().T@projected)
    result = {'blocks': len(records), 'old_rank': band.nuisance.rank, 'new_rank': nuisance.rank,
        'point_information_retained': retained.tolist(),
        'point_information_loss_db': (-10*np.log10(retained)).tolist(),
        'scene_mode_information_retained': mode_retained.tolist(),
        'scene_mode_information_loss_db': (-10*np.log10(mode_retained)).tolist(),
        'scope': 'Known-template energy after profiling unknown companion amplitudes, relative to stationary nuisance only. This is a conditional information cost, not a measured image-SNR loss.'}
    band.nuisance = nuisance
    band.clean = nuisance.remove(band.field.y.ravel()[band.selected])
    band.energy = float(np.vdot(band.clean, band.clean).real)
    return result
