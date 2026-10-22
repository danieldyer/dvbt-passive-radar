"""Estimate bin noise scale from orthogonal range/time innovations.

Within each short bin the mean has an arbitrary moving range vector and an
arbitrary stationary range vector. Their known temporal factors are constant
and the reversible seed deramp. Project both after whitening by the declared
row information and range covariance. For independent proper-complex Gaussian
noise, residual squared norm divided by its complex degrees of freedom is an
unbiased scalar variance estimate, independent of the fitted mean subspace.

That result is conditional: finite within-symbol covariance, variation of the
range vectors within a bin, moving interference and unknown range covariance
still need validation. No data or phase values are modified.
"""
import numpy as np

from integrated_stationary_road import RecordedBins
from range_fit import Field
from shared_target_geometry import RoadTarget


def residual_scale(observations, variance, temporal):
    observations, variance, temporal = map(np.asarray, (observations, variance, temporal))
    if (observations.ndim != 2 or temporal.ndim != 2 or variance.shape != (len(observations),)
            or len(temporal) != len(observations) or np.any(variance <= 0)
            or not np.isfinite(observations).all() or not np.isfinite(variance).all()):
        raise ValueError('Require finite observations and positive known relative row variances')
    root = 1/np.sqrt(variance)
    u, singular, _ = np.linalg.svd(root[:, None]*temporal, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    degrees = (len(observations)-rank)*observations.shape[1]
    if degrees <= 2:
        raise ValueError('Insufficient independent innovations to estimate a noise scale')
    white = root[:, None]*observations
    residual = white-u[:, :rank]@(u[:, :rank].conj().T@white)
    scale = float(np.vdot(residual, residual).real/degrees)
    if not scale > 0:
        raise ValueError('A zero innovation estimate cannot define a likelihood covariance')
    return scale, degrees, rank


def estimate(field, raw):
    bins = RecordedBins(field, raw['row_time'], raw['row_information'])
    weights = np.asarray(raw['row_information'])[bins.keep]
    values = np.asarray(raw['row_value'])[bins.keep]@field.whiten.T
    relative = (bins.time-field.seed['epoch'])/field.seed.get('scale', 15.)
    path = np.polynomial.polynomial.polyval(relative, field.seed['path'])
    speed = np.polynomial.polynomial.polyval(relative,
        np.polynomial.polynomial.polyder(field.seed['path']))/field.seed.get('scale', 15.)
    variance = (1+(field.k*speed*.001024)**2/12)/(8192*weights)
    temporal = np.column_stack((np.ones(len(values)),
        np.exp(1j*field.k*(path-field.seed['path'][0]))))
    if np.any(np.diff(bins.index) < 0):
        raise ValueError('Recorded row groups must retain time order')
    boundaries = np.r_[0, 1+np.flatnonzero(np.diff(bins.index)), len(values)]
    scales = np.zeros(bins.count); degrees = np.zeros(bins.count, int); ranks = np.zeros(bins.count, int)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        index = bins.index[left]
        scales[index], degrees[index], ranks[index] = residual_scale(
            values[left:right], variance[left:right], temporal[left:right])
    total = np.bincount(bins.index, weights=weights, minlength=bins.count)
    mean_variance = np.bincount(bins.index, weights=weights*weights*variance,
        minlength=bins.count)/total**2
    if np.any(scales <= 0):
        raise ValueError('A saved bin lacks a finite independent noise estimate')
    return scales*mean_variance, {'relative_scale_percentiles': np.percentile(scales, [1, 50, 99]).tolist(),
        'complex_degrees_minmax': [int(degrees.min()), int(degrees.max())],
        'temporal_rank_minmax': [int(ranks.min()), int(ranks.max())],
        'conditional_large_bin_count_weight_loss_db_max': float(-10*np.log10(
            (degrees.min()-2)/(degrees.min()-1))),
        'variance': (scales*mean_variance).tolist()}


def replace(source, patches):
    training, validation, checks = [], [], []
    for train, test, path in zip(source.training, source.validation, patches, strict=True):
        with np.load(path) as raw:
            variance, check = estimate(train, raw)
        data = {**train.data, 'variance': np.broadcast_to(variance[:, None], train.data['value'].shape).copy()}
        training.append(Field(data, train.pulse, train.seed, selected=train.selected))
        validation.append(Field(data, test.pulse, test.seed, selected=test.selected))
        checks.append(check)
    return RoadTarget(source.problem, training, validation, source.initial,
        source.transmitters, source.antennas), checks
