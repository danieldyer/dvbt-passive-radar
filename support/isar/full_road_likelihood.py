"""Full saved-bin road likelihood with explicitly unknown stationary returns.

Whitening is invertible on the selected bins. Profiling the fixed stationary
dictionary is exactly the Gaussian profile likelihood for arbitrary complex
stationary coefficients. Their delays are observed coordinates, not surveyed
calibration anchors. Both fast and original-row moving responses see the same
observations, covariance, static subspace and motion/geometry parameters.
"""
import copy

import numpy as np

from field import C, FS
from integrated_moving_road import IntegratedMovingField
from integrated_stationary_road import RecordedBins
from profiled_road_field import ProfiledRoadField
from road_gate_information import full_field
from shared_target_geometry import RoadTarget
from wide_road_templates import widen


def stationary_support(field, bins):
    seed = np.polynomial.polynomial.polyval(
        (bins.time-field.seed['epoch'])/field.seed.get('scale', 15.), field.seed['path'])
    delays = np.arange(np.floor(seed.min()*FS/C)+field.offsets.min()-5,
        np.ceil(seed.max()*FS/C)+field.offsets.max()+6)*C/FS
    maximum_lag = max(abs(field.offsets.min()-(delays.max()-seed.min())*FS/C),
                      abs(field.offsets.max()-(delays.min()-seed.max())*FS/C))
    return delays, float(maximum_lag)


def original_row_field(field, bins):
    """Keep a fixed static projection; integrate moving paths before it."""
    result = copy.copy(field)
    result.original = IntegratedMovingField(field.original, bins)
    result.time = bins.time
    return result


def build(source, patches, wide_records=None):
    clocks, required = [], []
    for field, path in zip(source.training, patches, strict=True):
        with np.load(path) as raw:
            bins = RecordedBins(field, raw['row_time'], raw['row_information'])
            np.testing.assert_allclose(bins.mean(raw['row_value'][bins.keep]),
                field.data['value'], rtol=1e-10, atol=1e-12)
        clocks.append(bins)
        required.append(stationary_support(field, bins)[1])
    extended = any(lag > getattr(f.pulse, 'support', 32)
                   for lag, f in zip(required, source.training))
    pulse_checks = None
    if extended:
        if wide_records is None:
            raise ValueError('Measured pulse tails are required for this vehicle interval')
        source, pulse_checks = widen(source, wide_records)
    fast, exact, channels = [[], []], [[], []], []
    for c, (train, test, bins) in enumerate(zip(source.training, source.validation, clocks)):
        delays, maximum_lag = stationary_support(train, bins)
        if maximum_lag > getattr(train.pulse, 'support', 32):
            raise ValueError('Static dictionary leaves measured pulse support')
        binned = bins.stationary(train, delays)
        for split, base in enumerate((train, test)):
            projected = ProfiledRoadField(full_field(base), delays, binned)
            fast[split].append(projected)
            exact[split].append(original_row_field(projected, bins))
        channels.append({'channel': c, 'required_pulse_lag_samples': maximum_lag,
            'static_columns': len(delays),
            'static_ranks': [group[c].basis.shape[1] for group in fast],
            'selected_bin_counts': [int(np.sum(f.selected)) for f in (train, test)],
            'original_rows': len(bins.time)})
    models = [RoadTarget(source.problem, a, b, source.initial,
        source.transmitters, source.antennas) for a, b in (fast, exact)]
    return (*models, {'channels': channels, 'extended_pulse': extended,
        'wide_template_checks': pulse_checks})
