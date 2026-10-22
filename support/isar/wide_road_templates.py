"""Wide measured PSFs for vehicle intervals with substantial range migration."""
import numpy as np
from scipy.interpolate import CubicSpline

from fit_shared_target_geometry import load_experiment  # also resolves calibration imports
from integrated_stationary_road import RecordedBins
from phase_history import C, FS
from range_fit import Field
from shared_target_geometry import RoadTarget
from stationary_road_control import StationaryRoad


class WidePulse:
    """Same 64-tap interpolation, using measured rather than extrapolated tails."""
    def __init__(self, samples):
        samples = np.asarray(samples)
        if len(samples) < 129 or len(samples) % 2 != 1:
            raise ValueError('A centred odd autocorrelation with at least 129 samples is required')
        radius = len(samples)//2
        self.support = radius-32
        self.samples = samples[radius-64:radius+65]
        grid = np.arange(-self.support, self.support+1/64, 1/32)
        indices = np.floor(grid).astype(int)[:, None]+np.arange(-32, 33)
        delta = grid[:, None]-indices
        weights = np.sinc(delta)*np.i0(8*np.sqrt(np.maximum(0, 1-(delta/32)**2)))/np.i0(8)
        weights *= abs(delta) <= 32
        weights /= weights.sum(axis=1)[:, None]
        self.curve = CubicSpline(grid, np.sum(weights*samples[indices+radius], axis=1), extrapolate=False)

    def __call__(self, lag, derivative=0):
        if np.any(abs(lag) > self.support):
            raise ValueError('Candidate exceeds measured wide-PSF interpolation support')
        return self.curve(lag, derivative)


def widen(source, records):
    """Replace only the template/corresponding range covariance; keep observations."""
    training, validation, diagnostics = [], [], []
    for c, (train, test) in enumerate(zip(source.training, source.validation)):
        start = train.seed['epoch']-train.seed['duration_s']/2
        end = train.seed['epoch']+train.seed['duration_s']/2
        keep = (records['channel'] == c)&(records['time'] >= start)&(records['time'] <= end)&(records['information'] > 0)
        if sum(keep) < 100 or records[keep]['time'][-1] < end-.1:
            raise ValueError('Wide-PSF replay does not cover the whole target interval')
        samples = np.average(records[keep]['psf'], axis=0, weights=records[keep]['information'])
        pulse = WidePulse(samples)
        discrepancy = float(np.linalg.norm(pulse.samples-train.pulse.samples)/np.linalg.norm(train.pulse.samples))
        if discrepancy > 1e-5:
            raise ValueError('Replayed central PSF differs materially from the original template')
        training.append(Field(train.data, pulse, train.seed, selected=train.selected))
        validation.append(Field(test.data, pulse, test.seed, selected=test.selected))
        diagnostics.append({'channel': c, 'central_psf_relative_change': discrepancy,
            'interpolation_lag_support_samples': pulse.support, 'replay_blocks': int(sum(keep))})
    return RoadTarget(source.problem, training, validation, source.initial, source.transmitters, source.antennas), diagnostics


def stationary(source, raw_patches, *, margin=5, cutoff=1e-10):
    """Construct physically constant-delay templates over the full migrated crop."""
    delays = []
    for field, origin in zip(source.training, source.origins):
        samples = field.seed_path*FS/C
        values = np.arange(np.floor(samples.min())+field.offsets.min()-margin,
            np.ceil(samples.max())+field.offsets.max()+margin+1)*C/FS-origin
        lag_min = field.offsets.min()-(values.max()+origin-field.seed_path.min())*FS/C
        lag_max = field.offsets.max()-(values.min()+origin-field.seed_path.max())*FS/C
        if max(abs(lag_min), abs(lag_max)) > field.pulse.support:
            raise ValueError('Export a wider measured PSF before constructing this stationary dictionary')
        delays.append(values)
    previous = StationaryRoad(source, delays=delays, cutoff=cutoff)
    binned = []
    for c, field in enumerate(previous.training):
        with np.load(raw_patches[c]) as raw:
            bins = RecordedBins(field.original, raw['row_time'], raw['row_information'])
            np.testing.assert_allclose(bins.mean(raw['row_value'][bins.keep]), field.original.data['value'],
                rtol=1e-10, atol=1e-12)
            binned.append(bins.stationary(field.original, field.delays, previous.origins[c]))
    return previous, binned
