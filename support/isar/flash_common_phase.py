"""Optimistic control: one reflection phase shared by all five flash bands.

This actually uses the constant interband phases, not just phase derivatives.
It assumes relative instrumental phase and reflector phase are calibrated.
Carrier integers remain unknown. Nearest-cycle branches at each baseline
geometry are enumerated over the full common-phase circle; this is not an
exhaustive search of every possible position/carrier ambiguity in the priors.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from compare_ais_phase import NAMES
from flash_geometry import Flash
from motion import C


def phase_observations(path, k, *, columns=(1,), width=5, epoch=164.5):
    """Phase and its cross-covariance with the retained physical observables."""
    phase, variance, cross = [], [], []
    for i, row in enumerate(json.loads(Path(path).read_text())):
        row = next(r for r in [row, *row.get('range_width_controls', [])] if len(r['offset_samples'])==width)
        u = (epoch-row['epoch_s'])/row['scale_s']
        b = np.array([0., 1., -k[i]*u, -k[i]*u*u])
        scale = row['scale_s']
        transform = np.array([[C/8e6,0.,u,u*u], [0.,0.,1/scale,2*u/scale], [0.,0.,0.,2/scale**2]])
        cov = np.array(row['conditional_parameter_covariance'])
        phase.append(b@row['parameters']);variance.append(b@cov@b);cross.append(b@cov@transform[list(columns)].T)
    return np.array(phase), np.array(variance), np.array(cross)


class CommonPhase:
    def __init__(self, base, observed_phase, phase_variance, phase_rate_covariance, nominal_path, k, cycles):
        self.base, self.phase, self.nominal, self.k, self.cycles = base, observed_phase, nominal_path, k, cycles
        cross = np.asarray(phase_rate_covariance).reshape(len(k),len(base.columns))
        self.regression = np.array([c@w.T@w for c,w in zip(cross,base.whiten)])
        conditional = phase_variance-np.sum(self.regression*cross,axis=1)
        if np.any(conditional <= 0):
            raise ValueError('Nonpositive conditional phase covariance')
        self.sigma = np.sqrt(conditional)

    def conditional_shift(self, predicted):
        return np.sum(self.regression*(predicted-self.base.measured)[:,self.base.columns],axis=1)

    def residual(self, p):
        predicted = self.base.prediction(p[:-1])
        phase = p[-1]-self.k*(predicted[:, 0]-self.nominal)
        error = (phase-self.phase-2*np.pi*self.cycles-self.conditional_shift(predicted))/self.sigma
        return np.r_[self.base.residual(p[:-1]), error]

    def fit(self, p):
        def phase(parameters):
            predicted = self.base.prediction(parameters)
            value = (self.phase+2*np.pi*self.cycles+self.k*(predicted[:, 0]-self.nominal)+
                     self.conditional_shift(predicted))
            return np.sum(value/self.sigma**2)/np.sum(1/self.sigma**2)
        def residual(parameters):
            return self.residual(np.r_[parameters, phase(parameters)])
        # The common phase is a linear nuisance for a fixed integer branch.
        # Profile it exactly instead of asking a stiff nonlinear solver to
        # alternate it with metre-scale geometry. A uniform integer shift is
        # merely the same reflection phase written on another 2pi branch.
        fit = least_squares(residual, p[:-1], jac='cs', bounds=self.base.bounds(), x_scale='jac',
                            max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-9)
        fit.x = np.r_[fit.x, phase(fit.x)]
        return fit


def run(result, options, track, utc, rotation, receiver, phase_report, output):
    case = next(c for c in result['cases'] if c['width_cells']==5 and not c['include_acceleration'])
    k = 2*np.pi*np.array([177.5, 184.5, 191.625, 219.5, 226.5])*1e6/C
    phase, variance, cross = phase_observations(phase_report, k)
    # Measured phases were deramped with the saved geometry. A changed receiver
    # prior changes the physical prediction, never this data reference.
    nominal = np.array(case['data_reference_ais_prediction'])[:, 0]
    records = []
    for record in case['records']:
        base = Flash(case['measurements'], case['covariance'], track, utc+164.5, rotation, receiver,
                     options[record['index']], include_range=False, include_acceleration=False,
                     **result.get('fit_options', {}))
        seed = np.array(record['parameters'])
        path = base.prediction(seed)[:, 0]
        # Each break changes the nearest carrier integer in one band. Sampling
        # every interval enumerates all nearest-cycle vectors as common phase
        # traverses [-pi, pi] at this fixed baseline geometry.
        breaks = np.angle(np.exp(1j*(k*(path-nominal)+phase+np.pi)))
        edges = np.unique(np.r_[-np.pi, np.sort(breaks), np.pi])
        origins = (edges[1:]+edges[:-1])/2
        trials = []
        for origin in origins:
            cycles = np.rint((origin-k*(path-nominal)-phase)/(2*np.pi))
            problem = CommonPhase(base, phase, variance, cross, nominal, k, cycles)
            fit = problem.fit(np.r_[seed, origin])
            trials.append((float(fit.fun@fit.fun), fit, cycles))
        score, fit, cycles = min(trials, key=lambda x:x[0])
        predicted = base.prediction(fit.x[:-1])
        observable_error = predicted-base.measured
        aligned = np.angle(np.exp(1j*(phase+k*(predicted[:, 0]-nominal))))
        errors = np.angle(np.exp(1j*(aligned-fit.x[-1])))
        records.append({'index': record['index'], 'sites': record['sites'],
                        'conditional_squared_cost': score, 'independent_phase_squared_cost': record['penalized_squared_residual'],
                        'phase_rms_deg': float(np.degrees(np.sqrt(np.mean(errors**2)))),
                        'unfitted_envelope_error_m': observable_error[:, 0].tolist(),
                        'fitted_path_rate_error_mps': observable_error[:, 1].tolist(),
                        'unfitted_path_acceleration_error_mps2': observable_error[:, 2].tolist(),
                        'geometry_corrected_phase_rad': aligned.tolist(), 'common_reflection_phase_rad': float(fit.x[-1]),
                        'parameters': fit.x.tolist(), 'carrier_cycles': cycles.astype(int).tolist(),
                        'antenna_fit': base.antennas.describe(fit.x[:18]),
                        'origin_branches_tested': len(trials), 'first_order_optimality': float(fit.optimality),
                        'estimated_remaining_gauss_newton_cost_decrease': base.remaining_cost_decrease(fit),
                        'converged':bool(fit.success), 'evaluations':fit.nfev,
                        'bound_parameter_indices':np.flatnonzero((fit.x[:-1]<base.lower+1e-4)|(fit.x[:-1]>base.upper-1e-4)).tolist(),
                        'trial_squared_costs': [s for s, _, _ in trials]})
        if record['index']%16 == 15:
            print('common-phase flash control', record['index']+1, '/', len(options), flush=True)
    ordered = sorted(records, key=lambda r:r['conditional_squared_cost'])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    minimum = ordered[0]['conditional_squared_cost']
    axes[0].plot(np.arange(len(ordered)), [r['conditional_squared_cost']-minimum for r in ordered], '.-', label='Common reflection phase fitted')
    axes[0].plot(np.arange(len(ordered)), [r['independent_phase_squared_cost']-minimum for r in ordered], '.', label='Independent band phase origins')
    axes[0].set_xlabel('Assignment rank under common-phase control');axes[0].set_ylabel('Conditional squared cost relative to best')
    axes[0].legend(fontsize=8)
    for row, colour in zip(ordered[:2], ['#0072b2', '#d55e00']):
        aligned = np.array(row['geometry_corrected_phase_rad'])
        error = np.angle(np.exp(1j*(aligned-row['common_reflection_phase_rad'])))
        label = '/'.join({'12013':'O','12015':'E','12014':'R'}[s] for s in row['sites'])
        axes[1].plot(np.arange(5), np.degrees(error), '.-', color=colour, label=label)
        offset = np.array(row['antenna_fit']['transmitter_offsets_enu_m'])
        axes[2].plot(np.arange(5), np.linalg.norm(offset, axis=1), '.-', color=colour, label=label)
    for ax in axes[1:]:
        ax.set_xticks(np.arange(5), NAMES, rotation=25);ax.legend(fontsize=8)
    axes[1].set_ylabel('Fitted midpoint phase minus common phase (degrees)')
    axes[1].set_title('Phase agreement at 164.5 s; a fitted quantity')
    axes[2].set_ylabel('TX correction from nominal antenna prior (m)')
    axes[2].set_title('Location corrections used by those fits')
    fig.suptitle('Midpoint phase + rate control · absolute range and phase curvature excluded\n'
                 'Common reflection phase and calibrated instrumental phase assumed; this is not whole-flash or withheld phase accuracy')
    fig.savefig(output/'10-common-phase-control.png', dpi=170);plt.close(fig)
    data = {'scope': __doc__, 'records': records,
            'phase_rms_definition': 'Unweighted RMS across the five fitted carrier-phase residuals at 164.5 s, after fitting geometry and one common phase. Not temporal RMS, withheld accuracy, or absolute range agreement.',
            'excluded_observables': ['absolute envelope range', 'phase curvature'],
            'best_index': ordered[0]['index'], 'runner_up_cost_gap': ordered[1]['conditional_squared_cost']-minimum,
            'accepted_calibration': False,
            'limitation': 'Nearest-cycle branches around each rate-only solution; not an exhaustive carrier-integer search. Equal reflection phase is an explicit optimistic hypothesis, not implied by temporal coherence alone.'}
    (output/'common-phase-control.json').write_text(json.dumps(data, indent=2))
    return data
