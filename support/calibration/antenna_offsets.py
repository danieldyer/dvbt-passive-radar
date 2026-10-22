"""Five electrical phase centres with correlated licensed-tower position priors.

The shared-mast model is a restricted initial hypothesis. Here each channel
has its own position correction. A common site survey error and stated
within-mast offset uncertainty give their prior covariance; no phase pattern
or per-target phase correction is added. All trajectory parameters still fit.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_triangular

from calibration_solver import solve
from coherent_fit import OTHER
from geometry import unit
from profiled_bands import ProfiledBandProblem
from road_fit import load_trial


class AntennaProblem(ProfiledBandProblem):
    scale = 10.

    def __init__(self, joint, antenna_sigma=(5., 5., 10.), site_sigma=10., *, nominal_offsets=None):
        if joint.range_bias:
            raise ValueError('Individual antenna positions and range-origin offsets are separate sensitivity trials')
        copied = copy.copy(joint)
        copied.problems = []
        nominal = (joint.problems[0].model.channel_offsets if nominal_offsets is None
                   else np.asarray(nominal_offsets, dtype=float))
        for source in joint.problems:
            problem = copy.copy(source)
            problem.model = copy.copy(source.model)
            problem.model.channel_offsets = nominal.copy()
            copied.problems.append(problem)
        super().__init__(copied)
        self.offset = self.count
        self.count += 15
        self.original_offsets = nominal.copy()
        self.assignment = joint.problems[0].model.channel_sites
        shared = self.assignment[:, None] == self.assignment[None, :]
        covariance = np.kron(shared, np.eye(3)*site_sigma**2)
        covariance += np.diag(np.tile(np.asarray(antenna_sigma)**2, 5))
        self.prior_covariance_m2 = covariance
        self.whiten = solve_triangular(np.linalg.cholesky(covariance)/self.scale, np.eye(15), lower=True)

    def initial(self, original):
        original = np.asarray(original)
        delta = original[:9].reshape(3, 3)*[30, 30, 75]
        result = np.r_[original[:self.offset], delta[self.assignment].ravel()/self.scale]
        result[:9] = 0
        return result

    def bounds(self):
        low, high = self.joint.bounds()
        width = 3*np.sqrt(np.diag(self.prior_covariance_m2))/self.scale
        return np.r_[low[:self.offset], -width], np.r_[high[:self.offset], width]

    def channel_model(self, p, index, channel, jacobian=False):
        problem = self.joint.problems[index]
        problem.model.channel_offsets = self.original_offsets+p[self.offset:].reshape(5, 3)*self.scale
        u, derivative = super().channel_model(p, index, channel, jacobian)
        if jacobian:
            joint = self.joint
            state = np.r_[p[9+index*joint.n:9+(index+1)*joint.n], p[:9]]
            field = problem.fields[channel]
            x, _, sites = problem.model.positions(state, field.time, problem.sites, True)
            tx = sites[self.assignment[channel]]+problem.model.channel_offsets[channel]
            rx = problem.model.receiver(channel)
            path = np.linalg.norm(x-tx, axis=1)+np.linalg.norm(x-rx, axis=1)-np.linalg.norm(tx-rx)
            dp = (-unit(x-tx)-unit(tx-rx))*self.scale
            _, local = field.template(path, dp)
            beta = 0 if channel == 1 else p[joint.beta_offset+int(np.flatnonzero(OTHER == channel)[0])]
            first = self.offset+3*channel
            derivative[:, first:first+3] = local*np.exp(1j*beta)
        return u, derivative

    def residual(self, p, jacobian=False, include_motion=True, include_site_prior=True):
        result = super().residual(p, jacobian, include_motion, include_site_prior)
        if not include_site_prior:
            return result
        prior = (np.c_[np.zeros((15, self.offset)), self.whiten] if jacobian
                 else self.whiten@p[self.offset:])
        return np.r_[result, prior]

    def information(self, p):
        expected = copy.copy(self)
        expected.joint = copy.copy(self.joint)
        expected.joint.problems = []
        for index, source in enumerate(self.joint.problems):
            problem = copy.copy(source)
            problem.model = copy.copy(source.model)
            _, amplitudes = self.target(p, index)
            problem.fields = []
            for ch, source_field in enumerate(source.fields):
                field = copy.copy(source_field)
                u, _ = self.channel_model(p, index, ch)
                field.y = amplitudes[ch]*u
                problem.fields.append(field)
            expected.joint.problems.append(problem)
        selected = np.r_[np.arange(self.joint.beta_offset, self.offset), np.arange(self.offset, self.count)]
        nuisance = np.arange(9, self.joint.beta_offset)
        result = {'order': 'four relative channel phases in radians, then five channel position corrections normalized by 10 m',
            'scope': 'Expected signal Jacobian at the fitted mean, after eliminating target paths, complex target phase and positive band amplitudes. Conditional on selected roads, scattering model and receiver.'}
        for motion, prior, name in ((False, False, 'data'), (True, True, 'data_motion_prior')):
            j = expected.residual(p, True, motion, prior)
            u, s, _ = np.linalg.svd(j[:, nuisance], full_matrices=False)
            u = u[:, s > s[0]*1e-10]
            projected = j[:, selected]-u@(u.T@j[:, selected])
            values, vectors = np.linalg.eigh(projected.T@projected)
            result[name+'_information_eigenvalues'] = values.tolist()
            if prior and values[0] > values[-1]*1e-12:
                covariance = (vectors/values)@vectors.T
                result['posterior_covariance_normalized'] = covariance.tolist()
                result['channel_position_standard_deviation_m'] = (np.sqrt(np.diag(covariance)[4:])*self.scale).reshape(5, 3).tolist()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('fit', 'histories', 'patches', 'templates', 'routes', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--antenna-sigma', type=float, nargs=3, default=[5., 5., 10.])
    parser.add_argument('--iterations', type=int, default=500)
    args = parser.parse_args()
    saved = json.loads(args.fit.read_text())
    joint = load_trial(saved, args.routes, args.histories, args.patches, args.templates)
    problem = AntennaProblem(joint, args.antenna_sigma)
    initial = problem.initial(saved['state'])
    active = np.arange(9, problem.count)
    low, high = problem.bounds()
    def residual(x, jacobian=False):
        p = initial.copy()
        p[active] = x
        value = problem.residual(p, jacobian)
        return value[:, active] if jacobian else value
    clock = time.monotonic()
    def progress(fit):
        if fit.nfev%10 == 0:
            print('eval', fit.nfev, 'cost', round(fit.cost, 3), 'elapsed', round(time.monotonic()-clock, 1), flush=True)
    fit = solve(residual, initial[active], (low[active], high[active]), args.iterations, progress)
    p = initial.copy()
    p[active] = fit.x
    state, absent = problem.expanded_state(p)
    offsets = problem.original_offsets+p[problem.offset:].reshape(5, 3)*problem.scale
    centres = np.array(saved['site_positions_enu_m'])[problem.assignment]+offsets
    result = {**saved, 'source_fit': str(args.fit), 'state': state.tolist(),
        'antenna_state': p.tolist(), 'cost': float(fit.cost), 'converged': bool(fit.success),
        'evaluations': fit.nfev, 'optimality': float(fit.optimality), 'termination': str(fit.message),
        'channel_offsets_enu_m': offsets.tolist(), 'site_corrections_enu_m': np.zeros((3, 3)).tolist(),
        'nominal_channel_offsets_enu_m': problem.original_offsets.tolist(),
        'effective_channel_centres_enu_m': centres.tolist(),
        'channel_position_corrections_enu_m': (p[problem.offset:].reshape(5, 3)*problem.scale).tolist(),
        'antenna_offset_prior_sigma_m': args.antenna_sigma, 'site_position_prior_sigma_m': 10.,
        'antenna_position_prior_covariance_m2': problem.prior_covariance_m2.tolist(),
        'zero_amplitude_channels_by_target': absent, 'profiled_band_gains': True,
        'information': problem.information(p),
        'status': 'Five electrical phase-centre hypothesis; prior sensitivity, site/road alternatives and held-out transfer remain required. No coordinates adopted.'}
    args.output.write_text(json.dumps(result, indent=2))
    print('finished', fit.cost, fit.success, 'elapsed', time.monotonic()-clock, flush=True)


if __name__ == '__main__':
    main()
