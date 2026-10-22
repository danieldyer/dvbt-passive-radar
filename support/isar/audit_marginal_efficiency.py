"""Full 30-second, five-channel nonlinear motion recovery under known truth.

This tests all 17 unfrozen physical spline coordinates of the recorded scene
operator. Original geometry and core responses define a synthetic mean;
diffuse coefficients and thermal noise are drawn from the declared model.
Every trial retains every failure and checks its final original-row score.
It is a local-refinement experiment, not gross ambiguity acquisition or a
statement that the actual ship has the stipulated fixed Gaussian reflectivity.
"""
import argparse
import fcntl
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
from audit_motion_efficiency import summarize
from ais import utc_seconds
from full_motion import Band
from imaging import motion_state
from marginal_refinement import CurvatureMotion, fit
from marginal_score import ScoreMotion
from mixed_experiment import load
from model_quadrature import quadrature
from scene_dictionary import fingerprint
from score_stopping import fit as fit_to_score


def quadrature_bands(experiment, nodes):
    """Change only evaluation epochs; copy the original fixed observations."""
    model, bands = experiment.model, []
    for band in experiment.bands:
        field = quadrature(band.field, nodes)
        prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+field.data['row_time'], model)
        reduced = Band(field, prior, band.reference_transmitter, band.receiver,
            band.points, model, band.selected_time, range_bias_m=band.range_bias)
        reduced.nuisance, reduced.clean, reduced.energy = band.nuisance, band.clean, band.energy
        bands.append(reduced)
    return bands


def synthetic_components(experiment, bound):
    means, diffuse, valid = [], [], []
    model = experiment.model
    for c, (band, weight, core, strength) in enumerate(zip(experiment.bands, experiment.weights,
            model['core_counts'], model['fixed_penalties'])):
        target, _ = band.design(model['parameters'])
        design = band.nuisance.remove(target[band.selected])*np.sqrt(weight)
        alpha = np.asarray(bound['channels'][c]['core_coefficients_real_imag'])@np.array([1., 1j])
        means.append(design[:, :core]@alpha)
        diffuse.append(np.empty((len(design), 0), complex) if strength is None else design[:, core:]/np.sqrt(strength))
        valid.append(np.repeat(band.field.root_information[band.selected_time] > 0, band.field.y.shape[1]))
    return means, diffuse, valid


def draw_trial(experiment, components, seed, trial):
    means, diffuse, valid = components
    rng = np.random.default_rng(np.random.SeedSequence([seed, trial]))
    def normal(n):
        return (rng.normal(size=n)+1j*rng.normal(size=n))/np.sqrt(2)
    for c, band in enumerate(experiment.bands):
        y = means[c]+diffuse[c]@normal(diffuse[c].shape[1])+normal(len(means[c]))*valid[c]
        band.clean = band.nuisance.remove(y)
        band.energy = float(np.vdot(band.clean, band.clean).real)
        experiment.fast[c].clean, experiment.fast[c].energy = band.clean, band.energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--information', type=Path, required=True)
    parser.add_argument('--score-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--seed', type=int, default=881924)
    parser.add_argument('--nodes', type=int, default=0, help='Gauss nodes per retained bin; 0 keeps the legacy representative epoch')
    parser.add_argument('--solver', choices=('BFGS', 'trust-exact'), default='BFGS')
    parser.add_argument('--score-stop-norm', type=float,
                        help='Inner stopping score, strictly below the unchanged .01 original-row gate')
    parser.add_argument('--resume-prefix', type=Path, help='Preserve all completed trials from an older implementation')
    parser.add_argument('--prefix-code', type=Path, help='Hash-verified implementation archive for the preserved prefix')
    args = parser.parse_args()
    if args.score_stop_norm is not None and not 0 < args.score_stop_norm < .01:
        raise ValueError('Inner score norm must be positive and below the final .01 gate')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fit_motion = fit if args.score_stop_norm is None else partial(fit_to_score, score_norm=args.score_stop_norm)
    bound, precision = [json.loads(p.read_text()) for p in (args.information, args.score_audit)]
    if (not bound['complete'] or not precision['complete'] or not precision['numerically_adequate_for_efficiency_trials']
            or bound['source_sha256'] != fingerprint(args.source) or precision['source_sha256'] != fingerprint(args.source)):
        raise ValueError('Finish the exact source information and numerical-precision audits first')
    experiment = load(args.source, accelerated=True)
    model = experiment.model
    if args.nodes:
        experiment.fast = quadrature_bands(experiment, args.nodes)
    active = np.asarray(bound['active_physical_indices'])
    truth = np.asarray(model['parameters'])[active]
    fisher = np.asarray(bound['unit_noise_information']['total'])
    covariance = np.asarray(bound['crlb_covariance'])
    eigen, vectors = np.linalg.eigh(fisher)
    if eigen.min() <= 0:
        raise ValueError('No finite joint bound for this declared model')
    transform = vectors/np.sqrt(eigen)
    np.testing.assert_allclose(transform.T@fisher@transform, np.eye(len(active)), atol=2e-7)
    extras = (model['parameters'], experiment.weights, model['core_counts'], model['fixed_penalties'])
    factory = CurvatureMotion if args.solver == 'trust-exact' else ScoreMotion
    exact = ScoreMotion(experiment.bands, *extras)
    fast, refinement_model = [factory(bands, *extras) for bands in (experiment.fast, experiment.bands)]
    np.testing.assert_array_equal(exact.active, active)
    # A fixed 2-cm smooth coordinate perturbation supplies no per-trial phase
    # estimate. The same local initialization is used for every realization.
    initial = truth+.02*np.sin(np.linspace(0., 2*np.pi, len(truth)))
    components = synthetic_components(experiment, bound)
    files = ('audit_marginal_efficiency.py', 'marginal_score.py', 'marginal_scene.py', 'mixed_scene.py',
        'mixed_experiment.py', 'full_motion.py', 'field.py', 'ridge_motion.py', 'spline_motion.py',
        'marginal_refinement.py', 'model_quadrature.py')
    if args.score_stop_norm is not None:
        files += ('score_stopping.py',)
    identity = {'source': str(args.source), 'source_sha256': fingerprint(args.source),
        'information': str(args.information), 'information_sha256': fingerprint(args.information),
        'score_audit': str(args.score_audit), 'score_audit_sha256': fingerprint(args.score_audit),
        'implementation_sha256': {n: fingerprint(Path(__file__).with_name(n)) for n in files},
        'seed': args.seed, 'iterations': args.iterations, 'solver': args.solver, 'quadrature_nodes': args.nodes,
        'inner_score_norm': args.score_stop_norm}
    result = {**identity, 'scope': __doc__, 'truth': truth.tolist(), 'fixed_initial': initial.tolist(),
        'active_physical_indices': active.tolist(), 'crlb_covariance': covariance.tolist(),
        'noise': 'Unit proper-complex thermal noise on all valid original retained observations, plus independent CN(0, 1/lambda) diffuse coefficients per trial and channel; fixed stationary contrasts.',
        'normalization': 'Unnormalized Gaussian negative log likelihood; no estimated noise scale or trial-specific amplitude rescaling.',
        'constraints': 'Yaw and the existing coordinate gauge are fixed. No kinematic inequalities or motion box constraints are applied; this is the unconstrained local bound.',
        'cases': [], 'complete': False}
    if args.resume_prefix:
        prefix = json.loads(args.resume_prefix.read_text())
        for key in ('source', 'source_sha256', 'information', 'information_sha256',
                    'score_audit', 'score_audit_sha256', 'seed', 'iterations', 'solver', 'quadrature_nodes'):
            if prefix[key] != identity[key]:
                raise ValueError('Prefix changed the source model or sampling experiment: '+key)
        for key in ('truth', 'fixed_initial', 'active_physical_indices', 'crlb_covariance'):
            np.testing.assert_array_equal(prefix[key], result[key])
        if [c['trial'] for c in prefix['cases']] != list(range(len(prefix['cases']))):
            raise ValueError('Prefix must retain every consecutive trial, including failures')
        archives = {}
        for name, digest in prefix['implementation_sha256'].items():
            path = Path(__file__).with_name(name)
            if fingerprint(path) != digest:
                if args.prefix_code is None:
                    raise ValueError('Archive the prior implementation before changing it')
                path = args.prefix_code/name
                if fingerprint(path) != digest:
                    raise ValueError('Prior implementation archive mismatch: '+name)
                archives[str(path)] = digest
        provenance = {'file': str(args.resume_prefix), 'sha256': fingerprint(args.resume_prefix),
            'completed_trials': len(prefix['cases']), 'archived_implementation_sha256': archives,
            'scope': 'Every completed prefix trial is retained unchanged. Later trials use the same seeds, source, initialization and independent original-row .01 acceptance gate, with the explicitly recorded inner stopping rule.'}
        identity['prefix'] = provenance
        result.update(prefix=provenance, cases=prefix['cases'])
        if 'statistics' in prefix:
            result['statistics'] = prefix['statistics']
        result['nonstationary_trials'] = prefix.get('nonstationary_trials', [])
    if args.output.exists():
        result = json.loads(args.output.read_text())
        for key, value in identity.items():
            if result[key] != value:
                raise ValueError(f'Cannot resume after a changed identity: {key}')
        if len(result['cases']) >= args.trials:
            raise ValueError('Requested trials already exist; preserve the result')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.output)
    result.update(requested_trials=args.trials, complete=False)
    save()
    for trial in range(len(result['cases']), args.trials):
        started = time.monotonic()
        draw_trial(experiment, components, args.seed, trial)
        exact.last = fast.last = refinement_model.last = None
        print(json.dumps({'trial': trial, 'stage': 'start'}), flush=True)
        def progress(stage):
            return lambda record: print(json.dumps({'trial': trial, 'stage': stage, **record}), flush=True)
        parameters, search = fit_motion(fast, initial, transform, iterations=args.iterations,
            method=args.solver, progress=progress('search'))
        original = exact.evaluate(parameters)
        original_score = float(np.linalg.norm(transform.T@original['gradient']))
        refinement = None
        if original_score > .01:
            parameters, refinement = fit_motion(refinement_model, parameters, transform, iterations=args.iterations,
                method=args.solver, progress=progress('original_refinement'))
        final = exact.evaluate(parameters)
        score = float(np.linalg.norm(transform.T@final['gradient']))
        error = parameters-truth
        case = {'trial': trial, 'parameters': parameters.tolist(), 'error': error.tolist(),
            'search': search, 'original_score_before_refinement': original_score, 'exact_refinement': refinement,
            'final_information_normalized_score_norm': score, 'stationary': bool(score <= .01),
            'mahalanobis_error_per_parameter': float(error@fisher@error/len(error)),
            'seconds': time.monotonic()-started}
        result['cases'].append(case)
        errors = np.array([c['error'] for c in result['cases']])
        if len(errors) >= 2:
            result['statistics'] = summarize(errors, covariance)
        result['nonstationary_trials'] = [c['trial'] for c in result['cases'] if not c['stationary']]
        result['complete'] = len(result['cases']) == args.trials
        save()
        print(json.dumps({k: case[k] for k in ('trial', 'stationary', 'final_information_normalized_score_norm',
            'mahalanobis_error_per_parameter', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
