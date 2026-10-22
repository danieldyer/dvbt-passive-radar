"""Recorded-operator derivatives and conditional mixed-scene efficiency.

The simulation stipulates the chosen diffuse prior and unit thermal noise as
truth. It verifies the linear estimator's conditional BLUP/MMSE calculation;
it does not validate that prior or covariance for the actual vessel return.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from ais import utc_seconds
from distributed_fixed_frame import quadrature
from field import Field
from full_motion import Band
from imaging import motion_state, sites
from mixed_scene import profile
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationary import Stationary


def efficiency(fit, design, prediction_design, core, strength, trials=8000):
    """Equivalent compressed draws; full training noise is represented by Uobsᴴ n."""
    u, singular, vh = fit['basis'], fit['singular'], fit['right']
    observed = u[:len(design)]
    q = observed.conj().T@observed
    error_prior = (observed.conj().T@design[:, core:]-singular[:, None]*vh[:, core:])/np.sqrt(strength)
    covariance = error_prior@error_prior.conj().T+q
    error = float(np.max(abs(covariance-np.eye(len(singular)))))
    response = vh.conj().T@((observed.conj().T@design[:, :core])/singular[:, None])
    desired = np.eye(design.shape[1])[:, :core]
    response_error = float(abs(response-desired).max())
    eigen, vectors = np.linalg.eigh((q+q.conj().T)/2)
    if eigen.min() < -1e-12 or eigen.max() > 1+1e-12:
        raise ValueError('The compressed noise covariance is not a projection contraction')
    rng = np.random.default_rng(281419)
    def noise(rows):
        return (rng.normal(size=(rows,trials))+1j*rng.normal(size=(rows,trials)))/np.sqrt(2)
    # Clipping removes only the explicitly bounded floating-point eigensolver
    # error; it is not fitted covariance regularization.
    factor = vectors*np.sqrt(np.clip(eigen,0.,1.))
    normalized_error = error_prior@noise(design.shape[1]-core)+factor@noise(len(singular))
    f = (prediction_design@vh.conj().T)/singular
    _, r = np.linalg.qr(f, mode='reduced')
    expected = float(np.sum(abs(r)**2))
    samples = np.sum(abs(r@normalized_error)**2,axis=0)
    measured = float(samples.mean())
    return {'trials': trials, 'maximum_whitened_covariance_identity_error': error,
        'maximum_unbiased_core_complex_response_error': response_error,
        'calculated_prediction_risk': expected,
        'calculated_prediction_risk_over_conditional_bound': float(np.sum((r@covariance)*r.conj()).real/expected),
        'measured_prediction_risk_over_conditional_bound': measured/expected,
        'monte_carlo_standard_error_of_ratio': float(samples.std(ddof=1)/np.sqrt(trials)/expected),
        'normalized_parameter_risk_ratio': float(np.mean(abs(normalized_error)**2)),
        'noise_covariance_eigenvalue_range': [float(eigen.min()),float(eigen.max())],
        'scope': 'Known physical motion, deterministic unknown core responses, stipulated CN(0, lambda^-1 I) diffuse coefficients and unit projected thermal noise. Expected MSE equals prediction-design times inverse augmented information times its adjoint. Not an empirical motion CRLB or calibration claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--channel', type=int, default=3)
    p.add_argument('--spacing', type=float, default=20.)
    args = p.parse_args()
    source = json.loads(args.training.read_text())
    if not source['complete'] or args.output.exists():
        raise ValueError('Complete training and use a fresh output')
    model = json.loads(Path(source['source_motion']).read_text())
    curve = SplineMotion(model)
    trial = next(t for t in source['channels'][args.channel]['trials'] if t['spacing_m']==args.spacing)
    strength = trial['tuning']['regularization_strength']
    if strength is None:
        raise ValueError('Choose a declared channel with a nonzero diffuse component')
    core = np.asarray(trial['core_points_enu_m'])
    diffuse, area = quadrature(source['bounds_enu_m'], args.spacing)
    scale = np.sqrt(np.r_[np.ones(len(core)), area])
    tx, rx, _ = sites(Path(model['geometry_directory']))
    with np.load(Path(source['fields'])/f'channel-{args.channel}.npz') as d:
        field = Field(dict(d), d['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
    prior = motion_state(None,utc_seconds(model['nominal_start_utc'])+field.data['row_time'],model)
    selected = np.floor((field.time-curve.interval[0])/source['holdout_seconds']).astype(int)%2==0
    band = Band(field,prior,tx[model['reference_channel']],rx,np.r_[core,diffuse],model,selected,
        range_bias_m=model['range_bias_m'][args.channel])
    started = wall_time.monotonic()
    parameters = np.array(model['parameters'])
    fit = profile(band,parameters,scale,len(core),strength,derivatives=True)
    direction = np.random.default_rng(73212).normal(size=len(parameters))
    direction[curve.sections[1]]=0.
    direction /= np.linalg.norm(direction)
    analytic = float(fit['gradient']@direction/band.energy)
    checks = []
    for step in (.01,.001,.0001):
        plus,minus=[profile(band,parameters+sign*step*direction,scale,len(core),strength)['value'] for sign in (1,-1)]
        numeric = (plus-minus)/(2*step*band.energy)
        checks.append({'step_m':step,'numeric':numeric,'analytic':analytic,'absolute_error':abs(numeric-analytic)})
        print(json.dumps(checks[-1]),flush=True)
    if min(c['absolute_error'] for c in checks)>1e-7:
        raise ValueError('Original-row normalized gradient does not pass')
    design=band.nuisance.remove(fit['target'][band.selected])*scale
    nuisance=Stationary(band.nuisance.design,~band.selected)
    prediction=nuisance.remove(fit['target'][~band.selected])*scale
    bound=efficiency(fit,design,prediction,len(core),strength)
    if bound['maximum_whitened_covariance_identity_error']>1e-7 or bound['maximum_unbiased_core_complex_response_error']>1e-7:
        raise ValueError('Conditional efficiency identity or core response check failed')
    result={'scope':__doc__,'training':str(args.training),'training_sha256':fingerprint(args.training),
        'channel':args.channel,'points':len(band.points),'core_points':len(core),'penalty':strength,
        'gradient_checks':checks,'conditional_efficiency':bound,'seconds':wall_time.monotonic()-started,
        'implementation_sha256':{name:fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_mixed_scene.py','mixed_scene.py','ridge_motion.py','full_motion.py','field.py','spline_motion.py')},
        'complete':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(bound),flush=True)


if __name__=='__main__':
    main()
