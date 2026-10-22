"""Complex sparse image on the unchanged physical waveform statistics.

One shared complex reflectivity image and a frozen five-band spectrum are
used. Inner two-second blocked prediction chooses the Laplace prior; the
original excluded fold is scored afterwards. The upstream motion remains
conditional. Every convex solve must pass its KKT and primal/dual gates.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from complex_lasso import ComplexLasso
from regularized_scene import Observation,score


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--statistics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--unit-spectrum',action='store_true');p.add_argument('--strength-relative',type=float,nargs='+',default=[1.,.5,.2,.1,.05,.02,.01,.005,.002,.001])
    p.add_argument('--maximum-iterations',type=int,default=10000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    description=json.loads((a.geometry/'analysis.json').read_text());motion=json.loads(Path(description['motion']).read_text())
    with np.load(a.geometry/'scene.npz') as f:points=f['points_body_m'];g=np.ones(5,complex) if a.unit_spectrum else f['spectrum_gains']
    beta=np.angle([complex(*z) for z in motion['physical_gains_real_imag']])
    with np.load(a.statistics) as f:
        observations={name:[Observation(f[f'{name}_gram_{c}'],f[f'{name}_rhs_{c}'],float(f[f'{name}_energy_{c}'])) for c in range(5)]
            for name in ['inner','validation','original','outer']}
    def problem(name):
        values=observations[name];gram=sum(abs(z)**2*o.gram for o,z in zip(values,g));rhs=sum(np.conj(z)*o.rhs for o,z in zip(values,g))
        return ComplexLasso(gram,rhs,sum(o.energy for o in values))
    inner=problem('inner');trials=[];previous=None;best=None
    for relative in sorted(a.strength_relative,reverse=True):
        strength=relative*inner.linear_scale;x,report=inner.solve(strength,previous,maximum_iterations=a.maximum_iterations)
        np.savez_compressed(a.output/f'coefficient-{relative:g}.npz',coefficient=x)
        if not report['numerical_convergence']:
            (a.output/'failed-solve.json').write_text(json.dumps(report,indent=2)+'\n')
            raise ValueError(f"Convex sparse image missed unchanged gates: KKT {report['maximum_scaled_kkt']}, relative gap {report['gap_relative_to_fitted_energy']}; full diagnostic saved")
        validation=score(observations['validation'],x,g);record=dict(relative_strength=relative,strength=strength,optimization=report,
            training=score(observations['inner'],x,g),inner_validation=validation)
        trials.append(record);previous=x
        if best is None or validation['cost']<best[0]:best=(validation['cost'],strength,x,relative)
        (a.output/'trials.json').write_text(json.dumps(dict(complete=False,scope=__doc__,trials=trials),indent=2)+'\n')
        print(json.dumps(dict(relative_strength=relative,training=record['training']['explained_fraction'],validation=validation['explained_fraction'],
            nonzero=report['nonzero_coefficients'],iterations=report['iterations'],kkt=report['maximum_scaled_kkt'],gap=report['gap_relative_to_fitted_energy'],seconds=time.monotonic()-started)),flush=True)
    _,strength,initial,relative=best;original=problem('original');x,report=original.solve(strength,initial,maximum_iterations=a.maximum_iterations)
    if not report['numerical_convergence']:raise ValueError('Original-training sparse image did not converge')
    result=dict(complete=True,scope=__doc__,geometry=str(a.geometry),statistics=str(a.statistics),motion=description['motion'],
        body_frame_hypothesis=description.get('body_frame_hypothesis','upstream-motion'),relative_strength=relative,strength=strength,
        prior='Uniform complex Laplace prior on the declared uniform physical mesh',relative_band_phases_fitted=False,
        selection='Minimum inner blocked complex prediction error; outer fold unused for prior selection',
        trials=trials,optimization=report,original_training=score(observations['original'],x,g),outer=score(observations['outer'],x,g),
        conditional_upstream_motion_and_calibration=True,regularization_grid_edge=relative in [min(a.strength_relative),max(a.strength_relative)],seconds=time.monotonic()-started)
    np.savez_compressed(a.output/'scene.npz',points_body_m=points,coefficient=x,spectrum_gains=g,spectrum_magnitudes=abs(g),band_phase_radians=beta+np.angle(g))
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(stage='final',outer=result['outer'],optimization=report)),flush=True)


if __name__=='__main__':main()
