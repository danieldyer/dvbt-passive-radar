"""Data-independent physical scene modes for joint image/phase acquisition.

Use the unprojected known-operator spectrum, reconstructed from exact
original-fold statistics and stationary means. A discarded-eigenvalue bound
controls the ridge fitted-data operator uniformly over common phase trials:
||K(K+lambda I)^-1 - Kr(Kr+lambda I)^-1|| <= ||K-Kr||/lambda.
Row phase is unitary and stationary projection is a contraction. The bound
also allows fixed band gains whose squared magnitude is at most the declared
maximum. This does not validate the separate phase-only geometry approximation.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from full_scene_context import open_scene,point_dictionary
from fit_distributed_physical_scene import operator_fingerprint,freeze_body_attitude
from profiled_phase_scene import ProfiledPhaseScene
from rigid_patch import profile_static


def means(h,root,mask,groups,phase=None):
    rows=[]
    for group in np.unique(groups):
        q=root*mask*(groups==group);norm=np.linalg.norm(q)
        if norm:
            q=q/norm
            if phase is not None:q=q*phase
            rows.append(np.einsum('t,trp->rp',q,h,optimize=True))
    return np.concatenate(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--minimum-ridge-relative',type=float,default=1e-4)
    p.add_argument('--maximum-predictor-error',type=float,default=1e-6);p.add_argument('--maximum-band-power',type=float,default=5.)
    p.add_argument('--maximum-modes',type=int,default=1024)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    description=json.loads((a.geometry/'analysis.json').read_text());source=Path(description['motion'])
    fit,_,scene,_=open_scene(source);bands=[s.point for s in scene]
    if description.get('body_frame_hypothesis')=='frozen-at-aperture-midpoint':freeze_body_attitude(bands,fit)
    with np.load(a.geometry/'scene.npz') as f:points=f['points_body_m']
    beta=np.angle([complex(*z) for z in fit['physical_gains_real_imag']]);fingerprint=operator_fingerprint(bands,points,beta,source)
    cache=np.load(a.geometry/'statistics.npz')
    if str(cache['operator_fingerprint'])!=fingerprint:raise ValueError('Known physical operator differs from its cached statistics')
    n=len(points);gram=np.zeros((n,n),complex);training_raw=np.zeros_like(gram)
    rng=np.random.default_rng(412);probe=rng.normal(size=n)+1j*rng.normal(size=n);direct_energy=0.
    for c,b in enumerate(bands):
        h=point_dictionary(b,points,c,phase=beta[c]);direct_energy+=float(np.vdot(h@probe,h@probe).real)
        for name,mask in [('original',b.training),('outer',~b.training)]:
            mean=means(h,b.field.root_information,mask,b.groups)
            raw=cache[f'{name}_gram_{c}']+mean.conj().T@mean
            gram+=raw
            if name=='original':training_raw+=raw
        del h
        print(json.dumps(dict(stage='unprojected-operator',channel=c,seconds=time.monotonic()-started)),flush=True)
    reconstructed=float(np.vdot(probe,gram@probe).real);closure=abs(reconstructed/direct_energy-1)
    if closure>1e-10:raise ValueError(f'Unprojected operator reconstruction failed: {closure}')
    eigen,vector=eigh((gram+gram.conj().T)/2);largest=float(eigen[-1]);minimum=a.minimum_ridge_relative*largest
    cutoff=minimum*a.maximum_predictor_error/a.maximum_band_power
    keep=eigen>cutoff;count=int(keep.sum());next_value=float(np.max(eigen[~keep],initial=0.))
    error_bound=a.maximum_band_power*max(next_value,0.)/minimum
    report=dict(complete=False,scope=__doc__,geometry=str(a.geometry),motion=str(source),operator_fingerprint=fingerprint,
        physical_pixels=n,modes=count,minimum_ridge_strength=minimum,minimum_ridge_relative=a.minimum_ridge_relative,
        maximum_band_gain_power=a.maximum_band_power,predictor_error_norm_bound=error_bound,
        discarded_operator_eigenvalue=next_value,unprojected_operator_closure_error=closure,
        smallest_operator_eigenvalue=float(eigen[0]),largest_operator_eigenvalue=largest,
        basis_depends_on_observed_complex_data=False,channels=[])
    np.save(a.output/'operator-eigenvalues.npy',eigen)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    if count>a.maximum_modes:raise ValueError(f'{count} modes required by the declared predictor-error bound, exceeding {a.maximum_modes}')
    v=vector[:,keep];np.save(a.output/'spatial-basis.npy',v);np.save(a.output/'points-body.npy',points)
    retained=np.sum(abs(v)**2*eigen[keep][None,:],axis=1)/gram.diagonal().real
    report['minimum_unprojected_point_information_retained']=float(retained.min())
    designs=[];data=[];roots=[];groups=[];selected=[];basis=[];trial=np.array([.11,-.07,.03])
    rhs=np.zeros(n,complex);trial_gram=training_raw.copy();energy=0.;mid=np.mean(fit['interval_s']);half=np.diff(fit['interval_s'])[0]/2
    for c,b in enumerate(bands):
        h=point_dictionary(b,points,c,phase=beta[c]);u=(b.t-mid)/half;curve=np.c_[u,u*u,u*u*u]
        phase=np.exp(-1j*b.field.k*(curve@trial));mean=means(h,b.field.root_information,b.training,b.groups,phase)
        trial_gram-=mean.conj().T@mean
        y=profile_static(b.field.y,b.field.root_information,b.training,b.groups)*b.training[:,None]
        energy+=float(np.vdot(y,y).real);rhs+=np.einsum('trp,tr,t->p',h.conj(),y,phase.conj(),optimize=True)
        reduced=(h.reshape(-1,n)@v).reshape(*h.shape[:2],count);del h
        np.save(a.output/f'channel-{c}-design.npy',reduced)
        designs.append(reduced);data.append(b.field.y);roots.append(b.field.root_information);groups.append(b.groups);selected.append(b.training);basis.append(curve)
        report['channels'].append(dict(channel=c,rows=len(b.t),ranges=len(b.field.offsets),rf_hz=b.field.metadata['rf_hz'],
            design=f'channel-{c}-design.npy',band_phase_radians=float(beta[c])))
        print(json.dumps(dict(stage='physical-modes',channel=c,modes=count,seconds=time.monotonic()-started)),flush=True)
    full=trial_gram.copy();full.flat[::n+1]+=minimum
    from scipy.linalg import cho_factor,cho_solve
    x=cho_solve(cho_factor((full+full.conj().T)/2,lower=True),rhs);expected=float((energy-np.vdot(x,rhs).real)/energy)
    model=ProfiledPhaseScene(designs,data,roots,groups,selected,basis,[b.field.k for b in bands],minimum)
    actual=model.objective(trial)[0];difference=abs(actual-expected)
    if difference>error_bound*1.05+1e-9:raise ValueError(f'Reduced phase profile exceeds its predictor bound: {difference}, {error_bound}')
    report.update(complete=True,phase_probe_parameters_m=trial.tolist(),full_phase_profile_objective=expected,
        reduced_phase_profile_objective=actual,phase_profile_objective_difference=difference,seconds=time.monotonic()-started)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');cache.close();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
