"""Finite-difference the off-grid profiled likelihood on recorded physical kernels."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from continuous_sparse_scene import ContinuousSparseScene


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    fit,_,scene,_=open_scene(a.motion);bands=[s.point.waveform_window(100,164) for s in scene]
    mid=np.mean(fit['interval_s']);half=np.diff(fit['interval_s'])[0]/2;basis=[];rates=[]
    for b in bands:
        u=(b.t-mid)/half;basis.append(np.c_[u,u*u,u*u*u]);rates.append(np.c_[np.ones_like(u),2*u,3*u*u]/half)
    beta=np.angle([complex(*z) for z in fit['physical_gains_real_imag']]);spectrum=np.exp(1j*beta)
    physical=np.array([[-113.2,-54.7,20.],[-66.4,-38.1,20.],[-21.3,-29.2,20.]])
    coefficient=np.array([1+.3j,-.4+.8j,.5-.2j]);parameters=np.r_[[.03,-.02,.01],physical[:,:2].ravel()]
    model=ContinuousSparseScene(bands,basis,rates,spectrum,1.)
    truth_bands=model.shifted_bands(parameters[:3]);observed=[]
    for c,(b,true) in enumerate(zip(bands,truth_bands)):
        other=copy.copy(b);other.field=copy.copy(b.field)
        other.field.y=sum(true.waveform(np.zeros(b.b.shape[1]),c,False,point-b.reference_point)*coefficient[j]*spectrum[c] for j,point in enumerate(physical))
        observed.append(other)
    # Declare the prior from the original training correlation, not the
    # test positions/finite-difference result or an image sharpening score.
    probe=ContinuousSparseScene(observed,basis,rates,spectrum,1.,kkt_tolerance=1e-11,gap_tolerance=1e-11)
    _,_,base=probe.evaluate(parameters,True)
    strength=.01*base['coefficient_optimization']['gram_coordinate_scale']
    probe=ContinuousSparseScene(observed,basis,rates,spectrum,strength,kkt_tolerance=1e-11,gap_tolerance=1e-11)
    at=parameters+np.r_[[.01,.015,-.005],[.25,-.12,-.2,.1,.15,.05]]
    value,gradient,report=probe.evaluate(at,True);fd=[];step=1e-4
    for e in np.eye(len(at)):fd.append((probe.evaluate(at+step*e)[0]-probe.evaluate(at-step*e)[0])/(2*step))
    fd=np.asarray(fd);relative=float(np.linalg.norm(fd-gradient)/max(np.linalg.norm(gradient),1e-30))
    result=dict(complete=True,scope=__doc__,motion=str(a.motion),original_representative_epochs_per_band=64,
        original_symbol_epochs_per_band=64*16,analytic_gradient=gradient.tolist(),finite_difference=fd.tolist(),
        relative_gradient_error=relative,maximum_absolute_gradient_error=float(max(abs(fd-gradient))),
        original_row_closure_error=report['original_row_closure_error'],coefficient_optimization=report['coefficient_optimization'],
        criterion='Relative gradient error below 2e-5; original row closure below 1e-9',passed=bool(relative<2e-5 and report['original_row_closure_error']<1e-9))
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['passed']:raise ValueError('Off-grid physical profile derivative failed')


if __name__=='__main__':main()
