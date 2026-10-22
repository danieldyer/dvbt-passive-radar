"""Test a common bounded cubic path-phase residual after scene inversion."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from full_scene_context import open_scene
from coherent_path_phase import PhaseProbe


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);fit,_,scene,_=open_scene(a.motion);bands=[s.point for s in scene]
    with np.load(a.scene/'scene.npz') as f:points=f['points_body_m'];coefficient=f['coefficient'];spectrum=f['spectrum_magnitudes'];beta=f['band_phase_radians']
    models=[]
    for c,b in enumerate(bands):
        value=np.zeros_like(b.field.y)
        for point,alpha in zip(points,coefficient):value+=b.waveform(np.zeros(b.b.shape[1]),c,False,point-b.reference_point)*alpha
        models.append(value*spectrum[c]*np.exp(1j*beta[c]))
    interval=np.asarray(fit['interval_s']);mid=interval.mean();half=np.diff(interval)[0]/2
    basis=[np.column_stack([((b.t-mid)/half)**j for j in [1,2,3]]) for b in bands]
    root=[b.field.root_information for b in bands];groups=[b.groups for b in bands];data=[b.field.y for b in bands];k=np.array([b.field.k for b in bands])
    training=PhaseProbe(models,data,root,groups,[b.training for b in bands],basis,k)
    excluded=PhaseProbe(models,data,root,groups,[~b.training for b in bands],basis,k)
    before=training.objective(np.zeros(3),True);old=excluded.objective(np.zeros(3),True)
    source=json.loads((a.scene/'analysis.json').read_text())
    if abs((1-old[0])-source['outer']['explained_fraction'])>1e-8:raise ValueError('Original-row scene prediction changed on reconstruction')
    seeds=[np.zeros(3),*(e*s for e in np.eye(3) for s in [-.5,.5])]
    trials=[minimize(training.objective,x,jac=True,method='L-BFGS-B',bounds=[(-.5,.5)]*3,
        options=dict(maxiter=150,gtol=1e-9,ftol=1e-13)) for x in seeds]
    best=min(trials,key=lambda r:r.fun);after=excluded.objective(best.x,True)
    result=dict(scope=__doc__,motion=str(a.motion),scene=str(a.scene),correction_adopted=False,
        phase_only_probe=True,range_envelope_unchanged=True,common_physical_path_across_all_bands=True,
        parameters_m=best.x.tolist(),coefficient_bound_m=.5,maximum_path_change_bound_m=1.5,
        training_before=1-before[0],training_after=1-best.fun,outer_before=1-old[0],outer_after=1-after[0],
        original_outer_channels=old[2],corrected_outer_channels=after[2],maximum_gradient=float(max(abs(best.jac))),
        trials=[dict(objective=float(r.fun),maximum_gradient=float(max(abs(r.jac))),message=str(r.message)) for r in trials])
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    arrays={}
    for c,b in enumerate(bands):
        arrays[f'time_s_{c}']=b.t;arrays[f'phase_probe_path_m_{c}']=basis[c]@best.x
    np.savez_compressed(a.output/'histories.npz',**arrays);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
