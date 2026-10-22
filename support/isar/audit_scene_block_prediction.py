"""Separate fixed-support model prediction from blind support acquisition.

The known-support diagnostic is not an input to the blind reconstruction.
Outer observations remain excluded from every fit. The inner two-second
blocks test interpolation across intervals substantially longer than the
original alternating half-second folds.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from fixed_scene_reconstruction import fit_scene,score_scene
from run_residual_scene_reconstruction import simulate
from audit_ship_scattering_envelopes import linear_basis
from scipy.sparse import csr_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--synthetic-experiment',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--amplitude-spacing-s',type=float,nargs='+',default=[5.])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    fit,_,scene,_=open_scene(a.motion);bands=[s.point for s in scene]
    beta=np.angle([complex(*z) for z in fit['physical_gains_real_imag']])
    points=np.asarray(fit['scene_points_body_m']);phi=np.asarray(fit['scene_point_phase_radians'])
    if a.synthetic_experiment:
        experiment=json.loads((a.synthetic_experiment/'analysis.json').read_text())
        west,east,south,north=experiment['grid_bounds_m'];d=experiment['grid_spacing_m']
        x=np.arange(west,east+d*.1,d);y=np.arange(south,north+d*.1,d);xx,yy=np.meshgrid(x,y)
        pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.)]
        bands,truth=simulate(bands,beta,pixels,experiment['truth']['kind'],experiment['truth']['per_point_integrated_snr_db'])
        points=np.asarray(truth['points_body_m']);phi=np.asarray(truth['source_phase_radians'])
    if points.ndim!=2:raise ValueError('This diagnostic requires one local persistent support')
    design=[np.stack([b.waveform(np.zeros(b.b.shape[1]),c,False,point-b.reference_point) for point in points],axis=2)
        for c,b in enumerate(bands)]
    original=[b.training.copy() for b in bands];first=min(b.t[0] for b in bands)
    inner=[old&(np.floor((b.t-first)/2).astype(int)%2==0) for b,old in zip(bands,original)]
    validation=[old&~mask for old,mask in zip(original,inner)];outer=[~old for old in original]
    result=dict(scope=__doc__,motion=str(a.motion),synthetic_experiment=str(a.synthetic_experiment),sources=len(points),models=[])
    for spacing in a.amplitude_spacing_s:
        if spacing<0:raise ValueError('Negative amplitude-basis spacing')
        views=[]
        for b in bands:
            view=copy.copy(b);basis,nodes=linear_basis(b.t,fit['interval_s'],spacing)
            view.amplitude=csr_matrix(basis);view.amplitude_nodes=nodes;views.append(view)
        for name,masks in [('inner-training',inner),('original-training',original)]:
            model=fit_scene(design,views,masks,beta,phi,iterations=500)
            record=dict(name=name,amplitude_spacing_s=spacing,optimization=model.optimization,training=score_scene(model,views,masks),
                inner_validation=score_scene(model,views,validation),outer=score_scene(model,views,outer))
            result['models'].append(record)
            arrays={}
            for c,(b,amplitude) in enumerate(zip(views,model.amplitude)):
                arrays[f'time_s_{c}']=b.t;arrays[f'amplitude_{c}']=b.amplitude@amplitude
            np.savez_compressed(a.output/f'{name}-{spacing:g}s.npz',**arrays)
            (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(record),flush=True)
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
