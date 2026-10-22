"""Blocked-prediction test of a regularized, time-invariant shared image.

The original physical path is held fixed. The default also fixes relative
band phase calibration; an explicit control fits one constant correction per
band, shared by every coefficient. Complex scene coefficients are shared
across frequencies. Five positive constant spectrum magnitudes are nuisance
parameters. Ridge strength is chosen on inner blocked observations; the
original outer fold is evaluated only after selection. This remains a
conditional exploratory test because upstream motion/support used the
original training fold, including the inner validation observations.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from regularized_scene import Observation,fit,score,FixedRidgePath
from run_residual_scene_reconstruction import simulate


def freeze_body_attitude(bands,source):
    """Hold relative source offsets fixed about the same measured reference.

    The reference point's anchor, velocity and radio path are unchanged.
    Only its rigid frame and frame derivative change; this never rotates a
    completed image or makes one band follow an independent trajectory.
    """
    from declared_ship_orientation import state,rotate_frame
    midpoint=float(np.mean(source['interval_s']))
    for b in bands:
        value=state(b.motion,np.array([midpoint]),source.get('orientation','existing'))
        value=rotate_frame(value,np.array([midpoint]),source.get('anchor_path_time_s',162.),
            source.get('angular_rate_offset_degps',0.),source.get('angular_acceleration_offset_degps2',0.))
        b.frame=np.broadcast_to(value['body_to_ecef'][0],b.frame.shape)
        b.frame_rate=np.zeros_like(b.frame_rate)


def operator_fingerprint(bands,points,beta,source):
    """Identify known operator inputs independently of observed complex data."""
    digest=hashlib.sha256(source.read_bytes())
    def add(value):
        value=np.asarray(value);digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    add(points);add(beta)
    for b in bands:
        for name in ['t','base_path','base_rate','frame','frame_rate','anchor','velocity','direction','direction_rate','reference_point','training','groups']:
            add(getattr(b,name))
        add(b.motion.tx);add(b.motion.rx);add(b.field.whiten);add(b.field.root_information)
        add(b.coordinate['native_time_s']);add(b.coordinate['native_information'])
        for name,value in sorted(b.field.data.items()):
            if name not in ['value','metadata_json']:digest.update(name.encode());add(value)
        blocks=b.field.pulse.row_blocks;add(blocks)
        add(b.field.pulse.kernels.coefficients[:,np.unique(blocks)])
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--radius-m',type=float,default=1.5)
    p.add_argument('--spacing-m',type=float,default=.75)
    p.add_argument('--spacing-y-m',type=float,help='Optional anisotropic rectangular body-y spacing')
    p.add_argument('--bounds-m',type=float,nargs=4,help='Complete rectangular body support instead of local patches')
    p.add_argument('--fixed-spectrum',action='store_true',help='Shared unit-magnitude spectrum; linear ridge solve without alternating magnitude estimation')
    p.add_argument('--spectrum-from',type=Path,help='Freeze previously training-estimated constant spectrum gains; requires fixed spectrum')
    p.add_argument('--synthetic-experiment',type=Path)
    p.add_argument('--synthetic-spectrum',choices=['pointwise-positive','shared'],default='pointwise-positive')
    p.add_argument('--statistics',type=Path,help='Explicit same-operator retry; original point-column norms/RHS are rechecked before reuse')
    p.add_argument('--operator-statistics',type=Path,help='Reuse only data-independent Gram matrices with a matching known-operator fingerprint')
    p.add_argument('--fit-band-phases',action='store_true',help='Estimate one constant relative calibration per band, shared by every image coefficient')
    p.add_argument('--freeze-body-attitude',action='store_true',help='Test constant physical orientation at aperture midpoint, preserving the measured reference trajectory')
    p.add_argument('--regularization-relative',type=float,nargs='+',default=np.logspace(-7,0,15).tolist())
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source,_,scene,_=open_scene(a.motion);bands=[b.point for b in scene];truth=None
    centres=np.asarray(source['scene_points_body_m'])
    if centres.ndim!=2 or a.radius_m<0 or a.spacing_m<=0:raise ValueError('Need one local scene and valid physical dictionary spacing')
    beta=np.angle([complex(*z) for z in source['physical_gains_real_imag']])
    if a.synthetic_experiment:
        experiment=json.loads((a.synthetic_experiment/'analysis.json').read_text());west,east,south,north=experiment['grid_bounds_m'];d=experiment['grid_spacing_m']
        x=np.arange(west,east+d*.1,d);y=np.arange(south,north+d*.1,d);xx,yy=np.meshgrid(x,y)
        grid=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.)]
        bands,truth=simulate(bands,beta,grid,experiment['truth']['kind'],experiment['truth']['per_point_integrated_snr_db'],spectral_model=a.synthetic_spectrum)
        centres=np.asarray(experiment['points_body_m'])
    if a.freeze_body_attitude:freeze_body_attitude(bands,source)
    offsets=np.arange(-a.radius_m,a.radius_m+a.spacing_m*.1,a.spacing_m);x,y=np.meshgrid(offsets,offsets)
    points=np.unique(np.round((centres[:,None,:]+np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]).reshape(-1,3),8),axis=0)
    if a.bounds_m:
        west,east,south,north=a.bounds_m
        if not west<east or not south<north:raise ValueError('Invalid rectangular scene bounds')
        dy=a.spacing_m if a.spacing_y_m is None else a.spacing_y_m
        if dy<=0:raise ValueError('Need positive body-y spacing')
        x,y=np.meshgrid(np.arange(west,east+a.spacing_m*.1,a.spacing_m),np.arange(south,north+dy*.1,dy))
        points=np.c_[x.ravel(),y.ravel(),np.full(x.size,20.)]
    if a.fixed_spectrum and a.fit_band_phases:raise ValueError('Fixed spectrum also fixes its phase calibration')
    if a.spectrum_from and not a.fixed_spectrum:raise ValueError('An upstream frozen spectrum requires the linear fixed-spectrum control')
    expected_design_bytes=max(len(b.t)*len(b.field.offsets) for b in bands)*len(points)*16
    if not 0<len(points)<=4096 or expected_design_bytes>4*1024**3:
        raise ValueError('Offline dense reference limit: 4096 points and at most 4 GiB for one exact design')
    first=min(b.t[0] for b in bands);observations={k:[] for k in ['inner','validation','original','outer']}
    fingerprint=operator_fingerprint(bands,points,beta,a.motion)
    if a.statistics and a.operator_statistics:raise ValueError('Choose a whole-observation retry or a Gram-only reuse')
    cached=None if a.statistics is None else np.load(a.statistics)
    operator=None if a.operator_statistics is None else np.load(a.operator_statistics)
    if operator is not None and str(operator['operator_fingerprint'])!=fingerprint:raise ValueError('Known operator changed; its Gram matrices cannot be reused')
    for c,b in enumerate(bands):
        check_indices=np.arange(len(points)) if cached is None else np.unique([0,len(points)//2,len(points)-1])
        h=np.stack([b.waveform(np.zeros(b.b.shape[1]),c,False,points[j]-b.reference_point) for j in check_indices],axis=2)*np.exp(1j*beta[c])
        inner=b.training&(np.floor((b.t-first)/2).astype(int)%2==0)
        masks=dict(inner=inner,validation=b.training&~inner,original=b.training,outer=~b.training)
        for name,mask in masks.items():
            observation=Observation.from_rows(h,b.field.y,b.field.root_information,mask,b.groups,
                gram=None if operator is None else operator[f'{name}_gram_{c}'])
            if cached is not None:
                stored=Observation(cached[f'{name}_gram_{c}'],cached[f'{name}_rhs_{c}'],float(cached[f'{name}_energy_{c}']))
                if stored.gram.shape!=(len(points),len(points)):raise ValueError('Cached dictionary size changed')
                np.testing.assert_allclose(stored.gram[np.ix_(check_indices,check_indices)],observation.gram,rtol=1e-10,atol=1e-8)
                np.testing.assert_allclose(stored.rhs[check_indices],observation.rhs,rtol=1e-10,atol=1e-8)
                np.testing.assert_allclose(stored.energy,observation.energy,rtol=1e-12,atol=0)
                observation=stored
            observations[name].append(observation)
        del h
        print(json.dumps(dict(stage='original-operator-statistics',channel=c,points=len(points),seconds=time.monotonic()-started)),flush=True)
    arrays={}
    for name,values in observations.items():
        for c,o in enumerate(values):arrays.update({f'{name}_gram_{c}':o.gram,f'{name}_rhs_{c}':o.rhs,f'{name}_energy_{c}':o.energy})
    arrays['operator_fingerprint']=np.array(fingerprint)
    if cached is None:np.savez_compressed(a.output/'statistics.npz',**arrays)
    else:cached.close()
    if operator is not None:operator.close()
    frozen_spectrum=np.ones(5)
    if a.spectrum_from:
        with np.load(a.spectrum_from/'scene.npz') as f:frozen_spectrum=f['spectrum_gains']
        if frozen_spectrum.shape!=(5,):raise ValueError('Upstream spectrum must retain all five RF identities')
    paths={name:FixedRidgePath(observations[name],frozen_spectrum) for name in ['inner','original']} if a.fixed_spectrum else None
    def solve(name,strength,initial=None):
        if paths is None:return fit(observations[name],strength,spectrum=initial,fit_band_phases=a.fit_band_phases,maximum_iterations=600)
        x=paths[name].solve(strength);g=frozen_spectrum
        gradient=paths[name].gram@x+strength*x-paths[name].rhs
        gradient/=sum(o.energy for o in observations[name])
        return x,g,dict(strength=float(strength),spectrum_fitted=False,numerical_convergence=bool(abs(gradient).max()<1e-8),
            maximum_scene_gradient=float(abs(gradient).max()),relative_band_phases_fitted=False,
            spectrum_magnitudes=abs(g).tolist(),constant_band_phase_correction_radians=np.angle(g).tolist(),scene_is_regularized_estimate=True)
    largest=float(paths['inner'].eigenvalue[-1]) if paths else float(np.linalg.eigvalsh(sum(o.gram for o in observations['inner']))[-1])
    if largest<=0:raise ValueError('No known-scene information in original training observations')
    trials=[];best=None
    for relative in a.regularization_relative:
        strength=largest*relative;x,g,report=solve('inner',strength)
        validation=score(observations['validation'],x,g);record=dict(relative_strength=relative,optimization=report,
            training=score(observations['inner'],x,g),inner_validation=validation)
        trials.append(record)
        if best is None or validation['cost']<best[0]:best=(validation['cost'],strength,g,relative)
        (a.output/'trials.json').write_text(json.dumps(dict(complete=False,scope=__doc__,trials=trials),indent=2)+'\n')
        print(json.dumps(dict(stage='blocked-regularization',relative_strength=relative,training=record['training']['explained_fraction'],validation=validation['explained_fraction'],spectrum=abs(g).tolist(),phase_correction_radians=np.angle(g).tolist(),seconds=time.monotonic()-started)),flush=True)
    _,strength,g,relative=best;x,g,report=solve('original',strength,g)
    result=dict(complete=True,scope=__doc__,motion=str(a.motion),points=len(points),radius_m=a.radius_m,spacing_m=a.spacing_m,
        relative_band_phases_fixed=not a.fit_band_phases,relative_strength=relative,strength=strength,trials=trials,
        original_training=score(observations['original'],x,g),outer=score(observations['outer'],x,g),optimization=report,
        regularization_grid_edge=relative in [a.regularization_relative[0],a.regularization_relative[-1]],seconds=time.monotonic()-started,
        selected_scene_is_physical_truth=False,statistics_reused=str(a.statistics),fixed_spectrum=a.fixed_spectrum,grid_bounds_m=a.bounds_m,
        grid_spacing_y_m=a.spacing_m if a.spacing_y_m is None else a.spacing_y_m,spectrum_source=str(a.spectrum_from),
        operator_fingerprint=fingerprint,operator_statistics_reused=str(a.operator_statistics),
        body_frame_hypothesis='frozen-at-aperture-midpoint' if a.freeze_body_attitude else 'upstream-motion',
        synthetic_control=truth,synthetic_true_support_given_to_estimator=False if truth and a.bounds_m else None)
    np.savez_compressed(a.output/'scene.npz',points_body_m=points,coefficient=x,spectrum_magnitudes=abs(g),spectrum_gains=g,band_phase_radians=beta+np.angle(g))
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(stage='final',outer=result['outer'],optimization=report,seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
