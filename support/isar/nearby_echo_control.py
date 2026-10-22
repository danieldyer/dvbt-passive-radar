"""Rebuild the completed nearby-feature controls on immutable observations."""
import json
from pathlib import Path

import numpy as np

from audit_joint_kernel_phase import known_band
from cluster_phase import SplinePath
from compare_ais_phase import NAMES,path_geometry
from fit_fixed_grid_phases import prepare_fixed
from report_measured_scatterer_phases import load_fields
from smooth_echo_cluster import SmoothEchoCluster
from shared_echo_cluster import ReferencedAmplitudeArc
from ship_site_signature import licensed_options
from waveform_kernels import read_kernels


ROOT=Path('docs/data/isar_20261017')


def load():
    folders=[f'main-return-extra-{n}-guided' for n in NAMES[:4]]+['main-return-extra-ABC-flexible-wide']
    sources=[ROOT/f/'fits.json' for f in folders];saved=[json.loads(p.read_text()) for p in sources]
    if any(not d['complete'] for d in saved):raise ValueError('Wait for all independent controls')
    reference=json.loads(Path(saved[0]['source']).read_text())
    first,last=reference['interval_s'];std=reference['validation_selected_jerk_std'];origins=saved[0]['origins_samples']
    for c,d in enumerate(saved):
        if (d['channel']!=c or d['interval_s']!=[first,last] or d['extra_curve']!='flexible' or
                d['origins_samples']!=origins or d['source']!=saved[0]['source']):
            raise ValueError('Same ordered observations, supports and flexible basis required')
    geometry=Path('docs/data/calibration_20261016_sites')
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    mean,parts,_=prepare_fixed(fields,geometry,first,last,stationary=True)
    nominal=[]
    for f,b in zip(fields,mean):
        rows=b.field.data['row_time'];indices=np.searchsorted(f['data']['row_time'],rows)
        np.testing.assert_array_equal(f['data']['row_time'][indices],rows)
        nominal.append(dict(position=f['data']['centre_ecef_m'][indices].copy(),
            velocity=f['data']['centre_velocity_ecef_mps'][indices].copy()))
    kernel=Path('build/isar-20261016/waveform-kernels-30s');_,records,pulses=read_kernels(kernel/'response.kernels')
    with np.load(kernel/'rows.npz') as grids:
        bands=[known_band(b,grids[f'channel_{c}'],records[c],pulses[c]) for c,b in enumerate(mean)]
    curve=SplinePath(first,last,reference['knot_spacing_s']);models=[];errors=[]
    for c,(b,d) in enumerate(zip(bands,saved)):
        m=SmoothEchoCluster([b],origins,(1,3,4),curve,shared=False,jerk_std=std)
        m.lower[8],m.upper[8]=d['extra_origin_bounds_samples']
        for f in d['trials']:
            p=np.array(f['parameters']);r,_=m.evaluate(p);errors.append(abs(r@r*m.energy-f['cost']))
        models.append(m)
    if max(errors)>1e-7:raise ValueError('Independent original-row objectives changed')
    return dict(sources=sources,saved=saved,bands=bands,parts=parts,curve=curve,origins=origins,
        jerk_std=std,knot_spacing_s=reference['knot_spacing_s'],independent_models=models,reproduced_objectives=len(errors),maximum_absolute_cost_error=max(errors),
        kernel=kernel,nominal=nominal,geometry=geometry)


def known_increment(position,velocity,transmitter,reference,receiver,time,epoch,observer_rate):
    """Change model geometry while leaving the captured C0/C1 reference fixed."""
    path,rate=path_geometry(position,velocity,transmitter,receiver)
    ref,ref_rate=path_geometry(position,velocity,reference,receiver)
    known=path-ref;known-=np.interp(epoch,time,known)
    return known,rate-ref_rate-observer_rate


def project_reference_change(curve,time,path_difference):
    """Represent a reference-path change without changing physical phases.

    The removed constant is a carrier gauge. Projection error is returned so
    callers cannot silently substitute an inaccurate new reference path.
    """
    time=np.asarray(time);delta=np.asarray(path_difference)
    if time.ndim!=1 or delta.shape!=time.shape or np.any(np.diff(time)<=0):
        raise ValueError('Ordered reference path samples required')
    centred=delta-np.interp(curve.epoch,time,delta)
    coefficient=np.linalg.lstsq(curve(time),centred,rcond=None)[0]
    error=float(np.max(abs(curve(time)@coefficient-centred)))
    return coefficient,error


def shared_model(control,assignment='00002',components=(1,4)):
    """Licensed-site hypothesis in the target model, never in the observations.

    A/C/D nuisance trajectories keep their original coordinate convention.
    Only selected target arcs receive the trial interband increment. Their
    original field, noise, averaging and C0/C1 combination remain immutable.
    """
    model=SmoothEchoCluster(control['bands'],control['origins'],(1,3,4),control['curve'],
        shared_components=components,jerk_std=control['jerk_std'])
    options=licensed_options(control['geometry']);codes={'0':'12013','1':'12015','2':'12014'}
    if len(assignment)!=5:raise ValueError('Five ordered licensed site codes required')
    selected=[]
    for c,code in enumerate(assignment):
        matches=[o for o in options if o['channel']==c and o['site_id']==codes.get(code)]
        if len(matches)!=1:raise ValueError('Unlicensed channel/site assignment')
        selected.append(np.asarray(matches[0]['transmitter_ecef_m']))
    receiver=np.array(json.loads((control['geometry']/'geometry-prior.json').read_text())['receiver_ecef_m'])
    for c,(b,cluster,state) in enumerate(zip(control['bands'],model.clusters,control['nominal'])):
        observer=-b.field.data['centre_omega'][0]/(b.field.k*.001024)
        path,rate=known_increment(state['position'],state['velocity'],selected[c],selected[2],receiver,
            b.field.data['row_time'],control['curve'].epoch,observer)
        arc=ReferencedAmplitudeArc(b.field,b.training,b.knots,control['curve'].epoch-control['curve'].scale,
            control['curve'].epoch+control['curve'].scale,path,rate,stationary_design=b.design,curve=control['curve'])
        for s in components:cluster.arcs[s]=arc
        for s in components:
            model.lower[10*c+2*s],model.upper[10*c+2*s]=(-2.5,2.5)
    return model


def independent_seed(model,control):
    """Map independent local coordinates into another explicit sharing layout."""
    p=np.zeros(model.count);n=model.components
    for c,(d,old) in enumerate(zip(control['saved'],control['independent_models'])):
        q=np.array(d['trials'][d['best_index']]['parameters'])
        p[2*n*c:2*n*(c+1)]=q[:2*n]
        for s,index in enumerate(model.local_path_indices[c]):
            if s not in model.shared_selected_components:p[index]=q[old.local_path_indices[0][s]]
    return p


def phase_seed(model,p,control,components=(1,4)):
    """Training-only complex phase acquisition; all later phases remain free."""
    p=np.asarray(p).copy();n=model.components
    for c,(d,old,cluster,index) in enumerate(zip(control['saved'],control['independent_models'],model.clusters,model.indices)):
        q=np.array(d['trials'][d['best_index']]['parameters']);b=cluster.base
        v=old.clusters[0].evaluate(q[old.indices[0]]);k=len(b.knots)
        others=sum((v['design'][:,s*k:(s+1)*k]@v['amplitude'][s*k:(s+1)*k]
            for s in range(n) if s not in components),start=np.zeros(b.field.y.size,complex))
        residual=b.clean-b.nuisance.remove(others[b.selected])
        phases=[2*n*c+2*s+1 for s in components];p[phases]=0.
        response=cluster.response_blocks(p[index])
        h=np.column_stack([b.nuisance.remove(b.reduced(response[s][0][:,:,None])[:,0][b.selected]) for s in components])
        coefficient=np.linalg.lstsq(h,residual,rcond=None)[0];p[phases]=np.angle(coefficient)
    return p
