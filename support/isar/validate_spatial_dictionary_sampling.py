"""Exact off-grid representation check for a shared spatial dictionary.

This supplies a local dictionary neighbourhood and tests only its ability to
represent a point between mesh nodes. It is neither blind imaging nor motion
recovery. Every band retains its original known pulse, epochs and fixed RF
calibration; no independent phase/gain alignment is applied.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from full_scene_context import open_scene
from regularized_scene import Observation
from bistatic_path_lift import lift
from motion import excess_path
from rigid_patch import profile_static


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--spacing-m',type=float,default=3.);p.add_argument('--radius-cells',type=int,default=3)
    p.add_argument('--spacing-y-m',type=float,help='Optional anisotropic body-y spacing')
    p.add_argument('--waveform-bandwidth-hz',type=float,default=8e6,help='Conservative complex baseband width, including waveform sidebands')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    fit,_,scene,_=open_scene(a.motion);bands=[s.point for s in scene];d=a.spacing_m;dy=d if a.spacing_y_m is None else a.spacing_y_m
    origin=np.round(np.asarray(fit['point_body_m'])[:2]/[d,dy])*[d,dy]
    x,y=np.meshgrid(np.arange(-a.radius_cells,a.radius_cells+1)*d,np.arange(-a.radius_cells,a.radius_cells+1)*dy)
    pixels=np.c_[x.ravel()+origin[0],y.ravel()+origin[1],np.full(x.size,20.)]
    truths=np.array([[origin[0]+fx*d,origin[1]+fy*dy,20.] for fx,fy in [(.17,.31),(.49,-.43),(-.38,.07)]])
    q=[];occupied=[];design=[];targets=[]
    for c,b in enumerate(bands):
        s=lift(b.anchor,b.velocity,b.direction,b.direction_rate,b.base_path,b.base_rate,b.motion.tx[3],b.motion.rx)
        _,g=excess_path(s['position_ecef_m'],b.motion.tx[c],b.motion.rx)
        wavevector=b.field.k*np.einsum('tji,tj->ti',b.frame,g);q.append(wavevector)
        fractional=a.waveform_bandwidth_hz/(2*b.field.metadata['rf_hz'])
        occupied.extend([wavevector*(1-fractional),wavevector*(1+fractional)])
        design.append(np.stack([b.waveform(np.zeros(b.b.shape[1]),c,False,p-b.reference_point) for p in pixels],axis=2))
        targets.append([b.waveform(np.zeros(b.b.shape[1]),c,False,p-b.reference_point) for p in truths])
    carrier_span=np.ptp(np.concatenate(q),axis=0);span=np.ptp(np.concatenate(occupied),axis=0)
    limit=np.divide(2*np.pi,span,out=np.full(3,np.inf),where=span>0)
    # Complex coefficients absorb one spatial carrier; the remaining
    # tangent-plane frequency span determines the mesh's Nyquist interval.
    records=[]
    for j,truth in enumerate(truths):
        gram=0.;rhs=0.;energy=0.
        for b,h,ys in zip(bands,design,targets):
            o=Observation.from_rows(h,ys[j],b.field.root_information,b.training,b.groups)
            gram=gram+o.gram;rhs=rhs+o.rhs;energy+=o.energy
        eigen,vector=eigh(gram);strength=eigen[-1]*1e-10
        coefficient=vector@((vector.conj().T@rhs)/(eigen+strength))
        errors=[];phase=[]
        for b,h,ys in zip(bands,design,targets):
            mask=~b.training;predicted=profile_static(h@coefficient,b.field.root_information,mask,b.groups)[mask]
            expected=profile_static(ys[j],b.field.root_information,mask,b.groups)[mask]
            error=float(np.vdot(predicted-expected,predicted-expected).real/np.vdot(expected,expected).real)
            weight=abs(expected)**2;delta=np.angle(predicted*expected.conj())
            errors.append(error);phase.append(float(np.degrees(np.sqrt(np.sum(weight*delta**2)/weight.sum()))))
        records.append(dict(true_point_body_m=truth.tolist(),excluded_relative_complex_squared_error=errors,
            excluded_information_weighted_phase_rms_degrees=phase,explicit_ridge_relative=1e-10))
    result=dict(complete=True,scope=__doc__,motion=str(a.motion),spacing_m=d,spacing_y_m=dy,local_dictionary_points=len(pixels),
        carrier_only_tangent_wavevector_span_radpm=carrier_span.tolist(),waveform_bandwidth_hz=a.waveform_bandwidth_hz,
        tangent_wavevector_span_radpm=span.tolist(),tangent_mesh_nyquist_interval_m=limit.tolist(),
        horizontal_tangent_nyquist_satisfied=bool(np.all(np.array([d,dy])<limit[:2])),checks=records,
        largest_excluded_relative_complex_squared_error=max(max(r['excluded_relative_complex_squared_error']) for r in records),
        largest_excluded_phase_rms_degrees=max(max(r['excluded_information_weighted_phase_rms_degrees']) for r in records))
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
