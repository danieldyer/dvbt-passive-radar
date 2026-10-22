"""Validate a reversible range-isolation operator and its information cost.

The exact sub-covariance whitener is used, not empirical cell normalization.
Noise isometry, original measured-data transfer, original waveform phase and
conditional point information are checked before a local autofocus control.
Native image formation continues to retain every recorded range cell.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from full_ship_field import select_range
from rigid_patch import profile_static


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--range-offsets',type=int,nargs=2,default=[-16,-5]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();fit,_,scene,_=open_scene(a.motion);records=[]
    for c,s in enumerate(scene):
        b=s.point;view=copy.copy(b);view.field=select_range(b.field,*a.range_offsets);f=b.field;small=view.field
        keep=np.flatnonzero(np.isin(f.offsets,small.offsets));operator=small.whiten@np.eye(len(f.offsets))[keep]@np.linalg.inv(f.whiten)
        isometry=float(np.max(abs(operator@operator.conj().T-np.eye(len(keep)))))
        measured_error=float(np.linalg.norm(small.y-f.y@operator.T)/np.linalg.norm(small.y))
        points=np.asarray(fit['scene_points_body_m']);norms=[];waveform_error=0.
        for point in points:
            h=b.waveform(np.zeros(b.b.shape[1]),c,False,point-b.reference_point)
            g=view.waveform(np.zeros(b.b.shape[1]),c,False,point-b.reference_point)
            waveform_error=max(waveform_error,float(np.linalg.norm(g-h@operator.T)/np.linalg.norm(g)))
            original=profile_static(h,f.root_information,b.training,b.groups)[b.training]
            restricted=profile_static(g,small.root_information,b.training,b.groups)[b.training]
            ratio=float(np.vdot(restricted,restricted).real/np.vdot(original,original).real)
            if ratio>1+1e-8:raise ValueError('Range restriction created information under the stated covariance')
            norms.append(ratio)
        if isometry>1e-10 or measured_error>1e-10 or waveform_error>1e-10:
            raise ValueError(f'Range isolation changed its declared linear operator: {isometry}, {measured_error}, {waveform_error}')
        records.append(dict(channel=c,noise_isometry_error=isometry,measured_relative_complex_error=measured_error,
            waveform_relative_complex_error=waveform_error,point_information_retained=norms,
            minimum_point_information_retained=min(norms),maximum_conditional_point_loss_db=float(-10*np.log10(min(norms)))))
    result=dict(passed=True,scope=__doc__,motion=str(a.motion),range_offsets=a.range_offsets,channels=records,
        independently_fitted_phase_or_scale=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
