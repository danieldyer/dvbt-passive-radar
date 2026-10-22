"""Compare gated waveform views to full-record evaluation without alignment."""
import argparse,json,time
from pathlib import Path
import numpy as np
from full_scene_context import open_scene


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval',type=float,nargs=2,default=[153.2720443125,183.2720443125])
    a=p.parse_args();fit,motion,bands,saved=open_scene(a.motion);records=[];started=time.monotonic()
    for c,band in enumerate(bands):
        b=band.point;first,last=np.searchsorted(b.t,a.interval)
        view=b.waveform_window(first,last)
        for offset in [np.zeros(3),np.array([70.,-10.,0.]),np.array([-70.,20.,0.])]:
            # Test the declared source only within this aspect support.
            # A constant off-axis point over the whole recording can leave
            # the retained pulse crop elsewhere; that is not a windowing error.
            offsets=np.zeros((len(b.t),3));offsets[first:last]=offset
            full=b.waveform(np.zeros(b.b.shape[1]),c,True,offsets,True)
            selected=view.waveform(np.zeros(b.b.shape[1]),c,True,offset,True)
            errors=[float(np.linalg.norm(x[first:last]-y)/max(np.linalg.norm(x[first:last]),1e-300)) for x,y in zip(full,selected)]
            h=full[0][first:last];z=selected[0];weight=abs(h)**2;phase=np.angle(h.conj()*z)
            rms=float(np.degrees(np.sqrt(np.sum(weight*phase**2)/weight.sum())))
            phase_origin=float(np.degrees(np.angle(np.vdot(h,z))))
            record=dict(channel=c,point_offset_body_m=offset.tolist(),waveform_and_derivative_relative_errors=errors,
                original_phase_origin_error_degrees=phase_origin,information_weighted_phase_rms_degrees=rms)
            if max(errors)>1e-6 or rms>1e-4:raise ValueError(f'Original-epoch waveform window changed the physical operator: {record}')
            records.append(record);print(json.dumps(record),flush=True)
    a.output.write_text(json.dumps(dict(complete=True,passed=True,scope=__doc__,motion=str(a.motion),interval_s=a.interval,
        phase_alignment_fitted=False,checks=records,seconds=time.monotonic()-started),indent=2)+'\n')


if __name__=='__main__':main()
