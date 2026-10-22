"""Score frozen physical scene fits on the full original range field.

All motion, phase and amplitude estimates stay frozen. A fit made in a
restricted range patch must transfer here before an increase in its local
explained-energy percentage is interpreted as an improvement.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import BSpline
from full_scene_context import open_scene,prediction
from fixed_scene_reconstruction import score_scene


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[];reference=None
    for path in a.motion:
        fit,_,scene,saved=open_scene(path,full_field=True);bands=[s.point for s in scene]
        raw=SimpleNamespace(model=[prediction(s,saved,fit,c) for c,s in enumerate(scene)])
        training=score_scene(raw,bands,[b.training for b in bands]);outer=score_scene(raw,bands,[~b.training for b in bands])
        curve=BSpline(fit['path_knots_s'],fit['path_coefficients_m'],3,extrapolate=False)
        if reference is None:reference=curve
        time=bands[3].t;delta=curve(time)-reference(time)
        records.append(dict(motion=str(path),fit_range_offsets=fit.get('fit_range_offsets'),training=training,outer=outer,
            maximum_path_change_from_first_m=float(max(abs(delta))),path_rms_change_from_first_m=float(np.sqrt(np.mean(delta**2))),
            numerical_convergence=fit['numerical_convergence'],physical_kkt_residual=fit.get('maximum_kkt_residual')))
    result=dict(complete=True,scope=__doc__,models=records)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
