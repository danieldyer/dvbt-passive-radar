"""Evaluate an optimizer checkpoint against its unchanged original scene.

Refuse a mismatched reference objective. Report physical constraint KKT,
including unrestricted phase components, rather than interpreting a raw
gradient from a constrained optimizer as its stationarity certificate.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from full_path_scene import ScenePath
from path_dynamics import constraint,certificate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maximum-acceleration',type=float,default=.15);p.add_argument('--maximum-correction',type=float,default=120.)
    a=p.parse_args();started=time.monotonic();raw=a.checkpoint.read_bytes();saved=json.loads(raw)
    reference,_,bands,_=open_scene(a.reference);likelihood=ScenePath(bands);x=np.asarray(saved['parameters'])
    value,g=likelihood.objective(x);expected=saved['reports'][-1]['objective']
    if abs(value-expected)>1e-7:raise ValueError(f'Reference operator differs from checkpoint: {value}, expected {expected}')
    count=sum((np.count_nonzero(b.point.training&(b.point.field.root_information>0))-len(np.unique(b.point.groups[b.point.training&(b.point.field.root_information>0)])))*len(b.point.field.offsets) for b in bands)
    scale=likelihood.energy/count
    bound=constraint(reference['path_knots_s'],reference['path_coefficients_m'],len(x),a.maximum_acceleration,a.maximum_correction,reference.get('anchor_path_time_s',162.))
    report=dict(scope=__doc__,reference=str(a.reference),checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint_iteration=saved['reports'][-1]['iteration'],objective=value,expected_objective=expected,optimizer_likelihood_scale=scale,
        raw_path_gradient_maximum=float(max(abs(g[:likelihood.n]))),raw_band_phase_gradient_maximum=float(max(abs(g[likelihood.n:likelihood.n+5]))),
        raw_source_phase_gradient_maximum=float(max(abs(g[likelihood.n+5:]))),seconds=time.monotonic()-started,
        **certificate(x,g*scale,bound))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
