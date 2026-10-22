"""Geometry sensitivity of the central echo measured by the new scene fits.

Use independently fitted envelope/rate/curvature summaries and their complete
conditional component covariance. Compare all licensed site assignments under
fixed and refined antenna priors. This screen must agree across scene models
before any location result can be adopted; thermal covariance alone does not
describe model mismatch or establish a conclusive survey.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH
from imaging import sites
from ship_site_signature import licensed_options


def main():
    root=Path('docs/data/isar_20261017');out=root/'trajectory-geometry-screen';out.mkdir(exist_ok=True)
    geometry=Path('docs/data/calibration_20261016_sites');_,receiver,_=sites(geometry)
    rotation=np.array(json.loads((geometry/'geometry-prior.json').read_text())['ecef_to_enu'])
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    options=assignments(licensed_options(geometry))
    water_path=Path('docs/data/isar_20261016/flash-reflection/analysis.json')
    water=json.loads(water_path.read_text())['water'];centre=np.asarray(water['centre_ecef_m']);radius=water['radius_m']
    code=lambda a:''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a)
    nominal=next(a for a in options if code(a)=='00002');cases=[];sources=[Path(__file__),water_path]
    for directory in ('shared-echo-cluster','shared-echo-cluster-context','shared-echo-cluster-split'):
        path=root/directory/'analysis.json';sources.append(path);analysis=json.loads(path.read_text())
        midpoint=np.mean(analysis['interval_s']);s=analysis['labels'].index('B');indices=np.arange(3*s,3*s+3)
        reference=Flash(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,EPOCH+midpoint,rotation,receiver,nominal,free_antennas=False)
        predicted=reference.prediction(np.zeros(reference.count));measured=[];covariance=[]
        for ch,record in enumerate(analysis['independent_conditional_observables']):
            if not record['identifiable']:raise ValueError('Cannot use a singular conditional covariance')
            observed=np.array(record['range_origin_m_rate_residual_mps_curvature_residual_mps2'])[s]
            with np.load(Path(f'build/isar-20261015/full-fields/channel-{ch}.npz')) as f:
                old=float(np.interp(midpoint,f['row_time'],f['centre_path_m']))
            measured.append([old+observed[0],predicted[ch,1]+observed[1],predicted[ch,2]+observed[2]])
            covariance.append(np.array(record['covariance'])[np.ix_(indices,indices)])
        measured=np.array(measured);covariance=np.array(covariance)
        case=dict(scene=directory,point='B',midpoint_s=midpoint,measured=measured.tolist(),covariance=covariance.tolist(),records=[])
        cases.append(case)
        for assignment in options:
            record=dict(assignment=code(assignment))
            for free in (False,True):
                problem=Flash(measured,covariance,track,EPOCH+midpoint,rotation,receiver,assignment,free_antennas=free,
                    receiver_sigma=(0.,0.,7.5),**(dict(receiver_up_bounds_m=(-10.,10.)) if free else {}))
                position=problem.receiver+problem.state(np.zeros(problem.count))[0]@rotation
                radial=(position-centre)/np.linalg.norm(position-centre)
                height=np.linalg.norm(position-centre)-radius;sensitivity=20*np.dot(radial,rotation[2])
                vertical=problem.nantenna+2
                problem.lower[vertical]=max(problem.lower[vertical],(8-height)/sensitivity)
                problem.upper[vertical]=min(problem.upper[vertical],(57-height)/sensitivity)
                seed=np.clip(np.zeros(problem.count),problem.lower+1e-9,problem.upper-1e-9)
                fit=problem.fit(seed);described=problem.describe(fit)
                record['refined' if free else 'fixed']=described
            case['records'].append(record)
        ranked=sorted(case['records'],key=lambda r:r['refined']['penalized_squared_residual'])
        case['best_assignment']=ranked[0]['assignment']
        print(directory,'best',[(r['assignment'],round(r['refined']['penalized_squared_residual'],2),r['refined']['bound_parameter_indices']) for r in ranked[:3]],flush=True)
    fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
    codes=sorted([code(a) for a in options])
    for case in cases:
        lookup={r['assignment']:r for r in case['records']}
        for ax,key in zip(axes,('fixed','refined')):
            cost=np.array([lookup[c][key]['penalized_squared_residual'] for c in codes])
            ax.plot(np.arange(len(codes)),cost-cost.min(),'.-',label=case['scene'])
            ax.set_xticks(range(len(codes)),codes,rotation=90,fontsize=7)
            ax.set(ylabel='Penalized conditional cost above that scene’s best',xlabel='Licensed assignment: Seven / SBS / Nine / Ten / ABC',title='Antennas '+key)
    axes[0].legend(fontsize=8)
    fig.suptitle('Does the new central-echo location inference survive scene changes?\n'
        '0 = Ornata, 1 = Eyre, 2 = Ridge. Scores are conditional sensitivity measures, not site probabilities.',fontsize=13)
    fig.savefig(out/'01-assignment-sensitivity.png',dpi=170);plt.close(fig)
    rows=''
    for case in cases:
        for r in sorted(case['records'],key=lambda r:r['refined']['penalized_squared_residual'])[:5]:
            v=r['refined'];rows+=f'<tr><td>{case["scene"]}</td><td>{r["assignment"]}</td><td>{v["data_squared_residual"]:.2f}</td><td>{v["prior_squared_residual"]:.2f}</td><td>{v["first_order_optimality"]:.2g}</td><td>{v["bound_parameter_indices"]}</td></tr>'
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Central echo geometry sensitivity</title>
<style>body{{font:16px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:7px;border:1px solid #ddd}}table{{border-collapse:collapse}}</style>
<h1>Central-echo geometry after resolving nearby returns</h1>
<p>This screen uses the independently fitted B envelope, phase rate and curvature from the new 17-cell three-component, 33-cell four-component and 33-cell split-cluster models. Each band’s full conditional component covariance is marginalized to B without taking a pseudoinverse. All 32 licensed assignments are fit under fixed and refined antenna priors, with the approved receiver horizontal coordinates fixed and height allowed ±10 m around the supplied value. The reflector midpoint height remains 8–57 m above the existing local water datum; this is a height constraint only, with no reflected propagation model.</p>
<p>The covariance assumes the fitted component model and thermal range noise. It omits coherent model discrepancy and association error, so the scores are not site probabilities or calibrated confidence intervals. The scene comparison is essential: a location result is not accepted merely because one model converges.</p>
<img src="01-assignment-sensitivity.png" alt="Site-assignment sensitivity to the surrounding scene model">
<table><tr><th>Scene</th><th>Assignment</th><th>Data cost</th><th>Prior cost</th><th>Optimality</th><th>Bounds</th></tr>{rows}</table>
<p><a href="fits.json">Every assignment, covariance and parameter fit</a> · <a href="../shared-echo-cluster-context/index.html">Measured component trajectories</a> · <a href="../index.html">Analysis index</a></p></html>''')
    result=dict(scope=__doc__,accepted_calibration=False,cases=cases,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(out/'index.html',flush=True)


if __name__=='__main__':main()
