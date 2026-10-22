"""Numerical/model gate for a common finite surface before antenna calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from compare_ais_phase import NAMES
from facet_geometry import FacetFlash
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH
from imaging import sites
from ship_site_signature import licensed_options


def publish(records,output):
    if not records:return
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for row in records:
        residual=np.array(row['fit']['predicted_observables'])-row['measurements']
        code='/'.join({'12013':'O','12015':'E','12014':'R'}[s] for s in row['sites'])
        label=f"{code}, start {row['start']}"
        for j,ax in enumerate(axes):
            ax.plot(np.arange(5),residual[:,j],'.-',label=label)
    for ax,title in zip(axes,['Envelope path residual (m)','Path-rate residual (m/s)','Path-curvature residual (m/s²)']):
        ax.set_xticks(np.arange(5),NAMES);ax.axhline(0,color='black',lw=.6);ax.set_ylabel(title)
    axes[0].legend(fontsize=7)
    fig.suptitle('Shared finite-side summary fit · preliminary model test\nOne surface and one trajectory; absolute interband carrier phase not fitted; no accepted antenna calibration')
    fig.savefig(output/'residuals.png',dpi=170);plt.close(fig)
    (output/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    rows=''
    for r in records:
        f=r['fit'];code='/'.join({'12013':'O','12015':'E','12014':'R'}[s] for s in r['sites'])
        rows+=f"<tr><td>{code}</td><td>{r['start']}</td><td>{f['data_squared_residual']:.2f}</td><td>{f['prior_squared_residual']:.2f}</td><td>{f['heading_deg']:.3f}°</td><td>{f['turn_rate_deg_s']:.4f}°/s</td><td>{f['first_order_optimality']:.3g}</td><td>{f['bound_parameter_indices']}</td></tr>"
    (output/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Finite-side joint-fit diagnostic</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px}}</style>
<h1>One finite side and one multiband motion: preliminary fit</h1><p>This is a model/numerical gate, not an accepted transmitter calibration. A finite vertical uniform-current PEC rectangle predicts envelope group path, phase rate and phase curvature across five frequencies. Receiver horizontal coordinates are fixed; the existing bounded receiver-height and transmitter priors are retained. One centre trajectory and a slowly turning surface replace independent per-band trajectory corrections. Heading describes this hypothesized facet; it does not rotate image pixels.</p>
<p>The inputs are the conditional local point-fit summaries and joint covariances from 164–165 s. An extended surface must eventually predict the original complex range vectors and unused samples, including amplitude, before those summaries can be trusted as its likelihood. Absolute interband carrier phases have not been included in this diagnostic. Only the listed assignments and starts have been tested; this is not an exhaustive site/shape search.</p>
<img src="residuals.png" alt="Residuals of common finite-surface fits">
<table><tr><th>Seven/SBS/Nine/Ten/ABC</th><th>Start</th><th>Data residual²</th><th>Prior residual²</th><th>Side heading</th><th>Turn rate</th><th>First-order optimality</th><th>Parameter bounds</th></tr>{rows}</table>
<p>O=Ornata, E=Eyre, R=Ridge. A small change in solver cost or a termination message is not sufficient numerical convergence; optimality, repeated starts and physical prediction all matter. Path, spreading and HH surface-current time derivatives are evaluated analytically with second-order Taylor arithmetic and checked against an independent temporal phase calculation. The parameter Jacobian uses absolute central-difference steps so near-zero coordinates remain resolved.</p>
<p><a href="summary.json">Fit records and source hashes</a>; <a href="../../isar_20261016/flash-reflection/index.html">physical geometry, sea-path scales and finite-facet analysis</a>.</p></html>''')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261016_sites'))
    p.add_argument('--point-fit',type=Path,default=Path('docs/data/isar_20261016/flash-siting-sites-updated/flash-siting.json'))
    p.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/facet-siting'))
    p.add_argument('--indices',type=int,nargs='+',default=[24])
    p.add_argument('--length-m',type=float,default=286.53)
    p.add_argument('--height-m',type=float,default=30.)
    p.add_argument('--reuse',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.point_fit.read_text())
    case=next(c for c in source['cases'] if c['width_cells']==5 and c['include_range'] and c['include_acceleration'])
    options=assignments(licensed_options(args.geometry))
    _,rx,_=sites(args.geometry)
    rotation=np.array(json.loads((args.geometry/'geometry-prior.json').read_text())['ecef_to_enu'])
    track=read_csv('docs/data/isar_20261013/rio-delta-ais.csv',reference_height_ellipsoid_m=5.)
    frequency=np.array([177.5,184.5,191.625,219.5,226.5])*1e6
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [args.point_fit,args.geometry/'geometry-prior.json',Path(__file__),Path(__file__).with_name('facet_geometry.py'),
         Path(__file__).with_name('facet_kinematics.py'),Path(__file__).with_name('specular_facet.py')]}
    records=[]
    for index in args.indices:
        point=next(r for r in case['records'] if r['index']==index)
        base=Flash(case['measurements'],case['covariance'],track,EPOCH+164.5,rotation,rx,options[index],**source['fit_options'])
        problem=FacetFlash(base,frequency,length_m=args.length_m,height_m=args.height_m)
        origins=[('nominal prior',np.zeros(base.count)),('point-fit geometry',np.array(point['parameters']))]
        seeds=[(label,problem.seed(origin,h,w)) for label,origin in origins
            for h,w in [(-.25,.1),(.25,-.1),(0.,.2)]]
        for i,(label,seed) in enumerate(seeds):
            path=args.output/f'assignment-{index}-start-{i}.json'
            if path.exists() and args.reuse:
                record=json.loads(path.read_text())
                if record['source_sha256']!=hashes or record['dimensions_m']!=[args.length_m,args.height_m]:
                    raise ValueError('Changed model/input: use a new output directory')
            else:
                if path.exists():raise ValueError(f'Result exists: {path}')
                started=time.perf_counter()
                fit,trials=problem.fit([seed])
                record={'index':index,'sites':point['sites'],'start':i,'dimensions_m':[args.length_m,args.height_m],
                    'seed_geometry':label,'initial_parameters':seed.tolist(),
                    'heading_seed':'Actual trial-geometry mean bistatic bisector, plus the declared side offset',
                    'source_sha256':hashes,'measurements':case['measurements'],'covariance':case['covariance'],
                    'fit':problem.describe(fit),'elapsed_s':time.perf_counter()-started,'accepted_calibration':False}
                path.write_text(json.dumps(record,indent=2,allow_nan=False))
            records.append(record);publish(records,args.output)
            f=record['fit']
            print(index,i,'data',round(f['data_squared_residual'],3),'prior',round(f['prior_squared_residual'],3),
                'optimality',f['first_order_optimality'],'bounds',f['bound_parameter_indices'],'s',round(record['elapsed_s'],1),flush=True)


if __name__=='__main__':main()
