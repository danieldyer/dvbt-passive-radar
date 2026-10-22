"""Run raw-data common-scattering-phase and carrier-cycle controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv
from cluster_phase import crop
from common_physical_arc import CommonPhysicalFit
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH
from imaging import sites
from physical_amplitude_arc import PhysicalAmplitudeArc,PhysicalAmplitudeFit
from ship_site_signature import licensed_options


def load_problem(record, intervals, geometry_directory=Path('docs/data/calibration_20261016_sites')):
    """Rebuild the exact raw likelihood from a completed independent fit."""
    interval=next(v for v in intervals['intervals'] if [v['first_s'],v['last_s']]==record['interval_s'])
    _,receiver,_=sites(geometry_directory)
    rotation=np.array(json.loads((geometry_directory/'geometry-prior.json').read_text())['ecef_to_enu'])
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    code=lambda a:''.join({'12013':'0','12015':'1','12014':'2'}[v['site_id']] for v in a)
    assignment=next(a for a in assignments(licensed_options(geometry_directory)) if code(a)==record['assignment'])
    midpoint=(interval['first_s']+interval['last_s'])/2
    geometry=Flash(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,EPOCH+midpoint,rotation,receiver,assignment,
        receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.))
    bands,testing=[],[]
    for ch,saved in enumerate(interval['channels']):
        fit=saved['alternatives'][saved['chosen_index']]
        with np.load(f'build/isar-20261015/full-fields/channel-{ch}.npz') as data:
            field=crop(data,interval['first_s'],interval['last_s'],np.arange(-2,3))
        bands.append(PhysicalAmplitudeArc(field,np.array(fit['test']['training']),fit['test']['amplitude_knots_s'],geometry,ch,midpoint))
        testing.append(np.array(fit['test']['testing']))
    return PhysicalAmplitudeFit(bands),testing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/ship-physical-arc-refined/fits.json'))
    parser.add_argument('--assignment',nargs='+')
    parser.add_argument('--branch-limit',type=int)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'fits.json').exists():raise ValueError('Use a fresh output')
    source=json.loads(args.source.read_text());intervals=json.loads(Path(source['independent_arc_source']).read_text())
    if not source['complete'] or not source['refine_antennas']:raise ValueError('A completed physical antenna fit is required')
    result={'scope':__doc__,'complete':False,'accepted_calibration':False,'independent_source':str(args.source),
        'interval_s':source['interval_s'],'trials':[],
        'phase_hypothesis':'Common scattering phase and calibrated relative instrumental phase assumed for this optimistic control. No per-band phase or rate correction; local carrier branches are not a globally exhaustive ambiguity search.'}
    start=time.perf_counter()
    for record in source['trials']:
        if args.assignment and record['assignment'] not in args.assignment:continue
        independent,testing=load_problem({**record,'interval_s':source['interval_s']},intervals)
        parameters=np.array(record['parameters']);common=CommonPhysicalFit(independent)
        if not np.isclose(independent.objective(parameters)[0],record['objective'],atol=1e-11,rtol=1e-10):
            raise ValueError('Rebuilt independent raw likelihood differs from saved fit')
        branches=common.phase_branches(parameters)
        for index,branch in enumerate(branches):
            branch['original_index']=index
            branch['initial_raw_objective']=common.objective(branch['parameters'])[0]
        branches.sort(key=lambda r:r['initial_raw_objective'])
        result.setdefault('phase_initialisations',[]).append({'assignment':record['assignment'],
            'branches':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in b.items()} for b in branches]})
        if args.branch_limit:branches=branches[:args.branch_limit]
        for index,branch in enumerate(branches):
            seed=branch['parameters'];value,gradient=common.objective(seed)
            if index==0:
                direction=np.sin(np.arange(len(seed))+.6)*.01;step=2e-5
                finite=(common.objective(seed+step*direction)[0]-common.objective(seed-step*direction)[0])/(2*step)
                relative=float(abs(finite-gradient@direction)/max(abs(finite),1e-8))
                if relative>1e-4:raise ValueError(f'Common physical gradient check failed: {relative}')
            fit=common.fit(seed);p=fit['parameters'];n=common.geometry.count
            lower,upper=common.geometry.bounds()
            fitted={**fit,'parameters':p.tolist(),'assignment':record['assignment'],'branch_index':branch['original_index'],
                'carrier_cycles':branch['carrier_cycles'],'phase_lift_cost':branch['phase_lift_cost'],
                'cost':fit['objective']*independent.energy,'independent_cost':record['cost'],
                'initial_objective':value,'prediction':common.prediction(p,testing),
                'bound_indices':np.flatnonzero((p[:n]<lower+1e-4)|(p[:n]>upper-1e-4)).tolist(),
                'antenna_fit':common.geometry.antennas.describe(p[:18]),
                'reflector_nuisance_standardised':p[18:n].tolist(),
                'predicted_observables':common.geometry.prediction(p[:n]).tolist(),
                'common_phase_rad':float(p[-1])}
            result['trials'].append(fitted);result['seconds']=time.perf_counter()-start
            (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))
            print(record['assignment'],index,'cost',round(fitted['cost'],2),'KKT',fit['maximum_scaled_kkt_residual'],
                'bounds',fitted['bound_indices'],'prediction',[round(v['test_complex_prediction_fraction'],3) for v in fitted['prediction']],flush=True)
    result['complete']=True
    sources=[args.source,Path(source['independent_arc_source']),Path(__file__),
        Path(__file__).with_name('common_physical_arc.py'),Path(__file__).with_name('physical_amplitude_arc.py'),Path(__file__).with_name('amplitude_arc.py')]
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sources}
    (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
