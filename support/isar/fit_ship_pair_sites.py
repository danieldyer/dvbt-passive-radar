"""All licensed site assignments with an explicit overlapping ship echo.

Two distinct primary/secondary associations seed each of the 32 assignments.
Coordinates retain the same user priors, bounds and nuisance terms. Training
cost chooses fits; unused samples are reported without refitting coefficients.
This single-interval comparison is not sufficient to accept a calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audit_ship_physical_pair import audit_fit
from fit_joint_sea_arcs import prepare_problem
from flash_geometry import Flash,assignments
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_pair import SharedPhysicalPair
from ship_site_signature import licensed_options


def model_for_assignment(case,index,assignment):
    old=case['geometries'][index]
    g=Flash(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),old.track,old.utc,old.rotation,old.receiver,assignment,
        receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.))
    g.lower[20],g.upper[20]=old.lower[20],old.upper[20]
    entry=case['descriptions'][index];midpoint=(entry['first_s']+entry['last_s'])/2
    bands=[PhysicalAmplitudeArc(f,tr,k,g,ch,midpoint) for ch,(f,tr,k) in
        enumerate(zip(case['fields'][index],case['training'][index],case['knots'][index]))]
    return SharedPhysicalPair(bands,origin_bounds=(-2.8,2.8))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/ship-physical-pair-F01-wider/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-physical-pair-sites-32'))
    parser.add_argument('--max-evaluations',type=int,default=300)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);output=args.output/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    source=json.loads(args.source.read_text());case=prepare_problem();i=next(i for i,d in enumerate(case['descriptions']) if d['label']=='F01')
    choices=assignments(licensed_options(case['geometry_directory']))
    if len(choices)!=32:raise ValueError('Expected exactly 32 licensed site assignments')
    seeds=[next(r for r in source['trials'] if r['start_rank']==rank) for rank in (1,2)]
    files=[args.source,Path(__file__),case['track_path'],case['geometry_directory']/'geometry-prior.json',
        case['geometry_directory']/'sites-main.json',case['geometry_directory']/'acma-transmitters.json']
    files += [Path(__file__).with_name(n+'.py') for n in ('physical_pair','amplitude_pair','physical_amplitude_arc',
        'flash_geometry','antenna_motion','stationarity','audit_ship_physical_pair')]
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,interval=case['descriptions'][i],
        assignments=32,starts_per_assignment=2,secondary_origin_bounds_samples=[-2.8,2.8],
        antenna_policy='Existing correlated site/individual antenna priors; receiver horizontal fixed and AGL 0–20 m.',
        validation_scope='Unused blocks in already inspected F01 data; no untouched target or time-interval validation.',
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files},trials=[])
    output.write_text(json.dumps(result,indent=2,allow_nan=False));started=time.perf_counter()
    for assignment in choices:
        code=''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in assignment)
        model=model_for_assignment(case,i,assignment);g=model.geometry
        for row in seeds:
            seed=np.asarray(row['parameters']);seed[:18]=0
            value=model.objective(seed)[0]
            fitted,first=model.fit(seed,free_antennas=True,max_evaluations=args.max_evaluations)
            fitted,audit=audit_fit(model,fitted,np.asarray(first['free_indices']))
            prediction=model.prediction(fitted,case['testing'][i]);sigma=g.nuisance_sigma
            bound=np.flatnonzero((fitted<model.lower+1e-4)|(fitted>model.upper-1e-4)).tolist()
            record=dict(assignment=code,initial_association=row['start_rank'],initial_objective=value,
                parameters=fitted.tolist(),first_order_fit=first,audit=audit,bound_indices=bound,
                cost=audit['polished_objective']*model.energy,data_energy=model.energy,
                prediction=prediction,primary_observables=g.prediction(fitted[:g.count]).tolist(),
                antenna_fit=g.antennas.describe(fitted[:18]),information=model.information(fitted),
                target_state=dict(position_correction_enu_m=(fitted[18:21]*sigma[:3]).tolist(),
                    velocity_correction_enu_mps=(fitted[21:24]*sigma[3:6]).tolist(),
                    acceleration_correction_enu_mps2=(fitted[24:27]*sigma[6:9]).tolist(),
                    ais_shift_s=float(fitted[27]*sigma[-1])))
            result['trials'].append(record);result['seconds']=time.perf_counter()-started
            output.write_text(json.dumps(result,indent=2,allow_nan=False))
            print(code,'association',row['start_rank'],'cost',round(record['cost'],3),'heldout',
                round(sum(p['heldout_error'] for p in prediction),3),'strict',audit['strict_local_minimum'],
                'bounds',bound,flush=True)
    result['complete']=True;output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
