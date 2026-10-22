"""Freeze F01-derived antennas when testing the separate flash interval.

Target state, two echo phases/brightnesses and secondary trajectories are
estimated from flash training data. Antenna coordinates cannot move. This
tests transfer of a calibration hypothesis, rather than another free refit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audit_ship_physical_pair import audit_fit
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from ship_site_signature import licensed_options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=Path,default=Path('docs/data/isar_20261017/ship-physical-pair-sites-32/fits.json'))
    parser.add_argument('--target',type=Path,default=Path('docs/data/isar_20261017/ship-physical-pair-Flash/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-pair-frozen-transfer'))
    args=parser.parse_args();calibration=json.loads(args.calibration.read_text());target=json.loads(args.target.read_text())
    if not calibration['complete'] or not target['complete']:raise ValueError('Source comparisons must finish')
    args.output.mkdir(parents=True,exist_ok=True);output=args.output/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    best={code:min([r for r in calibration['trials'] if r['assignment']==code],key=lambda r:r['cost'])
        for code in {r['assignment'] for r in calibration['trials']}}
    ranked=sorted(best,key=lambda c:best[c]['cost'])
    interior=[c for c in ranked if best[c]['audit']['strict_local_minimum'] and not any(k<18 for k in best[c]['bound_indices'])]
    codes=list(dict.fromkeys([*ranked[:3],'00002','02002',*interior]))
    case=prepare_problem();index=next(i for i,e in enumerate(case['descriptions']) if e['label']=='Flash')
    choices=assignments(licensed_options(case['geometry_directory']))
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a for a in choices}
    seeds=[next(r for r in target['trials'] if r['start_rank']==k) for k in (1,2)]
    files=[args.calibration,args.target,Path(__file__),Path(__file__).with_name('physical_pair.py'),
        Path(__file__).with_name('fit_ship_pair_sites.py'),Path(__file__).with_name('audit_ship_physical_pair.py')]
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,assignments=codes,
        selection_policy='Three smallest F01 training objectives, historical 00002/02002, and every strict training-selected fit without antenna-bound contacts.',
        source_interval='F01',target_interval=case['descriptions'][index],starts_per_assignment=2,
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files},trials=[])
    started=time.perf_counter();output.write_text(json.dumps(result,indent=2,allow_nan=False))
    for code in codes:
        model=model_for_assignment(case,index,lookup[code]);g=model.geometry
        fixed=np.asarray(best[code]['parameters'])[:18]
        for original in seeds:
            seed=np.asarray(original['parameters']);seed[:18]=fixed
            fitted,first=model.fit(seed,free_antennas=False,max_evaluations=300)
            fitted,audit=audit_fit(model,fitted,np.asarray(first['free_indices']))
            if not np.array_equal(fitted[:18],fixed):raise ValueError('Frozen calibration changed')
            prediction=model.prediction(fitted,case['testing'][index]);sigma=g.nuisance_sigma
            record=dict(assignment=code,seed_rank=original['start_rank'],parameters=fitted.tolist(),
                first_order_fit=first,audit=audit,cost=audit['polished_objective']*model.energy,
                prediction=prediction,antennas_unchanged=True,
                bound_indices=np.flatnonzero((fitted<model.lower+1e-4)|(fitted>model.upper-1e-4)).tolist(),
                antenna_fit=g.antennas.describe(fitted[:18]),primary_observables=g.prediction(fitted[:28]).tolist(),
                position_correction_enu_m=(fitted[18:21]*sigma[:3]).tolist(),
                velocity_correction_enu_mps=(fitted[21:24]*sigma[3:6]).tolist(),
                acceleration_correction_enu_mps2=(fitted[24:27]*sigma[6:9]).tolist())
            result['trials'].append(record);result['seconds']=time.perf_counter()-started
            output.write_text(json.dumps(result,indent=2,allow_nan=False))
            print(code,'seed',original['start_rank'],'cost',round(record['cost'],3),'heldout',
                round(sum(p['heldout_error'] for p in prediction),3),'strict',audit['strict_local_minimum'],flush=True)
    result['complete']=True;output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
