"""Test a common physical primary phase after resolving overlapping returns.

The independent-band-phase control is retained, as are explicit carrier-cycle
starts. The second echo remains independent in each band. Finally the best
common fit is relaxed back to independent phases, guarding against a misleading
comparison caused by a different local minimum of the larger model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audit_ship_physical_pair import audit_fit
from common_physical_pair import CommonPhysicalPair,antenna_information
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from ship_site_signature import licensed_options


def describe(model,parameters,first,audit,testing,*,common):
    g=model.geometry;n=g.count
    prediction=model.prediction(parameters,testing)
    phase=[n] if common else np.arange(n,n+5)
    secondary=np.arange(n+(1 if common else 5),model.count)
    return dict(parameters=parameters.tolist(),cost=model.objective(parameters)[0]*model.energy,
        data_energy=model.energy,first_order_fit=first,audit=audit,prediction=prediction,
        bound_indices=np.flatnonzero((parameters<model.lower+1e-4)|(parameters>model.upper-1e-4)).tolist(),
        primary_observables=g.prediction(parameters[:n]).tolist(),antenna_fit=g.antennas.describe(parameters[:18]),
        information=antenna_information(model,parameters,phase,secondary),
        position_correction_enu_m=(parameters[18:21]*g.nuisance_sigma[:3]).tolist(),
        velocity_correction_enu_mps=(parameters[21:24]*g.nuisance_sigma[3:6]).tolist(),
        acceleration_correction_enu_mps2=(parameters[24:27]*g.nuisance_sigma[6:9]).tolist())


def fitted_record(model,seed,testing,*,common,max_evaluations):
    fitted,first=model.fit(seed,free_antennas=True,max_evaluations=max_evaluations)
    fitted,audit=audit_fit(model,fitted,np.asarray(first['free_indices']))
    return describe(model,fitted,first,audit,testing,common=common)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-common-primary-pair'))
    parser.add_argument('--assignments',nargs='+',default=['00002','10002'])
    parser.add_argument('--intervals',nargs='+',choices=['F01','Flash'],default=['F01','Flash'])
    parser.add_argument('--adjacent-cycles',action='store_true')
    parser.add_argument('--max-evaluations',type=int,default=300)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);output=args.output/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    root=args.output.parent
    calibration_path=root/'ship-physical-pair-sites-32/fits.json';transfer_path=root/'ship-pair-frozen-transfer/fits.json'
    calibration=json.loads(calibration_path.read_text());transfer=json.loads(transfer_path.read_text())
    if not calibration['complete'] or not transfer['complete']:raise ValueError('Complete independent controls first')
    case=prepare_problem();options=assignments(licensed_options(case['geometry_directory']))
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a for a in options}
    if any(c not in lookup for c in args.assignments):raise ValueError('Unlicensed assignment')
    paths=[calibration_path,transfer_path,Path(__file__),case['track_path'],*set(case['field_paths'])]
    paths += [Path(__file__).with_name(n+'.py') for n in ('physical_pair','common_physical_pair','amplitude_pair',
        'physical_amplitude_arc','referenced_common_arc','common_physical_arc','amplitude_arc','field',
        'audit_ship_physical_pair','stationarity','fit_ship_pair_sites','fit_joint_sea_arcs')]
    paths += [case['geometry_directory']/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')]
    snapshot=args.output/'sources';snapshot.mkdir(exist_ok=True)
    for path in paths:
        if path.suffix=='.py':(snapshot/path.name).write_bytes(path.read_bytes())
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,assignments=args.assignments,
        intervals=args.intervals,reference_band=3,adjacent_cycles=args.adjacent_cycles,groups=[],
        phase_policy='One physical primary scattering phase across five RF bands, one constant secondary phase per band. No time-dependent band phase correction.',
        validation_scope='Same inspected samples and masks as independent-phase controls; coefficients frozen for unused blocks. Not untouched external validation.',
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths})
    started=time.perf_counter();output.write_text(json.dumps(result,indent=2,allow_nan=False))
    for label in args.intervals:
        i=next(i for i,e in enumerate(case['descriptions']) if e['label']==label)
        for code in args.assignments:
            rows=calibration['trials'] if label=='F01' else transfer['trials']
            source=min([r for r in rows if r['assignment']==code],key=lambda r:r['cost'])
            independent=model_for_assignment(case,i,lookup[code]);seed=np.asarray(source['parameters'])
            initial=independent.objective(seed)[0]*independent.energy
            if abs(initial-source['cost'])>1e-7:raise ValueError('Source independent objective did not reproduce')
            baseline=fitted_record(independent,seed,case['testing'][i],common=False,max_evaluations=args.max_evaluations)
            model=CommonPhysicalPair(independent,reference_band=3)
            branches=model.phase_branches(np.asarray(baseline['parameters']),adjacent=args.adjacent_cycles)
            group=dict(interval=label,assignment=code,initial_cost=initial,independent_baseline=baseline,common_trials=[])
            result['groups'].append(group)
            print('group',label,code,'independent',baseline['cost'],'branches',len(branches),flush=True)
            for branch_index,branch in enumerate(branches):
                seed=np.asarray(branch['parameters']);_,gradient=model.objective(seed)
                direction=np.sin(np.arange(model.count)+.4)*.01;step=5e-4
                finite=(model.objective(seed+step*direction)[0]-model.objective(seed-step*direction)[0])/(2*step)
                error=float(abs(finite-gradient@direction)/max(abs(finite),1e-7))
                if error>3e-4:raise ValueError(f'Common primary derivative check failed: {error}')
                record=fitted_record(model,seed,case['testing'][i],common=True,max_evaluations=args.max_evaluations)
                record.update(branch_index=branch_index,carrier_cycles=branch['carrier_cycles'],phase_lift_cost=branch['phase_lift_cost'],
                    initial_directional_gradient_relative_error=error)
                group['common_trials'].append(record);result['seconds']=time.perf_counter()-started
                output.write_text(json.dumps(result,indent=2,allow_nan=False))
                print(label,code,'branch',branch_index,'cost',round(record['cost'],3),'test',
                    round(sum(p['heldout_error'] for p in record['prediction']),3),'strict',record['audit']['strict_local_minimum'],flush=True)
            best=min(group['common_trials'],key=lambda r:r['cost']);mapped=model.mapping(np.asarray(best['parameters']))[0]
            if abs(independent.objective(mapped)[0]*independent.energy-best['cost'])>1e-7:raise ValueError('Nested likelihood failed')
            relaxed=fitted_record(independent,mapped,case['testing'][i],common=False,max_evaluations=args.max_evaluations)
            if relaxed['cost']>best['cost']+1e-6:raise ValueError('Releasing phases increased the minimized objective')
            group['relaxed_control']=relaxed;result['seconds']=time.perf_counter()-started
            output.write_text(json.dumps(result,indent=2,allow_nan=False))
            print(label,code,'relaxed cost',relaxed['cost'],'test',sum(p['heldout_error'] for p in relaxed['prediction']),flush=True)
    result['complete']=True;output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
