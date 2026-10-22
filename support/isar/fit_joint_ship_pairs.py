"""Share one antenna solution across disjoint intervals with explicit echoes.

Independent primary band phases provide a nested control. Common physical
phases use explicitly enumerated local carrier-cycle initializers; selection
and optimization see training samples only. A final phase release checks the
local-basin comparison. No antenna estimates are adopted by this experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audit_ship_physical_pair import audit_fit
from common_physical_pair import CommonPhysicalPair
from fit_joint_sea_arcs import prepare_problem
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from joint_physical_pairs import JointPhysicalPairs
from pair_diagnostics import diagnose
from propagated_pair import primary_propagation
from ship_site_signature import licensed_options


def fitted_record(model,seed,testing,max_evaluations):
    fitted,first=model.fit(seed,max_evaluations=max_evaluations)
    fitted,audit=audit_fit(model,fitted,np.asarray(first['free_indices']))
    return dict(parameters=fitted.tolist(),cost=model.objective(fitted)[0]*model.energy,
        data_energy=model.energy,first_order_fit=first,audit=audit,
        bound_indices=np.flatnonzero((fitted<model.lower+1e-4)|(fitted>model.upper-1e-4)).tolist(),
        prediction=model.prediction(fitted,testing),information=model.information(fitted),
        antenna_fit=model.arcs[0].geometry.antennas.describe(fitted[:18]),
        local_parameters=[fitted[index].tolist() for index in model.indices],
        primary_observables=[arc.geometry.prediction(fitted[index][:28]).tolist() for arc,index in zip(model.arcs,model.indices)],
        decomposition=[diagnose(arc,fitted[index]) for arc,index in zip(model.arcs,model.indices)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/ship-common-primary-pair/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/joint-ship-primary-pairs'))
    parser.add_argument('--assignments',nargs='+',default=['00002','10002'])
    parser.add_argument('--common-starts',type=int,default=6)
    parser.add_argument('--max-evaluations',type=int,default=300)
    parser.add_argument('--primary-family',choices=['direct','transmit','both'],default='direct')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);output=args.output/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    if args.common_starts<1:raise ValueError('At least one declared common-phase start is required')
    source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Finish interval controls first')
    case=prepare_problem();labels=['F01','Flash']
    selected=[next(i for i,e in enumerate(case['descriptions']) if e['label']==label) for label in labels]
    testing=[case['testing'][i] for i in selected]
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a
            for a in assignments(licensed_options(case['geometry_directory']))}
    paths=[args.source,Path(__file__),case['track_path'],*set(case['field_paths']),*case['source_paths']]
    paths += [Path(__file__).with_name(n+'.py') for n in ('physical_pair','common_physical_pair','joint_physical_pairs',
        'pair_diagnostics','amplitude_pair','physical_amplitude_arc','referenced_common_arc','common_physical_arc',
        'amplitude_arc','field','audit_ship_physical_pair','stationarity','fit_ship_pair_sites','fit_joint_sea_arcs',
        'propagated_pair','sea_physical_arc','differentiable_sea','spherical_sea')]
    paths += [case['geometry_directory']/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')]
    snapshot=args.output/'sources';snapshot.mkdir(exist_ok=True)
    for p in paths:
        if p.suffix=='.py':(snapshot/p.name).write_bytes(p.read_bytes())
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,assignments=args.assignments,
        intervals=[case['descriptions'][i] for i in selected],common_starts=args.common_starts,groups=[],
        primary_propagation=args.primary_family,
        water=case['water'] if args.primary_family!='direct' else None,
        reflection_policy='Declared smooth spherical HH sea, relative permittivity 80 and conductivity 4 S/m. Reflected rays share primary position and brightness; no fitted ray phase/gain.',
        phase_policy='One common physical primary phase per interval; each secondary has independent constant phase. No time-dependent phase corrections.',
        validation_scope='Same inspected data and masks as earlier controls; unused-block coefficients frozen. Not blind external validation.',
        initializer_policy='Enumerate nearest phase-origin cycle combinations at the independent joint solution; rank by actual training objective, fit the declared best starts. Not exhaustive over all carrier ambiguities.',
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths})
    started=time.perf_counter()
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(result,indent=2,allow_nan=False))
    save()
    for code in args.assignments:
        controls=[next(g for g in source['groups'] if g['interval']==label and g['assignment']==code)['independent_baseline'] for label in labels]
        parameters=[np.asarray(r['parameters']) for r in controls]
        arcs=[model_for_assignment(case,i,lookup[code]) for i in selected]
        if args.primary_family!='direct':arcs=[primary_propagation(a,case['water'],args.primary_family) for a in arcs]
        independent=JointPhysicalPairs(arcs)
        group=dict(assignment=code,independent_trials=[],common_seeds=[],common_trials=[]);result['groups'].append(group)
        for rank,antennas in enumerate((np.mean([p[:18] for p in parameters],axis=0),parameters[0][:18])):
            seed=np.r_[antennas,*[p[18:] for p in parameters]]
            record=fitted_record(independent,seed,testing,args.max_evaluations);record['start_rank']=rank
            group['independent_trials'].append(record);save()
            print(code,'independent',rank,'cost',round(record['cost'],3),'strict',record['audit']['strict_local_minimum'],flush=True)
        best=min(group['independent_trials'],key=lambda r:r['cost'])
        local=[np.asarray(p) for p in best['local_parameters']]
        model=JointPhysicalPairs([CommonPhysicalPair(arc) for arc in independent.arcs])
        branches=model.phase_initializers(local)
        for branch_index,branch in enumerate(branches):
            seed=branch['parameters']
            group['common_seeds'].append(dict(branch_index=branch_index,carrier_cycles=branch['carrier_cycles'],
                phase_lift_cost=branch['phase_lift_cost'],phase_lift_optimality=branch['phase_lift_optimality'],
                initial_training_cost=model.objective(seed)[0]*model.energy,parameters=seed.tolist()))
        chosen=sorted(group['common_seeds'],key=lambda r:r['initial_training_cost'])[:args.common_starts]
        group['chosen_seed_indices']=[r['branch_index'] for r in chosen];save()
        print(code,'ranked',len(branches),'cycle starts; fitting',group['chosen_seed_indices'],flush=True)
        for branch in chosen:
            seed=np.asarray(branch['parameters']);record=fitted_record(model,seed,testing,args.max_evaluations)
            record.update(branch_index=branch['branch_index'],carrier_cycles=branch['carrier_cycles'])
            group['common_trials'].append(record);save()
            print(code,'common',record['branch_index'],'cost',round(record['cost'],3),'test',
                round(sum(p['heldout_error'] for arc in record['prediction'] for p in arc),3),
                'strict',record['audit']['strict_local_minimum'],flush=True)
        best=min(group['common_trials'],key=lambda r:r['cost']);p=np.asarray(best['parameters'])
        local=[arc.mapping(p[index])[0] for arc,index in zip(model.arcs,model.indices)]
        seed=np.r_[local[0][:18],*[p[18:] for p in local]]
        if abs(independent.objective(seed)[0]*independent.energy-best['cost'])>1e-7:raise ValueError('Joint nested likelihood mismatch')
        group['relaxed_control']=fitted_record(independent,seed,testing,args.max_evaluations)
        if group['relaxed_control']['cost']>best['cost']+1e-6:raise ValueError('Releasing primary phases increased the fit objective')
        save()
    result['complete']=True;save()


if __name__=='__main__':main()
