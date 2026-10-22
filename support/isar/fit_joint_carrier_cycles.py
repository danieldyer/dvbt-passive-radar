"""Broader joint carrier-cycle search, followed by unchanged raw-data fits.

The K nearest lattice points of one local Gauss--Newton surrogate are fully
enumerated. Every point is projected through the nonlinear geometry with the
original bounds. Training samples rank the projected seeds; declared best
starts then use the original complex likelihood and strict stationarity audit.
This does not exhaust all nonlinear basins, assignments or scattering models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from carrier_cycles import CycleSurrogate,joint_mapping,cycle_constraints
from common_physical_pair import CommonPhysicalPair
from fit_joint_sea_arcs import prepare_problem
from fit_joint_ship_pairs import fitted_record
from fit_ship_pair_sites import model_for_assignment
from flash_geometry import assignments
from integer_least_squares import nearest_integers
from joint_physical_pairs import JointPhysicalPairs
from propagated_pair import primary_propagation
from ship_site_signature import licensed_options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/joint-ship-primary-pairs/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/joint-carrier-cycles'))
    parser.add_argument('--assignments',nargs='+',default=['00002','10002'])
    parser.add_argument('--candidates',type=int,default=64)
    parser.add_argument('--common-starts',type=int,default=6)
    parser.add_argument('--max-evaluations',type=int,default=300)
    parser.add_argument('--bounded-initializer',action='store_true')
    parser.add_argument('--whitened-projection',action='store_true')
    args=parser.parse_args()
    if args.common_starts<1 or args.candidates<args.common_starts:raise ValueError('Invalid search sizes')
    output=args.output/'fits.json'
    if output.exists():raise ValueError('Preserve existing results; use a new output directory')
    source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Complete independent controls first')
    case=prepare_problem();labels=['F01','Flash']
    selected=[next(i for i,e in enumerate(case['descriptions']) if e['label']==label) for label in labels]
    testing=[case['testing'][i] for i in selected]
    lookup={''.join({'12013':'0','12015':'1','12014':'2'}[o['site_id']] for o in a):a
            for a in assignments(licensed_options(case['geometry_directory']))}
    family=source.get('primary_propagation','direct')
    paths=[args.source,Path(__file__),case['track_path'],*set(case['field_paths']),*case['source_paths']]
    paths += [Path(__file__).with_name(n+'.py') for n in ('carrier_cycles','integer_least_squares','bounded_quadratic',
        'physical_pair','common_physical_pair','joint_physical_pairs','fit_joint_ship_pairs',
        'pair_diagnostics','amplitude_pair','physical_amplitude_arc','referenced_common_arc','common_physical_arc',
        'amplitude_arc','field','audit_ship_physical_pair','stationarity','fit_ship_pair_sites','fit_joint_sea_arcs',
        'propagated_pair','sea_physical_arc','differentiable_sea','spherical_sea','flash_geometry','antenna_motion')]
    paths += [case['geometry_directory']/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')]
    snapshot=args.output/'sources';snapshot.mkdir(parents=True)
    for path in paths:
        if path.suffix=='.py':(snapshot/path.name).write_bytes(path.read_bytes())
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,source=str(args.source),
        assignments=args.assignments,intervals=[case['descriptions'][i] for i in selected],groups=[],
        common_starts=args.common_starts,integer_candidates=args.candidates,primary_propagation=family,
        bounded_initializer=args.bounded_initializer,
        whitened_projection=args.whitened_projection,
        water=source.get('water'),
        validation_scope='Previously inspected data with unchanged unused-block masks; all nuisance coefficients frozen for prediction. No external validation.',
        cycle_policy='Integer search complete for one unconstrained local quadratic only. Bounds enforced during nonlinear surrogate projection and original likelihood refinement; final fit can change its initial integer branch.',
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths})
    started=time.perf_counter()
    def save():
        result['seconds']=time.perf_counter()-started
        output.write_text(json.dumps(result,indent=2,allow_nan=False))
    save()
    for code in args.assignments:
        original=next(g for g in source['groups'] if g['assignment']==code)
        control=min([*original['independent_trials'],original['relaxed_control']],key=lambda r:r['cost'])
        arcs=[model_for_assignment(case,i,lookup[code]) for i in selected]
        if family!='direct':arcs=[primary_propagation(a,source['water'],family) for a in arcs]
        independent=JointPhysicalPairs(arcs);common=JointPhysicalPairs([CommonPhysicalPair(a) for a in arcs])
        origin=np.asarray(control['parameters'])
        if abs(independent.objective(origin)[0]*independent.energy-control['cost'])>1e-7:
            raise ValueError('Original independent likelihood changed')
        surrogate=CycleSurrogate(independent,common,origin)
        lattice=nearest_integers(surrogate.float_cycles,surrogate.metric,args.candidates)
        group=dict(assignment=code,independent_trials=[control],common_seeds=[],common_trials=[],
            surrogate=dict(minimum_cost=surrogate.minimum_cost,jacobian_condition=surrogate.condition,
                float_cycles=surrogate.float_cycles.tolist(),metric=surrogate.metric.tolist(),
                metric_condition=float(np.linalg.cond(surrogate.metric))),lattice=lattice,
            previous_common_best=min(original['common_trials'],key=lambda r:r['cost']))
        result['groups'].append(group);save()
        print(code,'lattice nodes',lattice['visited_nodes'],'condition',round(group['surrogate']['metric_condition'],2),
            'penalty',round(lattice['candidates'][0]['cost'],6),flush=True)
        for rank,candidate in enumerate(lattice['candidates']):
            record=surrogate.project_whitened(candidate['integer']) if args.whitened_projection else surrogate.project(
                candidate['integer'],bounded_initializer=args.bounded_initializer)
            record['branch_index']=rank
            seed=np.asarray(record['parameters'])
            record['initial_training_cost']=common.objective(seed)[0]*common.energy
            record['carrier_cycles']=np.reshape(candidate['integer'],(len(labels),4)).tolist()
            group['common_seeds'].append(record)
            if rank%8==0 or rank==args.candidates-1:
                save();print(code,'projected',rank+1,'of',args.candidates,flush=True)
        chosen=sorted(group['common_seeds'],key=lambda r:r['initial_training_cost'])[:args.common_starts]
        group['chosen_seed_indices']=[r['branch_index'] for r in chosen];save()
        print(code,'fitting projected starts',group['chosen_seed_indices'],flush=True)
        for branch in chosen:
            row=fitted_record(common,np.asarray(branch['parameters']),testing,args.max_evaluations)
            mapped,_=joint_mapping(common,independent,np.asarray(row['parameters']))
            row.update(branch_index=branch['branch_index'],carrier_cycles=branch['carrier_cycles'],
                final_relative_cycles=cycle_constraints(common,independent,mapped)[0].tolist())
            group['common_trials'].append(row);save()
            height=max(abs(p[2]) for p in row['antenna_fit']['transmitter_offsets_enu_m'])
            print(code,'common',row['branch_index'],'cost',round(row['cost'],3),'test',
                round(sum(p['heldout_error'] for arc in row['prediction'] for p in arc),3),
                'height correction',round(height,2),'strict',row['audit']['strict_local_minimum'],flush=True)
        best=min(group['common_trials'],key=lambda r:r['cost'])
        mapped,_=joint_mapping(common,independent,np.asarray(best['parameters']))
        if abs(independent.objective(mapped)[0]*independent.energy-best['cost'])>1e-7:
            raise ValueError('Nested likelihood mismatch')
        group['relaxed_control']=fitted_record(independent,mapped,testing,args.max_evaluations)
        if group['relaxed_control']['cost']>best['cost']+1e-6:raise ValueError('Phase release worsened objective')
        save()
    result['complete']=True;save()


if __name__=='__main__':main()
