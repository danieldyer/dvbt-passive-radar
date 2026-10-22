"""Release constant band phases while preserving shared antennas and motion.

This nested control separates an optimistic common-scattering-phase
assumption from errors in the common physical range/rate/curvature model.
No band-specific timing, frequency or phase-rate correction is introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fit_joint_sea_arcs import prepare_problem, build_arcs
from joint_physical_arcs import JointPhysicalArcs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/joint-ship-sea-reference-phase/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/joint-ship-band-phase-control'))
    parser.add_argument('--max-evaluations',type=int,default=400)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'fits.json'
    if path.exists():raise ValueError('Use a fresh output directory')
    source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Source comparison must be complete')
    case=prepare_problem()
    files=[args.source,*case['source_paths'],case['track_path'],Path(__file__)]
    files += [Path(__file__).with_name(name+'.py') for name in ('fit_joint_sea_arcs','joint_physical_arcs','profiled_arc_solver',
        'physical_amplitude_arc','common_physical_arc','referenced_common_arc','sea_physical_arc','differentiable_sea')]
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,source=str(args.source),assignment=source['assignment'],
        intervals=source['intervals'],phase_policy='Independent constant midpoint phases per band and arc; same physical p/v/a and shared antennas.',
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files},trials=[])
    started=time.perf_counter()
    for record in source['trials']:
        common=build_arcs(case,record['family'],source.get('common_phase_reference_band'))
        models=[a.independent for a in common];joint=JointPhysicalArcs(models)
        parts=[np.asarray(record['parameters'])[:18]]
        for arc,prediction in zip(common,record['prediction']):
            mapped,_=arc.mapping(np.asarray(prediction['parameters']))
            parts.append(mapped[18:])
        seed=np.concatenate(parts);initial=joint.objective(seed)[0]*joint.energy
        if abs(initial-record['cost'])>1e-8*record['cost']:raise ValueError('Nested seed must reproduce the restricted objective')
        direction=np.cos(np.arange(len(seed))+.2)*.01;step=2e-5
        _,gradient=joint.objective(seed)
        finite=(joint.objective(seed+step*direction)[0]-joint.objective(seed-step*direction)[0])/(2*step)
        error=float(abs(finite-gradient@direction)/max(abs(finite),1e-7))
        if error>2e-4:raise ValueError(f'Joint gradient check failed: {error}')
        print('starting',record['family'],record['seed_cycle'],'cost',round(initial,3),'gradient_error',error,flush=True)
        fitted,audit=joint.fit(seed,max_evaluations=args.max_evaluations)
        predictions=[];states=[]
        for description,g,model,index,test in zip(case['descriptions'],case['geometries'],models,joint.indices,case['testing']):
            local=fitted[index];state=g.state(local[:28])
            predictions.append(dict(label=description['label'],parameters=local.tolist(),
                channels=model.prediction(local,test),observables=g.prediction(local[:28]).tolist()))
            sigma=g.nuisance_sigma
            states.append(dict(label=description['label'],position_correction_enu_m=(local[18:21]*sigma[:3]).tolist(),
                velocity_correction_enu_mps=(local[21:24]*sigma[3:6]).tolist(),
                acceleration_correction_enu_mps2=(local[24:27]*sigma[6:9]).tolist(),ais_shift_s=float(local[27]*sigma[-1]),
                position_local_enu_m=state[0].tolist(),velocity_local_enu_mps=state[1].tolist(),acceleration_local_enu_mps2=state[2].tolist()))
        row=dict(family=record['family'],seed_cycle=record['seed_cycle'],parameters=fitted.tolist(),
            common_model_cost=record['cost'],common_model_converged=record['audit']['converged'],initial_cost=initial,
            initial_gradient_relative_error=error,cost=audit['objective']*joint.energy,data_energy=joint.energy,audit=audit,
            prediction=predictions,target_states=states,antenna_fit=case['geometries'][0].antennas.describe(fitted[:18]),
            information=joint.information(fitted))
        result['trials'].append(row);result['seconds']=time.perf_counter()-started
        path.write_text(json.dumps(result,indent=2,allow_nan=False))
        print(row['family'],row['seed_cycle'],'cost',round(row['cost'],3),'converged',audit['converged'],
            'KKT',audit['polishing'].get('final_kkt_residual'),'bounds',audit['bound_indices'],flush=True)
    result['complete']=True;path.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
