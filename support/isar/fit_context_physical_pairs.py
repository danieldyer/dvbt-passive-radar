"""Revisit common physical motion after correcting the clutter range support.

Antennas remain fixed at the user priors in this control. One Cartesian
position/velocity/acceleration is common to the primary echo in all bands;
each band has one constant primary phase and an independent secondary echo.
Compare direct propagation with the complete declared HH four-path model.
No antenna coordinates or per-band phase corrections are adopted.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import least_squares

from audit_ship_physical_pair import audit_fit
from context_physical_pair import model_for_interval,background_status
from fit_joint_sea_arcs import prepare_problem
from fit_ship_physical_pair import raw_seed
from motion import C
from pair_diagnostics import diagnose


def component_seeds(model,source):
    """Resolution-scaled proposals only; no inverse of a degenerate fit FIM."""
    g=model.geometry;best=[ch['trials'][ch['best_index']] for ch in source['channels']]
    moments=np.asarray([r['observables'] for r in best])
    duration=model.bands[0].field.time[-1]-model.bands[0].field.time[0]
    wavelength=np.asarray([2*np.pi/b.field.k for b in model.bands])
    scales=np.column_stack([np.full(5,C/8e6),wavelength/duration,2*wavelength/duration**2])
    lo,hi=g.bounds();records=[]
    for association in itertools.product(range(2),repeat=5):
        observed=moments[np.arange(5),association]
        def residual(q):
            prediction=g.prediction(np.r_[np.zeros(18),q])
            return np.r_[((prediction-observed)/scales).ravel(),q]
        fit=least_squares(residual,np.clip(np.zeros(10),lo[18:],hi[18:]),jac='cs',
            bounds=(lo[18:],hi[18:]),max_nfev=150,ftol=1e-11,xtol=1e-11,gtol=1e-9)
        initial=dict(assignment=list(association),parameters=np.r_[np.zeros(18),fit.x].tolist(),
            proposal_cost=float(fit.fun@fit.fun),proposal_termination=str(fit.message))
        seed=raw_seed(model,source,initial)
        initial.update(seed=seed.tolist(),raw_initial_cost=model.objective(seed)[0]*model.energy)
        records.append(initial)
    return sorted(records,key=lambda r:r['raw_initial_cost']),scales


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/background-context-refits/fits.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/context-physical-controls'))
    parser.add_argument('--families',nargs='+',choices=['direct','transmit','receive','both'],default=['direct','both'])
    parser.add_argument('--starts',type=int,default=4)
    parser.add_argument('--max-evaluations',type=int,default=300)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);output=out/'fits.json'
    if output.exists():raise ValueError('Use a fresh output directory')
    source=json.loads(args.source.read_text())
    if not source['complete'] or not source['range_context']:raise ValueError('Require completed context component fits')
    case=prepare_problem();started=time.perf_counter()
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,groups=[],
        assignment='00002',antenna_policy='Fixed user Ornata/Ridge priors, including receiver height. This is a motion/propagation control, not an antenna fit.',
        phase_policy='Independent constant primary phase per band; no time-dependent phase freedom. Each secondary retains its own constant phase and path.',
        initialization='All 32 two-component associations propose a common physical state using range-cell and one-cycle/aperture scales, not claimed covariances. Original training likelihood ranks them. No unreliable inverse covariance of coalescing components is used. Source paths are proposals only.',
        selected_starts_per_family=args.starts,max_evaluations=args.max_evaluations,
        water=case['water'],sea_policy='Smooth spherical HH four-path model: direct, transmitter-sea, sea-receiver and both-sea rays; fixed Fresnel/spreading factors; no extra fitted ray phase or gain.',
        validation='Original unused time blocks with every estimated coefficient frozen. All epochs and range cells are previously inspected; this is not blind external validation.')
    paths=[args.source,Path(__file__),case['track_path'],*set(case['field_paths']),*case['source_paths']]
    paths += [Path('support/isar')/(n+'.py') for n in ('context_physical_pair','fit_joint_sea_arcs','fit_ship_physical_pair',
        'physical_pair','physical_amplitude_arc','amplitude_arc','coherent_arc','amplitude_pair','field','stationary',
        'propagated_pair','sea_physical_arc','differentiable_sea','spherical_sea','flash_geometry','antenna_motion',
        'audit_ship_physical_pair','stationarity','pair_diagnostics')]
    paths += [case['geometry_directory']/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')]
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for p in paths:
        if p.suffix=='.py':shutil.copyfile(p,snapshots/p.name)
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(result,indent=2,allow_nan=False))
    save()
    for interval in source['intervals']:
        description=interval['description'];label=description['label']
        i=next(i for i,e in enumerate(case['descriptions']) if e['label']==label)
        local=dict(channels=[next(m for m in c['modes'] if m['mode']=='local') for c in interval['channels']])
        for family in args.families:
            model=model_for_interval(case,i,family=family)
            starts,scales=component_seeds(model,local)
            group=dict(interval=description,family=family,background=background_status(model),
                initialization_scales=scales.tolist(),initial_states=starts,trials=[])
            result['groups'].append(group);save()
            for rank,start in enumerate(starts[:args.starts]):
                seed=np.asarray(start['seed']);r,j=model.evaluate(seed);direction=np.cos(np.arange(model.count))*.01;direction[:18]=0
                step=5e-4
                finite=(model.evaluate(seed+step*direction)[0]-model.evaluate(seed-step*direction)[0])/(2*step)
                derivative_error=float(np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-12))
                if derivative_error>3e-4:raise ValueError(f'Original-row derivative mismatch: {derivative_error}')
                parameters,first=model.fit(seed,free_antennas=False,max_evaluations=args.max_evaluations)
                try:
                    parameters,audit=audit_fit(model,parameters,np.asarray(first['free_indices']))
                except np.linalg.LinAlgError as error:
                    audit=dict(strict_local_minimum=False,reason='Singular residual information in the audit',exception=str(error))
                cost=model.objective(parameters)[0]*model.energy;prediction=model.prediction(parameters,case['testing'][i])
                record=dict(start_rank=rank,association=start['assignment'],parameters=parameters.tolist(),
                    first_order_fit=first,audit=audit,cost=cost,data_energy=model.energy,
                    data_cost=cost-float(parameters[:28]@parameters[:28]),
                    directional_residual_derivative_error=derivative_error,prediction=prediction,
                    primary_observables=model.geometry.prediction(parameters[:28]).tolist(),
                    decomposition=diagnose(model,parameters),
                    bound_indices=np.flatnonzero((parameters<model.lower+1e-4)|(parameters>model.upper-1e-4)).tolist())
                group['trials'].append(record);save()
                print(label,family,rank,'cost',round(cost,3),'unused',round(sum(p['heldout_error'] for p in prediction),3),
                    'strict',audit['strict_local_minimum'],'bounds',record['bound_indices'],flush=True)
            group['best_index']=int(np.argmin([r['cost'] for r in group['trials']]));save()
    result['complete']=True;save()


if __name__=='__main__':main()
