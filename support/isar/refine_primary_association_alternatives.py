"""Refine explicitly retained alternative associations on original observations.

The cheapest cold start need not retain the requested target: it can assign
the physical primary to another return and leave the target among independent
neighbours. Preserve alternative range/motion modes rather than interpreting
that label switch as vessel motion or forcing phase curves into agreement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from ais import read_csv
from fit_physical_primary_association import (assignment_code,geometry_for,make_model,prepare_window,
    free_parameters,diagnostics,measured)
from fit_physical_primary_coherence import derivative_check
from flash_geometry import assignments
from nearby_echo_control import load,ROOT
from physical_pair import fit_parameters
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'physical-primary-disjoint-arcs/fits.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--proposals',nargs='+',type=int,default=[31])
    parser.add_argument('--max-evaluations',type=int,default=200)
    args=parser.parse_args();source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Wait for complete original association trials')
    if args.output.exists():raise ValueError('Use a fresh directory')
    empirical=json.loads(Path(source['source']).read_text());control=load()
    track_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(track_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    options=assignments(licensed_options(control['geometry']));out=args.output;out.mkdir(parents=True)
    files=[args.source,Path(source['source']),Path(__file__),track_path,*[Path('support/isar')/(n+'.py') for n in
        ('fit_physical_primary_association','fit_physical_primary_coherence','physical_echo_cluster','physical_pair',
         'physical_amplitude_arc','component_phase','nearby_echo_control','phase_window')]]
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,groups=[],
        source=str(args.source),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        selection_policy='These starts deliberately retain moving-return alternatives to the former minimum cold cost. All-E is an initializer only, not an asserted common hull-point identity. No phase/range guide is added to the likelihood, no new gate supplies information and no data normalization changes. All original priors, guards, fixed nuisance coordinates and train/test splits remain unchanged.')
    snapshot=out/'sources';snapshot.mkdir()
    for p in files:
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    started=time.monotonic()
    def save():
        result['seconds']=time.monotonic()-started
        (out/'fits.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for old in source['groups']:
        window=empirical['windows'][old['window_index']]
        bands,parts,models,parameters,baseline=prepare_window(control,window)
        g=geometry_for(control,next(a for a in options if assignment_code(a)==old['assignment']),np.mean(old['interval_s']),track)
        model=make_model(bands,models,control,g,np.mean(old['interval_s']))
        previous=old['trials'][old['best_index']]['steps'][-1]
        r,_=model.evaluate(np.asarray(previous['parameters']))
        if abs(r@r*model.energy-previous['cost'])>1e-7:raise ValueError('Original competing mode changed')
        group=dict(interval_s=old['interval_s'],window_index=old['window_index'],assignment=old['assignment'],
            previous_best=previous,trials=[],independent_baseline_unused_error=old['independent_baseline_unused_error'])
        result['groups'].append(group);save()
        for index in args.proposals:
            proposal=old['proposals'][index];seed=np.asarray(proposal['parameters'])
            r,_=model.evaluate(seed)
            if abs(r@r*model.energy-proposal['training_cost'])>1e-7:raise ValueError('Alternative initializer changed')
            error=derivative_check(model,seed,free_parameters(model,None,nuisance_free=True))
            trial=dict(proposal_index=index,labels=proposal['labels'],directional_derivative_error=error,steps=[])
            group['trials'].append(trial);save()
            for release in (False,True):
                free=free_parameters(model,None,nuisance_free=release)
                p,audit=fit_parameters(model,seed,free,max_evaluations=args.max_evaluations)
                prediction=model.prediction(p,[part==3 for part in parts])
                record=dict(nuisance_paths_free=release,parameters=p,audit=audit,
                    cost=model.objective(p)[0]*model.energy,prior_cost=model.prior_cost(p),
                    prediction=prediction,diagnostics=diagnostics(model,p),measured_primary=measured(model,p),
                    primary_observables=g.prediction(p[:g.count]),
                    unused_change_from_independent_percent=100*(sum(v['heldout_error'] for v in prediction)/old['independent_baseline_unused_error']-1))
                trial['steps'].append(record);save();seed=p
                print(old['interval_s'],'proposal',index,'nuisance free',release,'cost',record['cost'],
                    'gradient',audit['projected_gradient'],'R/rate/acc',record['primary_observables'][2],flush=True)
        group['complete']=True;save()
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
