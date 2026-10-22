"""Test common RF scattering phase with flexible independent neighbours.

Both fixed and refined antenna controls use the same original data, nuisance
curves and priors. A common-phase optimum is always released back into the
independently phased model. Carrier-cycle proposals acquire only; they are not
additional observations or corrections to measured complex samples.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from ais import read_csv
from common_physical_pair import CommonPhysicalPair
from fit_physical_primary_association import (assignment_code,geometry_for,make_model,
    prepare_window,free_parameters,diagnostics,measured)
from flash_geometry import assignments
from nearby_echo_control import load,ROOT
from physical_pair import fit_parameters
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options


def independent_free(model,*,antennas):
    free=free_parameters(model,None,nuisance_free=True)
    return np.r_[np.arange(15),17,free] if antennas else free


def common_free(indices,geometry_count=28):
    """Pull the explicitly estimated coordinate set through phase restriction."""
    index=np.asarray(indices,int)
    return np.unique(np.where(index<geometry_count,index,
        np.where(index<geometry_count+5,geometry_count,index-4)))


def derivative_check(model,p,free):
    r,j=model.evaluate(p);direction=np.zeros(model.count)
    direction[free]=1e-4*np.cos(free+.2);step=.001
    finite=(model.evaluate(p+step*direction)[0]-model.evaluate(p-step*direction)[0])/(2*step)
    error=float(np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-15))
    if error>4e-4:raise ValueError(f'Original-row residual derivative failed: {error}')
    return error


def record(model,p,audit,testing,*,common):
    independent=model.independent if common else model
    mapped=model.mapping(p)[0] if common else p
    prediction=model.prediction(p,testing)
    return dict(parameters=p,cost=model.objective(p)[0]*model.energy,prior_cost=independent.prior_cost(mapped),
        audit=audit,prediction=prediction,total_unused_error=sum(v['heldout_error'] for v in prediction),
        primary_observables=model.geometry.prediction(p[:model.geometry.count]),
        antenna_fit=model.geometry.antennas.describe(p[:18]),
        diagnostics=diagnostics(independent,mapped),measured_primary=measured(independent,mapped))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'physical-primary-association/fits.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--finalists',type=int,default=2)
    parser.add_argument('--max-evaluations',type=int,default=180)
    parser.add_argument('--antenna-modes',nargs='+',choices=['fixed','refined'],default=['fixed','refined'])
    args=parser.parse_args();source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Wait for completed association controls')
    if args.output.exists():raise ValueError('Use a fresh output directory')
    empirical_path=Path(source['source']);empirical=json.loads(empirical_path.read_text());control=load()
    track_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(track_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    options=assignments(licensed_options(control['geometry']));out=args.output;out.mkdir(parents=True)
    files=[args.source,empirical_path,track_path,Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('fit_physical_primary_association','physical_echo_cluster','physical_amplitude_arc','physical_pair',
         'common_physical_pair','common_physical_arc','referenced_common_arc','component_phase',
         'amplitude_pair','joint_path_information','flash_geometry','nearby_echo_control')],
        *[control['geometry']/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')]]
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,groups=[],
        source=str(args.source),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        phase_policy='A genuinely common physical RF scattering phase is a testable optimistic restriction. Undo midpoint geometric phase with its complete chain rule. No per-band fitted phase correction is applied to the observations. Independent constant phases are the control; same positive real brightness and flexible independent A/C/D/nearby nuisance remain.',
        gauge_policy='Receiver horizontal position and redundant AIS delay are fixed. Fixed mode estimates no antennas; refined mode estimates antenna coordinates 0–14 and 17 with approved correlated priors and bounded receiver height. The two inherited Ten C nuisance path coefficients remain fixed in every compared model.',
        prior_policy='Proper-complex data squared residual plus half the real standardized geometry/white-jerk prior squared. Shared geometry prior counted once across bands. Carrier-lift 0.1 rad scale is acquisition only, never likelihood information.',
        validation_policy='Unchanged original rows, known symbol kernels, range metric, stationary observed-cell design, brightness knots and unused-block masks. Previously inspected two-second data provide an exploratory model comparison, not independent antenna calibration.')
    snapshot=out/'sources';snapshot.mkdir()
    for p in files:
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    started=time.monotonic()
    def save():
        result['seconds']=time.monotonic()-started
        (out/'fits.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for old_group in source['groups']:
        window=empirical['windows'][old_group['window_index']]
        bands,parts,models,parameters,baseline=prepare_window(control,window)
        midpoint=np.mean(window['interval_s']);assignment=next(a for a in options if assignment_code(a)==old_group['assignment'])
        geometry=geometry_for(control,assignment,midpoint,track)
        independent=make_model(bands,models,control,geometry,midpoint)
        old=old_group['trials'][old_group['best_index']]['steps'][-1];source_p=np.asarray(old['parameters'])
        r,_=independent.evaluate(source_p)
        if abs(r@r*independent.energy-old['cost'])>1e-7:raise ValueError('Original physical-primary objective changed')
        testing=[part==3 for part in parts]
        for mode in args.antenna_modes:
            antennas=mode=='refined';free=independent_free(independent,antennas=antennas)
            if antennas:
                p,audit=fit_parameters(independent,source_p,free,max_evaluations=args.max_evaluations)
                base=record(independent,p,audit,testing,common=False)
            else:p=source_p;base=record(independent,p,old['audit'],testing,common=False)
            group=dict(interval_s=window['interval_s'],window_index=old_group['window_index'],assignment=old_group['assignment'],
                antenna_mode=mode,independent=base,proposals=[],common_trials=[])
            result['groups'].append(group);save()
            model=CommonPhysicalPair(independent)
            # Unlike the default lift, frozen-antenna proposals must not move
            # antennas merely because the geometry structurally exposes them.
            lift=np.r_[np.arange(15),17,np.arange(18,21)] if antennas else np.arange(18,21)
            branches=model.phase_branches(p,adjacent=True,free_coordinates=lift)
            for branch in branches:
                seed=np.asarray(branch['parameters']);r,_=model.evaluate(seed)
                branch['original_training_cost']=float(r@r*model.energy);group['proposals'].append(branch)
            order=np.argsort([v['original_training_cost'] for v in group['proposals']]);group['proposal_order']=order;save()
            print(mode,'independent',base['cost'],'gradient',base['audit']['projected_gradient'],
                'proposals',len(order),'best raw',[group['proposals'][i]['original_training_cost'] for i in order[:args.finalists]],flush=True)
            estimated=common_free(free)
            for index in order[:args.finalists]:
                branch=group['proposals'][index];seed=np.asarray(branch['parameters'])
                error=derivative_check(model,seed,estimated)
                fitted,audit=fit_parameters(model,seed,estimated,max_evaluations=args.max_evaluations)
                fitted_record=record(model,fitted,audit,testing,common=True)
                fitted_record.update(proposal_index=int(index),carrier_cycles=branch['carrier_cycles'],derivative_error=error)
                group['common_trials'].append(fitted_record);save()
                print(mode,'common',int(index),'cost',fitted_record['cost'],'gradient',audit['projected_gradient'],
                    'unused%',100*(fitted_record['total_unused_error']/base['total_unused_error']-1),'bounds',audit['bound_indices'],flush=True)
            group['best_common_index']=int(np.argmin([t['cost'] for t in group['common_trials']]))
            best=group['common_trials'][group['best_common_index']];mapped=model.mapping(np.asarray(best['parameters']))[0]
            if abs(independent.objective(mapped)[0]*independent.energy-best['cost'])>1e-7:raise ValueError('Common likelihood is not nested')
            fitted,audit=fit_parameters(independent,mapped,free,max_evaluations=args.max_evaluations)
            relaxed=record(independent,fitted,audit,testing,common=False)
            if relaxed['cost']>best['cost']+1e-6:raise ValueError('Phase release increased the minimized objective')
            group['relaxed']=relaxed;group['complete']=True;save()
            print(mode,'relaxed',relaxed['cost'],'gradient',audit['projected_gradient'],
                'common minus best independent unused%',100*(best['total_unused_error']/min(base['total_unused_error'],relaxed['total_unused_error'])-1),flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
