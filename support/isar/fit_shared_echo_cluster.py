"""Resolve selected local ship echoes jointly before comparing phase motion.

The retained seventeen range cells constrain all components. Each band has
independent constant phase, range envelope origin and positive brightness.
Quadratic residual paths are either independent or common across bands.
All other processing uses the existing exact C0/C1 observation operator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from cluster_phase import crop
from compare_ais_phase import path_geometry,NAMES
from report_measured_scatterer_phases import load_fields,SELECTIONS,serial
from shared_echo_cluster import ReferencedAmplitudeArc,SharedEchoCluster
from ship_site_signature import licensed_options
from extract_full_aperture import load as load_aperture
from recorded_field_context import expand
from polynomial_phase_arc import Band,PolynomialPhaseArc


def prepare(fields,geometry,first,last,spacing=.25,range_extent=8,observation_directory=None):
    options=licensed_options(geometry);rx=np.asarray(json.loads((geometry/'geometry-prior.json').read_text())['receiver_ecef_m'])
    bands=[];partitions=[]
    for ch,f in enumerate(fields):
        field=crop(f['data'],first,last,np.arange(-8,9))
        if range_extent!=8:
            old=field
            aperture=load_aperture(Path(f'build/isar-20261015/full-geographic-aperture/channel-{ch}'))
            field=expand(field,aperture,np.arange(-range_extent,range_extent+1))
            indices=np.array([np.flatnonzero(field.offsets==o)[0] for o in old.offsets])
            if not np.array_equal(field.data['value'][:,indices],old.data['value']):
                raise ValueError('Expanded original observations changed existing measured cells')
        d=field.data;rows=d['row_time']
        if observation_directory is not None:
            observation_directory.mkdir(exist_ok=True)
            np.savez(observation_directory/f'channel-{ch}.npz',**{**d,'pulse_samples':field.pulse.samples})
        tx=np.asarray(next(o for o in options if o['channel']==ch and o['site_id']==('12014' if ch==4 else '12013'))['transmitter_ecef_m'])
        nominal,rate=path_geometry(d['centre_ecef_m'],d['centre_velocity_ecef_mps'],tx,rx)
        known=nominal-d['centre_path_m'];known-=np.interp((first+last)/2,rows,known)
        old_rate=-d['centre_omega']/(field.k*.001024)
        knots=np.r_[rows[0],np.arange(first+spacing,last,spacing),rows[-1]]
        partition=np.floor((field.time-first)/.125).astype(int)%4;training=partition%2==0
        bands.append(ReferencedAmplitudeArc(field,training,knots,first,last,known,rate-old_rate));partitions.append(partition)
    return bands,partitions


def seed_for_band(source,model,channel,labels,origins,first,last,cluster_source=None):
    n=model.components;p=np.zeros(model.count)
    for s,(label,origin) in enumerate(zip(labels,origins)):
        if cluster_source is not None:
            oldn=len(cluster_source['labels']);oldrecord=cluster_source['independent'][channel]
            old=np.asarray(oldrecord['trials'][oldrecord['best_index']]['parameters'])
            oldlabel='D' if label=='E' else label;oldindex=cluster_source['labels'].index(oldlabel)
            p[2*s:2*s+2]=old[2*oldindex:2*oldindex+2]
            coefficients=old[2*oldn+2*oldindex:2*oldn+2*oldindex+2]
            if label in ('D','E'):
                use=cluster_source['independent'][:4] if label=='D' else cluster_source['independent'][4:]
                coefficients=np.mean([np.asarray(c['trials'][c['best_index']]['parameters'])[-2*oldn:].reshape(oldn,2)[oldindex] for c in use],axis=0)
            p[2*n+2*s:2*n+2*s+2]=coefficients
            continue
        arc=next((a for a in source['arcs'] if a['point']==label and a['interval_s']==[first,last]),None)
        if arc is None:
            b=model.bands[0];column=int(np.flatnonzero(b.field.offsets==origin)[0]);d=b.field.data
            known=np.interp(b.field.time,d['row_time'],b.known_path)
            scalar=Band(b.field.time,d['value'][:,column],8192*d['information'],known,b.field.k,b.training,
                d['row_time'],d['row_information'],b.known_path,int(b.field.metadata['block_rows']))
            fit=PolynomialPhaseArc([scalar],first,last,2).fit()
        else:fit=next(m for m in arc['models'] if m['degree']==2)['independent'][channel]
        p[2*s:2*s+2]=[origin,fit['parameters'][0]]
        p[2*n+2*s:2*n+2*s+2]=fit['parameters'][-2:]
    return p


def subset_prediction(model,parameters,partitions):
    parameters=np.asarray(parameters);results={}
    for name,part in (('validation',1),('testing',3)):
        results[name]=model.prediction(parameters,[p==part for p in partitions])
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=float,default=164.);parser.add_argument('--last',type=float,default=166.)
    parser.add_argument('--window',default='W3')
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/shared-echo-cluster'))
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/piecewise-measured-phases/fits.json'))
    parser.add_argument('--starts',type=int,default=2);parser.add_argument('--max-evaluations',type=int,default=180)
    parser.add_argument('--range-extent',type=int,default=8,choices=[8,16])
    parser.add_argument('--extra-origin',type=int)
    parser.add_argument('--origin-radius',type=float,default=1.)
    parser.add_argument('--split-delayed-source',type=Path,help='Test two overlapping measured trajectories at component D, seeded by the four non-ABC and ABC independent fits')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);output=out/'fits.json'
    if output.exists():raise ValueError('Use a fresh output directory')
    geometry=Path('docs/data/calibration_20261016_sites');fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    source=json.loads(args.source.read_text());bands,partitions=prepare(fields,geometry,args.first,args.last,
        range_extent=args.range_extent,observation_directory=out/'observations')
    choices=list(SELECTIONS[args.window])
    if args.extra_origin is not None:choices.append(('D',args.extra_origin))
    cluster_source=json.loads(args.split_delayed_source.read_text()) if args.split_delayed_source else None
    if cluster_source is not None:
        if 'D' not in cluster_source['labels'] or args.extra_origin is None:raise ValueError('Split control requires an existing D component and --extra-origin')
        choices.append(('E',args.extra_origin))
    labels,origins=zip(*choices);independent=[];started=time.perf_counter()
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,multipath_fit=False,
        interval_s=[args.first,args.last],labels=labels,selected_origins_samples=origins,origin_radius_samples=args.origin_radius,
        range_extent_samples=args.range_extent,
        split_delayed_source=str(args.split_delayed_source) if args.split_delayed_source else None,
        split_policy='D/E start at the same observed range cluster with independent range/phase/brightness and different shared quadratic paths seeded from non-ABC and ABC measurements. No fixed component separation; singularity and cancellation diagnostics remain required.' if cluster_source is not None else None,
        brightness_knots_s=bands[0].knots.tolist(),phase_model='Independent constant phases and envelope origins per component/band; shared or independent path rate and curvature. No time-dependent phase screen.',
        background_policy='No additional stationary columns or projections. Archived capture-time frozen subtraction remains the input.',
        numerical_policy='All declared observed ranges, exact original-row C0/C1 response and range covariance. Expanded original C0/C1 extraction must reproduce all seventeen existing cells bit for bit. Real nonnegative brightness NNLS; exact profiled residual Jacobian.',
        validation_policy='125 ms repeating blocks: 0 and 2 train; 1 validation; 3 final frozen prediction. Starts select minimum training cost. Original candidate data have previously been inspected.',
        independent=[],common=[],source_sha256={str(f['source']):f['source_sha256'] for f in fields})
    paths=[Path(__file__),args.source,*[Path('support/isar')/(n+'.py') for n in ('shared_echo_cluster','test_shared_echo_cluster','amplitude_pair','amplitude_arc','coherent_arc','field','stationary','cluster_phase','recorded_field_context','polynomial_phase_arc')],
        *(out/'observations').glob('channel-*.npz'),
        geometry/'geometry-prior.json',geometry/'sites-main.json',geometry/'acma-transmitters.json']
    if args.split_delayed_source:paths.append(args.split_delayed_source)
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for p in paths:
        with p.open('rb') as f:result['source_sha256'][str(p)]=hashlib.file_digest(f,'sha256').hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for ch,band in enumerate(bands):
        model=SharedEchoCluster([band],origins,origin_radius=args.origin_radius);seed=seed_for_band(source,model,ch,labels,origins,args.first,args.last,cluster_source)
        # Verify the real-data derivative before any nonlinear fit.
        r,j=model.evaluate(seed);direction=np.cos(np.arange(model.count))*.01;step=5e-5
        finite=(model.evaluate(seed+step*direction)[0]-model.evaluate(seed-step*direction)[0])/(2*step)
        error=float(np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-12))
        if error>3e-4:raise ValueError(f'Real-data residual derivative mismatch {error}')
        record=dict(channel=ch,name=NAMES[ch],initial_derivative_error=error,trials=[])
        result['independent'].append(record);save()
        for start in range(args.starts):
            initial=seed.copy()
            if start:initial[1:2*model.components:2]+=.45*np.cos(np.arange(model.components)+start)
            fit=model.fit(initial,args.max_evaluations);fit['start']=start
            fit['prediction']=subset_prediction(model,fit['parameters'],[partitions[ch]])
            record['trials'].append(fit);save()
            print(NAMES[ch],start,'cost',round(fit['cost'],2),'gradient',fit['maximum_projected_gradient'],
                'bounds',fit['bound_indices'],'unused',round(fit['prediction']['testing'][0]['heldout_error'],2),flush=True)
        record['best_index']=int(np.argmin([t['cost'] for t in record['trials']]))
        independent.append(record['trials'][record['best_index']]);save()
    model=SharedEchoCluster(bands,origins,origin_radius=args.origin_radius);n=model.components;seed=np.zeros(model.count)
    coefficients=[];weights=[]
    for ch,fit in enumerate(independent):
        p=np.asarray(fit['parameters']);seed[ch*2*n:(ch+1)*2*n]=p[:2*n]
        coefficients.append(p[-2*n:]);weights.append(fit['data_energy'])
    seed[-2*n:]=np.average(coefficients,axis=0,weights=weights)
    for start in range(args.starts):
        initial=seed.copy()
        if start:
            for s,label in enumerate(labels):
                if cluster_source is not None and label in ('D','E'):
                    source_seed=seed_for_band(source,SharedEchoCluster([bands[0]],origins,origin_radius=args.origin_radius),0,labels,origins,args.first,args.last,cluster_source)
                    index=model.count-2*n+2*s;initial[index:index+2]=source_seed[-2*n:].reshape(n,2)[s]
                    continue
                arc=next((a for a in source['arcs'] if a['point']==label and a['interval_s']==[args.first,args.last]),None)
                if arc is not None:
                    index=model.count-2*n+2*s
                    initial[index:index+2]=next(m for m in arc['models'] if m['degree']==2)['common']['parameters'][-2:]
        fit=model.fit(initial,args.max_evaluations);fit['start']=start;fit['prediction']=subset_prediction(model,fit['parameters'],partitions)
        result['common'].append(fit);save()
        print('COMMON',start,'cost',round(fit['cost'],2),'gradient',fit['maximum_projected_gradient'],
            'bounds',fit['bound_indices'],'unused',round(sum(r['heldout_error'] for r in fit['prediction']['testing']),2),flush=True)
    result['best_common_index']=int(np.argmin([r['cost'] for r in result['common']]));result['complete']=True;save()
    print('complete',round(result['seconds'],1),'seconds',flush=True)


if __name__=='__main__':main()
