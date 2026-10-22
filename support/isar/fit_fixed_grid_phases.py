"""Separate measured moving phases from stationary profiles on a fixed grid.

Only nominal interband geometric differences are supplied by AIS. The fitted
quadratic is the radar-measured reference-channel path change, so AIS Hermite
acceleration at report knots cannot be mistaken for measured target motion.
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
from extract_full_aperture import load as load_aperture
from field import average
from fit_shared_echo_cluster import subset_prediction
from fixed_range_field import fixed_reference,stationary_profile_design
from report_measured_scatterer_phases import load_fields,serial
from shared_echo_cluster import ReferencedAmplitudeArc,SharedEchoCluster
from ship_site_signature import licensed_options


def prepare_fixed(fields,geometry,first,last,*,stationary,extent=16,observation_directory=None):
    options=licensed_options(geometry);geo=json.loads((geometry/'geometry-prior.json').read_text())
    receiver=np.asarray(geo['receiver_ecef_m']);mid=(first+last)/2
    reference_tx=np.asarray(next(o for o in options if o['channel']==2 and o['site_id']=='12013')['transmitter_ecef_m'])
    old=[];nominal=[];rates=[];transmitters=[]
    for c,f in enumerate(fields):
        field=crop(f['data'],first,last,np.arange(-8,9));d=field.data;old.append(field)
        tx=np.asarray(next(o for o in options if o['channel']==c and o['site_id']==('12014' if c==4 else '12013'))['transmitter_ecef_m'])
        transmitters.append(tx)
        path,rate=path_geometry(d['centre_ecef_m'],d['centre_velocity_ecef_mps'],tx,receiver)
        nominal.append(path);rates.append(rate)
    bands=[];partition=[];changes=[]
    for c,(field,path,rate) in enumerate(zip(old,nominal,rates)):
        rows=field.data['row_time']
        reference_path,reference_rate=path_geometry(field.data['centre_ecef_m'],
            field.data['centre_velocity_ecef_mps'],reference_tx,receiver)
        observer_rate=float(np.interp(mid,rows,rate));omega=-field.k*.001024*observer_rate
        aperture=load_aperture(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'))
        fixed,change=fixed_reference(field,aperture,mid,np.arange(-extent,extent+1),omega_reference=omega)
        # This known increment is tiny and geometric. It includes no assumed
        # common AIS trajectory, fitted phase rate or fitted curvature.
        known=path-reference_path;known-=np.interp(mid,rows,known)
        if np.array_equal(transmitters[c],reference_tx) and np.any(known!=0):
            raise ValueError('Co-sited bands must have exactly zero nominal interband path difference')
        known_rate=rate-reference_rate-observer_rate
        part=np.floor((fixed.time-first)/.125).astype(int)%4;training=part%2==0
        knots=np.r_[rows[0],np.arange(first+.25,last,.25),rows[-1]]
        design=stationary_profile_design(fixed) if stationary else None
        bands.append(ReferencedAmplitudeArc(fixed,training,knots,first,last,known,known_rate,stationary_design=design))
        partition.append(part);changes.append(change)
        if observation_directory:
            observation_directory.mkdir(exist_ok=True)
            np.savez(observation_directory/f'channel-{c}.npz',**{**fixed.data,'pulse_samples':fixed.pulse.samples})
    return bands,partition,changes


def acquisition_seeds(band,origin,radius=1.5):
    """Approximate midpoint tones acquire only; final fits use original rows."""
    rates=np.linspace(-3.,3.,121);candidates=[];f=band.field;use=band.training
    tone=np.exp(-1j*f.k*(f.time-band.epoch)[:,None]*rates)
    for offset in (origin-radius*.5,origin,origin+radius*.5):
        raw,_=band.response(np.array([offset,0.,0.,0.]))
        unit=band.reduced(raw[:,:,None])[:,0].reshape(f.y.shape)
        h=(unit[:,:,None]*tone[:,None,:]).reshape(f.y.size,len(rates))[band.selected]
        projected=band.nuisance.remove(h);norm=np.sum(abs(projected)**2,axis=0)
        before=np.sum(abs(h)**2,axis=0);retained=norm/np.maximum(before,1e-300)
        cross=projected.conj().T@band.clean
        score=np.where(retained>1e-3,abs(cross)**2/np.maximum(norm,1e-300),-np.inf)
        peaks=np.r_[0,np.flatnonzero((score[1:-1]>=score[:-2])&(score[1:-1]>=score[2:]))+1,len(rates)-1]
        for j in peaks:
            if np.isfinite(score[j]):
                candidates.append(dict(score=score[j],retained=retained[j],
                    parameters=[offset,float(np.angle(cross[j])),rates[j]*band.scale,0.]))
    ordered=sorted(candidates,key=lambda r:-r['score']);seeds=[]
    for item in ordered:
        if all(abs(item['parameters'][2]-s['parameters'][2])>.15*band.scale for s in seeds):seeds.append(item)
        if len(seeds)==3:break
    if not seeds:raise ValueError('No informative moving-template acquisition peak')
    return seeds


def diagnostics(model,p,partition):
    records=[]
    for cluster,index,part in zip(model.clusters,model.indices,partition):
        b=cluster.base;value=cluster.evaluate(p[index]);pred=value['design']@value['amplitude'];y=b.field.y.ravel()
        beta=b.nuisance.solve(y[b.selected]-pred[b.selected]);static=b.design@beta
        raw,_=cluster.response_blocks(p[index])[0];unit=b.reduced(raw[:,:,None])[:,0].reshape(b.field.y.shape)
        clean=(y-static).reshape(b.field.y.shape);predicted=pred.reshape(b.field.y.shape)
        product=np.sum(unit.conj()*clean,axis=1);mean=np.sum(unit.conj()*predicted,axis=1)
        phase=np.angle(product*mean.conj());weight=np.sum(abs(predicted)**2,axis=1);test=(part==3)&(weight>0)
        rms=float(np.degrees(np.sqrt(np.average(phase[test]**2,weights=weight[test])))) if test.any() else None
        template=pred[b.selected];projected=b.nuisance.remove(template)
        retained=float(np.vdot(projected,projected).real/np.vdot(template,template).real)
        signal_matrix=b.nuisance.remove(value['design'][b.selected])
        original_matrix=value['design'][b.selected]
        # Every brightness direction must keep its actual projected energy.
        column_retention=np.sum(abs(signal_matrix)**2,axis=0)/np.maximum(np.sum(abs(original_matrix)**2,axis=0),1e-300)
        singular=b.nuisance.singular_values
        records.append(dict(time_s=b.field.time,testing=part==3,phase_residual_deg=np.degrees(phase),
            conditional_phase_rms_deg=rms,primary_information=weight,
            retained_fitted_signal_energy=retained,brightness_column_retention=column_retention,
            stationary_rank=b.nuisance.rank,stationary_condition=float(singular.max()/singular.min()) if len(singular) else 1.,
            stationary_coefficients_real=beta.real,stationary_coefficients_imag=beta.imag,
            nuisance_policy='Constant profile coefficients fitted on training residual after the target; held fixed on all unused rows. Target and data receive the same training projection. Phase statistics are conditional and do not include source-association error.'))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window',choices=['AIS1','AIS2','AIS3','Flash'],default='AIS3')
    parser.add_argument('--first',type=float,help='Override measured-window start; use with --last')
    parser.add_argument('--last',type=float)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--modes',nargs='+',choices=['raw','stationary'],default=['raw','stationary'])
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);output=out/'fits.json'
    if output.exists():raise ValueError('Use a fresh output')
    geometry=Path('docs/data/calibration_20261016_sites');source=Path('docs/data/isar_20261017/ais-epoch-phase-candidates/measurements.json')
    chosen=json.loads(source.read_text())
    if args.window=='Flash':first,last,origin=163.,167.,0.
    else:
        window=next(w for w in chosen['windows'] if w['label']==args.window)
        first,last=window['interval_s'];origin=next(p['offset_samples'] for p in window['points'] if p['name']=='B')
    if args.first is not None or args.last is not None:
        if args.first is None or args.last is None:raise ValueError('Both interval overrides are required')
        first,last=args.first,args.last
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry);started=time.perf_counter()
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        window=args.window,interval_s=[first,last],selected_origin_samples=origin,groups=[],
        phase_policy='Fit the physical reference-band path change directly as a quadratic, with independent constant phases. Only nominal interband geometric differences use AIS. No common AIS interpolation slope or curvature is imposed.',
        stationary_policy='One constant complex vector over the actual fixed-grid observed cells. Its design is orthogonal; no unobserved delay prototypes or extrapolated coefficients. Target and data receive the same projection; retained target information is reported. No production correction adopted.',
        split_policy='125 ms blocks 0/2 train; block 3 predicts with all fitted coefficients frozen. Initial acquisitions and minimum-cost start selection use training only; these are internally inspected data.',
        source_sha256={str(f['source']):f['source_sha256'] for f in fields})
    files=[Path(__file__),source,*[Path('support/isar')/(n+'.py') for n in
        ('fixed_range_field','test_fixed_range_field','recorded_field_context','shared_echo_cluster','amplitude_arc','amplitude_pair','coherent_arc','field','stationary')]]
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for p in files:
        with p.open('rb') as stream:result['source_sha256'][str(p)]=hashlib.file_digest(stream,'sha256').hexdigest()
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    def save():
        result['seconds']=time.perf_counter()-started;output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for mode in args.modes:
        bands,part,changes=prepare_fixed(fields,geometry,first,last,stationary=mode=='stationary',observation_directory=out/(mode+'-observations'))
        group=dict(mode=mode,reference_changes=changes,independent=[],common=[]);result['groups'].append(group);save()
        best_parameters=[]
        for c,b in enumerate(bands):
            model=SharedEchoCluster([b],[origin],origin_radius=1.5);acquisition=acquisition_seeds(b,origin)
            entry=dict(channel=c,name=NAMES[c],acquisition=acquisition,trials=[]);group['independent'].append(entry);save()
            for start,initial in enumerate(acquisition):
                fit=model.fit(np.asarray(initial['parameters']),180);p=np.asarray(fit['parameters'])
                fit['start']=start;fit['prediction']=subset_prediction(model,p,[part[c]])
                fit['diagnostics']=diagnostics(model,p,[part[c]])
                entry['trials'].append(fit);save()
            entry['best_index']=int(np.argmin([f['cost'] for f in entry['trials']]))
            best=entry['trials'][entry['best_index']];best_parameters.append(best['parameters'])
            print(args.window,mode,NAMES[c],'unused',best['prediction']['testing'][0]['heldout_error'],
                'phase',best['diagnostics'][0]['conditional_phase_rms_deg'],
                'retained',best['diagnostics'][0]['retained_fitted_signal_energy'],flush=True)
        model=SharedEchoCluster(bands,[origin],origin_radius=1.5);seed=np.zeros(model.count)
        for c,p in enumerate(best_parameters):seed[2*c:2*c+2]=p[:2]
        coefficients=np.asarray(best_parameters)[:,2:]
        for start in range(2):
            seed[-2:]=np.average(coefficients,axis=0,weights=[b.energy for b in bands]) if start==0 else np.median(coefficients,axis=0)
            fit=model.fit(seed.copy(),240);p=np.asarray(fit['parameters'])
            fit['start']=start;fit['prediction']=subset_prediction(model,p,part);fit['diagnostics']=diagnostics(model,p,part)
            group['common'].append(fit);save()
        group['best_common_index']=int(np.argmin([f['cost'] for f in group['common']]))
        print(args.window,mode,'common complete',flush=True)
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
