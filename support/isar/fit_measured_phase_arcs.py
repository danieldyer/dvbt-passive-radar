"""Independent versus shared short polynomial paths for measured candidates.

Half the original retained epochs fit phase and real brightness; one quarter
selects quadratic/cubic degree; the final quarter predicts without refitting.
No stationary nuisance, multipath, antenna or time-varying phase screen enters.
The saved original-row phase averaging is included exactly. Fixed range cells
remain mixtures, so a good local fit alone is not a calibration certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from cluster_phase import crop
from polynomial_phase_arc import Band,PolynomialPhaseArc
from report_measured_scatterer_phases import load_fields,WINDOWS,SELECTIONS,serial


def make_bands(fields,first,last,origin):
    bands=[]
    for f in fields:
        data=f['data'];field=crop(data,first,last,np.array([origin]));d=field.data
        rows=d['row_time'];nominal=np.interp(rows,data['row_time'],f['nominal'])
        known=nominal-d['centre_path_m'];known-=np.interp((first+last)/2,rows,known)
        known_bin=np.interp(field.time,rows,known)
        partition=(np.floor((field.time-first)/.125).astype(int)%4)
        training=partition%2==0
        bands.append(Band(field.time,d['value'][:,0],8192*d['information'],known_bin,f['k'],training,
            rows,d['row_information'],known,int(field.metadata['block_rows'])))
    return bands


def subset_scores(model,parameters,first):
    parameters=np.asarray(parameters);results=[]
    for c,b in enumerate(model.bands):
        h,_=model.response(parameters,c);train=b.training&(b.weight>0)
        a=np.maximum(0.,(b.value[train]*h[train].conj()).real)/np.maximum(abs(h[train])**2,1e-300)
        amp=np.interp(b.time,b.time[train],a,left=np.nan,right=np.nan)
        prediction=amp*h;partition=(np.floor((b.time-first)/.125).astype(int)%4)
        record={}
        for name,p in (('validation',1),('testing',3)):
            use=(partition==p)&(b.weight>0)&np.isfinite(amp);w=b.weight[use];y=b.value[use];pred=prediction[use]
            error=np.sum(w*abs(y-pred)**2);energy=np.sum(w*abs(y)**2)
            phase=np.angle(y*h[use].conj());phasew=w*abs(pred)**2
            cross=np.sum(w*pred.conj()*y);norm=np.sum(w*abs(pred)**2)
            record[name]=dict(error=float(error),energy=float(energy),samples=int(use.sum()),
                explained_fraction=float(1-error/energy) if energy>0 else None,
                phase_rms_deg=float(np.degrees(np.sqrt(np.average(phase**2,weights=phasew)))) if phasew.sum()>0 else None,
                matched_power=float(abs(cross)**2/norm) if norm>0 else None,
                coherence_squared=float(abs(cross)**2/(norm*energy)) if norm*energy>0 else None,
                use=use.tolist())
        results.append(record)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields',type=Path,default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261016_sites'))
    p.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/piecewise-measured-phases'))
    p.add_argument('--windows',nargs='+',default=['W1','W2','W3','W4'])
    p.add_argument('--degrees',nargs='+',type=int,default=[2,3])
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    target=out/'fits.json'
    if target.exists() and not args.resume:raise ValueError('Use a fresh output directory or --resume')
    fields=load_fields(args.fields,args.geometry);started=time.perf_counter()
    result=dict(scope=__doc__,complete=False,arcs=[],selection_source='docs/data/isar_20261017/measured-scatterer-phases/measurements.json',
        accepted_calibration=False,antenna_fit=False,multipath_fit=False,production_changed=False,
        phase_model='Each band has its own constant phase and nonnegative real brightness at training epochs. Only residual polynomial path coefficients are shared in the common model.',
        operator='Original sixteen-row information-weighted phase average; brightness assumed constant within one retained bin. Range-cell envelope is not refitted: fitted paths remain phase-equivalent candidate trajectories.',
        validation_policy='Partition repeating 125 ms blocks: 0 and 2 train; 1 selects degree; 3 final prediction. Original candidate selections used already inspected data, so this is internal validation, not blind external evidence.',
        source_sha256={str(f['source']):f['source_sha256'] for f in fields})
    sources=[Path(__file__),Path(__file__).with_name('polynomial_phase_arc.py'),Path(__file__).with_name('test_polynomial_phase_arc.py'),
        Path(__file__).with_name('report_measured_scatterer_phases.py'),args.geometry/'geometry-prior.json',
        args.geometry/'sites-main.json',args.geometry/'acma-transmitters.json']
    for s in sources:
        with s.open('rb') as stream:result['source_sha256'][str(s)]=hashlib.file_digest(stream,'sha256').hexdigest()
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    if target.exists():
        previous=json.loads(target.read_text())
        for source,digest in result['source_sha256'].items():
            if source!=str(Path(__file__)) and previous['source_sha256'][source]!=digest:
                raise ValueError(f'Resumption would change model or observations: {source}')
        old=previous['source_sha256'][str(Path(__file__))]
        if old!=result['source_sha256'][str(Path(__file__))]:
            archive=snapshot/(Path(__file__).stem+'-'+old[:12]+'.py')
            if not archive.exists():shutil.copyfile(snapshot/Path(__file__).name,archive)
            previous.setdefault('runner_history',[]).append(dict(sha256=old,snapshot=str(archive),
                change='Resume completed numerical work after terminal logging encountered a missing phase statistic.'))
        previous['source_sha256']=result['source_sha256'];result=previous
        started-=result.get('seconds',0.)
    for s in sources:
        if s.suffix=='.py':shutil.copyfile(s,snapshot/s.name)
    def save():
        result['seconds']=time.perf_counter()-started;target.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    for window,begin,end in WINDOWS:
        if window not in args.windows:continue
        for first in np.arange(begin,end-1.,1.):
            last=first+2
            for name,origin in SELECTIONS[window]:
                existing=next((a for a in result['arcs'] if a['window']==window and a['point']==name and a['interval_s']==[first,last]),None)
                if existing is not None and 'selected_final_error_ratio_common_to_independent' in existing:continue
                bands=make_bands(fields,first,last,origin)
                arc=existing if existing is not None else dict(window=window,point=name,origin_samples=origin,interval_s=[first,last],models=[])
                if existing is None:result['arcs'].append(arc)
                save()
                for degree in args.degrees:
                    if any(m['degree']==degree for m in arc['models']):continue
                    independent=[]
                    for band in bands:
                        model=PolynomialPhaseArc([band],first,last,degree);fit=model.fit()
                        fit['scores']=subset_scores(model,fit['parameters'],first);independent.append(fit)
                    coefficients=np.array([fit['parameters'][-degree:] for fit in independent])
                    energies=[fit['data_energy'] for fit in independent]
                    common=PolynomialPhaseArc(bands,first,last,degree)
                    fit=common.fit(extra=[np.average(coefficients,axis=0,weights=energies)])
                    fit['scores']=subset_scores(common,fit['parameters'],first)
                    arc['models'].append(dict(degree=degree,common=fit,independent=independent));save()
                for family in ('common','independent'):
                    def validation(m):
                        rows=[m[family]] if family=='common' else m[family]
                        return sum(c['validation']['error'] for r in rows for c in r['scores'])
                    arc['selected_'+family+'_degree']=min(arc['models'],key=validation)['degree']
                common=next(m for m in arc['models'] if m['degree']==arc['selected_common_degree'])['common']
                independent=next(m for m in arc['models'] if m['degree']==arc['selected_independent_degree'])['independent']
                common_error=sum(r['testing']['error'] for r in common['scores'])
                independent_error=sum(r['scores'][0]['testing']['error'] for r in independent)
                arc['selected_final_error_ratio_common_to_independent']=common_error/independent_error
                arc['selected_test_phase_rms_deg']=[r['testing']['phase_rms_deg'] for r in common['scores']]
                save();print(window,name,f'{first:g}-{last:g}',
                    'degrees',arc['selected_common_degree'],arc['selected_independent_degree'],
                    'common/independent',round(common_error/independent_error,3),
                    'phase',[None if q is None else round(q,1) for q in arc['selected_test_phase_rms_deg']],flush=True)
    result['complete']=True;save();print('complete',len(result['arcs']),'arcs',round(result['seconds'],1),'seconds',flush=True)


if __name__=='__main__':main()
