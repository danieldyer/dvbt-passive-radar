"""Two-echo refits with disjoint-time background mean and finite uncertainty.

Unrestricted local clutter, fixed independent background (limiting control),
and a finite proper-complex Gaussian background model use the same ship data,
two original-row echo responses, brightness knots, bounds and initial paths.
The finite covariance is learned only from 30--90 s block means. Transfer of
that independent-block model is imperfect; this is a sensitivity diagnostic,
not an adopted background model or antenna calibration.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from amplitude_arc import AmplitudeArc
from amplitude_pair import AmplitudePair
from background_reference import field_reference
from cluster_phase import crop
from fit_joint_sea_arcs import prepare_problem
from fit_ship_amplitude_pair import phase_initialise
from gaussian_background import predictive_factors,set_background
from motion import C
from report_background_extrapolation import original_values
from ship_arc_coherence import archived_observables
from stationary import Stationary


def build_model(field,training,knots,reference,mode,stationary_delays=None):
    if mode=='local':
        base=AmplitudeArc(field,training,knots)
        if stationary_delays is not None:
            # Add observations without inventing further nuisance columns.
            base.delays=np.asarray(stationary_delays)
            base.design=np.column_stack([field.stationary(delay).ravel() for delay in base.delays])
            base.nuisance=Stationary(base.design,base.selected)
            base.clean=base.nuisance.remove(field.y.ravel()[base.selected])
            base.energy=float(np.vdot(base.clean,base.clean).real)
        return AmplitudePair(base),np.zeros_like(field.y)
    values=np.stack([field_reference(field,m,int(reference['first_range_sample'])).ravel()
        for m in reference['means'][:12]])
    mean,factor=predictive_factors(values,reference['information'][:12])
    if mode=='fixed':factor=factor[:,:0]
    elif mode!='finite':raise ValueError('Unknown background model')
    corrected=copy.copy(field);corrected.y=field.y-mean.reshape(field.y.shape)
    base=set_background(AmplitudeArc(corrected,training,knots),factor)
    return AmplitudePair(base),mean.reshape(field.y.shape)


def diagnostics(model,parameters,testing,mean,core=None):
    value=model.evaluate(parameters);b=model.base;k=len(b.knots)
    parts=np.stack([value['design'][:,j*k:(j+1)*k]@value['amplitude'][j*k:(j+1)*k] for j in range(2)])
    beta=b.nuisance.solve(b.field.y.ravel()[b.selected]-parts.sum(axis=0)[b.selected])
    static=b.design@beta;prediction=parts.sum(axis=0)+static+mean.ravel()
    full=np.r_[parts,static[None]];use=np.repeat(testing,len(b.field.offsets))
    denominator=max(float(np.sum(abs(prediction[use])**2)),1e-300)
    result=dict(component_plus_adjustable_background_energy_ratio=float(np.sum(abs(full[:,use])**2)/denominator),
        adjustable_background_unused_energy=float(np.sum(abs(static[use])**2)),
        fixed_reference_unused_energy=float(np.sum(abs(mean.ravel()[use])**2)),
        component_unused_energy=np.sum(abs(parts[:,use])**2,axis=1).tolist(),
        raw_training_error=float(np.sum(abs(b.field.y.ravel()[b.selected]-parts.sum(axis=0)[b.selected]-static[b.selected])**2)))
    if core is not None:
        columns=np.array([np.flatnonzero(b.field.offsets==offset)[0] for offset in core.offsets])
        raw=original_values(b.field,prediction.reshape(b.field.y.shape))
        residual=core.y-core.apply(raw[:,columns])
        result['core_unused_error']=float(np.sum(abs(residual[testing])**2))
        result['core_observed_unused_energy']=float(np.sum(abs(core.y[testing])**2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('docs/data/isar_20261017/background-reference'))
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/background-pair-refits'))
    parser.add_argument('--range-context',action='store_true',help='Use all 17 recorded cells with the original 5-cell nuisance delay grid')
    parser.add_argument('--modes',nargs='+',choices=['local','fixed','finite'],default=['local','fixed','finite'])
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    path=out/'fits.json'
    if path.exists():raise ValueError('Use a fresh output directory')
    case=prepare_problem();sources=[Path(__file__),Path('support/isar/gaussian_background.py'),
        Path('support/isar/background_reference.py'),Path('support/isar/amplitude_pair.py'),args.reference/'analysis.json']
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,intervals=[],
        reference=str(args.reference),background_modes=args.modes,range_context=args.range_context,
        initializations='Every saved path/phase start and its independent complex phase initialization; no converged phase seed is discarded.',
        reference_uncertainty='Finite mode uses the unbiased weighted covariance of twelve independent 5-second block means, multiplied by 1 + sum normalized_weight^2 for a future draw minus the learned mean. It is conditional on stationarity and proper complex Gaussian fluctuations; independent control intervals already show drift, so these are not calibrated confidence intervals.',
        selection='Minimum training objective within each background model only. Different covariance models are compared by original unused complex-data prediction error, not their differently weighted training objectives.')
    started=time.perf_counter()
    for label in ('F01','Flash'):
        i=next(i for i,e in enumerate(case['descriptions']) if e['label']==label)
        source=Path('docs/data/isar_20261017')/f'ship-amplitude-pair-{label}{"-wider" if label=="F01" else ""}'/'fits.json'
        previous=json.loads(source.read_text());sources.append(source)
        interval=dict(description=case['descriptions'][i],channels=[]);result['intervals'].append(interval)
        midpoint=(case['descriptions'][i]['first_s']+case['descriptions'][i]['last_s'])/2
        for ch,(f,tr,te,ks) in enumerate(zip(case['fields'][i],case['training'][i],case['testing'][i],case['knots'][i])):
            core=f;delays=None
            if args.range_context:
                delays=AmplitudeArc(core,tr,ks).delays
                with np.load(Path(f'build/isar-20261015/full-fields/channel-{ch}.npz')) as data:
                    f=crop(data,case['descriptions'][i]['first_s'],case['descriptions'][i]['last_s'],np.arange(-8,9))
            refpath=args.reference/f'channel-{ch}-means.npz';sources.append(refpath)
            with np.load(refpath) as data:reference=dict(data)
            channel=dict(channel=ch,modes=[]);interval['channels'].append(channel)
            seeds=[np.asarray(r['parameters']) for r in previous['channels'][ch]['trials']]
            for mode in result['background_modes']:
                model,mean=build_model(f,tr,ks,reference,mode,delays);base=model.base;trials=[]
                starts=[(start,phase,phase_initialise(base,seed) if phase else seed.copy())
                    for start,seed in enumerate(seeds) for phase in (False,True)]
                for start,phase,seed in starts:
                    fit=model.fit(seed,(-2.8,2.8));p=np.asarray(fit['parameters'])
                    fit['start_index']=start;fit['phase_reinitialized']=phase;fit['weighted_training_cost']=fit['objective']*model.energy
                    fit['prediction']=model.prediction(p,te);fit['decomposition']=diagnostics(model,p,te,mean,core)
                    u=(midpoint-base.epoch)/base.scale
                    transform=np.array([[C/8e6,0,u,u*u],[0,0,1/base.scale,2*u/base.scale],[0,0,0,2/base.scale**2]])
                    fit['observables']=(archived_observables(f.data,midpoint)[None]+p.reshape(2,4)@transform.T).tolist()
                    fit['numerically_regular']=bool(fit['information_rank']==8 and fit['projected_gradient']<1e-6 and
                        not fit['bound_indices'] and min(fit['curvature_eigenvalues'])>0)
                    trials.append(fit)
                best=int(np.argmin([r['objective'] for r in trials]));record=trials[best]
                channel['modes'].append(dict(mode=mode,trials=trials,best_index=best,energy=model.energy,
                    stationary_delays_m=None if mode!='local' else base.delays.tolist(),
                    stationary_condition=None if mode!='local' else float(base.nuisance.singular_values[0]/base.nuisance.singular_values[base.nuisance.rank-1])))
                print(label,ch,mode,'unused',round(record['prediction']['heldout_error'],3),
                    'regular',record['numerically_regular'],'including-background ratio',round(record['decomposition']['component_plus_adjustable_background_energy_ratio'],3),
                    'rates/curvatures',np.round(np.asarray(record['observables'])[:,1:],4).tolist(),flush=True)
                result['elapsed_s']=time.perf_counter()-started
                path.write_text(json.dumps(result,indent=2,allow_nan=False))
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in set(sources+case['field_paths'])}
    result['complete']=True;path.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
