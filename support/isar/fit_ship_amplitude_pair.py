"""Test overlapping ship echoes before treating either as a calibrator.

The original three-second fields and positive 0.5 s amplitude knots are kept.
Starts contain the shared physical trajectory and the distinct trajectory
preferred by Nine. Component paths may then move independently within the
same declared local search. This is an association/model diagnostic, not a
multichannel geometric fit or accepted radar-derived vessel motion.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from amplitude_arc import AmplitudeArc
from amplitude_pair import AmplitudePair
from fit_joint_sea_arcs import prepare_problem
from flash_geometry import observables
from motion import C
from report_joint_band_phase_control import independent_paths
from ship_arc_coherence import archived_observables


def phase_initialise(base,parameters):
    design=[]
    for p in parameters.reshape(2,4):
        trial=p.copy();trial[1]=0
        raw,_=base.response(trial)
        unit=base.reduced(raw[:,:,None])[:,0]
        design.append(base.nuisance.remove(unit[base.selected]))
    coefficients=np.linalg.lstsq(np.column_stack(design),base.clean,rcond=1e-10)[0]
    result=parameters.copy().reshape(2,4);result[:,1]=np.angle(coefficients)
    return result.ravel()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261017/ship-amplitude-pair-F01'))
    parser.add_argument('--interval',choices=['F01','F03','Flash'],default='F01')
    parser.add_argument('--origin-bounds',nargs=2,type=float,default=[-.8,2.8],metavar=('LOW','HIGH'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);output=args.output/'fits.json'
    if output.exists():raise ValueError('Use a new output directory')
    case=prepare_problem();i=next(i for i,e in enumerate(case['descriptions']) if e['label']==args.interval)
    single=independent_paths(case)[i]
    source=Path('docs/data/isar_20261017/joint-ship-band-phase-control/fits.json')
    joint=json.loads(source.read_text())
    row=min([r for r in joint['trials'] if r['family']=='direct' and r['audit']['converged']],key=lambda r:r['cost'])
    g=case['geometries'][i];local=np.asarray(row['prediction'][i]['parameters'])
    p,v,a,tx,rx=g.state(local[:28]);midpoint=(case['descriptions'][i]['first_s']+case['descriptions'][i]['last_s'])/2
    reference_nine=np.asarray(single['channels'][2]['observables'])
    geometry_mid=observables(p,v,a,tx,rx)
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,interval=case['descriptions'][i],
        origin_bounds_samples=args.origin_bounds,single_path=single,channels=[])
    sources=[source,Path(__file__),Path(__file__).with_name('amplitude_pair.py'),Path(__file__).with_name('amplitude_arc.py'),
        Path(__file__).with_name('fit_joint_sea_arcs.py')]
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sources}
    for ch,(f,tr,te,ks) in enumerate(zip(case['fields'][i],case['training'][i],case['testing'][i],case['knots'][i])):
        base=AmplitudeArc(f,tr,ks);model=AmplitudePair(base);dt=base.epoch-midpoint
        physical=observables(p+dt*v+.5*dt**2*a,v+dt*a,a,tx,rx)[ch]
        secondary=reference_nine+geometry_mid[ch]-geometry_mid[2]
        secondary=secondary+np.array([dt*secondary[1]+.5*dt*dt*secondary[2],dt*secondary[2],0])
        saved=archived_observables(f.data,base.epoch)
        def parameters(value):
            difference=value-saved
            return np.array([difference[0]/(C/8e6),0.,base.scale*difference[1],.5*base.scale**2*difference[2]])
        primary=parameters(physical);second=parameters(secondary);trials=[]
        for rate_offset in (0.,-.2,.2):
            seed=np.r_[primary,second];seed[6]+=rate_offset*base.scale
            seed=phase_initialise(base,seed)
            fit=model.fit(seed,args.origin_bounds)
            fit['initial_secondary_rate_offset_mps']=rate_offset
            fit['prediction']=model.prediction(np.asarray(fit['parameters']),te)
            pp=np.asarray(fit['parameters']).reshape(2,4)
            u=(midpoint-base.epoch)/base.scale
            transform=np.array([[C/8e6,0,u,u*u],[0,0,1/base.scale,2*u/base.scale],[0,0,0,2/base.scale**2]])
            fit['observables']=(archived_observables(f.data,midpoint)[None]+pp@transform.T).tolist()
            fit['numerically_regular']=bool(fit['information_rank']==8 and fit['projected_gradient']<1e-6 and
                not fit['bound_indices'] and min(fit['curvature_eigenvalues'])>0)
            trials.append(fit)
            print('channel',ch,'start',rate_offset,'objective',round(fit['objective'],6),
                'regular',fit['numerically_regular'],'rank',fit['information_rank'],'bounds',fit['bound_indices'],
                'heldout',round(fit['prediction']['heldout_explained_fraction'],4),flush=True)
        result['channels'].append(dict(channel=ch,trials=trials,best_index=int(np.argmin([t['objective'] for t in trials]))))
        output.write_text(json.dumps(result,indent=2,allow_nan=False))
    result['complete']=True;output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
