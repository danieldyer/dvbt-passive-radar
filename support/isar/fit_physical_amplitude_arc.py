"""Test a common physical ship point against the original complex arc data."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv
from cluster_phase import crop
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH,ais_observables
from imaging import sites
from motion import C
from physical_amplitude_arc import PhysicalAmplitudeArc,PhysicalAmplitudeFit
from ship_site_signature import licensed_options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/ship-phase-continuity/fits.json'))
    parser.add_argument('--fields',type=Path,default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261016_sites'))
    parser.add_argument('--interval',default='Flash context')
    parser.add_argument('--assignments',nargs='+')
    parser.add_argument('--refine-antennas',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'fits.json').exists():raise ValueError('Use a fresh output')
    source=json.loads(args.source.read_text())
    interval=next(v for v in source['intervals'] if v['label']==args.interval)
    first,last=interval['first_s'],interval['last_s'];midpoint=(first+last)/2
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    _,receiver,_=sites(args.geometry)
    rotation=np.array(json.loads((args.geometry/'geometry-prior.json').read_text())['ecef_to_enu'])
    options=assignments(licensed_options(args.geometry))
    code=lambda assignment:''.join({'12013':'0','12015':'1','12014':'2'}[v['site_id']] for v in assignment)
    if args.assignments:
        options=[v for v in options if code(v) in args.assignments]
        if len(options)!=len(args.assignments):raise ValueError('Unsupported assignment')
    fields,training,testing,knots,observed,covariance=[],[],[],[],[],[]
    sources=[args.source,Path(__file__),Path(__file__).with_name('physical_amplitude_arc.py'),Path(__file__).with_name('amplitude_arc.py'),
        Path(__file__).with_name('flash_geometry.py'),Path(__file__).with_name('field.py'),
        Path('docs/data/isar_20261013/rio-delta-ais.csv'),args.geometry/'geometry-prior.json',
        args.geometry/'sites-main.json',args.geometry/'acma-transmitters.json']
    for ch,row in enumerate(interval['channels']):
        path=args.fields/f'channel-{ch}.npz';sources.append(path)
        fit=row['alternatives'][row['chosen_index']]
        with np.load(path) as data:
            fields.append(crop(data,first,last,np.arange(-2,3)))
            reference=ais_observables(track,data['transmitter_ecef_m'][None,:],data['receiver_ecef_m'],midpoint)[0]
        training.append(np.array(fit['test']['training']));testing.append(np.array(fit['test']['testing']))
        knots.append(fit['test']['amplitude_knots_s'])
        u=(midpoint-row['epoch_s'])/row['scale_s'];s=row['scale_s']
        transform=np.array([[C/8e6,0,u,u*u],[0,0,1/s,2*u/s],[0,0,0,2/s**2]])
        observed.append(reference+transform@fit['parameters'])
        covariance.append(transform@fit['conditional_parameter_covariance']@transform.T)
    result={'scope':__doc__,'complete':False,'accepted_calibration':False,'interval_s':[first,last],
        'independent_arc_source':str(args.source),'independent_observables':np.array(observed).tolist(),
        'refine_antennas':args.refine_antennas,'trials':[]}
    started=time.perf_counter()
    for assignment in options:
        geometry=Flash(observed,covariance,track,EPOCH+midpoint,rotation,receiver,assignment,
            free_antennas=args.refine_antennas,receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.))
        seeds=[np.zeros(geometry.count)]
        for use_acceleration in (False,True):
            summary=Flash(observed,covariance,track,EPOCH+midpoint,rotation,receiver,assignment,
                free_antennas=args.refine_antennas,receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.),
                include_acceleration=use_acceleration)
            seeds.append(summary.fit().x)
        bands=[PhysicalAmplitudeArc(f,train,knot,geometry,ch,midpoint)
               for ch,(f,train,knot) in enumerate(zip(fields,training,knots))]
        model=PhysicalAmplitudeFit(bands)
        if not result['trials']:
            p=np.r_[seeds[0],np.zeros(5)];p[geometry.count:]=[b.phase_seed(p) for b in bands]
            _,grad=model.objective(p);checks=[]
            for angle in (0.,1.,2.):
                direction=np.sin(np.arange(len(p))+angle)*.1;step=1e-4
                numeric=(model.objective(p+step*direction)[0]-model.objective(p-step*direction)[0])/(2*step)
                checks.append({'numeric':numeric,'analytic':float(grad@direction),
                    'relative_error':float(abs(numeric-grad@direction)/max(abs(numeric),1e-8))})
            if max(r['relative_error'] for r in checks)>1e-4:raise ValueError(f'Raw geometry gradient failed: {checks}')
            result['raw_gradient_checks']=checks
        fit,starts=model.fit(seeds)
        value,grad=model.objective(fit.x);n=geometry.count
        lower,upper=geometry.bounds();projected=grad.copy()
        active=((fit.x[:n]<=lower+1e-6)&(grad[:n]>0))|((fit.x[:n]>=upper-1e-6)&(grad[:n]<0))
        projected[:n][active]=0
        record={'assignment':code(assignment),'parameters':fit.x.tolist(),'cost':value*model.energy,
            'objective':value,'data_energy':model.energy,'prior_cost':float(fit.x[:n]@fit.x[:n]),
            'maximum_projected_gradient':float(abs(projected).max()),'termination':str(fit.message),'iterations':fit.nit,
            'converged':bool(fit.success and abs(projected).max()<1e-6),
            'bound_indices':np.flatnonzero((fit.x[:n]<lower+1e-4)|(fit.x[:n]>upper-1e-4)).tolist(),
            'predicted_observables':geometry.prediction(fit.x[:n]).tolist(),
            'prediction':model.prediction(fit.x,testing),
            'initialisation_trials':[{'objective':float(f.fun),'gradient':float(abs(f.jac).max()),'iterations':f.nit,'termination':str(f.message)} for f in starts]}
        if args.refine_antennas:record['antenna_fit']=geometry.antennas.describe(fit.x[:18])
        record['reflector_nuisance_standardised']=fit.x[geometry.nantenna:n].tolist()
        result['trials'].append(record);result['seconds']=time.perf_counter()-started
        (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(record['assignment'],'cost',round(record['cost'],2),'pred',
            [round(v['test_complex_prediction_fraction'],3) for v in record['prediction']],
            'gradient',record['maximum_projected_gradient'],'bounds',record['bound_indices'],flush=True)
    result['complete']=True
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sources}
    (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
