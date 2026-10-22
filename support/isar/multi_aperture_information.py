"""Measure added site information from separated, fitted ship apertures.

Local motion and complex responses are eliminated per aperture. Antennas and
five range origins are shared; the origins are eliminated only after stacking
apertures. Fixed-scatterer and free-scatterer information are reported separately.
No optimizer bound or motion prior is counted as an observation.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import utc_seconds
from antenna_motion import Antennas
from audit_shared_geometry_information import summary
from field import Field
from full_motion import Band
from geometry_information import real_jacobian
from geometry_profile import GeometryProfile
from imaging import motion_state,sites
from scene_dictionary import fingerprint
from shared_origin_information import combined,local_root
from spline_motion import SplineMotion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,nargs='+',required=True)
    p.add_argument('--fields',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();started=time.monotonic()
    if args.output.exists():raise ValueError('Preserve existing results')
    models=[json.loads(path.read_text()) for path in args.models]
    first=models[0];prior=first['site_local_fit']
    antennas=Antennas(prior['ecef_to_enu'],prior['channel_sites'],site_sigma=prior['site_prior_sigma_enu_m'],
        antenna_sigma=prior['antenna_prior_sigma_enu_m'],receiver_sigma=prior['receiver_prior_sigma_enu_m'])
    tx,rx,_=sites(Path(first['geometry_directory']))
    intervals=sorted(m['interval_s'] for m in models)
    if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('Use disjoint apertures')
    fields=[]
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as d:fields.append(Field(dict(d),d['pulse_samples']))
    sources=[*args.models,*[args.fields/f'channel-{c}.npz' for c in range(5)],Path(__file__),
             *[Path(__file__).with_name(n) for n in ('geometry_information.py','profile_information.py',
                 'shared_origin_information.py','geometry_profile.py','physical_profile.py','full_motion.py')]]
    result={'complete':False,'scope':__doc__,'source_sha256':{str(f):fingerprint(f) for f in sources},
        'apertures':[],'summaries':[],'accepted_calibration':False,
        'limitation':'Local expected information with fitted amplitude/motion and the existing conditional range-noise metric. Independent apertures; same nominal assignment and shared origins. Fixed-point cases retain coarse geometric anchors; free 3-D point locations are the more conservative control. Weak predictive fits are not accepted calibration evidence.'}
    args.output.mkdir(parents=True);all_roots={}
    for index,(path,model) in enumerate(zip(args.models,models)):
        curve=SplineMotion(model)
        if not model['complete'] or not model['fit']['converged'] or model['original_row_kkt']>1e-7 or np.any(curve.yaw_coefficients):
            raise ValueError('Use completed original-row stationary, zero-yaw fits')
        np.testing.assert_allclose(model['range_bias_m'],first['range_bias_m'],rtol=0,atol=1e-10)
        if model.get('antenna_fit'):
            if max(np.max(abs(np.asarray(model['antenna_fit'][key]))) for key in
                   ('transmitter_offsets_enu_m','receiver_offset_enu_m'))>1e-6:
                raise ValueError('Keep nominal physical coordinates across all apertures')
        bands=[]
        for c,full in enumerate(fields):
            field=full.windowed(curve.interval[0]+.03,curve.interval[1]-.03)
            state=motion_state(None,utc_seconds(model['nominal_start_utc'])+field.data['row_time'],model)
            selected=np.floor((field.time-curve.interval[0])/.5).astype(int)%2==0
            bands.append(Band(field,state,tx[model['reference_channel']],rx,model['selected_points_body_m'][c],
                              model,selected,range_bias_m=model['range_bias_m'][c]))
        joint=GeometryProfile(bands,curve.parameters,antennas);parameters=joint.initial
        interest=np.r_[np.arange(len(parameters))[joint.antenna_slice],np.arange(len(parameters))[joint.range_slice]]
        yaw=(joint.active>=curve.sections[1].start)&(joint.active<curve.sections[1].stop)
        nuisance=np.setdiff1d(np.arange(joint.nmotion),np.flatnonzero(yaw))
        record={'source':str(path),'interval_s':model['interval_s'],'original_row_kkt':model['original_row_kkt'],
                'prediction':model.get('after_summary'),'cases':[]}
        roots={}
        for unknown in (False,True):
            for cutoff in (1e-8,1e-10,1e-12):
                j,ranks=real_jacobian(joint,parameters,point_coordinates=unknown,cutoff=cutoff)
                root,elimination=local_root(j,interest,nuisance,cutoff);del j
                key=f'points-{unknown}-cutoff-{cutoff:g}';roots[key]=root
                all_roots.setdefault(key,[]).append(root)
                information,rank=combined([root],antennas.count,cutoff)
                values=summary(information,antennas.factor)
                record['cases'].append({'point_coordinates_eliminated':unknown,'cutoff':cutoff,
                    'point_ranks':ranks,'motion_elimination':elimination,'shared_origin_elimination':rank,**values})
                print(json.dumps({'aperture':index,'free_points':unknown,'cutoff':cutoff,
                    'weak_directions':values['directions_weaker_than_declared_prior'],'seconds':time.monotonic()-started}),flush=True)
        output=args.output/f'aperture-{index}.npz';np.savez(output,**roots)
        record.update(arrays=str(output),arrays_sha256=fingerprint(output));result['apertures'].append(record)
        (args.output/'information.json').write_text(json.dumps(result,indent=2)+'\n')
    for key,roots in all_roots.items():
        cutoff=float(key.split('-cutoff-')[1]);cases={}
        for name,subset in [('original',[roots[0]]),('additional',roots[1:]),('combined',roots)]:
            information,rank=combined(subset,antennas.count,cutoff)
            cases[name]={**summary(information,antennas.factor),'shared_origin_elimination':rank}
        result['summaries'].append({'case':key,**cases})
    result.update(complete=True,seconds=time.monotonic()-started)
    (args.output/'information.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
