"""Check mixed-scene quadrature after physical motion fitting, without retuning."""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from ais import utc_seconds
from audit_distributed_motion import physical_check
from distributed_fixed_frame import prediction_metrics, quadrature
from field import Field
from full_motion import Band
from imaging import motion_state, sites
from midpoint_model import discrepancy
from mixed_scene import profile
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationary import Stationary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spacing',type=float,default=10.)
    args=p.parse_args()
    model=json.loads(args.motion.read_text())
    source=json.loads(Path(model['training']).read_text())
    initial=json.loads(Path(model['source_motion']).read_text())
    curve=SplineMotion(model)
    if not model['complete'] or not model['fixed_ne_frame'] or np.any(curve.yaw_coefficients) or args.output.exists():
        raise ValueError('Need a completed yaw-free fit and fresh output')
    tx,rx,_=sites(Path(model['geometry_directory']))
    started=wall_time.monotonic()
    result={'scope':__doc__,'motion':str(args.motion),'motion_sha256':fingerprint(args.motion),
        'original_spacing_m':model['spacing_m'],'refined_spacing_m':args.spacing,
        'physical_checks':physical_check(model,tx,rx),'channels':[],'complete':False,
        'implementation_sha256':{name:fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_mixed_motion.py','audit_distributed_motion.py','mixed_scene.py','full_motion.py','field.py')}}
    for c,strength in enumerate(model['fixed_penalties']):
        if strength is None:
            result['channels'].append({'channel':c,'core_only':True,'no_spatial_quadrature':True})
            continue
        core=np.asarray(model['core_points_enu_m'][c])
        with np.load(Path(source['fields'])/f'channel-{c}.npz') as d:
            field=Field(dict(d),d['pulse_samples']).windowed(curve.interval[0]+.03,curve.interval[1]-.03)
        prior=motion_state(None,utc_seconds(model['nominal_start_utc'])+field.data['row_time'],model)
        selected=np.floor((field.time-curve.interval[0])/source['holdout_seconds']).astype(int)%2==0
        trials,predictions=[],[]
        determinants=None
        for spacing in (model['spacing_m'],args.spacing):
            diffuse,area=quadrature(source['bounds_enu_m'],spacing)
            scale=np.sqrt(np.r_[np.ones(len(core)),area])
            band=Band(field,prior,tx[model['reference_channel']],rx,np.r_[core,diffuse],model,selected,
                range_bias_m=model['range_bias_m'][c])
            fit=profile(band,model['parameters'],scale,len(core),strength)
            if spacing==model['spacing_m']:
                determinants=[]
                for parameters in (initial['parameters'],model['parameters']):
                    target=fit['target'] if parameters is model['parameters'] else band.design(parameters)[0]
                    diffuse_design=band.nuisance.remove(target[band.selected])[:,len(core):]*scale[len(core):]
                    singular=np.linalg.svd(diffuse_design,compute_uv=False)
                    determinants.append(float(np.log1p(singular**2/strength).sum()))
            nuisance=Stationary(band.nuisance.design,~band.selected)
            y=nuisance.remove(field.y.ravel()[~band.selected]).reshape((~selected).sum(),-1)
            prediction=nuisance.remove((fit['target']@fit['physical_coefficient'])[~band.selected]).reshape(y.shape)
            predictions.append(prediction)
            trials.append({'spacing_m':spacing,'points':len(band.points),'effective_rank':fit['effective_rank'],
                'validation':prediction_metrics(y,prediction,field.time[~selected])})
            del fit
        old,new=[v['validation'] for v in trials]
        channel={'channel':c,'trials':trials,
            'diffuse_covariance_log_determinant_before_after':determinants,
            'log_determinant_change_per_training_energy':(determinants[1]-determinants[0])/band.energy,
            'matched_energy_change_db':float(10*np.log10(new['heldout_matched_energy']/old['heldout_matched_energy'])),
            'prediction_error_change_percent':100*(new['prediction_error']/old['prediction_error']-1),
            'prediction_discrepancy':discrepancy(predictions[0],predictions[1]),
            'original_score_reproduction_relative_error':abs(old['prediction_error']/model['after'][c]['prediction_error']-1)}
        result['channels'].append(channel)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'channel':c,'gain_change_db':channel['matched_energy_change_db'],
            'error_change_percent':channel['prediction_error_change_percent'],
            'seconds':wall_time.monotonic()-started}),flush=True)
    result.update(complete=True,seconds=wall_time.monotonic()-started)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
