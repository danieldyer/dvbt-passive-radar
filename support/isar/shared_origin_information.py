"""Location information with origins shared across independent targets.

Eliminate each target's own motion first, retain shared antenna AND origin
columns, then eliminate the origins after combining targets. Eliminating the
origins separately would incorrectly allow a different clock for each target.
No optimizer bounds or motion/location priors count as measured information.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_shared_geometry_information import summary
from fit_shared_target_geometry import load_experiment
from geometry_information import real_jacobian
from profile_information import from_jacobian
from scene_dictionary import fingerprint
from shared_origin_geometry import SharedOriginGeometry
from spline_motion import SplineMotion


def local_root(jacobian,interest,nuisance,cutoff):
    """Project local nuisance, then compress rows without dropping any columns."""
    jacobian=np.asarray(jacobian)
    local=jacobian[:,nuisance]
    norm=np.linalg.norm(local,axis=0)
    u,s,_=np.linalg.svd(local/np.where(norm>0,norm,1),full_matrices=False)
    keep=s>s.max(initial=0)*np.sqrt(cutoff)
    residual=jacobian[:,interest]-u[:,keep]@(u[:,keep].T@jacobian[:,interest])
    root=np.linalg.qr(residual,mode='r')
    return root,{'parameters':len(nuisance),'rank':int(keep.sum()),'cutoff':cutoff}


def combined(roots,antennas,cutoff):
    root=np.concatenate(roots)
    return from_jacobian(root,np.arange(antennas),np.arange(antennas,root.shape[1]),cutoff=cutoff)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','ship-fit','road-fit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fields',type=Path,default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261013'))
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve existing information audit')
    fit=json.loads(args.fit.read_text())
    if not fit['complete'] or not fit['fit']['converged'] or fit['original_row_maximum_kkt_residual']>1e-7:
        raise ValueError('Use a stationary completed fit')
    for group in ('sources','implementation_sha256'):
        for file,digest in fit['identity']['legacy_identity'][group].items():
            if fingerprint(Path(file))!=digest:raise ValueError('A source changed: '+file)
    for file,digest in fit['identity']['implementation_sha256'].items():
        if fingerprint(Path(file))!=digest:raise ValueError('Fit implementation changed: '+file)
    for file,digest in fit.get('numerical_refinement',{}).get('implementation_sha256',{}).items():
        if fingerprint(Path(file))!=digest:raise ValueError('Numerical refinement implementation changed: '+file)
    identity=fit['identity']['legacy_identity']
    legacy,ship,_,labels,_=load_experiment(args.ship_fit,args.road_fit,args.fields,args.geometry,identity['labels'],identity['nodes'])
    joint=SharedOriginGeometry(legacy);parameters=np.asarray(fit['parameters'])
    antennas=ship.antennas.count
    interest=np.r_[np.arange(joint.ship_count)[joint.antenna_slice],np.arange(joint.ship_count)[joint.range_slice]]
    curve=SplineMotion(ship.bands[0].model)
    yaw=(ship.active>=curve.sections[1].start)&(ship.active<curve.sections[1].stop)
    nuisance=np.setdiff1d(np.arange(ship.nmotion),np.flatnonzero(yaw))
    road_jacobians=[road.evaluate(parameters[section],parameters[joint.antenna_slice],parameters[joint.range_slice],
        include_motion_prior=False)['expected_real_jacobian'] for road,section in zip(joint.roads,joint.road_slices)]
    result={'scope':__doc__,'fit':str(args.fit),'fit_sha256':fingerprint(args.fit),'cases':[],'complete':False,
        'source_sha256':{str(f):fingerprint(f) for f in [Path(__file__),Path(__file__).with_name('shared_origin_geometry.py'),
            Path(__file__).with_name('geometry_information.py'),Path(__file__).with_name('profile_information.py')]}}
    for free_points in (False,True):
        for cutoff in (1e-8,1e-10,1e-12):
            jacobian,point_ranks=real_jacobian(ship,parameters[:joint.ship_count],point_coordinates=free_points,cutoff=cutoff)
            root,ship_rank=local_root(jacobian,interest,nuisance,cutoff)
            roads=[];road_ranks=[]
            for road,jac in zip(joint.roads,road_jacobians):
                r,rank=local_root(jac,np.arange(road.nmotion,jac.shape[1]),np.arange(road.nmotion),cutoff)
                roads.append(r);road_ranks.append(rank)
            summaries={}
            for name,roots in [('ship_only',[root]),('vehicles_only',roads),('combined',[root,*roads])]:
                information,rank=combined(roots,antennas,cutoff)
                summaries[name]={**summary(information,ship.antennas.factor),'shared_origin_elimination':rank}
            item={'ship_point_coordinates_eliminated':free_points,'cutoff':cutoff,'point_ranks':point_ranks,
                'ship_motion_elimination':ship_rank,'road_motion_elimination':road_ranks,**summaries}
            result['cases'].append(item)
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'free_points':free_points,'cutoff':cutoff,'weak_combined':summaries['combined']['directions_weaker_than_declared_prior']}),flush=True)
    result['complete']=True;args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
