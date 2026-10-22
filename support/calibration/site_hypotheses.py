"""Test licensed main/alternative sites before refining antenna coordinates.

Licensing establishes permitted configurations, not which backup was active
during a recording. These fits are offline hypotheses and never update capture
metadata. Current geometries use metadata coordinates and AGL heights shared
by all bands at a site. Historical geometries retain their licensed heights
for reproducibility. Licence roles are distinct from configured roles.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fit_geometry import observations
from geographic_io import read_profiles
from geometry import ecef
from phase_fit import load_histories
from polar_model import PolarTrajectory
from range_fit import RangeProblem,load_fields
from terrain import Terrain
from transmitter_config import nominal_indices, validate_assignment

NAMES=['Seven','SBS','Nine','Ten','ABC']
CALLS=['HSV6','SBS7','GTV8','ATV11','ABC12']
SITE_IDS=['12013','12015','12014']


class AboveGroundProblem(RangeProblem):
    """A feasibility penalty, separately reported from the data likelihood."""
    def __init__(self,*args,terrain,**kwargs):
        super().__init__(*args,**kwargs)
        self.terrain=terrain

    def auxiliary(self,p,free_sites=False,jacobian=False,columns=None,with_motion=True,with_coarse=False):
        pieces=super().auxiliary(p,free_sites,jacobian,columns,with_motion,with_coarse)
        if with_motion:
            time=self.model.seed['epoch']+np.linspace(-15,15,121)
            x,j,_=self.model.positions(p,time,self.sites,free_sites)
            h,gradient=self.terrain.clearance(x,True)
            value=np.minimum(h,0)/.1
            if jacobian:
                value=np.einsum('ti,tij->tj',gradient,j)*(h<0)[:,None]/.1
                if columns is not None:
                    value=value[:,columns]
            pieces.insert(len(pieces)-(1 if free_sites else 0),value)
        return pieces


def licensed_geometry(directory,terrain_directory,assignment,terrain=None):
    original=json.loads((directory/'geometry-prior.json').read_text())
    source=json.loads((directory/'acma-transmitters.json').read_text())
    validate_assignment(assignment, original)
    if 'site_configuration' in original:
        centres=np.array(original['enu_transmitters_m'])
        descriptions=[]
        for channel,index in enumerate(assignment):
            site=original['transmitters'][index]
            signal=original['site_configuration']['signals'][channel]
            device=next(d for d in source['devices'] if d['CALL_SIGN']==CALLS[channel] and d['SITE_ID']==site['acma_site_id'])
            descriptions.append({'channel':NAMES[channel], 'site_id':site['acma_site_id'],
                'configured_role':next(role for role,value in signal['transmitter_site_roles'].items() if value==site['id']),
                'station_type':device['STATION_TYPE'], 'licensed_height_m':float(device['HEIGHT']),
                'antenna_agl_m':site['antenna_agl_m'], 'device_id':device['SDD_ID']})
        return centres,np.zeros((len(assignment),3)),descriptions
    if terrain is None:
        terrain=Terrain(terrain_directory,original)
    receiver=np.array(original['receiver_ecef_m']);rotation=np.array(original['ecef_to_enu'])
    base_heights=[160.,110.,119.]
    centres=[];vertical=[]
    for site_id,height in zip(SITE_IDS,base_heights):
        site=source['sites'][site_id];lat,lon=float(site['LATITUDE']),float(site['LONGITUDE'])
        ground=terrain.elevation(lat,lon);geoid=terrain.undulation(lat,lon)
        point=ecef(lat,lon,ground+geoid+height)
        centres.append(rotation@(point-receiver))
        vertical.append(rotation@(ecef(lat,lon,ground+geoid+height+1)-point))
    channels=[];offsets=[]
    for channel,site_index in enumerate(assignment):
        device=next(d for d in source['devices'] if d['CALL_SIGN']==CALLS[channel] and d['SITE_ID']==SITE_IDS[site_index])
        offsets.append(vertical[site_index]*(float(device['HEIGHT'])-base_heights[site_index]))
        channels.append({'channel':NAMES[channel],'site_id':SITE_IDS[site_index],
                         'station_type':device['STATION_TYPE'],'licensed_height_m':float(device['HEIGHT']),
                         'device_id':device['SDD_ID']})
    return np.array(centres),np.array(offsets),channels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories',type=Path,required=True)
    parser.add_argument('--patches',type=Path,required=True)
    parser.add_argument('--templates',type=Path,required=True)
    parser.add_argument('--terrain',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',default=['A','B','C','E'])
    parser.add_argument('--hypotheses',nargs='+',default=['configured'])
    parser.add_argument('--above-ground',action='store_true')
    parser.add_argument('--tag',default='',help='Separate output suffix for additional candidate sets')
    args=parser.parse_args();_,records=read_profiles(args.templates)
    geometry=json.loads((args.output/'geometry-prior.json').read_text())
    hypotheses={'configured':nominal_indices(geometry),'main':[0,0,0,0,2]}
    for name in args.hypotheses:
        assignment=hypotheses[name]
        sites,offsets,description=licensed_geometry(args.output,args.terrain,assignment)
        result={'hypothesis':name,'channel_sites':assignment,'channel_offsets_enu_m':offsets.tolist(),
                'site_positions_enu_m':sites.tolist(),'channels':description,'targets':[],
                'status':'licensed configuration hypothesis; not an accepted calibration',
                'attribution':'Based on Australian Communications and Media Authority information'}
        for label,coarse in zip(args.labels,observations(args.histories,args.labels)):
            channels,desc=load_histories(args.histories,label,1.)
            options={}
            cls=RangeProblem
            if args.above_ground:
                cls=AboveGroundProblem
                options['terrain']=Terrain(args.terrain,json.loads((args.output/'geometry-prior.json').read_text()))
            problem=cls(channels,coarse,sites,load_fields(args.patches,label,desc,records),**options)
            problem.model=PolarTrajectory(channels[1]['seed'],channel_sites=assignment,channel_offsets=offsets)
            previous_path=args.output/f'{label}-range.json'
            previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
            seeds=problem.multistart(max_nfev=60)[:3]
            if args.above_ground:
                path=args.output/f'sites-{name}.json'
                old=json.loads(path.read_text()) if path.exists() else {'targets':[]}
                seeds.extend((0,np.array(t['state'])) for t in old['targets'] if t['label']==label)
            if previous is not None:
                seeds.append((0,np.array(previous['fixed_state'])))
            fits=[problem.fit(seed,max_nfev=450) for _,seed in seeds]
            fit=min(fits,key=lambda f:f.cost)
            data_cost=float(np.linalg.norm(problem.residual(fit.x,with_motion=False))**2/2)
            item={'label':label,'state':fit.x.tolist(),'cost':float(fit.cost),'data_cost':data_cost,
                  'converged':bool(fit.success),'curve':problem.describe(fit.x),
                  'trial_costs':[float(f.cost) for f in fits]}
            result['targets'].append(item)
            suffix='-physical' if args.above_ground else ''
            tag='-'+args.tag if args.tag else ''
            result['above_ground_constraint']=args.above_ground
            if args.above_ground:
                item['minimum_agl_m']=float(np.min(problem.terrain.clearance(np.array(item['curve']['position_enu_m']))))
            (args.output/f'sites-{name}{tag}{suffix}.json').write_text(json.dumps(result,indent=2))
            print(name,label,'cost',fit.cost,'converged',fit.success,'speed',item['curve']['max_speed_mps'],flush=True)


if __name__=='__main__':
    main()
