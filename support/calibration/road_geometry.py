"""Connected OSM centreline hypotheses for offline vehicle calibration.

Roads constrain cross-track position, not the radar phase. Along-road motion,
map/lane offset and reflector height remain fitted nuisance parameters. No
coordinates in this module are written to production metadata.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from geometry import unit

DRIVE_CLASSES = {'motorway','trunk','primary','secondary','tertiary','unclassified',
                 'residential','living_street','motorway_link','trunk_link',
                 'primary_link','secondary_link','tertiary_link'}


def connected_chains(ways):
    """Join same-named ways only through their actual shared OSM end nodes.

    Intersections with other named roads do not cut a straight road into tiny
    pieces. Branches within the same street remain distinct: no geometric
    nearest-neighbour connection, disconnected carriageway jump or turn guess.
    """
    groups=defaultdict(list)
    for way in ways:
        tags=way['tags']
        if tags.get('highway') not in DRIVE_CLASSES:
            continue
        key=(tags.get('name') or tags.get('ref') or str(way['id']),
             tags.get('bridge','no'),tags.get('tunnel','no'),tags.get('layer','0'))
        groups[key].append(way)
    for key,items in groups.items():
        adjacency=defaultdict(list)
        for i,way in enumerate(items):
            for end in (way['nodes'][0],way['nodes'][-1]):
                adjacency[end].append(i)
        used=set()
        starts=[(n,i) for n,edges in adjacency.items() if len(edges)!=2 for i in edges]
        starts += [(w['nodes'][0],i) for i,w in enumerate(items)]
        for node,index in starts:
            if index in used:
                continue
            ids=[];path=[];speeds=[];directions={-1,1};classes=set()
            while index not in used:
                used.add(index);way=items[index];tags=way['tags']
                forward=way['nodes'][0]==node
                part=way['nodes'] if forward else way['nodes'][::-1]
                path.extend(part if not path else part[1:]);ids.append(way['id'])
                classes.add(tags['highway'])
                one=tags.get('oneway','no')
                if one in ('yes','1','true','-1'):
                    direction=(1 if forward else -1)*(-1 if one=='-1' else 1)
                    directions &= {direction}
                if tags.get('maxspeed','').isdigit():
                    speeds.append(float(tags['maxspeed'])/3.6)
                node=part[-1]
                following=[i for i in adjacency[node] if i not in used]
                if len(adjacency[node])!=2 or len(following)!=1:
                    break
                index=following[0]
            if directions:
                yield {'name':key[0],'nodes':path,'ways':ids,'directions':sorted(directions),
                       'bridge':key[1],'tunnel':key[2],'layer':key[3],
                       'classes':sorted(classes),'posted_speed_mps':max(speeds) if speeds else None}


class RoadCurve:
    def __init__(self,description):
        self.description=description
        xyz=np.asarray(description['enu_m'],float)
        delta=np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1)
        good=np.r_[True,delta>1e-3];xyz=xyz[good]
        self.s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1))]
        self.length=float(self.s[-1])
        # Chord-based cubic interpolation is only a map prior. A lateral offset
        # and slope absorb lane choice and local centreline survey uncertainty.
        self.curve=CubicSpline(self.s,xyz,bc_type='natural',extrapolate=False)

    def centre(self,s,derivative=0):
        value=self.curve(s,derivative)
        if not np.all(np.isfinite(value)):
            raise ValueError('Trajectory extends beyond the connected road segment')
        return value

    def frame(self,s):
        tangent=self.centre(s,1);second=self.centre(s,2)
        speed=np.linalg.norm(tangent[:,:2],axis=1)
        normal=np.c_[-tangent[:,1],tangent[:,0],np.zeros(len(s))]/speed[:,None]
        derivative=np.c_[-second[:,1],second[:,0],np.zeros(len(s))]/speed[:,None]
        derivative-=normal*(np.sum(tangent[:,:2]*second[:,:2],axis=1)/speed**2)[:,None]
        return normal,derivative

    def positions(self,s,lateral=0.,height=2.):
        normal,_=self.frame(np.atleast_1d(s))
        result=self.centre(s)+np.asarray(lateral)[...,None]*normal
        result[:,2]+=height
        return result

    def crossings(self,length,transmitter,spacing=25.):
        s=np.linspace(0,self.length,max(3,int(np.ceil(self.length/spacing))+1))
        x=self.positions(s)
        def excess(q):
            return np.linalg.norm(q-transmitter,axis=-1)+np.linalg.norm(q,axis=-1)-np.linalg.norm(transmitter)
        y=excess(x)-length
        indices=np.flatnonzero(y[:-1]*y[1:]<0)
        return [brentq(lambda a:excess(self.positions(np.array([a]))[0])-length,s[i],s[i+1]) for i in indices]


def prepare(source,transmitter,maximum_range=6500.,minimum_length=180.):
    nodes=source['nodes'];result=[]
    for chain in connected_chains(source['ways']):
        xyz=np.array([nodes[str(n)]['enu_m'] for n in chain.pop('nodes')])
        if len(xyz)<2:
            continue
        length=np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1).sum()
        excess=np.linalg.norm(xyz-transmitter,axis=1)+np.linalg.norm(xyz,axis=1)-np.linalg.norm(transmitter)
        if length<minimum_length or excess.min()>maximum_range or excess.max()<450:
            continue
        chain['enu_m']=xyz.tolist();chain['length_m']=float(length)
        chain['excess_range_minmax_m']=[float(excess.min()),float(excess.max())]
        chain['elevation_uncertain']=chain['bridge'] not in ('no','0') or chain['tunnel'] not in ('no','0') or chain['layer']!='0'
        result.append(chain)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roads',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();source=json.loads(args.roads.read_text());prior=json.loads(args.prior.read_text())
    transmitter=np.mean(prior['enu_transmitters_m'],axis=0)
    routes=prepare(source,transmitter)
    with args.roads.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    result={'source_sha256':digest,'attribution':source['attribution'],'source_header':source['source_header'],
            'maximum_range_m':6500,'minimum_connected_length_m':180,'routes':routes}
    args.output.write_text(json.dumps(result,separators=(',',':')))
    print('Connected road hypotheses',len(routes),'length km',sum(r['length_m'] for r in routes)/1000,flush=True)


if __name__=='__main__':main()
