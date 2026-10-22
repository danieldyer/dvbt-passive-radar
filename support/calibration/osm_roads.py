"""Extract a dated, attributed road-centreline graph for offline calibration.

Only motor-road geometry and relevant way tags are retained. Coordinates are
map priors, not surveyed lane or scattering-centre positions. Node identifiers
preserve connectivity when OSM splits a street into several ways.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from geometry import ecef
from terrain import RasterTerrain

ROAD_CLASSES = {'motorway','trunk','primary','secondary','tertiary','unclassified',
    'residential','living_street','service','motorway_link','trunk_link',
    'primary_link','secondary_link','tertiary_link','road'}
KEEP_TAGS = ('name','ref','highway','oneway','maxspeed','lanes','width','bridge',
             'tunnel','layer','access','motor_vehicle','surface')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pbf',type=Path)
    for key in ['terrain','dem','prior','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--bbox',nargs=4,type=float,default=[144.93,-37.94,145.42,-37.71],
                   metavar=('WEST','SOUTH','EAST','NORTH'))
    args=p.parse_args()
    import osmium
    west,south,east,north=args.bbox
    class Roads(osmium.SimpleHandler):
        def __init__(self):
            super().__init__();self.ways=[];self.nodes={};self.incomplete=0
        def way(self,way):
            if way.tags.get('highway') not in ROAD_CLASSES or way.tags.get('area')=='yes':
                return
            if way.tags.get('motor_vehicle')=='no':
                return
            if any(not n.location.valid() for n in way.nodes):
                self.incomplete+=1;return
            coordinates=np.array([(n.lon,n.lat) for n in way.nodes])
            if not len(coordinates) or not np.any((coordinates[:,0]>=west)&(coordinates[:,0]<=east)&
                                                  (coordinates[:,1]>=south)&(coordinates[:,1]<=north)):
                return
            nodes=[n.ref for n in way.nodes]
            self.nodes.update(zip(nodes,coordinates.tolist()))
            self.ways.append({'id':way.id,'nodes':nodes,
                              'tags':{key:way.tags[key] for key in KEEP_TAGS if key in way.tags}})
    handler=Roads();handler.apply_file(str(args.pbf),locations=True)
    ids=np.array(list(handler.nodes));lonlat=np.array(list(handler.nodes.values()))
    prior=json.loads(args.prior.read_text());terrain=RasterTerrain(args.terrain,prior,args.dem)
    ground=terrain.elevation(lonlat[:,1],lonlat[:,0]);valid=np.isfinite(ground)
    xyz=(ecef(lonlat[:,1],lonlat[:,0],np.where(valid,ground,0)+terrain.undulation(lonlat[:,1],lonlat[:,0]))-
         np.array(prior['receiver_ecef_m']))@np.array(prior['ecef_to_enu']).T
    nodes={int(i):{'longitude_deg':float(ll[0]),'latitude_deg':float(ll[1]),
                  'ground_egm96_m':float(h),'enu_m':x.tolist()} for i,ll,h,x,ok in zip(ids,lonlat,ground,xyz,valid) if ok}
    # Preserve only contiguous pieces with real DEM support, never interpolate
    # across a missing terrain tile or invent a zero-height road node.
    ways=[]
    for way in handler.ways:
        chunks=[];current=[]
        for node in way['nodes']+[None]:
            if node in nodes:current.append(node)
            elif current:
                if len(current)>=2:chunks.append(current)
                current=[]
        ways.extend(dict(way,nodes=chunk,part=i) for i,chunk in enumerate(chunks))
    with osmium.io.Reader(str(args.pbf)) as reader:
        header=reader.header()
        source_header={key:header.get(key) for key in ('generator','osmosis_replication_timestamp','osmosis_replication_sequence_number')}
    result={'source_url':'https://download.bbbike.org/osm/bbbike/Melbourne/Melbourne.osm.pbf',
        'source_sha256':hashlib.file_digest(args.pbf.open('rb'),'sha256').hexdigest(),
        'source_header':source_header,'bbox_wsen':args.bbox,
        'attribution':'Map data © OpenStreetMap contributors, ODbL 1.0; extract by BBBike.org.',
        'license_url':'https://www.openstreetmap.org/copyright',
        'terrain':'Geoscience Australia 1 arcsecond DEM, EGM96 + EGM96 geoid to ECEF',
        'coordinates':'ENU relative to geometry-prior.json receiver; map centreline prior, not a lane survey',
        'missing_node_ways':handler.incomplete,'nodes':nodes,'ways':ways}
    args.output.write_text(json.dumps(result,separators=(',',':')))
    print('Road ways',len(ways),'nodes',len(nodes),'source',source_header,flush=True)


if __name__=='__main__':
    main()
