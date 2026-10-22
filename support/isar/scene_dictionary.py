"""Retain exact distributed-scene responses for repeatable temporal holdouts.

This is an offline numerical reference cache, not a production data format.
It contains the unchanged complex forward operator, observations and stationary
nuisance basis. Selection of training times happens only in subsequent fits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from assess_dense_scene import rectangular_grid
from field import Field
from full_motion import Band
from imaging import motion_state, sites
from spline_motion import SplineMotion


def union_grid(rectangles, spacing, height):
    return np.unique(np.concatenate([rectangular_grid(r, spacing, height) for r in rectangles]),axis=0)


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def column(field, state, point, range_bias, gain):
    position = state['position_ecef_m']+state['body_to_ecef']@point
    velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
    return gain*field.trajectory(position,velocity,range_bias_m=range_bias).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment','motion','ais','geometry-directory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    support = parser.add_mutually_exclusive_group(required=True)
    support.add_argument('--bounds-m',type=float,nargs=4,action='append',
        help='Union of declared rectangles; repeat to include a separate response region')
    support.add_argument('--point-support',type=Path,
        help='Use all points from a completed continuous-coordinate fit at this same motion')
    parser.add_argument('--spacing-m',type=float,default=2.)
    parser.add_argument('--height-m',type=float,default=20.)
    parser.add_argument('--interval', type=float, nargs=2,
        help='Cache a subaperture without changing the fitted motion spline or its reference')
    args = parser.parse_args()
    source, model = [json.loads(p.read_text()) for p in (args.alignment,args.motion)]
    interval = model['interval_s'] if args.interval is None else args.interval
    if not model['interval_s'][0]<=interval[0]<interval[1]<=model['interval_s'][1]:
        raise ValueError('Requested cache interval leaves the fitted motion support')
    if model.get('antenna_fit') or np.any(model.get('channel_path_rate_bias_mps',0.)):
        raise ValueError('This control holds the nominal antenna sites and channel rates fixed')
    if args.point_support:
        fitted = json.loads(args.point_support.read_text())
        if (not fitted.get('complete') or not fitted.get('fit', {}).get('converged')
                or Path(fitted['source_motion']).resolve() != args.motion.resolve()):
            raise ValueError('Use a completed point fit at the declared physical motion')
        points = np.asarray(fitted['points_body_m'], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Expected finite three-dimensional point coordinates')
    else:
        points = union_grid(args.bounds_m,args.spacing_m,args.height_m)
    if len(points)>7000:
        raise ValueError('Declared reference cache exceeds its bounded spatial dictionary')
    paths = [args.alignment,args.motion,args.ais,*[args.geometry_directory/n for n in ('geometry-prior.json','sites-main.json')],
             *[Path(source['input_fields'])/f'channel-{c}.npz' for c in range(5)],
             *[Path(__file__).parent/n for n in ('scene_dictionary.py','field.py','motion.py','imaging.py','spline_motion.py','symbol_moments.py')],
             Path(__file__).parent.parent/'calibration/range_fit.py']
    if args.point_support:
        paths.append(args.point_support)
    identity = {'source_sha256':{str(p.resolve()):fingerprint(p) for p in paths},
        'rectangles_m':args.bounds_m,'spacing_m':args.spacing_m,'height_m':args.height_m,'interval_s':interval,'numpy':np.__version__}
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = args.output/'dictionary.json'
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if result['identity'] != identity:
            raise ValueError('Cache inputs changed; use a new output directory')
    else:
        result = {'identity':identity,'source_motion':str(args.motion),'source_alignment':str(args.alignment),
            'source_support':str(args.point_support) if args.point_support else None,
            'points_body_m':points.tolist(),'interval_s':interval,'scope':__doc__,'channels':[],'complete':False}
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais,reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'],mmsi=563052200)
    curve = SplineMotion(model)
    started = time.perf_counter()
    for c in range(5):
        if any(v['channel']==c for v in result['channels']):
            continue
        with np.load(Path(source['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f),f['pulse_samples']).windowed(interval[0]+.02,interval[1]-.02)
        prior = motion_state(track,utc_seconds(model['nominal_start_utc'])+field.data['row_time'],model)
        state = curve.apply(field.data['row_time'],prior,tx[3],rx)
        gain = complex(*model['channel_gains_real_imag'][c])
        destination = args.output/f'channel-{c}-design.npy'
        temporary = destination.with_suffix('.building.npy')
        matrix = np.lib.format.open_memmap(temporary,mode='w+',dtype=np.complex128,
            shape=(field.y.size,len(points)),fortran_order=True)
        for j, point in enumerate(points):
            matrix[:,j] = column(field,state,point,model['range_bias_m'][c],gain)
        matrix.flush()
        probes = np.linspace(0,len(points)-1,7,dtype=int)
        band = Band(field,prior,tx[3],rx,points[probes],model,np.arange(len(field.time))%2==0,range_bias_m=model['range_bias_m'][c])
        expected = band.design(model['parameters'])[0]*gain
        difference = float(np.linalg.norm(matrix[:,probes]-expected)/np.linalg.norm(expected))
        if difference>1e-10:
            raise ValueError(f'Cached forward operator disagrees with the tested Band operator: {difference:g}')
        del matrix
        temporary.replace(destination)
        np.savez(args.output/f'channel-{c}-observation.npz',time=field.time,data=field.y,
            background=band.nuisance.design,range_offsets=field.data['offset_samples'],
            metadata_json=json.dumps(field.metadata))
        result['channels'].append({'channel':c,'shape':[field.y.size,len(points)],
            'forward_probe_relative_error':difference,'design_sha256':fingerprint(destination)})
        result['complete']=len(result['channels'])==5
        result['seconds']=time.perf_counter()-started
        manifest.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'channel':c,'points':len(points),'forward_relative_error':difference,'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    main()
