"""Fit multiple physical scatterers sharing short-arc translation on original data.

The flash control reuses immutable previous observations and background. A
separate aperture can use the full known-waveform archive. All starts fit the
same point/brightness/neighbour model; AIS supplies initialization and the
declared motion prior, never a correction to measured phase curves.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from ais import read_csv
from audit_joint_kernel_phase import known_band
from cluster_phase import SplinePath
from fit_fixed_grid_phases import prepare_fixed,acquisition_seeds
from fit_physical_primary_association import assignment_code,geometry_for,make_model,prepare_window,measured
from flash_geometry import assignments
from flash_siting_report import EPOCH
from motion import C
from nearby_echo_control import load,ROOT
from phase_window import window_band
from piecewise_ais import LocalAisReference
from physical_amplitude_arc import PhysicalAmplitudeArc
from report_measured_scatterer_phases import load_fields,serial
from shared_echo_cluster import ReferencedAmplitudeArc
from shared_translation_scene import SharedTranslationScene
from ship_site_signature import licensed_options
from waveform_kernels import read_kernels


def acquire_scene_phases(scene,parameters):
    p=np.asarray(parameters).copy()
    for c,(cluster,transform,private) in enumerate(zip(scene.clusters,scene.maps,scene.nuisance_indices)):
        b=cluster.base;phase=np.r_[scene.phase_indices[:,c],[index[1] for index in private]]
        p[phase]=0.
        matrix=np.column_stack([b.nuisance.remove(b.reduced(raw[:,:,None])[:,0][b.selected])
            for raw,_ in cluster.response_blocks(transform@p)])
        coefficient=np.linalg.lstsq(matrix,b.clean,rcond=None)[0];p[phase]=np.angle(coefficient)
    return p


def estimated_coordinates(scene,*,neighbours,antennas=False,fixed_ten_c=False):
    free=np.r_[np.arange(18,27),np.arange(28,scene.primary_count)]
    if antennas:free=np.r_[np.arange(15),17,free]
    if neighbours:
        free=np.r_[free,np.arange(scene.primary_count,scene.count)]
        if fixed_ten_c:free=np.setdiff1d(free,scene.nuisance_indices[3][1][2:])
    return free


def scene_diagnostics(scene,p):
    records=[]
    for cluster,transform in zip(scene.clusters,scene.maps):
        b=cluster.base;value=cluster.evaluate(transform@p);k=len(b.knots)
        matrix=b.nuisance.remove(value['design'][b.selected])[:,value['amplitude']>0]
        singular=np.linalg.svd(np.r_[matrix.real,matrix.imag],compute_uv=False)
        parts=np.stack([value['design'][:,s*k:(s+1)*k]@value['amplitude'][s*k:(s+1)*k]
            for s in range(cluster.components)])
        full=parts[:,b.selected];physical=full[:scene.points]
        responses=cluster.response_blocks(transform@p)
        h=[b.nuisance.remove(b.reduced(raw[:,:,None])[:,0][b.selected]) for raw,_ in responses[:scene.points]]
        denominator=np.linalg.norm(h[0])*np.linalg.norm(h[1]) if scene.points>1 else 0.
        retention=[float(np.sum(abs(b.nuisance.remove(a[b.selected]))**2)/max(np.sum(abs(a[b.selected])**2),1e-300)) for a in parts]
        records.append(dict(active_brightness_condition=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>0 else None,
            component_to_sum_energy_ratio=float(np.sum(abs(full)**2)/max(np.sum(abs(full.sum(axis=0))**2),1e-300)),
            physical_component_to_sum_ratio=float(np.sum(abs(physical)**2)/max(np.sum(abs(physical.sum(axis=0))**2),1e-300)),
            physical_template_correlation=float(abs(np.vdot(h[0],h[1]))/denominator) if denominator else None,
            physical_training_energies=np.sum(abs(physical)**2,axis=1),static_projection_retention=retention))
    return records


def measured_scene(scene,p):
    output=[]
    for channel in scene.measurements(p):
        points=[]
        for rec in channel:
            z=rec['complex_coefficient']
            points.append(dict(time_s=rec['time_s'],physical_path_change_m=rec['physical_path_change_m'],
                complex_coefficient_real_imag=np.c_[z.real,z.imag],conditional_information=rec['conditional_information']))
        output.append(points)
    return output


def seed_offsets(scene,base,range_difference_m,cross_range_m):
    p=base.copy();g=scene.geometry;position,velocity,acceleration,tx,rx=g.state(p[:28])
    vectors=(position-tx[2],position-rx)
    gradient=sum(v/np.linalg.norm(v) for v in vectors)[:2]
    direction=gradient/np.linalg.norm(gradient);cross=np.array([-direction[1],direction[0]])
    offset=range_difference_m/np.linalg.norm(gradient)*direction+cross_range_m*cross
    radius=np.linalg.norm(offset)
    if radius>=scene.relative_radius_m:raise ValueError('Acquisition offset outside declared support')
    p[scene.offset_indices[1]]=[radius/scene.relative_scale_m,np.arctan2(offset[1],offset[0]),p[20]]
    return p


def make_flash_problem(source,control,track,code):
    old_group=source['groups'][0];empirical=json.loads(Path(source['source']).read_text())
    window=empirical['windows'][old_group['window_index']]
    bands,parts,models,parameters,baseline=prepare_window(control,window)
    assignment=next(a for a in assignments(licensed_options(control['geometry'])) if assignment_code(a)==code)
    g=geometry_for(control,assignment,np.mean(window['interval_s']),track)
    old_model=make_model(bands,models,control,g,np.mean(window['interval_s']))
    fit=old_group['trials'][old_group['best_index']]['steps'][-1];q=np.asarray(fit['parameters'])
    checks={}
    if code==old_group['assignment']:
        r,_=old_model.evaluate(q);error=abs(r@r*old_model.energy-fit['cost'])
        if error>1e-7:raise ValueError('Original physical likelihood changed')
        fresh=measured(old_model,q);phase_error=0.
        for a,b in zip(fresh,fit['measured_primary']):
            for key in ('time_s','physical_path_change_m','complex_coefficient_real_imag','conditional_information'):
                np.testing.assert_array_equal(a[key],b[key]);phase_error=max(phase_error,float(np.max(abs(np.asarray(a[key])-np.asarray(b[key])))))
        checks=dict(original_cost_error=error,original_phase_diagnostic_error=phase_error)
    arcs=[c.nuisance_arcs[:3] for c in old_model.clusters]
    scene=SharedTranslationScene(old_model.bands,arcs,old_model.origins[:3],jerk_std=control['jerk_std'])
    p=np.zeros(scene.count);p[:28]=q[:28];p[scene.phase_indices[0]]=q[28:33]
    for c,private in enumerate(scene.nuisance_indices):
        local=q[old_model.indices[c]]
        for target,block in zip(private,old_model.clusters[c].nuisance_slices[:3]):p[target]=local[block]
    return scene,p,parts,dict(interval_s=window['interval_s'],fixed_ten_c=True,checks=checks,
        source_one_physical_cost=fit['cost'] if code==old_group['assignment'] else None,
        source_one_physical_unused_error=sum(v['heldout_error'] for v in fit['prediction']) if code==old_group['assignment'] else None,
        source_independent_unused_error=old_group['independent_baseline_unused_error'])


def make_new_problem(first,last,track,code,kernel,geometry):
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    mean,_,_=prepare_fixed(fields,geometry,first,last,stationary=True)
    _,records,pulses=read_kernels(kernel/'response.kernels');bands=[];parts=[]
    with np.load(kernel/'rows.npz') as grids:
        for c,b in enumerate(mean):
            b,part,_=window_band(b,first,last,complete_rows=True)
            bands.append(known_band(b,grids[f'channel_{c}'],records[c],pulses[c]));parts.append(part)
    assignment=next(a for a in assignments(licensed_options(geometry)) if assignment_code(a)==code)
    geo=json.loads((geometry/'geometry-prior.json').read_text())
    reference=LocalAisReference(track,EPOCH+(first+last)/2,np.asarray(geo['ecef_to_enu']),np.asarray(geo['receiver_ecef_m']))
    g=geometry_for({'geometry':geometry},assignment,(first+last)/2,track,ais_reference=reference)
    physical=[PhysicalAmplitudeArc(b.field,b.training,b.knots,g,c,(first+last)/2,stationary_design=b.design) for c,b in enumerate(bands)]
    curve=SplinePath(first,last,1.);neighbours=[]
    for b in bands:
        flexible=ReferencedAmplitudeArc(b.field,b.training,b.knots,first,last,b.known_path,b.known_rate,
            stationary_design=b.design,curve=curve)
        neighbours.append((b,b,flexible))
    scene=SharedTranslationScene(physical,neighbours,[-7.,6.,13.],jerk_std=.1)
    p=np.zeros(scene.count);p[20]=np.clip(p[20],g.lower[20],g.upper[20])
    acquisitions=[]
    for b,arcs,indices in zip(bands,neighbours,scene.nuisance_indices):
        acquired=[]
        for origin,arc,index in zip(scene.origins,arcs,indices):
            seed=acquisition_seeds(b,float(origin))[0];rate=seed['parameters'][2]/b.scale
            coefficients=np.linalg.lstsq(arc.base.basis,rate*(b.field.data['row_time']-arc.epoch),rcond=None)[0]
            p[index]=np.r_[seed['parameters'][:2],coefficients];acquired.append(seed)
        acquisitions.append(acquired)
    p=np.clip(p,scene.lower,scene.upper)
    return scene,p,parts,dict(interval_s=[first,last],fixed_ten_c=False,checks={},
        kernel=str(kernel),nuisance_acquisition=acquisitions,source_one_physical_unused_error=None,
        ais_reference_maximum_error_m=reference.maximum_error_m,ais_reference_piece_boundaries_s=reference.edges,
        source_independent_unused_error=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'physical-primary-association/fits.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interval',nargs=2,type=float,help='Use a fresh independent aperture, rather than the flash control')
    parser.add_argument('--kernel',type=Path,default=Path('build/isar-20261017/waveform-kernels-full'))
    parser.add_argument('--assignment',default='00002')
    parser.add_argument('--starts-per-motion',type=int,default=1)
    parser.add_argument('--max-evaluations',type=int,default=180)
    args=parser.parse_args();out=args.output
    if out.exists():raise ValueError('Use a fresh output')
    track_path=Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track=read_csv(track_path,reference_height_ellipsoid_m=5.,mmsi=563052200)
    geometry=Path('docs/data/calibration_20261016_sites')
    if args.interval:
        scene,base,parts,case=make_new_problem(*args.interval,track,args.assignment,args.kernel,geometry)
    else:
        source=json.loads(args.source.read_text())
        if not source['complete']:raise ValueError('Wait for complete physical-primary reference')
        scene,base,parts,case=make_flash_problem(source,load(),track,args.assignment)
    out.mkdir(parents=True);snapshot=out/'sources';snapshot.mkdir()
    files=[Path(__file__),track_path,*[Path('support/isar')/(n+'.py') for n in
        ('shared_translation_scene','physical_amplitude_arc','amplitude_pair','component_phase','joint_path_information',
         'physical_pair','fit_physical_primary_association','phase_window','waveform_kernels','flash_geometry')],
        *[geometry/n for n in ('geometry-prior.json','sites-main.json','acma-transmitters.json')]]
    if args.interval:files+=[args.kernel/'request.json',args.kernel/'response.kernels',args.kernel/'rows.npz']
    else:files.append(args.source)
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,production_changed=False,
        case=case,assignment=args.assignment,source=str(args.source) if not args.interval else None,
        proposals=[],trials=[],
        model_policy='Two physical points, one common NEU velocity/acceleration and separate midpoint positions. Each point has independent constant phase and nonnegative brightness knots in each band. Exact geometry links envelope, carrier phase and symbol Doppler. No yaw, moving band phase screen, sea/multipath model or arbitrary range offset. A/C/D remain independent observed-cell nuisance returns.',
        prior_policy='Proper-complex squared data residual plus half real standardized prior squared. Shared 28-state geometry prior counted once, relative horizontal radius /150 m and height difference /20 m once, each independent nuisance jerk prior once. Relative horizontal radius bounded by 300 m; each point has the same declared midpoint height support. A static polar point coordinate is not fitted ship heading.',
        validation_policy='Original complete bins, native rows, known kernels, original range covariance and 125 ms split. Cold score chooses starts within each predeclared source/AIS-motion family. Phase controls remove constants only. Unused predictions are exploratory on previously inspected data.',
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files})
    for p in files:
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    started=time.monotonic()
    def save():
        result['seconds']=time.monotonic()-started
        (out/'fits.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    save()
    families=['AIS motion'] if args.interval else ['Source motion','AIS motion']
    for family in families:
        for delta in (-45.,-20.,20.,45.):
            for cross in (-100.,0.,100.):
                p=base.copy()
                if family=='AIS motion':p[21:27]=0.
                p=seed_offsets(scene,p,delta,cross);p=acquire_scene_phases(scene,p)
                r,_=scene.evaluate(p)
                result['proposals'].append(dict(family=family,range_offset_seed_m=delta,cross_range_seed_m=cross,
                    parameters=p,training_cost=float(r@r*scene.energy)))
    save();print('prepared',case['interval_s'],'proposals',len(result['proposals']),flush=True)
    for family in families:
        choices=sorted([i for i,p in enumerate(result['proposals']) if p['family']==family],
            key=lambda i:result['proposals'][i]['training_cost'])[:args.starts_per_motion]
        for index in choices:
            proposal=result['proposals'][index];seed=np.asarray(proposal['parameters'])
            free=estimated_coordinates(scene,neighbours=True,fixed_ten_c=case['fixed_ten_c'])
            r,j=scene.evaluate(seed);direction=np.zeros(scene.count);direction[free]=1e-4*np.cos(free+.2);step=.03
            finite=(scene.evaluate(seed+step*direction)[0]-scene.evaluate(seed-step*direction)[0])/(2*step)
            error=float(np.linalg.norm(finite-j@direction)/max(np.linalg.norm(finite),1e-15))
            if error>4e-4:raise ValueError(f'Original-data shared-motion derivative failed: {error}')
            trial=dict(proposal_index=index,family=family,directional_derivative_error=error,steps=[]);result['trials'].append(trial);save()
            for release in (False,True):
                free=estimated_coordinates(scene,neighbours=release,fixed_ten_c=case['fixed_ten_c'])
                p,audit=scene.fit(seed,free,max_evaluations=args.max_evaluations)
                prediction=scene.prediction(p,[part==3 for part in parts])
                record=dict(neighbour_paths_free=release,parameters=p,audit=audit,
                    cost=scene.objective(p)[0]*scene.energy,prior_cost=scene.prior_cost(p),
                    prediction=prediction,total_unused_error=sum(v['heldout_error'] for v in prediction),
                    point_summary=scene.point_summary(p),diagnostics=scene_diagnostics(scene,p),measured_points=measured_scene(scene,p))
                trial['steps'].append(record);save();seed=p
                print(family,index,'neighbours free',release,'cost',record['cost'],'gradient',audit['projected_gradient'],
                    'unused',record['total_unused_error'],'bounds',audit['bound_indices'],flush=True)
    result['best_index']=int(np.argmin([t['steps'][-1]['cost'] for t in result['trials']]))
    result['complete']=True;save();print('complete',result['seconds'],flush=True)


if __name__=='__main__':main()
