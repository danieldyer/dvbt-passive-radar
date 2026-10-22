"""Fit one shared low-order rigid motion to the recorded ship patch.

This is an image-regression recovery experiment. The earlier image supplies
candidate support only; its angular history is one initializer, not a frozen
heading. Alternative zero/report-course starts use the same radar likelihood.
The final model is checked through original native symbols and known pulses.
"""
import argparse
import json
from pathlib import Path
import time
from itertools import product

import numpy as np

from ais import read_csv,utc_seconds
from block_pulse import BlockPulse
from cluster_phase import crop
from coherent_rigid_patch import CoherentPatch
from extract_full_aperture import load
from fixed_range_field import fixed_reference
from imaging import motion_state,sites,attitude_basis
from midpoint_model import representative,discrepancy
from motion import enu_basis
from report_measured_scatterer_phases import load_fields,serial
from rigid_patch import RigidPatch,PatchBand,fit,polish
from spline_motion import SplineMotion
from waveform_kernels import KernelField,bind_subset,read_kernels
from ship_site_signature import assigned_rays,licensed_options
from shared_scene import solve as shared_scene,phase_coherent
from phase_rigid_patch import PhasePatch
from dense_patch_image import DenseImages,focus,entropy


def candidates(directory,count=8,*,north_up=False,association_frame=None):
    z=[]
    for c in range(5):
        a=np.load(directory/(f'channel-{c}.npz' if north_up else f'channel-{c}-image.npz'));z.append(8192*abs(a['complex_image'])**2*a['information'])
    xx,yy=np.meshgrid(a['east_m' if north_up else 'along_m'],a['north_m' if north_up else 'port_m']);power=np.sum(z,axis=0)
    if north_up:
        if association_frame is None:raise ValueError('North-up candidate association requires a declared vessel corridor')
        forward=association_frame[:,0];port=association_frame[:,1]
        u=xx*forward[0]+yy*forward[1];v=xx*port[0]+yy*port[1]
        allowed=(abs(u)<=330.)&(abs(v)<=100.)
    else:allowed=(xx>-150)&(xx<260)&(yy>-60)&(yy<100)
    chosen=[]
    for _ in range(count):
        at=np.unravel_index(np.argmax(np.where(allowed,power,-np.inf)),xx.shape)
        chosen.append([xx[at],yy[at],20.]);allowed&=((xx-xx[at])**2+(yy-yy[at])**2>=35**2)
    return np.asarray(chosen)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluations',type=int,default=100);parser.add_argument('--points',type=int,default=6)
    parser.add_argument('--reference',choices=['ais','golden','reported'],default='golden')
    parser.add_argument('--initial-result',type=Path);parser.add_argument('--polish',action='store_true')
    parser.add_argument('--coherent',action='store_true')
    parser.add_argument('--geometry-directory',type=Path);parser.add_argument('--support-images',type=Path)
    parser.add_argument('--assignment',default='00002')
    parser.add_argument('--site-sweep',action='store_true')
    parser.add_argument('--phase-coherent',action='store_true')
    parser.add_argument('--initial-rotation-rates',nargs='+',type=float)
    parser.add_argument('--image-autofocus',action='store_true');parser.add_argument('--fit-band-phases',action='store_true')
    parser.add_argument('--focus-spacing-m',type=float,default=1.)
    parser.add_argument('--translation-only',action='store_true');parser.add_argument('--zero-angular-correction',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    baseline=Path('build/isar-20261015/heading-five-142-172');description=json.loads((baseline/'image-description.json').read_text())
    old=description['motion_model'];epoch=utc_seconds(old['nominal_start_utc']);mid=157.;curve=SplineMotion(old)
    reference_tx,reference_rx,_=sites(Path(old['geometry_directory']))
    tx,rx=(assigned_rays(args.geometry_directory,args.assignment) if args.geometry_directory else (reference_tx,reference_rx))
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    anchor=track.sample(np.array([epoch+mid]),max_gap_s=180.);local=enu_basis(anchor['latitude'][0],anchor['longitude'][0])
    anchor_state=curve.apply(np.array([mid]),motion_state(track,epoch+mid,old),reference_tx[3],reference_rx)
    reference_model={**old,'attitude_prior':'reported-course-prior'} if args.reference=='reported' else old
    reference_anchor=curve.apply(np.array([mid]),motion_state(track,epoch+mid,reference_model),reference_tx[3],reference_rx)
    association=(local.T@attitude_basis(track,anchor,'reported-course-prior')[0])[:2,:2]
    points=(candidates(args.support_images,args.points,north_up=True,association_frame=association) if args.support_images else
        candidates(baseline,args.points)@anchor_state['body_to_ecef'][0].T@local)
    records=load_fields(Path('build/isar-20261015/full-fields'),Path('docs/data/calibration_20261016_sites'))
    kernels=Path('build/isar-20261017/waveform-kernels-full');_,kr,ks=read_kernels(kernels/'response.kernels');grid=np.load(kernels/'rows.npz')
    native=[];fast=[]
    for c in range(5):
        # Keep complete original 16-symbol bins inside the legacy curve's
        # support; selecting by bin midpoint alone can cross an endpoint.
        observed=crop(records[c]['data'],142.02,171.98,np.arange(-8,9))
        raw=load(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'))
        f,_=fixed_reference(observed,raw,mid,np.arange(-16,17),omega_reference=0.)
        f=bind_subset(f,grid[f'channel_{c}']['time'],grid[f'channel_{c}']['information'],ks[c],np.asarray([r['valid'] for r in kr[c]]))
        approximate=representative(f);blocks=f.pulse.row_blocks.reshape(len(f.time),f.metadata['block_rows'])
        if np.any(blocks!=blocks[:,0,None]):raise ValueError('Representative model must preserve each known waveform identity')
        approximate=KernelField(approximate,f.pulse.kernels,blocks[:,0])
        train=np.floor((f.time-142.)/.5).astype(int)%2==0
        for field,target in [(f,native),(approximate,fast)]:
            state=track.sample(epoch+field.data['row_time'],max_gap_s=180.)
            options={}
            if args.reference!='ais':
                state=curve.apply(field.data['row_time'],motion_state(track,epoch+field.data['row_time'],reference_model),reference_tx[3],reference_rx)
                gauge=reference_anchor['body_to_ecef'][0].T@local
                options=dict(reference_frame=state['body_to_ecef']@gauge,reference_frame_rate=state['frame_velocity']@gauge)
            kin=RigidPatch(field.data['row_time'],mid,state['position_ecef_m'],state['velocity_ecef_mps'],local,tx[3],rx,**options)
            target.append(PatchBand(field,kin,points,train,tx[c],rx))
    t=np.linspace(142.,172.,301);dt=t-mid;path=curve.components(t)[:,0]
    initial_path=np.linalg.lstsq(np.c_[dt,.5*dt**2,dt**3/6],path,rcond=None)[0]
    s=curve.apply(t,motion_state(track,epoch+t,old),reference_tx[3],reference_rx);direction=s['body_to_ecef'][:,:,0]@local
    heading=np.unwrap(np.arctan2(direction[:,0],direction[:,1]));angular=-np.polyfit(dt,heading,2)*180/np.pi
    initials=[np.r_[initial_path,angular[1],2*angular[0]],np.r_[np.zeros(3),.2,0.],np.r_[initial_path,0.,0.]]
    if args.reference!='ais':initials=[np.zeros(5),np.array([.01,0.,0.,.03,0.]),np.array([-.01,0.,0.,-.03,0.])]
    if args.initial_rotation_rates:initials=[np.r_[np.zeros(3),w,0.] for w in args.initial_rotation_rates]
    if args.initial_result:
        previous=json.loads(args.initial_result.read_text())
        if previous['reference']!=args.reference:raise ValueError('Motion reference changed between refinement stages')
        if 'phase_coherent_fit' in previous:
            initials=[np.asarray(previous['phase_coherent_fit']['motion_parameters'])]
            points=np.asarray(previous['phase_coherent_fit']['points_east_north_up_m'])
            for b in native+fast:b.points=points.copy()
        elif 'coherent_fit' in previous:
            initials=[np.asarray(previous['coherent_fit']['motion_parameters'])]
            points=np.asarray(previous['coherent_fit']['points_east_north_up_m'])
            for b in native+fast:b.points=points.copy()
        else:initials=[np.asarray(min(previous['trials'],key=lambda r:r['objective'])['parameters'])]
    bounds=(np.array([-1.,-.1,-.01,-.75,-.02]),np.array([1.,.1,.01,.75,.02]))
    result=dict(complete=False,interval_s=[142.,172.],epoch_s=mid,local_to_ecef=local,points_east_north_up_m=points,reference=args.reference,reference_motion=reference_model,
        transmitter_ecef_m=tx,receiver_ecef_m=rx,nominal_start_utc=old['nominal_start_utc'],
        translation_reference_transmitter_ecef_m=tx[3],translation_reference_receiver_ecef_m=rx,
        geometry_directory=str(args.geometry_directory),support_images=str(args.support_images),assignment=args.assignment,
        candidate_association=dict(frame_forward_port_to_east_north=association,forward_limits_m=[-330.,330.],port_limits_m=[-100.,100.],
            scope='Broad AIS-supported target association only. No hull mask is applied to the likelihood observations or rendered image.'),
        parameter_order=['path_rate_mps','path_acceleration_mps2','path_jerk_mps3','rotation_deg_per_s','rotation_acceleration_deg_per_s2'],
        support_source='Training-conditional candidate support from the earlier all-data image; this is not independent target discovery',trials=[])
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    save()
    if args.image_autofocus:
        if not args.initial_result:raise ValueError('Supply the declared motion and band-response initializer')
        s=previous.get('phase_coherent_fit')
        if s is not None:
            gamma=np.asarray(s['coefficient_real_imag']);gamma=gamma[...,0]+1j*gamma[...,1]
            gain=np.asarray([complex(*z) for z in s['gain_phase_real_imag']])
        else:
            gamma=np.asarray([r['coefficient_real_imag'] for r in previous['native_validation']]);gamma=gamma[...,0]+1j*gamma[...,1]
            gain=np.exp(1j*np.angle(gamma[:,0]*gamma[3,0].conjugate()))
        info=np.asarray([np.sum(b.field.root_information**2) for b in fast])
        response=np.sqrt(info[:,None])*abs(gamma);_,vec=np.linalg.eigh(response@response.T);amplitude=vec[:,-1]
        if amplitude.sum()<0:amplitude=-amplitude
        amplitude/=np.sqrt(info);amplitude/=amplitude.max();gain*=amplitude
        u=np.arange(-330.,330.01,args.focus_spacing_m);v=np.arange(-100.,100.01,args.focus_spacing_m);uu,vv=np.meshgrid(u,v)
        pixel=np.c_[np.c_[uu.ravel(),vv.ravel()]@association.T,np.full(uu.size,20.)]
        model=DenseImages(fast,pixel,out/'dense-training',training=True);initial=np.asarray(initials[0])
        if args.zero_angular_correction:initial[3:]=0.
        lo=np.array([-1.,-.1,-.01,-.15,-.005]);hi=-lo
        if args.translation_only:lo[3:]=hi[3:]=initial[3:]
        def progress(record):
            result.setdefault('image_autofocus_trace',[]).append(record);save();print('dense-focus',serial(record),flush=True)
        try:
            p,record,formed=focus(model,initial,gain,bounds=(lo,hi),
                iterations=args.evaluations,fit_phases=args.fit_band_phases,checkpoint=progress)
        finally:model.close()
        result['image_autofocus_fit']=record;save()
        np.savez_compressed(out/'dense-training-images.npz',pixels_east_north_up_m=pixel,forward_m=u,port_m=v,
            image=formed['image'][:,0],image_derivative=formed['image'][:,1:],combined=formed['combined'])
        validation=DenseImages(fast,pixel,out/'dense-unused',training=False)
        try:before=validation.evaluate(initial);after=validation.evaluate(p[:5])
        finally:validation.close()
        final_gain=np.asarray([complex(*z) for z in record['gains_real_imag']])
        normal=np.sqrt(np.sum(abs(gain)**2*info));z0=np.einsum('c,cp->p',gain.conj(),before[:,0])/normal
        z1=np.einsum('c,cp->p',final_gain.conj(),after[:,0])/normal
        result['dense_unused_validation']=dict(initial_entropy=entropy(z0),final_entropy=entropy(z1),
            individual_entropy_before=[entropy(a) for a in before[:,0]],individual_entropy_after=[entropy(a) for a in after[:,0]],
            scope='Different retained epochs; common pose and band responses frozen from training. Entropy is a focusing statistic, not a CRLB or an absolute calibration certificate.')
        np.savez_compressed(out/'dense-unused-images.npz',pixels_east_north_up_m=pixel,forward_m=u,port_m=v,
            before=before[:,0],after=after[:,0],combined_before=z0,combined_after=z1)
        result['complete']=True;save();return
    if args.phase_coherent:
        common=PhasePatch(fast,initials[0]);tic=time.monotonic()
        lo=np.r_[bounds[0],np.full(common.size-5,-6.)];hi=np.r_[bounds[1],np.full(common.size-5,6.)]
        p,record=common.fit(bounds=(lo,hi),max_iterations=args.evaluations)
        result['phase_coherent_fit']=record;result['seconds']=time.monotonic()-tic;save()
        original=PhasePatch(native,record['motion_parameters'])
        result['native_coherent_validation']=original.validation(p)
        result['complete']=True;save();print('phase-coherent',serial(record),flush=True);return
    if args.site_sweep:
        if args.geometry_directory is None:raise ValueError('Site comparisons need the explicit licensed primary/backup priors')
        options=licensed_options(args.geometry_directory);designs=[];codes={'12013':'0','12015':'1','12014':'2'}
        for c,base in enumerate(fast):
            band_options={}
            for option in (o for o in options if o['channel']==c):
                b=PatchBand(base.field,base.kinematics,points,base.training,np.asarray(option['transmitter_ecef_m']),rx)
                h,_=b.design(initials[0]);a=b.project(h).reshape(-1,len(points))[b.selected]
                selected=np.repeat(~b.training,len(b.field.offsets));at=b.project(h,~b.training).reshape(-1,len(points))[selected]
                yt=b.project(b.field.y,~b.training).ravel()[selected]
                band_options[codes[option['site_id']]]=dict(gram=a.conj().T@a,rhs=a.conj().T@b.data,energy=b.energy,
                    test_gram=at.conj().T@at,test_rhs=at.conj().T@yt,test_energy=np.vdot(yt,yt).real)
            designs.append(band_options)
        scores=[]
        for assignment in product(*[list(d) for d in designs]):
            d=[item[code] for item,code in zip(designs,assignment)]
            fitted=shared_scene([a['gram'] for a in d],[a['rhs'] for a in d],[a['energy'] for a in d])
            gamma,alpha=fitted['gain'],fitted['scene'];test=0.
            for c,a in enumerate(d):
                test+=a['test_energy']-2*np.real(gamma[c].conjugate()*np.vdot(alpha,a['test_rhs']))+abs(gamma[c])**2*np.vdot(alpha,a['test_gram']@alpha).real
            scores.append(dict(assignment=''.join(assignment),training_rss=fitted['cost'],unused_rss=test,
                gains_real_imag=np.c_[gamma.real,gamma.imag],scene_real_imag=np.c_[alpha.real,alpha.imag],iterations=fitted['iterations']))
            phase=phase_coherent([a['gram'] for a in d],[a['rhs'] for a in d],[a['energy'] for a in d]);test=0.
            for c,a in enumerate(d):
                z=phase['coefficient'][c]
                test+=a['test_energy']-2*np.vdot(z,a['test_rhs']).real+np.vdot(z,a['test_gram']@z).real
            scores[-1]['phase_coherent']=dict(training_rss=phase['cost'],unused_rss=test,
                gain_phase_real_imag=np.c_[phase['gain'].real,phase['gain'].imag],
                coefficient_real_imag=np.stack([phase['coefficient'].real,phase['coefficient'].imag],axis=-1),
                converged=phase['converged'],maximum_gradient=phase['maximum_gradient'])
        result['site_sweep']=sorted(scores,key=lambda r:r['training_rss']);result['complete']=True;save()
        print('site-sweep',serial(result['site_sweep']),flush=True);return
    if args.coherent:
        if not args.initial_result:raise ValueError('Supply the checked common-motion initializer')
        common=CoherentPatch(fast,initials[0]);tic=time.monotonic()
        lo=np.r_[bounds[0],np.full(common.nposition,-4.),np.full(common.ng,-5.)]
        hi=np.r_[bounds[1],np.full(common.nposition,4.),np.full(common.ng,5.)]
        parameters,record=common.fit(bounds=(lo,hi),max_evaluations=args.evaluations)
        result['coherent_fit']=record;result['seconds']=time.monotonic()-tic;save()
        native_model=CoherentPatch(native,record['motion_parameters'])
        result['native_coherent_validation']=native_model.validation(parameters)
        result['complete']=True;save();print('coherent',serial(record),flush=True);return
    for j,initial in enumerate(initials):
        tic=time.monotonic();initial=np.clip(initial,bounds[0]+1e-8,bounds[1]-1e-8)
        p,record=(polish(fast,initial,bounds=bounds,max_iterations=args.evaluations) if args.polish else
            fit(fast,initial,bounds=bounds,max_evaluations=args.evaluations))
        record.update(start=j,initial=initial,seconds=time.monotonic()-tic)
        result['trials'].append(record);save();print('trial',j,serial(record),flush=True)
    best=min(result['trials'],key=lambda r:r['objective']);parameters=np.asarray(best['parameters'])
    result['selected_start']=best['start'];result['native_validation']=[];result['approximation']=[]
    for c,(a,b) in enumerate(zip(native,fast)):
        result['native_validation'].append(a.validation(parameters));ha,_=a.design(parameters);hb,_=b.design(parameters)
        result['approximation'].append([discrepancy(ha[...,j],hb[...,j]) for j in range(len(points))]);save()
        print('native',c,serial(result['native_validation'][-1]),flush=True)
    result['complete']=True;save()


if __name__=='__main__':main()
