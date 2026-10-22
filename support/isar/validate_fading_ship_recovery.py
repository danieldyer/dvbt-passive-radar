"""Recover a known shared carrier path with measured fading and real apertures.

Known-waveform kernels, original epochs, five VHF frequencies and information
weights come from the recording. Synthesis uses independent Cartesian norms
and explicit original-epoch carrier averaging. The estimator uses the physical
path lift and analytic derivatives. Point support is given: this is a local
motion-identifiability control, not a blind geographic or hull-recovery claim.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize,LinearConstraint

from full_ship_field import load
from full_path_point import PointBand,initial_curve
from full_path_scene import SceneBand,ScenePath,RefinedScenePath
from joint_ship_motion import JointMotion
from local_ship_motion import load_motion
from path_dynamics import constraint,certificate
from rigid_patch import profile_static
from scene_phase_coordinates import transform
from report_measured_scatterer_phases import serial


def nuisance_energy_scale(base_energy,cross,nuisance_energy,target_energy):
    """One positive field amplitude giving the requested total crop energy."""
    if nuisance_energy<=0 or target_energy<=base_energy:return 0.
    return float((-cross+np.sqrt(cross*cross+nuisance_energy*(target_energy-base_energy)))/nuisance_energy)


def independent_response(field,coordinate,state,point,tx,rx,extra=None):
    pos=state['position_ecef_m']+np.einsum('tij,j->ti',state['body_to_ecef'],point)
    vel=state['velocity_ecef_mps']+np.einsum('tij,j->ti',state['frame_velocity'],point)
    if extra is not None:pos=pos+extra
    path=np.full(len(pos),-np.linalg.norm(tx-rx));rate=np.zeros(len(pos))
    for antenna in [tx,rx]:
        ray=pos-antenna;distance=np.sqrt(np.sum(ray*ray,axis=1))
        path+=distance;rate+=np.sum(ray*vel,axis=1)/distance
    dt=coordinate['native_time_s'].reshape(-1,16)-field.time[:,None]
    weight=coordinate['native_information'].reshape(-1,16);total=weight.sum(axis=1)
    factor=np.divide(np.sum(weight*np.exp(-1j*field.k*rate[:,None]*dt),axis=1),total,out=np.ones(len(total),complex),where=total>0)
    delta=path-field.data['centre_path_m'];omega=-field.k*.001024*rate
    return field.apply(field.path_response(delta,omega)*factor[:,None])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--brightness',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=120)
    p.add_argument('--noise-objective',action='store_true',help='Optimize the Gaussian log-likelihood difference per noise observation, independent of unrelated received power')
    p.add_argument('--information-preconditioner',action='store_true')
    p.add_argument('--initial-path-error-m',type=float,default=.7)
    p.add_argument('--thermal-only',action='store_true')
    p.add_argument('--chirp-initializer',action='store_true')
    p.add_argument('--acquisition-brightness',choices=['constant','magnitude','truth'],default='magnitude')
    p.add_argument('--acquisition-mode',choices=['joint','best-point'],default='joint')
    p.add_argument('--acquisition-subaperture-s',type=float,default=0.)
    p.add_argument('--acquisition-range-offsets',type=int,nargs=2)
    p.add_argument('--point-initial-error-m',type=float,default=0.,help='Perturb supplied associations and refine common horizontal source coordinates')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    d=json.loads(a.motion.read_text());fading=json.loads(a.brightness.read_text());truth=load_motion(a.motion)
    motion=JointMotion(Path(d['motion']));motion.interval=d['interval_s'];mid=float(np.mean(motion.interval))
    fields,coordinates,_=load(d['field']);brightness=np.load(a.brightness.parent/'trajectory.npz')
    strength=sum(np.sum(brightness[f'amplitude_{c}']**2,axis=0) for c in range(5));selected=np.argsort(strength)[-6:][::-1]
    points=np.asarray(fading['scene_points_body_m'])[selected];phi=np.asarray(fading['scene_point_phase_radians'])[selected]
    point_error=np.zeros_like(points);point_error[:,:2]=np.random.default_rng(8819).normal(size=(len(points),2))*a.point_initial_error_m
    initial_points=points+point_error
    beta=np.angle([complex(*q) for q in fading['physical_gains_real_imag']])+phi[0];phi=phi-phi[0]
    sample=np.linspace(*motion.interval,2001);r=truth.path_curve(sample);v=truth.path_curve(sample,1)
    grid,base=initial_curve(sample,r,v,motion.interval,3.);design=BSpline.design_matrix(sample,grid,3).toarray()
    perturb=np.linalg.lstsq(design,a.initial_path_error_m*np.sin(2*np.pi*(sample-mid)/30),rcond=None)[0]
    origin=BSpline(grid,np.eye(len(base)),3)(mid);perturb-=origin@perturb
    initial_curve_coeff=base+perturb;reports=[];histories={}
    for competing in ([False] if a.thermal_only else [False,True]):
        rng=np.random.default_rng(72811);bands=[];closure=[];source_brightness=[];nuisances=[];real_energy=[]
        for c,(full,coordinate) in enumerate(zip(fields,coordinates)):
            field=full.windowed(*motion.interval);index=np.searchsorted(full.time,field.time)
            coord={k:coordinate[k].reshape(-1,16)[index].ravel() for k in ['native_time_s','native_information']}
            coord['training']=coordinate['training'][index]
            if 'stationary_groups' in coordinate:coord['stationary_groups']=coordinate['stationary_groups'][index]
            b=PointBand(field,coord,motion,d['point_body_m'],grid,initial_curve_coeff,mid,5.,d['orientation'])
            real_energy.append(b.energy[True])
            band=SceneBand(b,initial_points);truth_band=SceneBand(b,points);state=truth.state(b.t);signal=np.zeros_like(field.y)
            amplitude=np.column_stack([np.interp(b.t,brightness[f'time_s_{c}'],brightness[f'amplitude_{c}'][:,j]) for j in selected])
            source_brightness.append(amplitude)
            independent=np.stack([independent_response(field,coord,state,point,motion.tx[c],motion.rx) for point in points],axis=-1)
            exact_coefficient=(-perturb[:-1]+perturb[-1]);calculated=truth_band.design(exact_coefficient,c,False)
            error=float(np.linalg.norm(calculated-independent)/np.linalg.norm(independent));closure.append(error)
            if error>2e-7:raise AssertionError(f'Independent Cartesian synthesis disagrees with the physical fitter: {error}')
            signal=np.sum(independent*(amplitude*np.exp(1j*(phi+beta[c])))[:,None,:],axis=-1)
            noise=(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))/np.sqrt(2)
            noise*=field.root_information[:,None]>0
            if competing:
                at=truth.state(np.array([mid]));centre=at['position_ecef_m'][0]
                g=sum((centre-ant)/np.linalg.norm(centre-ant) for ant in [motion.tx[3],motion.rx]);shift=g*450/np.dot(g,g)
                nuisance=independent_response(field,coord,state,points[0],motion.tx[c],motion.rx,shift)
                nuisances.append(float(amplitude.max())*np.exp(1j*(beta[c]+.7))*nuisance)
            field.y=signal+noise+field.root_information[:,None]*(.1+.2j)
            for train in [True,False]:
                mask=b.training if train else ~b.training;clean=profile_static(field.y,field.root_information,mask,b.groups)*mask[:,None]
                b.data[train]=clean;b.energy[train]=float(np.vdot(clean,clean).real)
            bands.append(band)
        nuisance_scale=0.
        if competing:
            norms=0.;cross=0.;base_energy=sum(b.point.energy[True] for b in bands)
            for band,value in zip(bands,nuisances):
                b=band.point;v=profile_static(value,b.field.root_information,b.training,b.groups)*b.training[:,None]
                norms+=float(np.vdot(v,v).real);cross+=float(np.vdot(b.data[True],v).real)
            nuisance_scale=nuisance_energy_scale(base_energy,cross,norms,sum(real_energy))
            for band,value in zip(bands,nuisances):
                b=band.point;b.field.y+=nuisance_scale*value
                for train in [True,False]:
                    mask=b.training if train else ~b.training;clean=profile_static(b.field.y,b.field.root_information,mask,b.groups)*mask[:,None]
                    b.data[train]=clean;b.energy[train]=float(np.vdot(clean,clean).real)
        likelihood=(RefinedScenePath if a.point_initial_error_m else ScenePath)(bands);n=likelihood.n;phase_end=n+5+len(points)-1
        size=phase_end+(2*len(points) if a.point_initial_error_m else 0)
        initial=np.r_[np.zeros(n),beta+rng.normal(size=5)*.2,phi[1:]+rng.normal(size=len(points)-1)*.2]
        true_parameters=np.r_[-perturb[:-1]+perturb[-1],beta,phi[1:]]
        if a.point_initial_error_m:
            initial=np.r_[initial,np.zeros(2*len(points))];true_parameters=np.r_[true_parameters,-point_error[:,:2].ravel()]
        true_cost=likelihood.objective(true_parameters)[0]
        if a.point_initial_error_m:likelihood.set_points(initial)
        acquisition=None
        if a.chirp_initializer:
            from polynomial_phase_acquisition import acquire
            weights=(source_brightness if a.acquisition_brightness=='truth' else [np.ones_like(q) for q in source_brightness] if a.acquisition_brightness=='constant' else None)
            acquisition_bands=bands;retention=[]
            if a.acquisition_range_offsets:
                from full_ship_field import select_range
                acquisition_bands=[]
                for band in bands:
                    b=band.point;field=b.field
                    value=np.divide(field.y,field.root_information[:,None],out=np.zeros_like(field.y),where=field.root_information[:,None]>0)@np.linalg.inv(field.whiten.T)
                    field.data=dict(field.data);field.data['value']=value
                    small=select_range(field,*a.acquisition_range_offsets)
                    point_band=PointBand(small,b.coordinate,motion,d['point_body_m'],grid,initial_curve_coeff,mid,5.,d['orientation'])
                    acquisition_bands.append(SceneBand(point_band,initial_points))
                for c,(full,small) in enumerate(zip(bands,acquisition_bands)):
                    h0=profile_static(full.design(np.zeros(n),c,False),full.point.field.root_information,full.point.training,full.point.groups)
                    h1=profile_static(small.design(np.zeros(n),c,False),small.point.field.root_information,small.point.training,small.point.groups)
                    retention.append((np.sum(abs(h1[small.point.training])**2,axis=(0,1))/np.sum(abs(h0[full.point.training])**2,axis=(0,1))).tolist())
            polynomial,acquisition=acquire(acquisition_bands,mid,(motion.interval[1]-motion.interval[0])/2,amplitude_weights=weights,mode=a.acquisition_mode,subaperture_s=a.acquisition_subaperture_s)
            acquisition.update(acquisition_range_offsets=a.acquisition_range_offsets,coarse_template_information_fraction=retention,final_likelihood_uses_full_field=True)
            u=(sample-mid)/((motion.interval[1]-motion.interval[0])/2)
            path=np.c_[u,u*u,u*u*u]@polynomial;delta=np.linalg.lstsq(design,path,rcond=None)[0]
            initial[:n]=delta[:-1]-delta[-1]
            (a.output/f'chirp-initializer-{int(competing)}.json').write_text(json.dumps(serial(acquisition),indent=2)+'\n')
            print(json.dumps(dict(competing_return=competing,stage='polynomial-phase-acquisition',coefficients_m=polynomial.tolist(),zero_score=acquisition['zero_correction_score_fraction'],acquired_score=acquisition['best_score_fraction'],seconds=time.monotonic()-started)),flush=True)
        degrees=sum(np.count_nonzero(b.point.training&(b.point.field.root_information>0))*len(b.point.field.offsets) for b in bands)
        numerical_scale=likelihood.energy/degrees if a.noise_objective else 1.
        objective_multiplier=1. if a.noise_objective else 100.
        matrix=transform(likelihood,size)
        if a.information_preconditioner:
            if not a.noise_objective:raise ValueError('Information coordinates require the noise objective')
            from full_scene_information import information,coordinates as information_coordinates
            fisher,checked_gradient,details=information(likelihood,initial)
            error=float(max(abs(checked_gradient-likelihood.objective(initial)[1])))
            if error>1e-6:raise ValueError(f'Projected Gaussian Jacobian failed original gradient check: {error}')
            matrix,diagnostic=information_coordinates(fisher,matrix,degrees)
            np.save(a.output/f'conditional-information-{int(competing)}.npy',fisher)
            diagnostic.update(maximum_original_gradient_error=error,channels=details)
            (a.output/f'information-coordinates-{int(competing)}.json').write_text(json.dumps(serial(diagnostic),indent=2)+'\n')
            print(json.dumps(dict(competing_return=competing,stage='information-coordinates',gradient_error=error,seconds=time.monotonic()-started)),flush=True)
        bound=constraint(grid,initial_curve_coeff,size,.15,3.,mid)
        if a.point_initial_error_m:
            position_matrix=np.eye(size)[phase_end:]
            bound=LinearConstraint(np.r_[bound.A,position_matrix],np.r_[bound.lb,np.full(2*len(points),-5.)],np.r_[bound.ub,np.full(2*len(points),5.)])
        bounded=LinearConstraint(bound.A@matrix,bound.lb,bound.ub);trace=[]
        def objective(q):
            value,g=likelihood.objective(matrix@q);return ((value-1) if a.noise_objective else value)*objective_multiplier*numerical_scale,(g@matrix)*objective_multiplier*numerical_scale
        def callback(q):
            value,g=objective(q);trace.append(dict(iteration=len(trace)+1,cost=value/objective_multiplier,seconds=time.monotonic()-started))
            if len(trace)%10==0:print(json.dumps(dict(competing_return=competing,**trace[-1])),flush=True)
        result=minimize(objective,np.linalg.solve(matrix,initial),jac=True,method='SLSQP',constraints=[bounded],callback=callback,options=dict(maxiter=a.iterations,ftol=1e-11 if a.noise_objective else 1e-9))
        actual=matrix@result.x;cost,g,models=likelihood.objective(actual,True);stationarity=certificate(actual,g,bound)
        fitted_points=likelihood.set_points(actual) if a.point_initial_error_m else initial_points
        curve=BSpline(grid,initial_curve_coeff+np.r_[actual[:n],0.]-origin@np.r_[actual[:n],0.],3)
        error=curve(sample)-truth.path_curve(sample)
        start_coefficient=np.r_[initial[:n],0.];before=BSpline(grid,initial_curve_coeff+start_coefficient-origin@start_coefficient,3)(sample)-truth.path_curve(sample)
        phase_records=[];cross=np.zeros(len(points),complex);total=np.zeros(len(points));path_weight=np.zeros_like(sample)
        for c,band in enumerate(bands):
            b=band.point;field=b.field;state=truth.state(b.t);actual_h=band.design(actual[:n],c,False)
            known_h=np.stack([independent_response(field,b.coordinate,state,point,motion.tx[c],motion.rx) for point in points],axis=-1)
            amp=np.column_stack([np.interp(b.t,brightness[f'time_s_{c}'],brightness[f'amplitude_{c}'][:,j]) for j in selected])
            information=np.sum(abs(known_h)**2,axis=1)*amp**2;mask=~b.training
            phasor=np.exp(1j*np.angle(np.sum(known_h.conj()*actual_h,axis=1)))
            weight=information[mask];summed=weight.sum(axis=0);coherence=np.divide(abs(np.sum(weight*phasor[mask],axis=0))**2,summed**2,out=np.zeros_like(summed),where=summed>0)
            cross+=np.sum(weight*phasor[mask],axis=0)*np.exp(1j*(actual[n+c]-beta[c]));total+=summed
            phase_records.append(dict(channel=c,temporal_coherent_fraction=coherence.tolist(),information=summed.tolist()))
            path_weight+=np.interp(sample,b.t,information.sum(axis=1),left=0,right=0)
        combined=np.divide(abs(cross)**2,total**2,out=np.zeros_like(total),where=total>0)
        common=float(np.sum(combined*total)/np.sum(total));offset=float(np.average(error,weights=path_weight))
        observable_rms=float(np.sqrt(np.average((error-offset)**2,weights=path_weight)))
        report=dict(competing_return=competing,numerical_convergence=bool(result.success and stationarity['maximum_kkt_residual']<1e-6),message=str(result.message),iterations=result.nit,objective=cost,
            known_truth_profile_objective=true_cost,objective_gap_from_known_truth=cost-true_cost,noise_observation_degrees=degrees,optimizer_scale=numerical_scale,chirp_initializer_used=bool(acquisition),
            competing_amplitude_scale=nuisance_scale,measured_original_crop_energy=sum(real_energy),synthetic_crop_energy=likelihood.energy,competing_strength_policy='Total stationary-projected training energy matched to the recorded crop; noise and weak target retained',
            independent_forward_relative_errors=closure,initial_path_rms_m=float(np.sqrt(np.mean(before**2))),recovered_path_rms_m=float(np.sqrt(np.mean(error**2))),maximum_path_error_m=float(max(abs(error))),
            information_weighted_constant_path_offset_m=offset,information_weighted_gauge_removed_path_rms_m=observable_rms,
            phase_only_five_band_coherent_fraction_by_point=combined,phase_only_five_band_coherent_fraction_information_weighted=common,
            phase_only_five_band_loss_db=float(-10*np.log10(max(common,1e-30))),phase_records=phase_records,parameters=actual,path_coefficients_m=curve.c,**stationarity)
        report.update(fitted_points_body_m=fitted_points,maximum_source_position_error_m=float(np.max(np.linalg.norm(fitted_points-points,axis=1))),source_centroid_shift_m=np.mean(fitted_points-points,axis=0),information_preconditioner=a.information_preconditioner)
        difference=fitted_points-points;report.update(centroid_aligned_source_rms_m=float(np.sqrt(np.mean(np.sum((difference-difference.mean(axis=0))**2,axis=1)))),optimizer_objective_multiplier=objective_multiplier)
        histories[f'path_error_m_{int(competing)}']=error;histories[f'initial_error_m_{int(competing)}']=before
        reports.append(report);print(json.dumps(serial({k:v for k,v in report.items() if k not in ['phase_records','parameters','path_coefficients_m']})),flush=True)
        (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=False,scope=__doc__,cases=reports)),indent=2)+'\n')
    histories['time_s']=sample;np.savez_compressed(a.output/'histories.npz',**histories)
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True)
    axes[0].plot(sample,histories['initial_error_m_0'],label='Initial physical path error')
    for index,report in enumerate(reports):
        axes[0].plot(sample,histories[f'path_error_m_{index}'],label='Recovered: '+('strong moving return 450 m farther' if report['competing_return'] else 'thermal noise and stationary nuisance'))
        axes[1].plot(sample,np.degrees(fields[3].k*histories[f'path_error_m_{index}']))
    axes[0].legend();axes[0].set_ylabel('Path error from known truth (m)');axes[1].set(xlabel='Capture time (s)',ylabel='Equivalent Ten-band phase error (degrees)')
    fig.suptitle('Known-motion recovery with real fading, five VHF bands and exact recorded epochs\nIndependent Cartesian synthesis; point support given; no time-varying reflector phase')
    fig.savefig(a.output/'known-fading-motion-recovery.png',dpi=140);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=True,scope=__doc__,motion=str(a.motion),brightness=str(a.brightness),noise_objective=a.noise_objective,acquisition_brightness=a.acquisition_brightness,initial_path_error_amplitude_m=a.initial_path_error_m,point_initial_error_sigma_m=a.point_initial_error_m,points_body_m=points,initial_points_body_m=initial_points,cases=reports,
        recovery_within_5cm_rms=all(q['recovered_path_rms_m']<.05 for q in reports),seconds=time.monotonic()-started)),indent=2)+'\n')


if __name__=='__main__':main()
