"""Native-row north/east BP of a fitted injected point, without image alignment.

Only the viewport and oracle control read the injection truth. Estimated
motion and constant complex band coefficients come from the blind fit. Every
input C0/C1 row is passed to the existing Vulkan backprojector unchanged.
"""
import argparse,json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from channel_update import ChannelUpdates
from flash_siting_report import EPOCH
from imaging import geometry_from_state
from piecewise_ais import LocalAisReference
from recorded_injection_fields import read_profiles
from recovery_experiment import SCALE
from report_measured_scatterer_phases import serial
from coordinated_turn import trajectory as turn_trajectory
from native_folds import native_indices
from rigid_patch import RigidPatch


def cartesian_state(time,state,mid,rotation,receiver,heading_rate_degps=None,kinematics='quadratic'):
    time=np.asarray(time);dt=time-mid;p,v,a=np.asarray(state)
    if kinematics=='coordinated-turn':
        position,velocity,course_rate=turn_trajectory(dt,p,v,a)
    elif kinematics=='quadratic':
        position=p+dt[:,None]*v+.5*dt[:,None]**2*a
        velocity=v+dt[:,None]*a;course_rate=0.
    else:raise ValueError('Unknown motion family')
    # Heading is clockwise from north; Cartesian N/E rotation has the opposite
    # sign. Pixel axes remain fixed N/E at the aperture midpoint.
    rate=course_rate if heading_rate_degps is None else -np.radians(heading_rate_degps)
    angle=rate*dt;c,s=np.cos(angle),np.sin(angle)
    basis=np.zeros((len(time),3,3));derivative=np.zeros_like(basis)
    basis[:,0,0]=c;basis[:,0,1]=-s;basis[:,1,0]=s;basis[:,1,1]=c;basis[:,2,2]=1.
    derivative[:,0,0]=-rate*s;derivative[:,0,1]=-rate*c;derivative[:,1,0]=rate*c;derivative[:,1,1]=-rate*s
    return dict(position_ecef_m=receiver+position@rotation,velocity_ecef_mps=velocity@rotation,
        body_to_ecef=rotation.T@basis,frame_velocity=rotation.T@derivative)


def fitted_state(time,state,mid,rotation,receiver,fit,heading_rate_degps=None):
    model=fit.get('motion_model','quadratic')
    if model!='rigid-patch':return cartesian_state(time,state,mid,rotation,receiver,heading_rate_degps,model)
    if heading_rate_degps is not None:raise ValueError('A fitted rigid angular history cannot also have a heading override')
    base=cartesian_state(time,state,mid,rotation,receiver)
    kin=RigidPatch(np.asarray(time),mid,base['position_ecef_m'],base['velocity_ecef_mps'],rotation.T,
        np.asarray(fit['reference_transmitter_ecef_m']),receiver,
        translation_direction=fit.get('translation_direction_ecef'))
    position,velocity=kin.point_state(np.asarray(fit['rigid_parameters']),np.zeros(3))
    frame,frame_velocity=kin.orientation(np.asarray(fit['rigid_parameters']))
    return dict(position_ecef_m=position,velocity_ecef_mps=velocity,body_to_ecef=frame,frame_velocity=frame_velocity)


def brightness_template(model,component,channel,time):
    """A training-fitted real envelope and its constant complex band gain."""
    from amplitude_arc import linear_basis
    band=model['bands'][channel];coefficients=np.asarray(band['coefficients'],float)
    if not 0<=component<len(coefficients):raise ValueError('Unsupported brightness component')
    a=coefficients[component]
    if not np.isfinite(a).all() or np.any(a<0) or a.mean()<=0:raise ValueError('Brightness component must have finite nonnegative support')
    mean=a.mean();weight=linear_basis(np.asarray(time),np.asarray(band['knots_s']))@a/mean
    gain=mean*np.exp(1j*(band['band_phase_rad']+model['point_phase_rad'][component]))
    return weight,gain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--oracle',action='store_true')
    parser.add_argument('--subject',choices=['injected-point','ship','echo'],default='injected-point')
    parser.add_argument('--extent-m',type=float,nargs=2,default=[240.,440.])
    parser.add_argument('--heading-rate-degps',type=float)
    parser.add_argument('--stationary-mean-control',action='store_true',
        help='Diagnostic only: remove the information-weighted constant native profile before BP; retain original power scale')
    parser.add_argument('--spacing-m',type=float,default=1.)
    parser.add_argument('--first',type=float,default=142.)
    parser.add_argument('--last',type=float,default=172.)
    parser.add_argument('--fold',choices=['all','training','testing'],help='Select complete native bins in the declared quarter-second validation fold')
    parser.add_argument('--windowed-read',action='store_true',help='Read a bounded context from a long native-profile export')
    parser.add_argument('--brightness-fit',type=Path,help='Apply one training-fitted real brightness envelope as a conditional temporal matched template')
    parser.add_argument('--brightness-component',type=int,default=0)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    if args.oracle and args.brightness_fit:raise ValueError('Oracle coefficients and inferred brightness templates are separate controls')
    read_interval=(args.first,args.last) if args.windowed_read else None
    header,channels=read_profiles(args.profiles,interval_s=read_interval)
    plan=json.loads(args.plan.read_text());fit=json.loads(args.fit.read_text())
    mid=fit['midpoint_s'];rotation=np.asarray(header['plan']['ecef_to_enu'])
    rx=np.asarray(header['plan']['receiver_ecef_m']);tx=np.asarray(header['plan']['transmitters_ecef_m'])
    # The coarse motion is independently reconstructed from the declared AIS
    # anchor, just as in the fitter. Do not use the plan's state as a fit seed.
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    reference=LocalAisReference(track,EPOCH+mid,rotation,rx)
    state=np.asarray(fit.get('coarse_state_enu',[reference(0.,0),reference(0.,1),np.zeros(3)])).copy()
    chosen=fit['fits']['constant_velocity_gls'];parameters=np.asarray(chosen['parameters'])
    for i in range(3):state[i,:2]+=parameters[2*i:2*i+2]*SCALE[2*i:2*i+2]
    gains=np.asarray(chosen['coefficients_real_imag'])[:,0];gains=gains[:,0]+1j*gains[:,1]
    brightness=None
    if args.brightness_fit:
        brightness_fit=json.loads(args.brightness_fit.read_text())
        np.testing.assert_array_equal(brightness_fit['fit']['parameters'][:5],fit['rigid_parameters'])
        np.testing.assert_array_equal(brightness_fit['reference_state_enu'],state)
        if brightness_fit['midpoint_s']!=mid:raise ValueError('Brightness template uses another motion epoch')
        brightness=brightness_fit['phase_histories']['brightness_model']
        for c in range(5):
            if brightness['bands'][c]['rf_hz']!=header['applied_channels'][c]['rf_hz']:raise ValueError('Brightness band frequency differs')
            _,gains[c]=brightness_template(brightness,args.brightness_component,c,np.array([mid]))
    if args.oracle:
        state=np.asarray(plan['state_enu']);gains=np.asarray(plan['gains_real_imag']);gains=gains[:,0]+1j*gains[:,1]
    # Truth frames only the validation viewport. All cases share these absolute
    # geographic coordinates at the midpoint, with north up and zero yaw.
    centre=np.asarray(plan['state_enu'])[0]
    east=np.arange(-args.extent_m[0]/2,args.extent_m[0]/2+args.spacing_m*.01,args.spacing_m)
    north=np.arange(-args.extent_m[1]/2,args.extent_m[1]/2+args.spacing_m*.01,args.spacing_m)
    xx,yy=np.meshgrid(east,north)
    pixels=np.c_[xx.ravel()+centre[0]-state[0,0],yy.ravel()+centre[1]-state[0,1],np.zeros(xx.size),np.zeros(xx.size)]
    baseline=None
    if args.baseline:
        baseline_header,baseline=read_profiles(args.baseline,interval_s=read_interval)
        if baseline_header['first_range_sample']!=header['first_range_sample']:raise ValueError('Paired range origins differ')
    total=np.zeros((5,len(pixels)),complex);information=np.zeros(5);counts=[]
    fold=args.fold or ('all' if brightness is not None else None)
    worker=ChannelUpdates(pixels,out/'worker')
    try:
        for c in range(5):
            ch=channels[c]
            use=(native_indices(ch['rows'],ch['information'],args.first,args.last,fold) if fold else
                 np.flatnonzero((ch['rows'][:,1]>=args.first)&(ch['rows'][:,1]<args.last)))
            if baseline is not None:np.testing.assert_array_equal(ch['rows'][:,1],baseline[c]['rows'][:,1])
            counts.append(len(use));rf=header['applied_channels'][c]['rf_hz']
            static=0.
            if args.stationary_mean_control:
                weights=ch['information'][use]*ch['rows'][use,4]
                sample=ch['values'][:,use]
                if baseline is not None:sample=sample-baseline[c]['values'][:,use]
                static=np.einsum('mnr,n->mr',sample,weights)/weights.sum()
            for first in range(0,len(use),256):
                at=use[first:first+256];time=ch['rows'][at,1]
                weight=ch['information'][at]*ch['rows'][at,4]
                template=None
                if brightness is not None:
                    template,_=brightness_template(brightness,args.brightness_component,c,time)
                data=ch['values'][:,at]
                if baseline is not None:data=data-baseline[c]['values'][:,at]
                if args.stationary_mean_control:data=data-static[:,None,:]
                g,_=geometry_from_state(fitted_state(time,state,mid,rotation,rx,fit,args.heading_rate_degps),tx[c],rx,rf,header['first_range_sample'],
                    weight if template is None else weight*template)
                g[:,5,3]=2*np.pi*rf/299792458.;g[:,6,3]=c;g[:,7,3]=1.;g[:,8,3]=1.
                z,w=worker.update(g,data);total+=z
                if template is None:information+=w
                else:information[c]+=np.dot(weight,template**2)
            print('Rendered channel',c,'native rows',len(use),flush=True)
    finally:worker.close()
    if np.any(information<=0):raise ValueError('Incomplete channel support')
    amplitude=total/information[:,None]
    summed=np.einsum('c,cp->p',gains.conj(),total)
    combined_information=np.sum(abs(gains)**2*information)
    combined=summed/combined_information
    power=np.r_[8192*abs(total)**2/information[:,None],(8192*abs(summed)**2/combined_information)[None]]
    power=power.reshape(6,len(north),len(east));db=10*np.log10(np.maximum(power,1e-30))
    np.savez(out/'images.npz',east_m=east,north_m=north,complex_amplitude=amplitude.reshape(5,len(north),len(east)),
        information=information,combined_amplitude=combined.reshape(len(north),len(east)),power=power,gains=gains)
    high=float(db.max())+.1;low=high-45
    fig,axes=plt.subplots(2,3,figsize=(12,12),sharex=True,sharey=True,layout='constrained')
    names=['Seven','SBS','Nine','Ten','ABC','Coherent five-band sum']
    peaks=[]
    for ax,label,value in zip(axes.ravel(),names,db):
        im=ax.imshow(value,origin='lower',extent=[east[0],east[-1],north[0],north[-1]],cmap='magma',vmin=low,vmax=high,interpolation='nearest')
        ax.plot(*plan.get('marker_east_north_m',[0.,0.]),'+',color='#00ffff',ms=9,mew=1.)
        row,col=np.unravel_index(np.argmax(value),value.shape);peaks.append([east[col],north[row]])
        ax.set(xlabel=('East from injected point (m)' if args.subject=='injected-point' else 'East from fitted point (m)' if 'viewport_label' in plan else 'East from AIS antenna (m)'),ylabel='North (m)',title=label)
        ax.set(xlim=(east[0],east[-1]),ylim=(north[0],north[-1]))
    fig.colorbar(im,ax=axes.ravel().tolist(),label=('Coherent power after projection / original information (dB)' if args.stationary_mean_control else 'Conditional matched power (dB)'),shrink=.7)
    marker='injected point' if args.subject=='injected-point' else plan.get('marker_label','AIS antenna prior')
    title='Oracle motion and band coefficients' if args.oracle else ('Ship-motion hypothesis and frozen band coefficients' if args.subject=='ship' else 'Echo-motion hypothesis and frozen band coefficients' if args.subject=='echo' else 'Fitted motion and frozen band coefficients')
    fig.suptitle(title+
        f' · {args.last-args.first:g} s native-row BP\n'+('Stationary-mean diagnostic' if args.stationary_mean_control else 'Paired point increment' if baseline is not None else 'Complete observed scene')+f' · north up · cyan = {marker}'+
        (f'\nConditional real brightness template for candidate {args.brightness_component}' if brightness is not None else ''))
    fig.savefig(out/'point-images.png',dpi=150);plt.close(fig)
    result=dict(scope=__doc__,profiles=str(args.profiles),fit=str(args.fit),plan=str(args.plan),oracle_control=args.oracle,
        paired_baseline=None if args.baseline is None else str(args.baseline),native_rows=counts,
        interval_s=[args.first,args.last],motion_state_enu=state,midpoint_s=mid,
        gains_real_imag=np.c_[gains.real,gains.imag],peak_east_north_m=peaks,peak_power_db=db.max(axis=(1,2)),
        colour_scale_db=[low,high],image_registration_applied=False,heading_rate_hypothesis_degps=args.heading_rate_degps,
        motion_model=fit.get('motion_model','quadratic'),yaw_fitted=fit.get('angular_motion_fitted',fit.get('motion_model')=='coordinated-turn'))
    result['stationary_mean_control']=args.stationary_mean_control
    result['stationary_projection_information_correction_applied']=False
    result['native_validation_fold']=fold
    result['brightness_fit']=None if args.brightness_fit is None else str(args.brightness_fit)
    result['brightness_component']=None if brightness is None else args.brightness_component
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
