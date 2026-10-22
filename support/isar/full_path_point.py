"""Conditional whole-record point likelihood with one physical carrier path.

Range, RF phase, Doppler and native-bin attenuation all follow the same
position and velocity. Per-band phase is constant; only real nonnegative
brightness envelopes vary. Stationary range vectors are nuisance parameters
profiled at every trial, separately on training and excluded epochs.
"""
import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline
from scipy.sparse import csr_matrix

from audit_ship_scattering_envelopes import linear_basis
from bistatic_path_lift import lift, curvature
from motion import excess_path
from positive_envelope_scene import solve
from rigid_patch import RigidPatch, profile_static
from spline_motion import knots
from declared_ship_orientation import state as orientation_state,rotate_frame


def initial_curve(time, path, rate, interval, spacing):
    grid=knots(interval,spacing);curve=CubicHermiteSpline(time,path,rate,extrapolate=False)
    t=np.linspace(*interval,4001);clamped=np.clip(t,time[0],time[-1]);v=curve(clamped,1)
    r=curve(clamped)+v*(t-clamped)
    b=BSpline.design_matrix(t,grid,3).toarray()
    bd=BSpline(grid,np.eye(b.shape[1]),3)(t,1)
    # This is only an initializer. The linked range/rate observations enter
    # the final complex likelihood, not a second penalty or a motion prior.
    coef=np.linalg.lstsq(np.r_[b/20,bd/.1],np.r_[r/20,v/.1],rcond=None)[0]
    return grid,coef


def carrier_bin(time,rate,native_time,native_information,k):
    dt=np.asarray(native_time).reshape(-1,16)-np.asarray(time)[:,None]
    weight=np.asarray(native_information).reshape(-1,16);total=weight.sum(axis=1)
    weight=np.divide(weight,total[:,None],out=np.zeros_like(weight,dtype=float),where=total[:,None]>0)
    phasor=np.exp(-1j*k*rate[:,None]*dt)
    value=np.sum(weight*phasor,axis=1);value[total==0]=1
    return value,np.sum(weight*phasor*(-1j*k*dt),axis=1)


class PointBand:
    def __init__(self,field,coordinate,motion,point,grid,base_coeff,anchor,amplitude_spacing,orientation='existing',course_time_offset_s=0.,angular_rate_offset_degps=0.,angular_acceleration_offset_degps2=0.,free_path_offset=False):
        self.field=field;self.coordinate=coordinate;self.motion=motion;self.t=field.time
        self.range_bias_m=0.
        b=BSpline.design_matrix(self.t,grid,3).toarray();bd=BSpline(grid,np.eye(b.shape[1]),3)(self.t,1)
        origin=BSpline(grid,np.eye(b.shape[1]),3)(anchor)
        self.b=csr_matrix((b-origin)[:,:-1]);self.bd=csr_matrix(bd[:,:-1]);self.base_path=b@base_coeff;self.base_rate=bd@base_coeff
        self.free_path_offset=free_path_offset
        if free_path_offset:
            # Absolute envelope delay is observable even when constant RF
            # phase is unknown. Do not mistake a phase gauge for known range.
            from scipy.sparse import hstack
            self.b=hstack([self.b,np.ones((len(self.t),1))],format='csr')
            self.bd=hstack([self.bd,csr_matrix((len(self.t),1))],format='csr')
        self.amplitude,self.amplitude_nodes=linear_basis(self.t,motion.interval,amplitude_spacing)
        self.amplitude=csr_matrix(self.amplitude)
        state=orientation_state(motion,self.t,orientation,course_time_offset_s)
        self.reference_point=np.asarray(point)
        self.frame,self.frame_rate=state['body_to_ecef'],state['frame_velocity']
        self.anchor=state['position_ecef_m']+np.einsum('tij,j->ti',state['body_to_ecef'],point)
        self.velocity=state['velocity_ecef_mps']+np.einsum('tij,j->ti',state['frame_velocity'],point)
        rotated=rotate_frame(state,self.t,anchor,angular_rate_offset_degps,angular_acceleration_offset_degps2)
        self.frame,self.frame_rate=rotated['body_to_ecef'],rotated['frame_velocity']
        kin=RigidPatch(self.t,anchor,self.anchor,self.velocity,motion.local,motion.tx[3],motion.rx)
        self.direction,self.direction_rate=kin.direction,kin.direction_rate
        self.training=np.asarray(coordinate['training'],bool)
        self.groups=np.asarray(coordinate.get('stationary_groups',np.zeros(len(self.t),int)))
        self.data={};self.energy={}
        for selected in [True,False]:
            mask=self.training if selected else ~self.training
            y=profile_static(field.y,field.root_information,mask,self.groups)*mask[:,None]
            self.data[selected]=y;self.energy[selected]=float(np.vdot(y,y).real)

    def waveform_window(self,first,last):
        """Exact contiguous original-epoch view for a time-gated source.

        This is a waveform view only. Global likelihood/nuisance projection
        remains in the caller, so windowing never resets a carrier origin or
        introduces an independent stationary-clutter fit.
        """
        import copy
        if not 0<=first<last<=len(self.t):raise ValueError('Invalid waveform epoch window')
        value=copy.copy(self);sl=slice(first,last)
        value.field=self.field.windowed(self.t[first],np.nextafter(self.t[last-1],np.inf))
        np.testing.assert_array_equal(value.field.time,self.t[sl])
        for name in ['t','base_path','base_rate','frame','frame_rate','anchor','velocity','direction','direction_rate','training','groups']:
            setattr(value,name,getattr(self,name)[sl])
        value.b,value.bd=self.b[sl],self.bd[sl]
        value.coordinate={k:self.coordinate[k].reshape(-1,16)[sl].ravel() for k in ['native_time_s','native_information']}
        value.data=value.energy=None
        return value

    def waveform(self,coefficient,channel,derivatives,point_offset=None,spatial_derivatives=False):
        path=self.base_path+self.b@coefficient;rate=self.base_rate+self.bd@coefficient
        s=lift(self.anchor,self.velocity,self.direction,self.direction_rate,path,rate,self.motion.tx[3],self.motion.rx)
        position,velocity=s['position_ecef_m'],s['velocity_ecef_mps'];f=self.field;tx=self.motion.tx[channel];rx=self.motion.rx
        if point_offset is not None:
            point_offset=np.asarray(point_offset)
            if point_offset.ndim==1:point_offset=np.broadcast_to(point_offset,(len(self.t),3))
            if point_offset.shape!=(len(self.t),3):raise ValueError('Each active physical point needs one offset per original epoch')
            # A dictionary slot can refer to different time-gated physical
            # points. There is no velocity of the identity switch: inside
            # each declared support it is the fixed point's rigid velocity.
            position=position+np.einsum('tij,tj->ti',self.frame,point_offset)
            velocity=velocity+np.einsum('tij,tj->ti',self.frame_rate,point_offset)
        delta,omega,g=f.path_kinematics(position,velocity,transmitter=tx,receiver=rx,position_offset=position-f.data['centre_ecef_m'])
        radial_speed=np.sum(g*velocity,axis=1)
        factor,dfactor=carrier_bin(self.t,radial_speed,self.coordinate['native_time_s'],self.coordinate['native_information'],f.k)
        if not derivatives and not spatial_derivatives:return f.apply(f.path_response(delta,omega,range_bias_m=self.range_bias_m)*factor[:,None])
        raw,dl,domega,_=f.path_response(delta,omega,range_bias_m=self.range_bias_m,derivatives=True)
        dp=s['position_path_jacobian'];dv=s['velocity_path_jacobian'];gdp=np.sum(g*dp,axis=1)
        dvr=np.sum(g*dv+curvature(position,dp,tx,rx)*velocity,axis=1)
        dr=(dl*gdp[:,None]-f.k*.001024*domega*dvr[:,None])*factor[:,None]+raw*(dfactor*dvr)[:,None]
        dv=(-f.k*.001024*domega*gdp[:,None])*factor[:,None]+raw*(dfactor*gdp)[:,None]
        result=(f.apply(raw*factor[:,None]),f.apply(dr),f.apply(dv))
        if spatial_derivatives:
            partial=[]
            for j in range(2):
                point_dp=self.frame[:,:,j];point_dv=self.frame_rate[:,:,j]
                path_derivative=np.sum(g*point_dp,axis=1)
                speed_derivative=np.sum(g*point_dv+curvature(position,point_dp,tx,rx)*velocity,axis=1)
                derivative=(dl*path_derivative[:,None]-f.k*.001024*domega*speed_derivative[:,None])*factor[:,None]+raw*(dfactor*speed_derivative)[:,None]
                partial.append(f.apply(derivative))
            result+=tuple(partial)
        return result

    def evaluate(self,coefficient,phase,channel,*,training=True,derivatives=True,frozen_amplitude=None):
        result=self.waveform(coefficient,channel,derivatives);h,dr,dv=result if derivatives else (result,None,None)
        f=self.field;mask=self.training if training else ~self.training;y=self.data[training];b=self.amplitude
        norm=np.sum(abs(h)**2,axis=1)*mask
        means=[]
        for group in np.unique(self.groups):
            q=f.root_information*mask*(self.groups==group);denom=np.vdot(q,q).real
            if denom:means.append(np.asarray((b.T@(h*(q/np.sqrt(denom))[:,None])).T))
        mean=np.concatenate(means,axis=0)
        gram=(b.T@b.multiply(norm[:,None])).toarray()-mean.conj().T@mean
        rhs=b.T@np.sum(h.conj()*y,axis=1)
        if frozen_amplitude is None:
            solution=solve((gram+gram.conj().T)/2,rhs,self.energy[training],np.zeros(b.shape[1],int),np.array([phase]))
            amplitude=solution['amplitude']
        else:amplitude=np.asarray(frozen_amplitude);solution=None
        envelope=b@amplitude;gain=envelope*np.exp(1j*phase)
        prediction=profile_static(h*gain[:,None],f.root_information,mask,self.groups)*mask[:,None]
        residual=prediction-y;cost=float(np.vdot(residual,residual).real)
        value=dict(cost=cost,energy=self.energy[training],amplitude=amplitude,
            explained_fraction=1-cost/self.energy[training],solution=solution)
        if derivatives:
            gr=2*np.real(np.sum(residual.conj()*dr,axis=1)*gain)
            gv=2*np.real(np.sum(residual.conj()*dv,axis=1)*gain)
            gp=2*np.real(np.vdot(residual,1j*prediction))
            value['gradient']=np.r_[np.asarray(self.b.T@gr+self.bd.T@gv).ravel(),gp]
        return value


class PointPath:
    def __init__(self,bands):self.bands=bands;self.n=bands[0].b.shape[1];self.energy=sum(b.energy[True] for b in bands)

    def objective(self,parameters,full=False):
        values=[b.evaluate(parameters[:self.n],parameters[self.n+c],c) for c,b in enumerate(self.bands)]
        cost=sum(v['cost'] for v in values)/self.energy;gradient=np.zeros_like(parameters)
        for c,v in enumerate(values):gradient[:self.n]+=v['gradient'][:-1];gradient[self.n+c]=v['gradient'][-1]
        gradient/=self.energy
        return (cost,gradient,values) if full else (cost,gradient)
