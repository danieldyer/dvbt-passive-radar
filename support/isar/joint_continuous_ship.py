"""Shared smooth ship motion and correlated static antenna phase centres.

Physical antenna changes affect excess delay, carrier phase, Doppler and the
original 16-symbol bin response together. The received data and its carrier
reference are immutable. A receiver change never translates the ship.
"""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,hstack
from continuous_ship_fit import MotionCoordinates,GlobalPointBand
from full_path_point import carrier_bin
from bistatic_path_lift import curvature
from motion import excess_path


class JointCoordinates:
    def __init__(self,motion,antennas):
        self.base=MotionCoordinates(motion);self.motion=motion;self.antennas=antennas
        self.nmotion=self.base.n;self.height_scale=20.;self.location_start=self.nmotion+1
        self.n=self.location_start+antennas.count
        self.gauge_point=motion.state([motion.result['anchor_s']])['position_ecef_m'][0]

    def constraints(self):
        matrix,lo,hi=self.base.constraints()
        return np.pad(matrix,((0,0),(0,self.n-self.nmotion))),lo,hi

    def reference_phase(self,q,channel,derivatives=False):
        tx0=self.motion.tx[channel];rx0=self.motion.rx
        offset=self.antennas.offsets(q[self.location_start:],channel);tx,rx=tx0+offset[0],rx0+offset[1]
        point=self.gauge_point[None]
        delta=float(excess_path(point,tx,rx)[0][0]-excess_path(point,tx0,rx0)[0][0])
        if not derivatives:return delta
        baseline=(tx-rx)/np.linalg.norm(tx-rx);a=self.gauge_point-tx;b=self.gauge_point-rx
        return delta,np.r_[-a/np.linalg.norm(a)-baseline,-b/np.linalg.norm(b)+baseline]

    def export(self,q,path,report):
        self.base.export(q[:self.nmotion],path,report);d=json.loads(Path(path).read_text())
        offsets=[self.antennas.offsets(q[self.location_start:],c) for c in range(len(self.motion.tx))]
        receiver_offset=offsets[0][1]
        for offset in offsets:np.testing.assert_array_equal(offset[1],receiver_offset)
        d['transmitters_ecef_m']=(self.motion.tx+np.array([o[0] for o in offsets])).tolist()
        d['receiver_ecef_m']=(self.motion.rx+receiver_offset).tolist()
        # Exported position coefficients are receiver-relative. Compensate
        # this coordinate origin so changing Rx does not move the vessel.
        pc=np.asarray(d['position_coefficients_enu_m'])-receiver_offset@self.motion.rotation.T
        pc[:,2]+=self.height_scale*q[self.nmotion]
        d['position_coefficients_enu_m']=pc.tolist()
        d['antenna_refinement']=self.antennas.describe(q[self.location_start:])
        d['antenna_refinement']['common_scatterer_height_shift_m']=self.height_scale*q[self.nmotion]
        d['antenna_refinement']['template_phase_reference_ecef_m']=self.gauge_point.tolist()
        d['antenna_refinement']['accepted_as_calibration']=False
        Path(path).write_text(json.dumps(d,indent=2)+'\n')


class JointPointBand(GlobalPointBand):
    def __init__(self,field,coordinate,model,channel,amplitude_spacing=2.):
        super().__init__(field,coordinate,model.base,amplitude_spacing)
        self.model=model;self.channel=channel;self.nmotion=model.nmotion
        padding=csr_matrix((len(self.t),model.n-self.nmotion))
        bases=[hstack([b,padding],format='csr') for b in self.motion_bases]
        height=np.zeros(model.n);height[self.nmotion]=model.height_scale
        bases.append(csr_matrix(np.broadcast_to(height,(len(self.t),model.n))))
        mapping=model.antennas.mapping(channel)
        for row in mapping:
            basis=np.r_[np.zeros(model.location_start),row]
            bases.append(csr_matrix(np.broadcast_to(basis,(len(self.t),model.n))))
        self.motion_bases=tuple(bases);self.b,self.bd=bases[:2]

    def point_state(self,q,point):
        position,velocity,offset,dp,dv=super().point_state(q,point)
        up=self.motion.rotation[2]
        delta=up*(self.model.height_scale*q[self.nmotion])
        dp=np.concatenate([dp,np.broadcast_to(up,(len(self.t),3))[:,:,None]],axis=2)
        dv=np.concatenate([dv,np.zeros((len(self.t),3,1))],axis=2)
        return position+delta,velocity,offset+delta,dp,dv

    def responses(self,q,channel,derivatives,point_offset=None):
        if channel!=self.channel:raise ValueError('Antenna derivative mapping belongs to a different channel')
        f=self.field;m=self.motion
        position,velocity,offset,dp,dv=self.point_state(q,self.reference_point if point_offset is None else point_offset)
        antenna=self.model.antennas.offsets(q[self.model.location_start:],channel)
        tx,rx=m.tx[channel]+antenna[0],m.rx+antenna[1]
        kw=dict(transmitter=tx,receiver=rx,position_offset=offset)
        delay,omega,g=f.path_kinematics(position,velocity,**kw)
        speed=np.sum(g*velocity,axis=1)
        factor,dfactor=carrier_bin(self.t,speed,self.coordinate['native_time_s'],self.coordinate['native_information'],f.k)
        reference=self.model.reference_phase(q,channel,derivatives)
        delta,dref=reference if derivatives else (reference,None)
        phase=np.exp(1j*f.k*delta)
        if not derivatives:return f.apply(f.path_response(delay,omega)*factor[:,None]*phase)
        raw,dl,domega,_=f.path_response(delay,omega,derivatives=True)
        # Range whitening is linear and acts independently at each epoch.
        # Every real parameter changes only path, path rate and a constant
        # phase origin. Whiten those three responses once, then chain the
        # scalar geometric derivatives; do not whiten 13 large tensors.
        value=f.apply(raw*factor[:,None]*phase)
        path_partial=f.apply(dl*factor[:,None]*phase)
        speed_partial=f.apply((-f.k*.001024*domega*factor[:,None]+raw*dfactor[:,None])*phase)
        path_partials=list(np.einsum('ti,tij->tj',g,dp).T);speed_partials=[]
        for j in range(7):
            speed_partials.append(np.sum(g*dv[:,:,j]+curvature(position,dp[:,:,j],tx,rx)*velocity,axis=1))
        baseline=(tx-rx)/np.linalg.norm(tx-rx)
        for antenna,sign in [(tx,-1),(rx,1)]:
            ray=position-antenna;distance=np.linalg.norm(ray,axis=1);unit=ray/distance[:,None]
            path_partials.extend((-unit+sign*baseline).T)
            hessian_velocity=-(velocity-unit*np.sum(unit*velocity,axis=1)[:,None])/distance[:,None]
            speed_partials.extend(hessian_velocity.T)
        return value,path_partial,speed_partial,np.column_stack(path_partials),np.column_stack(speed_partials),dref

    def waveform(self,q,channel,derivatives,point_offset=None,spatial_derivatives=False):
        if spatial_derivatives:raise ValueError('Joint fit uses its declared shared physical source support')
        response=self.responses(q,channel,derivatives,point_offset)
        if not derivatives:return response
        value,path_partial,speed_partial,dl,ds,dref=response
        partials=[path_partial*a[:,None]+speed_partial*b[:,None] for a,b in zip(dl.T,ds.T)]
        for j in range(6):partials[7+j]+=value*(1j*self.field.k*dref[j])
        return (value,*partials)

    def pullback(self,q,channel,point,residual,gain):
        """Exact likelihood adjoint without materialising 13 range tensors."""
        value,path,speed,dl,ds,dref=self.responses(q,channel,True,point)
        cotangent=residual.conj()*gain[:,None]
        a=2*np.real(np.sum(cotangent*path,axis=1));b=2*np.real(np.sum(cotangent*speed,axis=1))
        physical=dl*a[:,None]+ds*b[:,None]
        phase=2*np.real(np.sum(cotangent*(1j*value),axis=1))
        physical[:,7:]+=phase[:,None]*(self.field.k*dref)[None,:]
        return sum(np.asarray(basis.T@physical[:,j]).ravel() for j,basis in enumerate(self.motion_bases))


class JointPrior:
    def __init__(self,model,motion_prior,*,locations=True):
        self.model=model;self.motion_prior=motion_prior;self.locations=locations
        self.gram=np.zeros((model.n,model.n));n=model.nmotion
        self.gram[:n,:n]=motion_prior.gram
        self.gram[n,n]=1.
        if locations:self.gram[n+1:,n+1:]=np.eye(model.antennas.count)

    def evaluate(self,q):
        n=self.model.nmotion;value,g=self.motion_prior.evaluate(q[:n]);gradient=np.r_[g,np.zeros(len(q)-n)]
        value+=.5*q[n]**2;gradient[n]=q[n]
        if self.locations:value+=.5*float(q[n+1:]@q[n+1:]);gradient[n+1:]=q[n+1:]
        return value,gradient

    def diagnostics(self,q):
        n=self.model.nmotion
        return dict(motion=self.motion_prior.diagnostics(q[:n]),location_negative_log_prior=.5*float(q[n+1:]@q[n+1:]) if self.locations else 0.,
            height_negative_log_prior=.5*float(q[n]**2),common_scatterer_height_shift_m=self.model.height_scale*q[n],
            locations=self.model.antennas.describe(q[n+1:]),scope='Declared Gaussian location/height uncertainty, not an extra radar measurement')
