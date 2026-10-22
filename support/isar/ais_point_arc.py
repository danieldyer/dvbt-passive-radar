"""AIS-tied physical point motion with a fixed antenna-to-scatterer lever arm.

The heading-change control either keeps that lever arm in fixed N/E axes or
rotates it by the prescribed change in interpolated AIS course. The latter is
an explicit hypothesis about heading change, not a true-heading measurement.
There is no fitted yaw, free velocity or free acceleration in this model.
"""
import numpy as np
from scipy.linalg import solve_triangular

from amplitude_arc import AmplitudeArc
from antenna_motion import Antennas
from flash_geometry import observables
from piecewise_ais import LocalAisReference


class AisPointGeometry:
    def __init__(self,measured,covariance,track,utc,rotation,receiver,assignment,*,follow_course_change=True):
        self.measured=np.asarray(measured);self.rotation=np.asarray(rotation);self.receiver=np.asarray(receiver)
        self.local_transmitters=(np.array([a['transmitter_ecef_m'] for a in assignment])-self.receiver)@self.rotation.T
        self.antennas=Antennas(rotation,[a['site_id'] for a in assignment],receiver_sigma=(0.,0.,7.5))
        self.nantenna=18;self.count=22;self.follow_course_change=follow_course_change
        self.reference=LocalAisReference(track,utc,rotation,receiver,half_width_s=15.)
        self.nuisance_sigma=np.array([100.,100.,20.,2.])
        self.lower=np.r_[np.full(18,-3.),[-3.5,-3.5,-3.,-5.]]
        self.upper=-self.lower
        self.lower[17]=-10/7.5;self.upper[17]=10/7.5
        self.whiten=[solve_triangular(np.linalg.cholesky(c),np.eye(3),lower=True) for c in covariance]

    def bounds(self):return self.lower,self.upper

    def trajectory(self,parameters,time):
        q=np.asarray(parameters[18:])*self.nuisance_sigma
        t=np.atleast_1d(time)+q[3]
        position,velocity,acceleration,jerk=[self.reference.sample(t,j) for j in range(4)]
        offset=q[:3]
        if not self.follow_course_change:
            return position+offset,velocity,acceleration
        v=velocity[:,:2];a=acceleration[:,:2];j=jerk[:,:2]
        speed=np.sqrt(np.sum(v*v,axis=1));u=v/speed[:,None]
        vmid=self.reference(q[3],1)[:2];u0=vmid/np.sqrt(vmid@vmid)
        speed_rate=np.sum(u*a,axis=1);du=(a-u*speed_rate[:,None])/speed[:,None]
        speed_second=np.sum(du*a+u*j,axis=1)
        ddu=(j-2*du*speed_rate[:,None]-u*speed_second[:,None])/speed[:,None]
        def rotate(direction,vertical):
            cosine=direction@u0;sine=u0[0]*direction[:,1]-u0[1]*direction[:,0]
            return np.column_stack([cosine*offset[0]-sine*offset[1],
                sine*offset[0]+cosine*offset[1],np.full(len(t),vertical)])
        return position+rotate(u,offset[2]),velocity+rotate(du,0.),acceleration+rotate(ddu,0.)

    def antennas_at(self,parameters):
        delta=self.antennas.factor@parameters[:18]
        return self.local_transmitters+delta[:15].reshape(5,3),delta[15:]

    def state(self,parameters):
        p,v,a=self.trajectory(parameters,[0.]);tx,rx=self.antennas_at(parameters)
        return p[0],v[0],a[0],tx,rx

    def prediction(self,parameters):return observables(*self.state(parameters))

    def data_residual(self,parameters):
        return np.concatenate([w@e for w,e in zip(self.whiten,self.prediction(parameters)-self.measured)])


class AisPointArc(AmplitudeArc):
    def __init__(self,field,training,knots,geometry,channel,midpoint_s,*,stationary_paths=None,stationary_design=None):
        super().__init__(field,training,knots,stationary_paths=stationary_paths,stationary_design=stationary_design)
        self.geometry,self.channel,self.midpoint=geometry,channel,midpoint_s
        self.reference_local=(field.data['centre_ecef_m']-geometry.receiver)@geometry.rotation.T
        self.reference_path=np.interp(midpoint_s,field.data['row_time'],field.data['centre_path_m'])

    def response(self,parameters,*,range_bias_m=0.,envelope_derivative=False):
        g,c=self.geometry,self.channel;q=np.asarray(parameters[:g.count])
        time=self.field.data['row_time']-self.midpoint
        local,velocity,_=g.trajectory(q,time);tx,rx=g.antennas_at(q)
        length=g.prediction(q)[:,0]
        dp=np.empty(local.shape+(g.count,));dv=np.empty_like(dp)
        dtx=np.empty(tx.shape+(g.count,));drx=np.empty(rx.shape+(g.count,));dlength=np.empty((5,g.count))
        for j in range(g.count):
            probe=q.astype(complex);probe[j]+=1e-20j
            p,v,_=g.trajectory(probe,time);t,r=g.antennas_at(probe)
            dp[:,:,j]=p.imag/1e-20;dv[:,:,j]=v.imag/1e-20
            dtx[:,:,j]=t.imag/1e-20;drx[:,j]=r.imag/1e-20
            dlength[:,j]=g.prediction(probe)[:,0].imag/1e-20
        offset=(local-self.reference_local)@g.rotation
        position=self.field.data['centre_ecef_m']+offset
        raw,derivative=self.field.raw_trajectory(position,velocity@g.rotation,
            position_jacobian=np.einsum('nij,ik->nkj',dp,g.rotation),
            velocity_jacobian=np.einsum('nij,ik->nkj',dv,g.rotation),
            transmitter=g.receiver+tx[c]@g.rotation,receiver=g.receiver+rx@g.rotation,
            antenna_derivatives=True,position_offset=offset,range_bias_m=range_bias_m,
            range_bias_derivative=envelope_derivative)
        envelope=derivative[:,:,-1] if envelope_derivative else None
        antenna=np.r_[g.rotation.T@dtx[c],g.rotation.T@drx]
        derivative=derivative[:,:,:g.count]+derivative[:,:,g.count:g.count+6]@antenna
        derivative+=1j*self.field.k*raw[:,:,None]*dlength[c]
        rotation=np.exp(1j*(self.field.k*(length[c]-self.reference_path)+parameters[g.count+c]))
        raw,derivative=raw*rotation,derivative*rotation
        phase=np.zeros(raw.shape+(5,),complex);phase[:,:,c]=1j*raw
        changes=[derivative,phase]
        if envelope_derivative:changes.append((envelope*rotation)[:,:,None])
        return raw,np.concatenate(changes,axis=2)
