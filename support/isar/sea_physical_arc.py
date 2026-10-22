"""Original-row positive-amplitude physical point with declared HH sea rays."""
import numpy as np

from differentiable_sea import paths
from physical_amplitude_arc import PhysicalAmplitudeArc,state_derivatives


class SeaPhysicalArc(PhysicalAmplitudeArc):
    def __init__(self,*args,water,family='transmit',**kwargs):
        super().__init__(*args,**kwargs)
        self.water=water
        self.rays={'direct':(0,),'transmit':(0,1),'receive':(0,2),'both':(0,1,2,3)}[family]
        self.family=family

    def response(self,parameters):
        g,c=self.geometry,self.channel
        state,jac,length,dlength=state_derivatives(g,parameters[:g.count])
        p,v,a,tx,rx=state;dp,dv,da,dtx,drx=jac
        dt=self.field.data['row_time']-self.midpoint
        local=p+dt[:,None]*v+.5*dt[:,None]**2*a
        position=g.receiver+local@g.rotation
        velocity=(v+dt[:,None]*a)@g.rotation
        dx=np.einsum('nij,ik->nkj',dp[None]+dt[:,None,None]*dv+.5*dt[:,None,None]**2*da,g.rotation)
        dvelocity=np.einsum('nij,ik->nkj',dv[None]+dt[:,None,None]*da,g.rotation)
        values=paths(position,velocity,g.receiver+tx[c]@g.rotation,g.receiver+rx@g.rotation,
            np.asarray(self.water['centre_ecef_m']),self.water['radius_m'],dx,dvelocity,
            g.rotation.T@dtx[c],g.rotation.T@drx,self.field.metadata['rf_hz'])
        ls,vs,gs,dls,dvs,dgs=values
        raw=np.zeros((len(position),len(self.field.offsets)),complex)
        derivative=np.zeros(raw.shape+(g.count,),complex)
        for ray in self.rays:
            value,delay,doppler,_=self.field.path_response(ls[ray]-self.field.data['centre_path_m'],
                -self.field.k*.001024*vs[ray],derivatives=True)
            raw+=value*gs[ray,:,None]
            change=delay[:,:,None]*dls[ray,:,None,:]-doppler[:,:,None]*(self.field.k*.001024*dvs[ray,:,None,:])
            derivative+=change*gs[ray,:,None,None]+value[:,:,None]*dgs[ray,:,None,:]
        derivative+=1j*self.field.k*raw[:,:,None]*dlength[c]
        rotation=np.exp(1j*(self.field.k*(length[c]-self.reference_path)+parameters[g.count+c]))
        raw,derivative=raw*rotation,derivative*rotation
        phase=np.zeros(raw.shape+(5,),complex);phase[:,:,c]=1j*raw
        return raw,np.concatenate([derivative,phase],axis=2)
