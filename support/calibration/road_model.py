"""Continuous vehicle trajectory on a mapped road, with finite map uncertainty."""
import numpy as np
from scipy.interpolate import BSpline

from geometry import unit
from range_fit import RangeProblem


class RoadTrajectory:
    def __init__(self,seed,road,coarse_state,channel_sites,channel_offsets,controls=12):
        self.seed,self.road=seed,road
        self.receiver_enu_m=np.zeros(3)
        self.receiver_channel_offsets_enu_m=np.zeros((5,3))
        self.channel_sites=np.asarray(channel_sites);self.channel_offsets=np.asarray(channel_offsets)
        self.half=seed['duration_s']/2
        self.coarse=np.asarray(coarse_state)
        knots=np.r_[np.full(4,-1.),np.linspace(-1,1,controls-2)[1:-1],np.full(4,1.)]
        self.radial_count=controls
        self.spline=BSpline(knots,np.eye(controls),3,extrapolate=False)
        self.count=controls+3
        self.probe=seed['epoch']+np.linspace(-self.half,self.half,121)
        critical=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(self.coarse))
        interior=[root.real for root in critical if abs(root.imag)<1e-10 and -1<root.real<1]
        base=np.polynomial.polynomial.polyval(np.r_[-1.,interior,1.],self.coarse)
        self.lower=max(-100.,-float(base.min())+1e-4)
        self.upper=min(100.,road.length-float(base.max())-1e-4)
        if self.lower>=0 or self.upper<=0:
            raise ValueError('Coarse trajectory lacks full connected-road support')

    def length_basis(self,time):
        t=(np.asarray(time)-self.seed['epoch'])/self.half
        return np.polynomial.polynomial.polyval(t,self.coarse),self.spline(t)

    def positions(self,p,time,sites,free_sites=False):
        t=(np.asarray(time)-self.seed['epoch'])/self.half
        base,b=self.length_basis(time)
        s=base+b@p[:self.radial_count]
        lateral=p[self.radial_count]+t*p[self.radial_count+1]
        normal,dnormal=self.road.frame(s)
        x=self.road.centre(s)+lateral[:,None]*normal
        x[:,2]+=2+p[self.radial_count+2]
        j=np.zeros((len(t),3,len(p)))
        j[:,:,:self.radial_count]=(self.road.centre(s,1)+lateral[:,None]*dnormal)[:,:,None]*b[:,None,:]
        j[:,:,self.radial_count]=normal
        j[:,:,self.radial_count+1]=t[:,None]*normal
        j[:,2,self.radial_count+2]=1.
        tx=np.asarray(sites).copy()
        if free_sites:tx+=p[self.count:].reshape(3,3)*[30,30,75]
        return x,j,tx

    def paths(self,p,time,channel,sites,free_sites=False):
        x,jx,tx=self.positions(p,time,sites,free_sites)
        transmitter=tx[self.channel_sites[channel]]+self.channel_offsets[channel]
        receiver=self.receiver(channel)
        a=x-transmitter;b=x-receiver;direct=transmitter-receiver
        path=np.linalg.norm(a,axis=1)+np.linalg.norm(b,axis=1)-np.linalg.norm(direct,axis=1)
        j=np.einsum('ni,nij->nj',unit(a)+unit(b),jx)
        if free_sites:
            derivative=np.zeros((len(time),3,3))
            derivative[np.arange(len(time)),self.channel_sites[channel]]=(-unit(a)-unit(direct))*[30,30,75]
            j[:,self.count:]+=derivative.reshape(len(time),9)
        return path,j

    def receiver(self,channel):
        return self.receiver_enu_m+self.receiver_channel_offsets_enu_m[channel]

    def receiver_derivative(self,p,time,channel,sites,free_sites=False):
        x,_,tx=self.positions(p,time,sites,free_sites)
        transmitter=tx[self.channel_sites[channel]]+self.channel_offsets[channel]
        receiver=self.receiver(channel)
        return -unit(x-receiver)+unit(transmitter-receiver)

    def kinematics(self,p,sites,free_sites=False):
        x,j,_=self.positions(p,self.probe,sites,free_sites)
        step=self.probe[1]-self.probe[0]
        values,derivatives=[],[]
        for _ in range(3):
            x=np.gradient(x,step,axis=0,edge_order=2);j=np.gradient(j,step,axis=0,edge_order=2)
            values.append(x[4:-4]);derivatives.append(j[4:-4])
        return values,derivatives

    def bounds(self,free_sites=False):
        lo=np.r_[np.full(self.radial_count,self.lower),-15.,-8.,-10.]
        hi=np.r_[np.full(self.radial_count,self.upper),15.,8.,15.]
        if free_sites:lo,hi=np.r_[lo,np.full(9,-3.)],np.r_[hi,np.full(9,3.)]
        return lo,hi


class RoadProblem(RangeProblem):
    def __init__(self,channels,coarse,sites,fields,model,motion=(40.,3.,3.)):
        super().__init__(channels,coarse,sites,fields,motion=motion)
        self.model=model

    def auxiliary(self,p,free_sites=False,jacobian=False,columns=None,with_motion=True,with_coarse=False):
        pieces=[]
        if with_motion:
            values,derivatives=self.model.kinematics(p,self.sites,free_sites)
            for value,j,scale in zip(values,derivatives,self.motion):
                pieces.append(j.reshape(-1,j.shape[-1])/(scale*np.sqrt(len(value))) if jacobian else value.ravel()/(scale*np.sqrt(len(value))))
            # Cross-track OSM/lane error and reflecting height are nuisance
            # priors, explicitly separate from the measured complex likelihood.
            scale=np.array([5.,3.,5.]);indices=np.arange(self.model.radial_count,self.model.count)
            pieces.append(np.eye(len(p))[indices]/scale[:,None] if jacobian else p[indices]/scale)
            base,b=self.model.length_basis(self.model.probe)
            v=(np.polynomial.polynomial.polyval((self.model.probe-self.model.seed['epoch'])/self.model.half,
                np.polynomial.polynomial.polyder(self.model.coarse))+self.model.spline((self.model.probe-self.model.seed['epoch'])/self.model.half,1)@p[:self.model.radial_count])/self.model.half
            posted=self.model.road.description['posted_speed_mps']
            vmax=40. if posted is None else max(posted*1.2,posted+3.)
            allowed=self.model.road.description['directions']
            lo=-vmax if -1 in allowed else 0.;hi=vmax if 1 in allowed else 0.
            active=(v<lo)|(v>hi)
            if jacobian:
                j=np.zeros((len(v),len(p)))
                j[:,:self.model.radial_count]=self.model.spline((self.model.probe-self.model.seed['epoch'])/self.model.half,1)/self.model.half
                pieces.append(j*active[:,None]/np.sqrt(len(v)))
            else:pieces.append((v-np.clip(v,lo,hi))/np.sqrt(len(v)))
        if free_sites:
            pieces.append(np.c_[np.zeros((9,self.model.count)),np.eye(9)] if jacobian else p[self.model.count:])
        if columns is not None and jacobian:pieces=[v[:,columns] for v in pieces]
        return pieces
