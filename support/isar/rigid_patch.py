"""Low-order rigid-target likelihood with shared differential motion.

The image axes are north/east at the aperture midpoint. A common angular
history rotates physical scatterer offsets during formation, not the image
afterward. Translation is a common path-rate/acceleration/jerk correction to
the coarse Cartesian prior. Every band uses the same points and motion, with
independent constant complex scattering coefficients and stationary profiles.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares,minimize,LinearConstraint

from motion import excess_path


def profile_static(value,root_information,selected,groups=None):
    if groups is not None:
        groups=np.asarray(groups)
        if groups.shape!=np.asarray(root_information).shape:raise ValueError('Stationary-reference identities must accompany each epoch')
        unique=np.unique(groups)
        if len(unique)>1:
            a=np.asarray(value);result=a.copy()
            for group in unique:
                q=np.asarray(root_information)*np.asarray(selected)*(groups==group);norm=np.linalg.norm(q)
                if norm:
                    q=q/norm;result-=q.reshape((-1,)+(1,)*(a.ndim-1))*np.tensordot(q.conj(),a,axes=1)
            return result
    q=np.asarray(root_information)*np.asarray(selected)
    q=q/np.linalg.norm(q)
    a=np.asarray(value)
    return a-q.reshape((-1,)+(1,)*(a.ndim-1))*np.tensordot(q.conj(),a,axes=1)


@dataclass
class RigidPatch:
    """Exact original-row kinematics; parameters v, a, jerk, spin, spin-rate.

    Path coefficients use m/s, m/s², m/s³; angular coefficients degrees/s and
    degrees/s². The midpoint orientation is a coordinate gauge fixed to zero.
    """
    time: np.ndarray
    epoch: float
    position: np.ndarray
    velocity: np.ndarray
    local_to_ecef: np.ndarray
    reference_transmitter: np.ndarray
    receiver: np.ndarray
    reference_frame: np.ndarray | None = None
    reference_frame_rate: np.ndarray | None = None
    translation_direction: np.ndarray | None = None

    def __post_init__(self):
        self.dt=np.asarray(self.time)-self.epoch
        self.frame=(np.broadcast_to(self.local_to_ecef,(len(self.time),3,3)) if self.reference_frame is None else np.asarray(self.reference_frame))
        self.frame_rate=(np.zeros_like(self.frame) if self.reference_frame_rate is None else np.asarray(self.reference_frame_rate))
        if self.frame.shape!=(len(self.time),3,3) or self.frame_rate.shape!=self.frame.shape:
            raise ValueError('Reference orientation and derivative must accompany every original epoch')
        _,g=excess_path(self.position,self.reference_transmitter,self.receiver)
        gd=np.zeros_like(g)
        for ant in (self.reference_transmitter,self.receiver):
            d=self.position-ant;n=np.linalg.norm(d,axis=1);unit=d/n[:,None]
            gd+=(self.velocity-unit*np.sum(unit*self.velocity,axis=1)[:,None])/n[:,None]
        up=self.local_to_ecef[:,2]
        g-=np.sum(g*up,axis=1)[:,None]*up
        gd-=np.sum(gd*up,axis=1)[:,None]*up
        n=np.sum(g*g,axis=1)
        self.direction=g/n[:,None]
        self.direction_rate=gd/n[:,None]-2*g*np.sum(g*gd,axis=1)[:,None]/n[:,None]**2
        if self.translation_direction is not None:
            direction=np.asarray(self.translation_direction,float)
            if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:
                raise ValueError('A constant Cartesian translation direction must be a finite nonzero vector')
            self.direction=np.broadcast_to(direction,self.position.shape)
            self.direction_rate=np.zeros_like(self.direction)

    def point_state(self,parameters,point,*,derivatives=False,origin=None):
        """Return position/velocity, optionally relative to a fixed origin.

        Subtract the declared observation origin before adding fitted small
        displacements. This avoids rounding optimisation increments through
        Earth-sized coordinates; velocity and parameter derivatives are unchanged.
        """
        v,a,j,w,b=np.asarray(parameters);dt=self.dt;deg=np.pi/180
        shift=v*dt+.5*a*dt**2+j*dt**3/6
        rate=v+a*dt+.5*j*dt**2
        theta=(w*dt+.5*b*dt**2)*deg;td=(w+b*dt)*deg
        c,s=np.cos(theta),np.sin(theta);e,n,z=np.asarray(point)
        offset=np.c_[c*e-s*n,s*e+c*n,np.full(len(dt),z)]
        first=np.c_[-s*e-c*n,c*e-s*n,np.zeros(len(dt))]
        second=np.c_[-c*e+s*n,-s*e-c*n,np.zeros(len(dt))]
        offset_rate=np.einsum('nij,nj->ni',self.frame_rate,offset)
        first_rate=np.einsum('nij,nj->ni',self.frame_rate,first)
        offset=np.einsum('nij,nj->ni',self.frame,offset)
        first=np.einsum('nij,nj->ni',self.frame,first);second=np.einsum('nij,nj->ni',self.frame,second)
        base=self.position if origin is None else self.position-np.asarray(origin)
        pos=base+shift[:,None]*self.direction+offset
        vel=self.velocity+rate[:,None]*self.direction+shift[:,None]*self.direction_rate+offset_rate+td[:,None]*first
        if not derivatives:return pos,vel
        h=np.c_[dt,.5*dt**2,dt**3/6];hd=np.c_[np.ones(len(dt)),dt,.5*dt**2]
        angular=np.c_[dt,.5*dt**2]*deg;angular_rate=np.c_[np.ones(len(dt)),dt]*deg
        dp=np.concatenate([self.direction[:,:,None]*h[:,None,:],first[:,:,None]*angular[:,None,:]],axis=2)
        dv=np.concatenate([self.direction[:,:,None]*hd[:,None,:]+self.direction_rate[:,:,None]*h[:,None,:],
            first_rate[:,:,None]*angular[:,None,:]+second[:,:,None]*(td[:,None]*angular)[:,None,:]+first[:,:,None]*angular_rate[:,None,:]],axis=2)
        return pos,vel,dp,dv

    def orientation(self,parameters):
        w,b=np.asarray(parameters)[3:];dt=self.dt;theta=np.radians(w*dt+.5*b*dt**2);rate=np.radians(w+b*dt)
        c,s=np.cos(theta),np.sin(theta);r=np.zeros((len(dt),3,3));dr=np.zeros_like(r)
        r[:,0,0]=r[:,1,1]=c;r[:,1,0]=s;r[:,0,1]=-s;r[:,2,2]=1
        dr[:,0,0]=dr[:,1,1]=-s*rate;dr[:,1,0]=c*rate;dr[:,0,1]=-c*rate
        return self.frame@r,self.frame_rate@r+self.frame@dr


class PatchBand:
    def __init__(self,field,kinematics,points,training,transmitter,receiver,*,range_origin_m=0.):
        self.field,self.kinematics,self.points=field,kinematics,np.asarray(points)
        self.training=np.asarray(training,bool);self.tx,self.rx=transmitter,receiver;self.origin=range_origin_m
        self.selected=np.repeat(self.training,len(field.offsets))
        self.data=self.project(field.y).ravel()[self.selected]
        self.energy=np.vdot(self.data,self.data).real

    def project(self,value,selected=None):
        return profile_static(value,self.field.root_information,self.training if selected is None else selected)

    def design(self,parameters,*,derivatives=False,points=None,point_derivatives=False):
        values=[];partials=[];f=self.field
        frame,frame_rate=self.kinematics.orientation(parameters)
        for point in self.points if points is None else points:
            state=self.kinematics.point_state(parameters,point,derivatives=derivatives)
            pos,vel=state[:2]
            if derivatives and point_derivatives:
                state=(*state[:2],np.concatenate([state[2],frame[:,:,:2]],axis=-1),
                    np.concatenate([state[3],frame_rate[:,:,:2]],axis=-1))
            result=f.trajectory(pos,vel,transmitter=self.tx,receiver=self.rx,range_bias_m=self.origin,
                position_offset=pos-f.data['centre_ecef_m'],
                **({'position_jacobian':state[2],'velocity_jacobian':state[3]} if derivatives else {}))
            if derivatives:
                h,dh=result;partials.append(dh)
            else:h=result
            values.append(h)
        return np.stack(values,axis=-1),np.stack(partials,axis=-2) if derivatives else None

    def evaluate(self,parameters,*,derivatives=True):
        h,dh=self.design(parameters,derivatives=derivatives)
        a=self.project(h).reshape(-1,len(self.points))[self.selected]
        u,s,vh=np.linalg.svd(a,full_matrices=False)
        if s[-1]<s[0]*1e-10:raise ValueError('Rigid patch has unresolved scene columns')
        coef=vh.conj().T@((u.conj().T@self.data)/s)
        r=self.data-a@coef
        result=dict(residual=r,coefficient=coef,condition=s[0]/s[-1],value=np.vdot(r,r).real)
        if derivatives:
            da=self.project(dh).reshape(-1,len(self.points),5)[self.selected]
            fixed=np.einsum('ncp,c->np',da,coef)
            cross=np.einsum('ncp,n->cp',da.conj(),r)
            jac=-fixed+u@(u.conj().T@fixed)-u@((vh@cross)/s[:,None])
            result.update(jacobian=jac,gradient=2*np.real(jac.conj().T@r))
        return result

    def validation(self,parameters):
        model=self.evaluate(parameters,derivatives=False);h,_=self.design(parameters)
        test=~self.training
        a=self.project(h,test).reshape(-1,len(self.points))[np.repeat(test,len(self.field.offsets))]
        y=self.project(self.field.y,test).ravel()[np.repeat(test,len(self.field.offsets))]
        predicted=a@model['coefficient'];residual=y-predicted
        free=np.linalg.lstsq(a,y,rcond=1e-10)[0];best=a@free
        return dict(unused_rss=np.vdot(residual,residual).real,unused_data_energy=np.vdot(y,y).real,
            unused_prediction_energy=np.vdot(predicted,predicted).real,
            unused_free_scene_energy=np.vdot(best,best).real,
            conditional_scene_transfer_loss_db=10*np.log10(np.vdot(predicted,predicted).real*np.vdot(best,best).real/abs(np.vdot(predicted,best))**2),
            coefficient_real_imag=np.c_[model['coefficient'].real,model['coefficient'].imag].tolist(),condition=model['condition'])


def fit(bands,initial,*,bounds,scale=None,max_evaluations=100):
    scale=np.asarray([.05,.005,.0005,.05,.005] if scale is None else scale)
    cache=[None,None]
    energy=sum(b.energy for b in bands)
    def evaluate(p):
        if cache[0] is None or not np.array_equal(p,cache[0]):
            cache[:]=[p.copy(),[b.evaluate(p*scale) for b in bands]]
        return cache[1]
    def fun(p):return np.concatenate([np.r_[r['residual'].real,r['residual'].imag] for r in evaluate(p)])/np.sqrt(energy)
    def jac(p):return np.concatenate([np.r_[r['jacobian'].real,r['jacobian'].imag] for r in evaluate(p)])*scale/np.sqrt(energy)
    r=least_squares(fun,np.asarray(initial)/scale,jac=jac,bounds=(np.asarray(bounds[0])/scale,np.asarray(bounds[1])/scale),
        max_nfev=max_evaluations,ftol=1e-11,xtol=1e-11,gtol=1e-9,x_scale='jac')
    p=r.x*scale
    return p,dict(parameters=p.tolist(),objective=float(2*r.cost),maximum_scaled_gradient=float(r.optimality),
        success=bool(r.success and r.optimality<1e-7),evaluations=r.nfev,message=r.message,
        angular_sweep_degrees=float(p[3]*(bands[0].kinematics.time[-1]-bands[0].kinematics.time[0])),
        validation=[b.validation(p) for b in bands])


def polish(bands,initial,*,bounds,max_iterations=300):
    """Whiten all five local motion directions, retaining the same likelihood.

    Column scaling alone leaves near-parallel path/rotation derivatives. An
    invertible SVD change of optimizer coordinates removes that correlation;
    it neither drops a mode nor changes any physical bound or observation.
    """
    initial=np.asarray(initial);energy=sum(b.energy for b in bands)
    evaluation=[b.evaluate(initial) for b in bands]
    j=np.concatenate([np.r_[e['jacobian'].real,e['jacobian'].imag] for e in evaluation])/np.sqrt(energy)
    scale=np.array([.05,.005,.0005,.05,.005]);_,singular,vh=np.linalg.svd(j*scale,full_matrices=False)
    if singular[-1]<singular[0]*1e-10:raise ValueError('Local rigid motion has an unresolved information direction')
    transform=scale[:,None]*vh.T/singular
    cache=[None,None];evaluations=0
    def objective(q):
        nonlocal evaluations
        if cache[0] is None or not np.array_equal(q,cache[0]):
            p=initial+transform@q;local=[b.evaluate(p) for b in bands];evaluations+=1
            value=sum(e['value'] for e in local)/energy
            gradient=sum(e['gradient'] for e in local)/energy
            cache[:]=[q.copy(),(value,gradient@transform)]
        return cache[1]
    r=minimize(objective,np.zeros(5),jac=True,method='SLSQP',
        constraints=[LinearConstraint(transform/scale[:,None],(np.asarray(bounds[0])-initial)/scale,(np.asarray(bounds[1])-initial)/scale)],
        options={'maxiter':max_iterations,'ftol':1e-13})
    p=initial+transform@r.x;local=[b.evaluate(p) for b in bands]
    gradient=sum(e['gradient'] for e in local)/energy
    g=gradient*scale;at_lower=p<np.asarray(bounds[0])+1e-8;at_upper=p>np.asarray(bounds[1])-1e-8
    g=np.where(at_lower,np.minimum(g,0),np.where(at_upper,np.maximum(g,0),g))
    return p,dict(parameters=p.tolist(),objective=float(r.fun),maximum_scaled_gradient=float(np.max(abs(g))),
        success=bool(r.success and np.max(abs(g))<1e-7),evaluations=evaluations,message=r.message,
        optimizer='SLSQP in invertible initial-Jacobian SVD coordinates; unchanged physical bounds and likelihood',
        initial_jacobian_singular_values=singular.tolist(),
        validation=[b.validation(p) for b in bands])
