"""Continuous, shared physical radar motion with fixed inter-band calibration.

Positive reflectivity envelopes and a common reflector phase per declared
aspect interval are nuisance parameters. They never correct the recorded
samples or introduce a time-varying band-dependent phase history. Derivative
bounds on B-spline control coefficients certify the kinematic limits between
samples too, by the convex-hull property of the derivative B-spline.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix,hstack
from scipy.optimize import minimize,LinearConstraint,linprog
from audit_ship_scattering_envelopes import linear_basis
from bistatic_path_lift import curvature
from full_path_point import carrier_bin
from full_path_scene import SceneBand,warm_phases
from full_ship_field import load
from rigid_patch import profile_static
from motion import excess_path
from continuous_ship_motion import ContinuousShipMotion,initialise
from kinematic_prior import KinematicPrior


class MotionCoordinates:
    def __init__(self,motion):
        self.motion=motion;d=motion.result
        self.pk=np.asarray(d['position_knots_s']);self.ak=np.asarray(d['angle_knots_s'])
        self.pc=np.asarray(d['position_coefficients_enu_m']);self.ac=np.asarray(d['angle_coefficients_rad'])
        self.np=len(self.pc);self.na=len(self.ac);self.n=2*self.np+self.na-1
        self.position_basis=BSpline(self.pk,np.eye(self.np),3,extrapolate=False)
        self.angle_basis=BSpline(self.ak,np.eye(self.na),3,extrapolate=False)
        self.anchor=self.angle_basis(d['anchor_s'])
        state=motion.state([d['anchor_s']]);_,g=excess_path(state['position_ecef_m'],motion.tx[3],motion.rx)
        radial=(g@motion.rotation.T)[0];radial[2]=0;radial/=np.linalg.norm(radial)
        self.axes=np.array([radial,[-radial[1],radial[0],0.]])
        # Invertible optimizer coordinates, not likelihood regularisation.
        self.scale=np.r_[np.ones(self.np),np.full(self.np,10.),np.full(self.na-1,.01)]

    def matrices(self,t,derivative=0):
        b=self.position_basis(t,derivative);a=self.angle_basis(t,derivative)
        if derivative==0:a=a-self.anchor
        return csr_matrix(b),csr_matrix(a[:,:-1])

    def coefficients(self,q):
        x=np.asarray(q)*self.scale
        position=self.pc+x[:self.np,None]*self.axes[0]+x[self.np:2*self.np,None]*self.axes[1]
        angle=np.r_[x[2*self.np:],0.];angle-=self.anchor@angle
        return position,self.ac+angle

    def constraints(self):
        """Conservative componentwise physical limits, valid at every epoch."""
        p=np.zeros((self.np,3,self.n));p[:,:, :self.np]=self.axes[0][None,:,None]*np.eye(self.np)[:,None,:]
        p[:,:,self.np:2*self.np]=self.axes[1][None,:,None]*np.eye(self.np)[:,None,:]
        p*=self.scale[None,None,:]
        a=np.zeros((self.na,self.n));a[:,2*self.np:]=(np.eye(self.na)-np.ones((self.na,1))*self.anchor[None,:])[:,:-1]*self.scale[2*self.np:][None,:]
        blocks=[];lower=[];upper=[]
        for order,limit in [(1,15.),(2,.15),(3,.03)]:
            base=BSpline(self.pk,self.pc,3).derivative(order)
            correction=BSpline(self.pk,p,3).derivative(order)
            count=len(base.t)-base.k-1
            blocks.append(correction.c[:count,:2].reshape(-1,self.n))
            value=base.c[:count,:2].ravel();lower.extend(-limit-value);upper.extend(limit-value)
        for order,limit in [(1,np.radians(.6)),(2,np.radians(.06))]:
            base=BSpline(self.ak,self.ac,3).derivative(order);correction=BSpline(self.ak,a,3).derivative(order)
            count=len(base.t)-base.k-1;blocks.append(correction.c[:count]);value=base.c[:count]
            lower.extend(-limit-value);upper.extend(limit-value)
        matrix=np.concatenate(blocks);return matrix,np.asarray(lower),np.asarray(upper)

    def export(self,q,path,report):
        position,angle=self.coefficients(q);result=dict(self.motion.result)
        result.update(position_coefficients_enu_m=position.tolist(),angle_coefficients_rad=angle.tolist(),
            complete=True,stage='Full-capture physical radar likelihood',radar_fit=report)
        Path(path).write_text(json.dumps(result,indent=2)+'\n')


class GlobalPointBand:
    def __init__(self,field,coordinate,model,amplitude_spacing=2.):
        self.field,self.coordinate,self.model=field,coordinate,model;self.t=field.time
        m=model.motion;self.motion=m;self.reference_point=np.zeros(3)
        b,a=model.matrices(self.t);bd,ad=model.matrices(self.t,1)
        zp=csr_matrix((len(self.t),model.np));za=csr_matrix((len(self.t),model.na-1))
        s=model.scale
        self.motion_bases=tuple(hstack(parts,format='csr').multiply(s).tocsr() for parts in
            [(b,zp,za),(bd,zp,za),(zp,b,za),(zp,bd,za),(zp,zp,a),(zp,zp,ad)])
        self.b=self.motion_bases[0];self.bd=self.motion_bases[1]
        self.base=m.state(self.t);self.offset=self.base['position_ecef_m']-field.data['centre_ecef_m']
        self.axes=model.axes@m.rotation
        self.training=np.asarray(coordinate['training'],bool)
        if not self.training.all():raise ValueError('Continuous radar fit uses every native epoch')
        self.groups=np.zeros(len(self.t),int);self.amplitude,self.amplitude_nodes=linear_basis(self.t,m.interval,amplitude_spacing)
        self.amplitude=csr_matrix(self.amplitude);self.refresh_data()

    def refresh_data(self):
        y=profile_static(self.field.y,self.field.root_information,self.training,self.groups)
        self.data={True:y};self.energy={True:float(np.vdot(y,y).real)}

    def point_state(self,q,point):
        values=[basis@q for basis in self.motion_bases[:6]];r,rd,t,td,angle,rate=values
        c,s=np.cos(angle),np.sin(angle);point=np.broadcast_to(np.asarray(point),(len(self.t),3))
        rotated=np.c_[c*point[:,0]-s*point[:,1],s*point[:,0]+c*point[:,1],point[:,2]]
        first=np.c_[-rotated[:,1],rotated[:,0],np.zeros(len(self.t))]
        second=np.c_[-rotated[:,0],-rotated[:,1],np.zeros(len(self.t))]
        f,fd=self.base['body_to_ecef'],self.base['frame_velocity']
        displacement=np.einsum('tij,tj->ti',f,rotated);turn=np.einsum('tij,tj->ti',f,first)
        turn_rate=np.einsum('tij,tj->ti',fd,first)+rate[:,None]*np.einsum('tij,tj->ti',f,second)
        offset=self.offset+r[:,None]*self.axes[0]+t[:,None]*self.axes[1]+displacement
        velocity=self.base['velocity_ecef_mps']+rd[:,None]*self.axes[0]+td[:,None]*self.axes[1]+np.einsum('tij,tj->ti',fd,rotated)+rate[:,None]*turn
        dp=np.zeros((len(self.t),3,6));dv=np.zeros_like(dp)
        dp[:,:,0]=dv[:,:,1]=self.axes[0];dp[:,:,2]=dv[:,:,3]=self.axes[1]
        dp[:,:,4]=turn;dv[:,:,4]=turn_rate;dv[:,:,5]=turn
        return self.field.data['centre_ecef_m']+offset,velocity,offset,dp,dv

    def waveform(self,q,channel,derivatives,point_offset=None,spatial_derivatives=False):
        if spatial_derivatives:raise ValueError('Global fit keeps its declared common reference scatterers fixed')
        f=self.field;m=self.motion;position,velocity,offset,dp,dv=self.point_state(q,self.reference_point if point_offset is None else point_offset)
        kw=dict(transmitter=m.tx[channel],receiver=m.rx,position_offset=offset)
        raw=f.raw_trajectory(position,velocity,**kw,**(dict(position_jacobian=dp,velocity_jacobian=dv) if derivatives else {}))
        _,_,g=f.path_kinematics(position,velocity,**kw)
        speed=np.sum(g*velocity,axis=1);factor,dfactor=carrier_bin(self.t,speed,self.coordinate['native_time_s'],self.coordinate['native_information'],f.k)
        if not derivatives:return f.apply(raw*factor[:,None])
        raw,jac=raw;partials=[]
        for j in range(6):
            speed_derivative=np.sum(g*dv[:,:,j]+curvature(position,dp[:,:,j],m.tx[channel],m.rx)*velocity,axis=1)
            partials.append(f.apply(jac[:,:,j]*factor[:,None]+raw*(dfactor*speed_derivative)[:,None]))
        return (f.apply(raw*factor[:,None]),*partials)


class ContinuousScene:
    def __init__(self,bands,model,band_phases,workers=1,prior=None,fit_band_phases=False,profile_brightness=False):
        self.bands,self.model=bands,model;self.n=model.n;self.p=bands[0].phase_count
        self.beta=np.asarray(band_phases);self.energy=sum(b.point.energy[True] for b in bands)
        self.calls=0;self.workers=workers;self.cached=None;self.prior=prior
        reference=min(3,len(bands)-1)
        self.reference_band=reference
        self.free_bands=[c for c in range(len(bands)) if c!=reference] if fit_band_phases else []
        self.profile_brightness=profile_brightness

    def phase(self,p,c):
        delta=p[self.n+self.p+self.free_bands.index(c)] if c in self.free_bands else 0.
        return p[self.n:self.n+self.p]+self.beta[c]+delta

    def phase_coordinates(self,p):
        """Represent the same prediction with one source phase fixed to zero."""
        phi=p[self.n:self.n+self.p];beta=self.beta.copy()
        for i,c in enumerate(self.free_bands):beta[c]+=p[self.n+self.p+i]
        return np.r_[p[:self.n],beta+phi[0],phi[1:]-phi[0]]

    def from_phase_coordinates(self,p):
        beta=p[self.n:self.n+len(self.bands)];phi=p[self.n+len(self.bands):]
        common=beta[self.reference_band]-self.beta[self.reference_band]
        return np.r_[p[:self.n],common+np.r_[0.,phi],[beta[c]-self.beta[c]-common for c in self.free_bands]]

    def objective(self,p,full=False):
        if self.cached is not None and np.array_equal(p,self.cached[0]):
            value=self.cached[1]
            return value if full else value[:2]
        def evaluate(c):return self.bands[c].evaluate(p[:self.n],self.phase(p,c),c)
        if self.workers==1:values=list(map(evaluate,range(len(self.bands))))
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as pool:values=list(pool.map(evaluate,range(len(self.bands))))
        gradient=np.zeros_like(p)
        for c,v in enumerate(values):
            gradient[:self.n]+=v['path_gradient'];gradient[self.n:self.n+self.p]+=v['phase_gradient']
            if c in self.free_bands:gradient[self.n+self.p+self.free_bands.index(c)]=v['phase_gradient'].sum()
        self.calls+=1;cost=sum(v['cost'] for v in values)/self.energy
        if self.prior is not None:
            penalty,partial=self.prior.evaluate(p[:self.n]);cost+=penalty/self.energy;gradient[:self.n]+=partial
        value=(cost,gradient/self.energy,values);self.cached=(np.asarray(p).copy(),value)
        return value if full else value[:2]

    def mean_information(self,p,include_prior=False,profile_brightness=False):
        """Conditional mean Fisher matrix; optionally profile active brightness.

        Includes exact stationary projection and all explicit phase/motion
        derivatives. Unknown source coordinates remain conditional. The
        optional Schur term eliminates the current positive amplitude face.
        """
        _,_,solutions=self.objective(p,True);size=len(p);gram=np.zeros((size,size))
        for c,(scene,solution) in enumerate(zip(self.bands,solutions)):
            b=scene.point;value=scene.design(p[:self.n],c,True);h=value[0]
            gain=(b.amplitude@solution['amplitude'])*np.exp(1j*scene.phase_at_time(self.phase(p,c)))
            partials=[np.sum(d*gain[:,None,:],axis=-1) for d in value[1:]]
            bases=[hstack([a,csr_matrix((len(b.t),self.p+len(self.free_bands)))],format='csr') for a in b.motion_bases]
            for j in range(scene.points_per_window):
                partials.append(1j*h[:,:,j]*gain[:,None,j])
                phase_basis=csr_matrix((np.ones(len(b.t)),(np.arange(len(b.t)),scene.phase_ids[scene.window_index,j])),shape=(len(b.t),self.p))
                bases.append(hstack([csr_matrix((len(b.t),self.n)),phase_basis,csr_matrix((len(b.t),len(self.free_bands)))],format='csr'))
            if c in self.free_bands:
                partials.append(1j*np.sum(h*gain[:,None,:],axis=-1))
                beta_basis=csr_matrix((np.ones(len(b.t)),(np.arange(len(b.t)),np.full(len(b.t),self.n+self.p+self.free_bands.index(c)))),shape=(len(b.t),size))
                bases.append(beta_basis)
            root=b.field.root_information;root=root/np.linalg.norm(root);mean=np.zeros((len(b.field.offsets),size),complex)
            for j,(a,d) in enumerate(zip(bases,partials)):
                mean+=np.asarray(a.T@(d*root[:,None])).T
                for k in range(j,len(bases)):
                    weight=2*np.real(np.einsum('tr,tr->t',d.conj(),partials[k]))
                    term=(a.T@bases[k].multiply(weight[:,None])).toarray();gram+=term
                    if k!=j:gram+=term.T
            gram-=2*np.real(mean.conj().T@mean)
            if profile_brightness:
                from scipy.linalg import cho_factor,cho_solve
                from sparse_positive_envelope import Normal
                normal=Normal(h,b.field.y,b.field.root_information,b.training,b.amplitude,b.groups)
                phase=self.phase(p,c);rotation=np.exp(1j*phase[scene.phase_groups]);points=scene.points_per_window
                cross=np.zeros((size,len(rotation)))
                for a,d in zip(bases,partials):
                    at_time=np.einsum('trp,tr->tp',h.conj(),d)
                    for point in range(points):
                        term=(a.T@b.amplitude.multiply(at_time[:,point,None])).toarray()
                        cross[:,point::points]+=2*np.real(term*rotation[point::points].conj()[None,:])
                amplitude_mean=normal.mean*rotation[None,:]
                cross-=2*np.real(mean.conj().T@amplitude_mean)
                active=np.flatnonzero(solution['amplitude'].ravel()>0)
                if len(active):
                    local=normal.local_csr[active][:,active].toarray()
                    block=2*np.real(rotation[active].conj()[:,None]*local*rotation[active][None,:]-amplitude_mean[:,active].conj().T@amplitude_mean[:,active])
                    block=(block+block.T)/2;rhs=cross[:,active].T
                    try:projected=cho_solve(cho_factor(block,lower=True),rhs)
                    except np.linalg.LinAlgError:
                        eigen,vector=np.linalg.eigh(block);tolerance=64*np.finfo(float).eps*len(active)*max(abs(eigen).max(),1e-300)
                        if eigen[0]<-tolerance:raise ValueError('Active brightness information is not positive semidefinite')
                        keep=eigen>tolerance
                        projected=(vector[:,keep]/eigen[keep][None,:])@(vector[:,keep].T@rhs)
                    gram-=cross[:,active]@projected
        gram=(gram+gram.T)/2
        if include_prior and self.prior is not None:gram[:self.n,:self.n]+=self.prior.gram
        return gram

    def preconditioner(self,p,objective_scale=1000.):
        """Invertible coordinates; brightness remains profiled in the fit."""
        gram=self.mean_information(p,include_prior=True,profile_brightness=self.profile_brightness)*objective_scale/self.energy
        eigen,vector=np.linalg.eigh(gram)
        if eigen[-1]<=0:raise ValueError('No motion curvature is available for numerical preconditioning')
        if eigen[0]<-1e-8*eigen[-1]:raise ValueError('Conditional motion Fisher matrix is not positive semidefinite')
        floor=eigen[-1]*1e-8;coordinate=(vector/np.sqrt(np.maximum(eigen,floor))[None,:])@vector.T
        return coordinate,dict(conditional_fisher_eigenvalue_range=eigen[[0,-1]].tolist(),preconditioning_floor=floor,
            brightness_nuisance_profiled_on_active_face=self.profile_brightness,
            scope='Strictly invertible numerical coordinates; no curvature term, loading or penalty added to the likelihood')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--field',type=Path,required=True);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=100);p.add_argument('--aspect-seconds',type=float,default=30.);p.add_argument('--amplitude-seconds',type=float,default=2.)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--phase-iterations',type=int,default=30)
    p.add_argument('--anchor-reviewed',action='store_true',help='Condition on the reviewed aperture midpoint position, velocity and angular rate')
    p.add_argument('--precondition',action='store_true',help='Use invertible conditional Fisher coordinates for the same likelihood and bounds')
    p.add_argument('--acceleration-change-mps2',type=float,help='WNOJ prior: per-axis acceleration-change standard deviation over the motion timescale')
    p.add_argument('--angular-rate-change-degps',type=float,help='Prior on angular-rate increments over the motion timescale')
    p.add_argument('--motion-timescale-s',type=float,default=30.)
    p.add_argument('--resume',type=Path);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    motion=ContinuousShipMotion(args.motion);model=MotionCoordinates(motion);fields,coordinates,_=load(args.field)
    points=np.asarray(motion.result['reference_points_east_north_up_m']);edges=np.r_[np.arange(motion.interval[0],motion.interval[1],args.aspect_seconds),motion.interval[1]]
    bands=[]
    for c,(field,coordinate) in enumerate(zip(fields,coordinates)):
        b=GlobalPointBand(field,coordinate,model,args.amplitude_seconds)
        bands.append(SceneBand(b,np.broadcast_to(points,(len(edges)-1,*points.shape)),edges,args.amplitude_seconds))
        print(json.dumps(dict(stage='loaded',channel=c,bins=len(b.t))),flush=True)
    prior=KinematicPrior(model,acceleration_change_mps2=args.acceleration_change_mps2,angular_rate_change_degps=args.angular_rate_change_degps,timescale_s=args.motion_timescale_s)
    scene=ContinuousScene(bands,model,np.angle(motion.gains),args.workers,prior);at=np.zeros(model.n+scene.p)
    reference=json.loads((Path(motion.result['reference_fit'])/'analysis.json').read_text())['phase_histories']['brightness_model']
    at[model.n:]=np.tile(reference['point_phase_rad'],len(edges)-1)
    if args.resume:at=np.asarray(json.loads(args.resume.read_text())['parameters'])
    matrix,lo,hi=model.constraints();matrix=np.c_[matrix,np.zeros((len(matrix),scene.p))]
    equalities=[]
    if args.anchor_reviewed:
        b,a=model.matrices(np.array([motion.result['anchor_s']]));bd,ad=model.matrices(np.array([motion.result['anchor_s']]),1)
        zp=np.zeros(model.np);za=np.zeros(model.na-1)
        rows=np.array([np.r_[b.toarray()[0],zp,za],np.r_[bd.toarray()[0],zp,za],
            np.r_[zp,b.toarray()[0],za],np.r_[zp,bd.toarray()[0],za],np.r_[zp,zp,ad.toarray()[0]]])*model.scale
        rows=np.c_[rows,np.zeros((len(rows),scene.p))]
        equalities=[LinearConstraint(rows,0.,0.)]
    bounds=[(-150.,150.)]*model.np+[(-50.,50.)]*model.np+[(-np.radians(20)/.01,np.radians(20)/.01)]*(model.na-1)+[(None,None)]*scene.p
    started=time.monotonic();history=[];initial=scene.objective(at)[0];best=np.inf
    if args.iterations and not args.resume:
        # A nuisance-block solve of the SAME likelihood, before geometry.
        # Only a common phase can change: all relative band phases stay fixed.
        phase=at[model.n:];temporary=np.r_[at[:model.n],scene.beta+phase[0],phase[1:]-phase[0]]
        def phase_progress(record):
            r=dict(record);phase=np.asarray(r.pop('phase_parameters'));common=phase[0]-scene.beta[0]
            values=np.r_[at[:model.n],common+np.r_[0.,phase[5:]]]
            (out/'phase-progress.json').write_text(json.dumps(dict(parameters=values.tolist(),**r),indent=2)+'\n')
            if record['calls']==1 or record['calls']%10==0:print(json.dumps(dict(stage='reflectivity-phase',**r)),flush=True)
        fitted,phase_report=warm_phases(scene,temporary,args.phase_iterations,phase_progress,fixed_band_offsets=scene.beta,workers=args.workers)
        common=fitted[model.n]-scene.beta[0]
        at=np.r_[fitted[:model.n],common+np.r_[0.,fitted[model.n+5:]]]
        (out/'phase-initialization.json').write_text(json.dumps(phase_report,indent=2)+'\n')
        print(json.dumps(dict(stage='reflectivity-phase-initialization',**phase_report)),flush=True)
    def callback(x):
        nonlocal best
        value,gradient,values=scene.objective(x,True);record=dict(iteration=len(history)+1,objective=value,radar_rss_fraction=sum(v['cost'] for v in values)/scene.energy,
            normalized_motion_prior=prior.evaluate(x[:model.n])[0]/scene.energy,maximum_scaled_gradient=float(max(abs(gradient))),seconds=time.monotonic()-started,evaluations=scene.calls)
        history.append(record);(out/'progress.json').write_text(json.dumps(dict(parameters=x.tolist(),history=history,aspect_edges_s=edges.tolist()),indent=2)+'\n')
        violation=max(np.max(lo-matrix@x),np.max(matrix@x-hi),0.)
        if value<best and violation<1e-7:
            best=value;(out/'best-progress.json').write_text(json.dumps(dict(parameters=x.tolist(),history=history,aspect_edges_s=edges.tolist()),indent=2)+'\n')
        print(json.dumps(record),flush=True)
    origin=at.copy();transform=None
    if args.precondition:
        transform,conditioning=scene.preconditioner(at);(out/'preconditioning.json').write_text(json.dumps(conditioning,indent=2)+'\n')
    def expand(q):return q if transform is None else origin+transform@q
    def optimizer(q):
        value,gradient=scene.objective(expand(q))
        if transform is not None:gradient=gradient@transform
        return 1000*value,1000*gradient
    constraints=[LinearConstraint(matrix,lo,hi),*equalities];optimizer_bounds=bounds;start=at
    if transform is not None:
        constraints=[LinearConstraint(c.A@transform,np.asarray(c.lb)-c.A@origin,np.asarray(c.ub)-c.A@origin) for c in constraints]
        lower=np.array([-np.inf if a is None else a for a,b in bounds]);upper=np.array([np.inf if b is None else b for a,b in bounds])
        keep=np.isfinite(lower)|np.isfinite(upper)
        constraints.append(LinearConstraint(transform[keep],lower[keep]-origin[keep],upper[keep]-origin[keep]));optimizer_bounds=None;start=np.zeros_like(at)
    result=minimize(optimizer,start,jac=True,method='SLSQP',bounds=optimizer_bounds,constraints=constraints,callback=lambda q:callback(expand(q)),
        options=dict(maxiter=args.iterations,ftol=2e-8)) if args.iterations else None
    at=at if result is None else expand(result.x);value,gradient,values=scene.objective(at,True)
    constraint=matrix@at;violation=float(max(np.max(lo-constraint),np.max(constraint-hi),0.))
    report=dict(complete=True,initial_objective=initial,objective=value,iterations=0 if result is None else result.nit,numerical_convergence=False if result is None else bool(result.success),
        message='assessed initializer' if result is None else str(result.message),maximum_scaled_gradient=float(max(abs(gradient))),maximum_kinematic_constraint_violation=violation,
        aspect_edges_s=edges.tolist(),points_east_north_up_m=points.tolist(),parameters=at.tolist(),seconds=time.monotonic()-started,
        per_channel=[dict(cost=v['cost'],energy=v['energy'],explained_fraction=v['explained_fraction'],maximum_amplitude_kkt=v['maximum_amplitude_kkt']) for v in values],
        scope=__doc__,in_sample=True,spatial_geometry_conditioned_on_reference=True,anchor_reviewed=args.anchor_reviewed,
        initial_motion=str(args.motion),field=str(args.field),resume_checkpoint=None if args.resume is None else str(args.resume),
        amplitude_spacing_s=args.amplitude_seconds,aspect_spacing_s=args.aspect_seconds,optimizer_preconditioned=args.precondition)
    report.update(radar_rss_fraction=sum(v['cost'] for v in values)/scene.energy,motion_prior=prior.diagnostics(at[:model.n]),
        objective_scope='Radar Gaussian residual plus explicit physical motion prior; radar residual reported separately')
    if violation>1e-7:raise ValueError('Physical kinematic bounds were not satisfied')
    lower=np.array([-np.inf if a is None else a for a,b in bounds]);upper=np.array([np.inf if b is None else b for a,b in bounds])
    equal=np.concatenate([c.A for c in equalities]) if equalities else None
    certificate=linprog(gradient,A_ub=np.r_[matrix,-matrix],b_ub=np.r_[hi-matrix@at,matrix@at-lo],
        A_eq=equal,b_eq=None if equal is None else -equal@at,
        bounds=list(zip(np.maximum(-1,lower-at),np.minimum(1,upper-at))),method='highs')
    gap=None if not certificate.success else float(max(0.,-certificate.fun))
    report.update(solver_success=False if result is None else bool(result.success),maximum_linearized_feasible_improvement=gap,
        numerical_convergence=bool(result is not None and result.success and gap is not None and gap<1e-7),
        stationarity_scope='Maximum linearized improvement in the normalized posterior objective within one optimizer-coordinate unit and all physical bounds; no claim of global identifiability')
    for c,v in enumerate(values):np.save(out/f'brightness-{c}.npy',v['amplitude'])
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');model.export(at[:model.n],out/'motion.json',report)

if __name__=='__main__':main()
