"""A shared physical radio path and persistent positive-envelope scatterers."""
import numpy as np
from rigid_patch import profile_static
from sparse_positive_envelope import Normal
from scipy.optimize import minimize
from audit_ship_scattering_envelopes import linear_basis
from scipy.sparse import csr_matrix


class SceneBand:
    def __init__(self,point_band,points,window_edges=None,amplitude_spacing=2.,active_points=None,*,adjoint_gradient=False):
        self.point=point_band;self.points=np.asarray(points);self.warm=None;self.window_edges=window_edges
        self.adjoint_gradient=adjoint_gradient
        if self.points.ndim==2:
            self.points_per_window=len(self.points);self.phase_count=len(self.points);self.window_index=np.zeros(len(point_band.t),int)
            self.phase_ids=np.arange(self.phase_count)[None,:]
            self.active_points=np.ones((1,len(self.points)),bool)
            self.phase_groups=np.tile(np.arange(self.phase_count),point_band.amplitude.shape[1])
        elif self.points.ndim==3 and window_edges is not None:
            edges=np.asarray(window_edges);windows,p,_=self.points.shape
            if len(edges)!=windows+1 or edges[0]>point_band.t[0] or edges[-1]<point_band.t[-1]:raise ValueError('Each epoch needs exactly one declared scattering support')
            self.window_index=np.clip(np.searchsorted(edges,point_band.t,side='right')-1,0,windows-1)
            active=np.ones((windows,p),bool) if active_points is None else np.asarray(active_points,bool)
            if active.shape!=(windows,p) or np.any(active.sum(axis=1)==0):raise ValueError('Each declared aspect needs its explicit nonempty source support')
            self.active_points=active;self.phase_count=int(active.sum());self.phase_ids=np.zeros((windows,p),int)
            self.phase_ids[active]=np.arange(self.phase_count)
            self.points_per_window=p;rows=[];columns=[];values=[];groups=[];nodes=[];start=0
            for w in range(windows):
                chosen=np.flatnonzero(self.window_index==w);basis,at=linear_basis(point_band.t[chosen],edges[w:w+2],amplitude_spacing)
                r,c=np.nonzero(basis);rows.extend(chosen[r]);columns.extend(start+c);values.extend(basis[r,c]);start+=len(at);nodes.extend(at)
                groups.extend(np.tile(self.phase_ids[w],len(at)))
            point_band.amplitude=csr_matrix((values,(rows,columns)),shape=(len(point_band.t),start));point_band.amplitude_nodes=np.asarray(nodes)
            self.phase_groups=np.asarray(groups)
        else:raise ValueError('Scene points must be fixed points or declared nonoverlapping aspect supports')

    def phase_at_time(self,phase):
        return np.asarray(phase)[self.phase_ids[self.window_index]]

    def design(self,coefficient,channel,derivatives,spatial_derivatives=False):
        active=np.broadcast_to(self.points,(len(self.point.t),*self.points.shape)) if self.points.ndim==2 else self.points[self.window_index]
        motion_bases=getattr(self.point,'motion_bases',(self.point.b,self.point.bd))
        values=[];width=1+len(motion_bases)+(2 if spatial_derivatives else 0)
        for p in range(self.points_per_window):
            enabled=self.active_points[self.window_index,p]
            if enabled.all():value=self.point.waveform(coefficient,channel,derivatives,active[:,p]-self.point.reference_point,spatial_derivatives)
            else:
                result=[np.zeros_like(self.point.field.y) for _ in range(width)] if derivatives or spatial_derivatives else np.zeros_like(self.point.field.y)
                edges=np.diff(np.r_[False,enabled,False].astype(int));first=np.flatnonzero(edges==1);last=np.flatnonzero(edges==-1)
                for start,stop in zip(first,last):
                    view=self.point.waveform_window(start,stop)
                    part=view.waveform(coefficient,channel,derivatives,active[start:stop,p]-self.point.reference_point,spatial_derivatives)
                    if derivatives or spatial_derivatives:
                        for j in range(width):result[j][start:stop]=part[j]
                    else:result[start:stop]=part
                value=tuple(result) if derivatives or spatial_derivatives else result
            values.append(value)
        active=self.active_points[self.window_index][:,None,:]
        if derivatives or spatial_derivatives:return tuple(np.stack([x[j] for x in values],axis=-1)*active for j in range(width))
        return np.stack(values,axis=-1)*active

    def evaluate(self,coefficient,phase,channel,*,training=True,derivatives=True,frozen_amplitude=None,spatial_derivatives=False):
        if self.adjoint_gradient and spatial_derivatives:raise ValueError('This adjoint path keeps the declared common source support fixed')
        materialise=derivatives and not self.adjoint_gradient
        value=self.design(coefficient,channel,materialise,spatial_derivatives);h=value[0] if materialise else value
        b=self.point;mask=b.training if training else ~b.training
        normal=Normal(h,b.field.y,b.field.root_information,mask,b.amplitude,b.groups)
        if frozen_amplitude is None:
            result=normal.solve(phase,self.warm,self.phase_groups)
            if not result['numerical_convergence']:raise ValueError(f'Envelope NNLS did not converge: KKT {result["maximum_scaled_kkt"]}')
            amplitude=result['amplitude'].reshape(normal.knots,normal.points);self.warm=amplitude.copy()
        else:amplitude=np.asarray(frozen_amplitude);result=None
        gain=(b.amplitude@amplitude)*np.exp(1j*self.phase_at_time(phase))
        model=h*gain[:,None,:];prediction=profile_static(model.sum(axis=-1),b.field.root_information,mask,b.groups)*mask[:,None]
        residual=prediction-b.data[training];cost=float(np.vdot(residual,residual).real)
        output=dict(cost=cost,energy=b.energy[training],explained_fraction=1-cost/b.energy[training],amplitude=amplitude,
            maximum_amplitude_kkt=None if result is None else result['maximum_scaled_kkt'])
        if derivatives:
            bases=getattr(b,'motion_bases',(b.b,b.bd));gradient=np.zeros(b.b.shape[1])
            if self.adjoint_gradient:
                active=np.broadcast_to(self.points,(len(b.t),*self.points.shape)) if self.points.ndim==2 else self.points[self.window_index]
                if not self.active_points.all():raise ValueError('The joint adjoint requires its declared nonempty full source support')
                for j in range(self.points_per_window):gradient+=b.pullback(coefficient,channel,active[:,j]-b.reference_point,residual,gain[:,j])
            else:
                for basis,partial in zip(bases,value[1:1+len(bases)]):
                    at_time=2*np.real(np.sum(residual.conj()*np.sum(partial*gain[:,None,:],axis=-1),axis=1))
                    gradient+=np.asarray(basis.T@at_time).ravel()
            output['path_gradient']=gradient
            time_gradient=2*np.real(np.einsum('tr,trp->tp',residual.conj(),1j*model))
            groups=self.phase_ids[self.window_index]
            output['phase_gradient']=np.bincount(groups.ravel(),time_gradient.ravel(),minlength=self.phase_count)
            if spatial_derivatives:
                spatial=[]
                for partial in value[1+len(bases):]:
                    time_gradient=2*np.real(np.einsum('tr,trp->tp',residual.conj(),partial*gain[:,None,:]))
                    spatial.append(np.bincount(groups.ravel(),time_gradient.ravel(),minlength=self.phase_count))
                output['point_gradient']=np.column_stack(spatial).ravel()
        return output


class ScenePath:
    def __init__(self,bands):
        self.bands=bands;self.n=bands[0].point.b.shape[1];self.p=bands[0].phase_count;self.energy=sum(b.point.energy[True] for b in bands)

    def objective(self,parameters,full=False):
        beta=parameters[self.n:self.n+5];phi=np.r_[0.,parameters[self.n+5:]]
        if len(phi)!=self.p:raise ValueError('One scatterer fixes the phase-origin gauge')
        values=[b.evaluate(parameters[:self.n],phi+beta[c],c) for c,b in enumerate(self.bands)]
        gradient=np.zeros_like(parameters)
        for c,v in enumerate(values):
            gradient[:self.n]+=v['path_gradient'];gradient[self.n+c]=v['phase_gradient'].sum();gradient[self.n+5:]+=v['phase_gradient'][1:]
        cost=sum(v['cost'] for v in values)/self.energy;gradient/=self.energy
        return (cost,gradient,values) if full else (cost,gradient)


class RefinedScenePath(ScenePath):
    """Same physical likelihood with bounded horizontal source coordinates.

    Source locations are constant within their declared aspect supports and
    common to every band. A candidate image pixel is an initializer, not an
    exact known reflector position. No per-band spatial offset is introduced.
    """
    def __init__(self,bands):
        super().__init__(bands)
        if not np.all(bands[0].active_points):raise ValueError('Off-grid refinement requires unpadded source supports')
        self.reference_points=bands[0].points.copy();self.phase_end=self.n+5+self.p-1

    def set_points(self,parameters):
        points=self.reference_points.copy()
        points[...,:2]+=np.asarray(parameters[self.phase_end:]).reshape(points.shape[:-1]+(2,))
        for band in self.bands:band.points=points
        return points

    def objective(self,parameters,full=False):
        self.set_points(parameters)
        beta=parameters[self.n:self.n+5];phi=np.r_[0.,parameters[self.n+5:self.phase_end]]
        values=[b.evaluate(parameters[:self.n],phi+beta[c],c,spatial_derivatives=True) for c,b in enumerate(self.bands)]
        gradient=np.zeros_like(parameters)
        for c,v in enumerate(values):
            gradient[:self.n]+=v['path_gradient'];gradient[self.n+c]=v['phase_gradient'].sum()
            gradient[self.n+5:self.phase_end]+=v['phase_gradient'][1:]
            gradient[self.phase_end:]+=v['point_gradient']
        cost=sum(v['cost'] for v in values)/self.energy;gradient/=self.energy
        return (cost,gradient,values) if full else (cost,gradient)


def solve_normal_phases(normals,initial,phase_groups,iterations,progress=None,fixed_band_offsets=None,workers=1):
    """One constant band phase plus tied source phases; real NNLS envelopes."""
    energy=sum(n.energy for n in normals);warm=[None]*len(normals);calls=0;best=np.inf
    def objective(x,full=False):
        nonlocal calls,best
        phi=np.r_[0.,x[5:]];gradient=np.zeros_like(x);cost=0.;solutions=[]
        def evaluate(c):return normals[c].solve(phi+x[c],warm[c],phase_groups[c])
        if workers==1:values=list(map(evaluate,range(len(normals))))
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:values=list(pool.map(evaluate,range(len(normals))))
        for c,r in enumerate(values):
            if not r['numerical_convergence']:raise ValueError(f'Phase initialization amplitude solve failed KKT: {r["maximum_scaled_kkt"]}; direct solve {r["direct_cholesky_nnls_polish"]}')
            warm[c]=r['amplitude'];solutions.append(r);cost+=r['cost'];gradient[c]=r['phase_gradient'].sum();gradient[5:]+=r['phase_gradient'][1:]
        calls+=1
        if progress and cost<best:
            best=cost;progress(dict(calls=calls,objective=cost/energy,maximum_gradient=float(max(abs(gradient))/energy),phase_parameters=x.copy()))
        return (cost/energy,gradient/energy,solutions) if full else (cost/energy,gradient/energy)
    def expand(q):
        return q if fixed_band_offsets is None else np.r_[np.asarray(fixed_band_offsets)+q[0],q[1:]]
    def gradient(g):return g if fixed_band_offsets is None else np.r_[g[:5].sum(),g[5:]]
    at=initial if fixed_band_offsets is None else np.r_[np.mean(initial[:5]-fixed_band_offsets),initial[5:]]
    def optimizer(q):
        value,g=objective(expand(q));return value,gradient(g)
    before=optimizer(at)[0]
    result=minimize(optimizer,at,jac=True,method='L-BFGS-B',options=dict(maxiter=iterations,gtol=1e-8,ftol=1e-12,maxcor=30))
    fitted=expand(result.x);value,g,solutions=objective(fitted,True);projected=gradient(g)
    return fitted,solutions,dict(initial_objective=before,objective=value,maximum_gradient=float(max(abs(projected))),iterations=result.nit,message=str(result.message),numerical_convergence=bool(result.success and max(abs(projected))<1e-6),relative_band_phases_fixed=fixed_band_offsets is not None)


def warm_phases(likelihood,parameters,iterations,progress=None,fixed_band_offsets=None,workers=1):
    """Solve the same training likelihood's phase block before moving geometry."""
    normals=[];n=likelihood.n
    for c,band in enumerate(likelihood.bands):
        b=band.point;h=band.design(parameters[:n],c,False)
        normals.append(Normal(h,b.field.y,b.field.root_information,b.training,b.amplitude,b.groups))
    phase,solutions,report=solve_normal_phases(normals,parameters[n:].copy(),[b.phase_groups for b in likelihood.bands],iterations,progress,fixed_band_offsets,workers)
    answer=parameters.copy();answer[n:]=phase
    for band,solution in zip(likelihood.bands,solutions):band.warm=solution['amplitude'].reshape(-1,band.points_per_window)
    return answer,report
