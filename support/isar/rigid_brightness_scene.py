"""Shared rigid motion and point phases with positive native-row brightness.

Stationary coefficients and real brightness knots are profiled on fitting samples.
Band calibration is frozen. One phase per physical point is shared across
bands; no band-specific point phase or time-dependent phase screen is fitted.
"""
import numpy as np
from scipy.optimize import minimize
from amplitude_arc import AmplitudeArc,qr_nnls


class RigidBrightnessBand(AmplitudeArc):
    def __init__(self,field,training,knots,kinematics,points,band_phase,stationary_design,*,position_gauge=None):
        super().__init__(field,training,knots,stationary_design=stationary_design)
        self.kinematics=kinematics;self.points=np.asarray(points);self.band_phase=band_phase
        self.motion_size=getattr(kinematics,'parameter_count',5)
        self.size=self.motion_size+len(points)
        self.position_gauge=None if position_gauge is None else np.asarray(position_gauge)
        if self.position_gauge is not None:
            if self.position_gauge.shape!=(len(points),2):raise ValueError('A common spatial phase gauge is required for every point')
            self.size+=2*(len(points)-1)
        self._last_parameters=None
        self._last_scene=None

    def responses(self,parameters,derivatives=True):
        f=self.field;responses=[];partial=[]
        m=self.motion_size
        frame,frame_rate=self.kinematics.orientation(parameters[:m])
        phase_end=m+len(self.points)
        for j,original in enumerate(self.points):
            q=original.copy();offset=np.zeros(2)
            if self.position_gauge is not None and j:
                start=phase_end+2*(j-1);offset=np.asarray(parameters)[start:start+2];q[:2]+=offset
            state=self.kinematics.point_state(parameters[:m],q,derivatives=derivatives,origin=f.data['centre_ecef_m'])
            options=dict(position_offset=state[0])
            if derivatives:
                dp,dv=state[2:]
                if self.position_gauge is not None and j:
                    dp=np.concatenate([dp,frame[:,:,:2]],axis=2);dv=np.concatenate([dv,frame_rate[:,:,:2]],axis=2)
                options.update(position_jacobian=dp,velocity_jacobian=dv)
            result=f.raw_trajectory(f.data['centre_ecef_m']+state[0],state[1],**options)
            if derivatives:raw,change=result
            else:raw=result
            phi=self.band_phase+parameters[m+j]
            if self.position_gauge is not None:phi+=np.dot(self.position_gauge[j],offset)
            phase=np.exp(1j*phi);responses.append(raw*phase)
            if derivatives:
                derivative=np.zeros(raw.shape+(self.size,),complex)
                derivative[:,:,:m]=change[:,:,:m]*phase;derivative[:,:,m+j]=1j*raw*phase
                if self.position_gauge is not None and j:
                    derivative[:,:,start:start+2]=(change[:,:,m:]+1j*raw[:,:,None]*self.position_gauge[j])*phase
                partial.append(derivative)
        return responses,partial

    def evaluate_scene(self,parameters,*,full=False):
        if self._last_parameters is not None and np.array_equal(parameters,self._last_parameters):
            value=self._last_scene
            return value.copy() if full else {k:value[k] for k in ('rss','gradient')}
        raw,change=self.responses(parameters)
        design=np.concatenate([self.reduced(h[:,:,None]*self.amplitude_basis[:,None,:]) for h in raw],axis=1)
        projected=self.nuisance.remove(design[self.selected]);matrix=np.r_[projected.real,projected.imag]
        a=self.solve_brightness(matrix,np.r_[self.clean.real,self.clean.imag])
        residual=self.clean-projected@a
        mean=np.zeros_like(change[0]);coefficients=a.reshape(len(self.points),len(self.knots))
        for derivative,coefficient in zip(change,coefficients):
            mean+=derivative*(self.amplitude_basis@coefficient)[:,None,None]
        jacobian=self.nuisance.remove(self.reduced(mean)[self.selected])
        result=dict(rss=float(np.vdot(residual,residual).real),gradient=-2*np.real(jacobian.conj().T@residual),
            design=design,amplitude=a,prediction=design@a,jacobian=jacobian)
        self._last_parameters=np.asarray(parameters).copy();self._last_scene=result
        return result.copy() if full else {k:result[k] for k in ('rss','gradient')}

    def solve_brightness(self,matrix,right):
        return qr_nnls(matrix,right)

    def prediction(self,parameters):
        if np.all(self.selected):raise ValueError('Prediction requires samples unused by this fit')
        fitted=self.evaluate_scene(parameters,full=True);predicted=fitted['prediction'];y=self.field.y.ravel()
        beta=self.nuisance.solve(y[self.selected]-predicted[self.selected])
        baseline=self.nuisance.solve(y[self.selected]);test=~self.selected
        error=np.sum(abs(y[test]-self.design[test]@beta-predicted[test])**2)
        base=np.sum(abs(y[test]-self.design[test]@baseline)**2)
        return dict(test_rss=float(error),stationary_only_test_rss=float(base),explained_fraction=float(1-error/base),
            brightness_coefficients=fitted['amplitude'].reshape(len(self.points),len(self.knots)))

    def residual_diagnostics(self,parameters):
        """In-sample residual energy, explicitly not an independent prediction."""
        fitted=self.evaluate_scene(parameters,full=True)
        return dict(rss=fitted['rss'],nuisance_only_rss=float(self.energy),
            explained_fraction=float(1-fitted['rss']/self.energy),
            fitted_complex_coordinates=int(np.count_nonzero(self.selected)),
            unused_complex_coordinates=int(np.count_nonzero(~self.selected)),
            scope='Residual on the samples used for estimation; not evidence of out-of-sample recovery or optimal SNR.')

    def frozen_components(self,parameters):
        """Return fitted component signals and nuisance-clean data."""
        fitted=self.evaluate_scene(parameters,full=True);k=len(self.knots)
        components=np.column_stack([fitted['design'][:,j*k:(j+1)*k]@fitted['amplitude'][j*k:(j+1)*k]
            for j in range(len(self.points))])
        y=self.field.y.ravel();beta=self.nuisance.solve(y[self.selected]-components[self.selected].sum(axis=1))
        return components,y-self.design@beta

    def component_transfer(self,parameters):
        """Measure independent complex multipliers without applying them.

        Native brightness envelopes, physical trajectories and point phases
        stay frozen. Joint deblending avoids attributing a neighbour's phase
        to a varying mixture in an individual range/Doppler cell.
        """
        components,clean=self.frozen_components(parameters);result=[]
        fitted_label='fitting' if np.all(self.selected) else 'training'
        for mask,label in [(self.selected,fitted_label),(~self.selected,'excluded')]:
            if not np.any(mask):continue
            h=components[mask];norm=np.linalg.norm(h,axis=0);supported=norm>max(norm)*1e-10
            unit=h[:,supported]/norm[supported]
            coefficient=np.zeros(len(self.points),complex)
            if np.any(supported):coefficient[supported]=np.linalg.lstsq(unit,clean[mask],rcond=1e-10)[0]/norm[supported]
            condition=float(np.linalg.cond(unit)) if np.any(supported) else np.inf
            result.append(dict(fold=label,multiplier_real_imag=[[z.real,z.imag] if use else None for z,use in zip(coefficient,supported)],
                phase_error_deg=[float(np.degrees(np.angle(z))) if use else None for z,use in zip(coefficient,supported)],expected_component_energy=norm**2,
                supported=supported,condition=condition if np.isfinite(condition) else None))
        return result

    def component_histories(self,parameters,first_s,*,block_s=.25):
        """Diagnose local complex multipliers; never use them for formation.

        Geometry, brightness and stationary estimates stay frozen. Every
        temporal bin jointly deblends all supported components. A phase is
        undefined when that fitted component has no predicted energy.
        """
        if block_s<=0:raise ValueError('Diagnostic blocks must have positive duration')
        components,clean=self.frozen_components(parameters);t=self.field.time
        bins=np.floor((t-first_s)/block_s).astype(int);result=[]
        for index in np.unique(bins):
            keep=bins==index;mask=np.repeat(keep,len(self.field.offsets));h=components[mask]
            norm=np.linalg.norm(h,axis=0);supported=norm>max(norm.max()*1e-10,np.finfo(float).tiny)
            unit=h[:,supported]/norm[supported];coefficient=np.zeros(len(self.points),complex)
            if np.any(supported):coefficient[supported]=np.linalg.lstsq(unit,clean[mask],rcond=1e-10)[0]/norm[supported]
            condition=float(np.linalg.cond(unit)) if np.any(supported) else None
            result.append(dict(time_s=float(np.mean(t[keep])),training=bool(np.all(self.training[keep])),
                multiplier_real_imag=[[z.real,z.imag] if use else None for z,use in zip(coefficient,supported)],
                phase_error_deg=[float(np.degrees(np.angle(z))) if use else None for z,use in zip(coefficient,supported)],
                expected_component_energy=norm**2,condition=condition if condition is None or np.isfinite(condition) else None))
        return result


class RigidBrightnessFit:
    def __init__(self,bands):
        self.bands=bands;self.energy=sum(b.energy for b in bands);self.size=bands[0].size

    def objective(self,parameters):
        values=[b.evaluate_scene(parameters) for b in self.bands]
        return sum(v['rss'] for v in values)/self.energy,sum(v['gradient'] for v in values)/self.energy

    def settings(self,position_bound_m=6.):
        n=len(self.bands[0].points);extra=self.size-5-n
        scale=np.r_[.05,.01,.005,.1,.01,np.full(n,.3),np.full(extra,.5)]
        limit=np.r_[.5,.12,.05,.5,.05,np.full(n,np.inf),np.full(extra,position_bound_m)]
        return scale,limit

    def assess(self,parameters,*,free=None,position_bound_m=6.):
        """Check the exact objective/gradient without claiming optimisation."""
        p=np.asarray(parameters);value,gradient=self.objective(p)
        scale,limit=self.settings(position_bound_m);free=np.arange(self.size) if free is None else np.asarray(free)
        gradient=gradient[free]*scale[free]
        active=np.isclose(abs(p[free]),limit[free],atol=1e-6)
        gradient[((p[free]<=-limit[free]+1e-6)&(gradient>0))|((p[free]>=limit[free]-1e-6)&(gradient<0))]=0
        return dict(parameters=p,objective=float(value),success=False,iterations=0,
            maximum_scaled_projected_gradient=float(max(abs(gradient))),active_bounds=free[active],free_indices=free,
            message='Native-symbol objective and gradient verified; no native optimisation requested')

    def gradient_check(self,parameters,*,free=None):
        """Check a real-data checkpoint, including the inner NNLS optimality."""
        p=np.asarray(parameters);scale,_=self.settings();_,gradient=self.objective(p)
        free=np.arange(self.size) if free is None else np.asarray(free)
        largest=int(free[np.argmax(abs(gradient[free]*scale[free]))]);directions=[np.eye(self.size)[largest]]
        for phase in (0.,1.):
            d=np.zeros(self.size);d[free]=np.sin(np.arange(len(free))+phase+.3);directions.append(d/np.linalg.norm(d))
        checks=[]
        for d in directions:
            direction=d*scale;expected=float(gradient@direction);steps=[]
            for h in (1e-3,1e-4,1e-5):
                upper=self.objective(p+h*direction)[0];lower=self.objective(p-h*direction)[0]
                steps.append(dict(scaled_step=h,finite_difference=float((upper-lower)/(2*h))))
            checks.append(dict(scaled_direction=d,analytic=expected,central_differences=steps))
        kkt=[]
        for b in self.bands:
            value=b.evaluate_scene(p,full=True);a=value['amplitude'];design=b.nuisance.remove(value['design'][b.selected])
            residual=b.clean-design@a;derivative=-np.real(design.conj().T@residual);active=a>0
            violation=max(np.max(abs(derivative[active]),initial=0.),np.max(-derivative[~active],initial=0.))
            normal=np.linalg.norm(design,axis=0).max()*np.linalg.norm(residual)
            kkt.append(float(violation/max(normal,np.finfo(float).tiny)))
        return dict(directions=checks,nnls_relative_kkt_violation_by_band=kkt,
            scope='Central differences at a frozen real-data checkpoint; NNLS is re-profiled at each perturbation. No fit or phase correction is adopted.')

    def information(self,parameters):
        """Local mean-response information after explicit nuisance projection.

        Every real brightness coefficient is an unconstrained local nuisance,
        including fitted zeros. This avoids claiming extra information from
        assumed zero brightness. The result remains conditional on the point
        model, known thermal covariance, fixed anchor and transverse prior.
        """
        known=[];unknown=[];scale,_=self.settings();m=self.bands[0].motion_size
        for b in self.bands:
            v=b.evaluate_scene(parameters,full=True)
            a=b.nuisance.remove(v['design'][b.selected]);j=v['jacobian']
            a=np.r_[a.real,a.imag];j=np.r_[j.real,j.imag]
            mean=b.nuisance.remove(v['prediction'][b.selected]);phase=np.r_[-mean.imag,mean.real]
            # Static complex coefficients were already projected above.
            all_columns=np.c_[j,phase]
            all_columns-=a@np.linalg.lstsq(a,all_columns,rcond=1e-12)[0]
            j,phase=all_columns[:,:self.size],all_columns[:,-1]
            known.append(j)
            norm=np.linalg.norm(phase)
            unknown.append(j-np.outer(phase/norm,(phase/norm)@j) if norm>0 else j)
        result={}
        for label,blocks in [('fixed_band_calibration',known),('unknown_constant_band_phases',unknown)]:
            j=np.sqrt(2.)*np.concatenate(blocks)*scale
            nuisance=j[:,m:];motion=j[:,:m].copy()
            motion-=nuisance@np.linalg.lstsq(nuisance,motion,rcond=1e-12)[0]
            _,singular,vh=np.linalg.svd(motion,full_matrices=False)
            positive=singular>singular[0]*1e-12 if singular[0]>0 else np.zeros(m,bool)
            null_weight=np.sum(abs(vh[~positive])**2,axis=0)
            covariance=(vh[positive].T/singular[positive]**2)@vh[positive] if np.any(positive) else np.zeros((m,m))
            sigma=scale[:m]*np.sqrt(np.maximum(np.diag(covariance),0.))
            result[label]=dict(joint_information_scaled=j.T@j,marginal_motion_information_scaled=motion.T@motion,
                marginal_motion_singular_values=singular,
                numerical_motion_ranks={str(r):int(np.count_nonzero(singular>singular[0]*r)) if singular[0]>0 else 0 for r in [1e-8,1e-10,1e-12]},
                conditional_motion_sigma=[float(s) if n<1e-10 else None for s,n in zip(sigma,null_weight)],
                scaled_motion_axes=vh,parameter_scale=scale)
        result['motion_coordinates']=['excess_path_rate_mps','excess_path_acceleration_mps2','excess_path_jerk_mps3',
            'counterclockwise_angular_rate_degps','angular_acceleration_degps2']
        result['scope']=self.information.__doc__
        return result

    def fit(self,initial,*,max_iterations=180,free=None,position_bound_m=6.,progress=None):
        scale,limit=self.settings(position_bound_m)
        free=np.arange(self.size) if free is None else np.asarray(free)
        bounds=list(zip(-limit[free]/scale[free],limit[free]/scale[free]))
        seed=np.asarray(initial).copy()
        evaluations=0;iterations=0
        def objective(q):
            nonlocal evaluations
            evaluations+=1
            p=seed.copy();p[free]=q*scale[free]
            value,gradient=self.objective(p);return value,gradient[free]*scale[free]
        def checkpoint(intermediate_result):
            nonlocal iterations
            iterations+=1;p=seed.copy();p[free]=intermediate_result.x*scale[free]
            progress(dict(iterations=iterations,evaluations=evaluations,parameters=p,objective=float(intermediate_result.fun)))
        fit=minimize(objective,seed[free]/scale[free],jac=True,method='L-BFGS-B',bounds=bounds,
            callback=checkpoint if progress is not None else None,
            options=dict(maxiter=max_iterations,maxcor=20,gtol=1e-8,ftol=2e-13))
        p=seed.copy();p[free]=fit.x*scale[free];gradient=fit.jac.copy()
        active=np.isclose(abs(p[free]),limit[free],atol=1e-6)
        gradient[((p[free]<=-limit[free]+1e-6)&(gradient>0))|((p[free]>=limit[free]-1e-6)&(gradient<0))]=0
        return dict(parameters=p,objective=float(fit.fun),success=bool(fit.success),iterations=fit.nit,
            maximum_scaled_projected_gradient=float(max(abs(gradient))),active_bounds=free[active],free_indices=free,
            evaluations=fit.nfev,message=str(fit.message))
