"""Rigid motion with common reflector phases and band-dependent brightness."""
import numpy as np
from scipy.optimize import minimize,LinearConstraint

from shared_scene import phase_coherent


class PhasePatch:
    def __init__(self,bands,initial_motion):
        self.bands=bands;self.points=bands[0].points.copy();self.count=len(self.points)
        self.size=5+2*(self.count-1);self.initial=np.r_[initial_motion,np.zeros(self.size-5)]
        self.energy=sum(b.energy for b in bands)

    def unpack(self,p):
        points=self.points.copy();points[1:,:2]+=np.asarray(p)[5:].reshape(-1,2)
        return np.asarray(p)[:5],points

    def evaluate(self,p,*,derivatives=True,precondition=False):
        motion,points=self.unpack(p);local=[];gram=[];rhs=[]
        for b in self.bands:
            h,dh=b.design(motion,points=points,derivatives=derivatives,point_derivatives=True)
            a=b.project(h).reshape(-1,self.count)[b.selected]
            da=b.project(dh).reshape(-1,self.count,7)[b.selected] if derivatives else None
            local.append((a,da));gram.append(a.conj().T@a);rhs.append(a.conj().T@b.data)
        scene=phase_coherent(gram,rhs,[b.energy for b in self.bands])
        result=dict(value=scene['cost']/self.energy,scene=scene)
        if not derivatives:return result
        gradient=np.zeros(self.size);jacobian=[]
        for c,((a,da),b) in enumerate(zip(local,self.bands)):
            coefficient=scene['coefficient'][c];residual=b.data-a@coefficient
            fixed=np.zeros((len(a),self.size),complex)
            fixed[:,:5]=np.einsum('ncp,c->np',da[:,:,:5],coefficient)
            for j in range(1,self.count):fixed[:,5+2*(j-1):5+2*j]=coefficient[j]*da[:,j,5:]
            gradient-=2*np.real(fixed.conj().T@residual)/self.energy
            if precondition:
                # This is solely an invertible optimizer coordinate choice.
                # The likelihood still uses the constrained coherent scene.
                q,_=np.linalg.qr(a,mode='reduced');fixed-=q@(q.conj().T@fixed)
                jacobian.append(np.r_[fixed.real,fixed.imag]/np.sqrt(self.energy))
        result['gradient']=gradient
        if precondition:result['preconditioner_jacobian']=np.concatenate(jacobian)
        return result

    def validation(self,p):
        fitted=self.evaluate(p,derivatives=False);motion,points=self.unpack(p);records=[]
        for c,b in enumerate(self.bands):
            h,_=b.design(motion,points=points);selected=np.repeat(~b.training,len(b.field.offsets))
            a=b.project(h,~b.training).reshape(-1,self.count)[selected]
            y=b.project(b.field.y,~b.training).ravel()[selected];coefficient=fitted['scene']['coefficient'][c]
            prediction=a@coefficient;free=a@np.linalg.lstsq(a,y,rcond=1e-10)[0]
            records.append(dict(unused_rss=np.vdot(y-prediction,y-prediction).real,
                unused_data_energy=np.vdot(y,y).real,unused_prediction_energy=np.vdot(prediction,prediction).real,
                unused_independent_scene_energy=np.vdot(free,free).real,
                unused_complex_prediction_overlap=abs(np.vdot(prediction,free))**2/(np.vdot(prediction,prediction).real*np.vdot(free,free).real)))
        return records

    def fit(self,*,bounds,max_iterations=200):
        initial=self.initial.copy();scale=np.r_[.05,.005,.0005,.05,.005,np.full(self.size-5,.5)]
        r=self.evaluate(initial,precondition=True);_,singular,vh=np.linalg.svd(r['preconditioner_jacobian']*scale,full_matrices=False)
        # A floor only limits the optimizer step amplification. Every physical
        # parameter is retained, and no penalty enters the statistical model.
        floor=max(singular[0]*1e-6,1e-12);transform=scale[:,None]*vh.T/np.maximum(singular,floor)
        cache=[None,None];evaluations=0
        def objective(q):
            nonlocal evaluations
            if cache[0] is None or not np.array_equal(q,cache[0]):
                value=self.evaluate(initial+transform@q);evaluations+=1
                cache[:]=[q.copy(),(value['value'],value['gradient']@transform)]
            return cache[1]
        fitted=minimize(objective,np.zeros(self.size),jac=True,method='SLSQP',
            constraints=[LinearConstraint(transform/scale[:,None],(np.asarray(bounds[0])-initial)/scale,(np.asarray(bounds[1])-initial)/scale)],
            options={'maxiter':max_iterations,'ftol':1e-13})
        p=initial+transform@fitted.x;r=self.evaluate(p);gradient=r['gradient']*scale
        gradient=np.where(p<np.asarray(bounds[0])+1e-7,np.minimum(gradient,0),
            np.where(p>np.asarray(bounds[1])-1e-7,np.maximum(gradient,0),gradient))
        motion,points=self.unpack(p);scene=r['scene']
        return p,dict(parameters=p.tolist(),motion_parameters=motion.tolist(),points_east_north_up_m=points.tolist(),
            gain_phase_real_imag=np.c_[scene['gain'].real,scene['gain'].imag].tolist(),
            coefficient_real_imag=np.stack([scene['coefficient'].real,scene['coefficient'].imag],axis=-1).tolist(),
            objective=r['value'],maximum_scaled_gradient=float(abs(gradient).max()),
            success=bool(fitted.success and abs(gradient).max()<1e-7 and scene['converged']),
            inner_phase_gradient=scene['maximum_gradient'],evaluations=evaluations,message=fitted.message,
            optimizer_singular_values=singular.tolist(),optimizer_step_floor=floor,validation=self.validation(p))
